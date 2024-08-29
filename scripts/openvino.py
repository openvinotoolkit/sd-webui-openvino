# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: AGPL-3.0
from contextlib import closing
import cv2
import os
import torch
import time
import functools
from collections import defaultdict
from enum import Enum
import gradio as gr
from modules.ui import plaintext_to_html
import numpy as np
import sys, gc, os, shutil
from PIL import Image, ImageOps
from typing import Dict, Optional, Tuple, List
from einops import rearrange

import modules
import modules.paths as paths
from modules import scripts, script_callbacks
from modules import processing, sd_unet
from modules import images, devices, extra_networks, masking, shared, sd_models_config, prompt_parser
from modules.sd_vae import vae_dict, base_vae, loaded_vae_file
from modules.processing import (
    StableDiffusionProcessing, Processed, apply_overlay, apply_color_correction,
    get_fixed_seed, create_infotext, setup_color_correction
)
from modules.sd_models import CheckpointInfo, get_checkpoint_state_dict
from modules.shared import opts, state
from modules.ui_common import create_refresh_button
from modules.timer import Timer

import openvino
from openvino.runtime import Core, Type, PartialShape, serialize
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    AutoencoderKL,
)

current_extension_directory = scripts.basedir() + '/../sd-webui-controlnet'
print('current_extension_directory', current_extension_directory)
sys.path.append(current_extension_directory)
#from internal_controlnet import external_code  # noqa: F403
#from internal_controlnet.external_code import ControlNetUnit, Preprocessor

#ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




class ModelState:
    def __init__(self):
        self.enable_ov_extension = False
        self.enable_caching = False
        self.recompile = True
        self.device = "CPU"
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.mode = 0
        self.partition_id = 0
        self.model_name = ""
        self.control_models = []
        self.is_sdxl = False
        self.lora_model = "None"
        self.vae_ckpt = "None"
        self.refiner_ckpt = "None"
        
        
model_state = ModelState()

def cond_stage_key(self):
    return None



#sdxl invisible-watermark pixel artifact workaround
class NoWatermark:
    def apply_watermark(self, img):
        return img



def on_change(mode):
    return gr.update(visible=mode)

from modules.processing import StableDiffusionProcessing

POSITIVE_MARK_TOKEN = 1024
NEGATIVE_MARK_TOKEN = - POSITIVE_MARK_TOKEN
MARK_EPS = 1e-3


def prompt_context_is_marked(x):
    t = x[..., 0, :]
    m = torch.abs(t) - POSITIVE_MARK_TOKEN
    m = torch.mean(torch.abs(m)).detach().cpu().float().numpy()
    return float(m) < MARK_EPS


def mark_prompt_context(x, positive):
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = mark_prompt_context(x[i], positive)
        return x
    if isinstance(x, MulticondLearnedConditioning):
        x.batch = mark_prompt_context(x.batch, positive)
        return x
    if isinstance(x, ComposableScheduledPromptConditioning):
        x.schedules = mark_prompt_context(x.schedules, positive)
        return x
    if isinstance(x, ScheduledPromptConditioning):
        if isinstance(x.cond, dict):
            cond = x.cond['crossattn']
            if prompt_context_is_marked(cond):
                return x
            mark = POSITIVE_MARK_TOKEN if positive else NEGATIVE_MARK_TOKEN
            cond = torch.cat([torch.zeros_like(cond)[:1] + mark, cond], dim=0)
            return ScheduledPromptConditioning(end_at_step=x.end_at_step, cond=dict(crossattn=cond, vector=x.cond['vector']))
        else:
            cond = x.cond
            if prompt_context_is_marked(cond):
                return x
            mark = POSITIVE_MARK_TOKEN if positive else NEGATIVE_MARK_TOKEN
            cond = torch.cat([torch.zeros_like(cond)[:1] + mark, cond], dim=0)
            return ScheduledPromptConditioning(end_at_step=x.end_at_step, cond=cond)
    return x


disable_controlnet_prompt_warning = True
# You can disable this warning using disable_controlnet_prompt_warning.


def unmark_prompt_context(x):
    if not prompt_context_is_marked(x):
        # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
        # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
        # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
        # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
        # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
        # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
        # After you mark the prompts, the mismatch errors will disappear.
        if not disable_controlnet_prompt_warning:
            logger.warning('ControlNet Error: Failed to detect whether an instance is cond or uncond!')
            logger.warning('ControlNet Error: This is mainly because other extension(s) blocked A1111\'s \"process.sample()\" and deleted ControlNet\'s sample function.')
            logger.warning('ControlNet Error: ControlNet will shift to a backup backend but the results will be worse than expectation.')
            logger.warning('Solution (For extension developers): Take a look at ControlNet\' hook.py '
                  'UnetHook.hook.process_sample and manually call mark_prompt_context to mark cond/uncond prompts.')
        mark_batch = torch.ones(size=(x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
        context = x
        return mark_batch, [], [], context
    mark = x[:, 0, :]
    context = x[:, 1:, :]
    mark = torch.mean(torch.abs(mark - NEGATIVE_MARK_TOKEN), dim=1)
    mark = (mark > MARK_EPS).float()
    mark_batch = mark[:, None, None, None].to(x.dtype).to(x.device)

    mark = mark.detach().cpu().numpy().tolist()
    uc_indices = [i for i, item in enumerate(mark) if item < 0.5]
    c_indices = [i for i, item in enumerate(mark) if not item < 0.5]

    StableDiffusionProcessing.cached_c = [None, None]
    StableDiffusionProcessing.cached_uc = [None, None]

    return mark_batch, uc_indices, c_indices, context

df_pipe = None # pipeline from single file of diffusers

OV_df_pipe = None # OpenVINO diffusers pipeline

OV_df_vae = None # vae diffusers model

class OVUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str):
        self.label = f"[OV] {name}"
        self.model_name = name
        self.configs = None

    def create_unet(self):
        return OVUnet(self.model_name)


from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing

def set_numpy_seed(p: processing.StableDiffusionProcessing) -> Optional[int]:
    """
    Set the random seed for NumPy based on the provided parameters.

    Args:
        p (processing.StableDiffusionProcessing): The instance of the StableDiffusionProcessing class.

    Returns:
        Optional[int]: The computed random seed if successful, or None if an exception occurs.

    This function sets the random seed for NumPy using the seed and subseed values from the given instance of
    StableDiffusionProcessing. If either seed or subseed is -1, it uses the first value from `all_seeds`.
    Otherwise, it takes the maximum of the provided seed value and 0.

    The final random seed is computed by adding the seed and subseed values, applying a bitwise AND operation
    with 0xFFFFFFFF to ensure it fits within a 32-bit integer.
    """
    try:
        # to do: fix multiple seeds in x y z plot or prompt matrix
        tmp_seed = int(p.all_seeds[0]) # if len(p.seed) == 1 and p.seed == -1 else max((p.seed), 0))
        tmp_subseed = int(p.all_seeds[0]) # if len(p.subseed) == 1 and p.subseed == -1 else max((p.subseed), 0))
        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
        np.random.seed(seed)
        return seed
    except Exception as e:
        logger.warning(e)
        logger.warning('Warning: Failed to use consistent random seed.')
        return None

class ControlModelType(Enum):
    """
    The type of Control Models (supported or not).
    """

    ControlNet = "ControlNet, Lvmin Zhang"
    T2I_Adapter = "T2I_Adapter, Chong Mou"
    T2I_StyleAdapter = "T2I_StyleAdapter, Chong Mou"
    T2I_CoAdapter = "T2I_CoAdapter, Chong Mou"
    MasaCtrl = "MasaCtrl, Mingdeng Cao"
    GLIGEN = "GLIGEN, Yuheng Li"
    AttentionInjection = "AttentionInjection, Lvmin Zhang"  # A simple attention injection written by Lvmin
    StableSR = "StableSR, Jianyi Wang"
    PromptDiffusion = "PromptDiffusion, Zhendong Wang"
    ControlLoRA = "ControlLoRA, Wu Hecong"
    ReVision = "ReVision, Stability"
    IPAdapter = "IPAdapter, Hu Ye"
    Controlllite = "Controlllite, Kohya"
    InstantID = "InstantID, Qixun Wang"
    SparseCtrl = "SparseCtrl, Yuwei Guo"

    @property
    def is_controlnet(self) -> bool:
        """Returns whether the control model should be treated as ControlNet."""
        return self in (
            ControlModelType.ControlNet,
            ControlModelType.ControlLoRA,
            ControlModelType.InstantID,
        )

    @property
    def allow_context_sharing(self) -> bool:
        """Returns whether this control model type allows the same PlugableControlModel
        object map to multiple ControlNetUnit.
        Both IPAdapter and Controlllite have unit specific input (clip/image) stored
        on the model object during inference. Sharing the context means that the input
        set earlier gets lost.
        """
        return self not in (
            ControlModelType.IPAdapter,
            ControlModelType.Controlllite,
        )

    @property
    def supports_effective_region_mask(self) -> bool:
        return (
            self
            in {
                ControlModelType.IPAdapter,
                ControlModelType.T2I_Adapter,
            }
            or self.is_controlnet
        )

class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"

def get_control(
    p: StableDiffusionProcessing,
    unit, #: ControlNetUnit,
    idx: int,
    control_model_type, #: ControlModelType,
    preprocessor, #: Preprocessor,
):
    sys.path.append(current_extension_directory)
    from internal_controlnet import external_code  # noqa: F403
    """Get input for a ControlNet unit."""
    '''
    if unit.is_animate_diff_batch:
        unit = add_animate_diff_batch_input(p, unit)
    '''

    high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)
    h, w, hr_y, hr_x = Script.get_target_dimensions(p)
    input_image, resize_mode = Script.choose_input_image(p, unit, 0)
    #print('input_image:', input_image) # good
    if isinstance(input_image, list):
        print('input image is list')
        assert unit.accepts_multiple_inputs or unit.is_animate_diff_batch
        input_images = input_image
    else: # Following operations are only for single input image.
        print('input image not list')
        input_image = Script.try_crop_image_with_a1111_mask(p, unit, input_image, resize_mode)
        input_image = np.ascontiguousarray(input_image.copy()).copy() # safe numpy
        if unit.module == 'inpaint_only+lama' and resize_mode == ResizeMode.OUTER_FIT:
            # inpaint_only+lama is special and required outpaint fix
            _, input_image = Script.detectmap_proc(input_image, unit.module, resize_mode, hr_y, hr_x)
        input_images = [input_image]
        print('image after crop:', input_images)

    if unit.pixel_perfect:
        unit.processor_res = external_code.pixel_perfect_resolution(
            input_images[0],
            target_H=h,
            target_W=w,
            resize_mode=resize_mode,
        )
    # Preprocessor result may depend on numpy random operations, use the
    # random seed in `StableDiffusionProcessing` to make the
    # preprocessor result reproducable.
    # Currently following preprocessors use numpy random:
    # - shuffle
    seed = set_numpy_seed(p)
    print(f"Use numpy seed {seed}.")
    print(f"Using preprocessor: {unit.module}")
    print(f'preprocessor resolution = {unit.processor_res}')

    detected_maps = []
    def store_detected_map(detected_map, module: str) -> None:
        if unit.save_detected_map:
            detected_maps.append((detected_map, module))

    def preprocess_input_image(input_image: np.ndarray):
        """ Preprocess single input image. """
        #print('input befre preprocessor cached call:', input_image) # good
        print('preprocessor:', preprocessor)
        result = preprocessor.cached_call(
            input_image,
            resolution=unit.processor_res,
            slider_1=unit.threshold_a,
            slider_2=unit.threshold_b,
            low_vram=(
                ("clip" in unit.module or unit.module == "ip-adapter_face_id_plus") and
                shared.opts.data.get("controlnet_clip_detector_on_cpu", False)
            ),
            model=unit.model,
        )
        print('result.value:',result.value)
        print('result.display_images:',result.display_images)
        detected_map = result.value
        is_image = preprocessor.returns_image
        # TODO: Refactor img control detection logic.
        if high_res_fix:
            if is_image:
                hr_control, hr_detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, hr_y, hr_x)
                store_detected_map(hr_detected_map, unit.module)
            else:
                hr_control = detected_map
        else:
            hr_control = None

        if is_image:
            print('is_image is true')
            print('detected map:', detected_map) # None
            control, detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
            print('detected map after:', detected_map) # None
            print('control after detectmap_proc =', control)
            store_detected_map(detected_map, unit.module)
        else:
            control = detected_map
            for image in result.display_images:
                store_detected_map(image, unit.module)

        if control_model_type == ControlModelType.T2I_StyleAdapter:
            control = control['last_hidden_state']

        if control_model_type == ControlModelType.ReVision:
            control = control['image_embeds']

        if is_image and unit.is_animate_diff_batch: # AnimateDiff save VRAM
            control = control.cpu()
            if hr_control is not None:
                hr_control = hr_control.cpu()

        return control, hr_control

    def optional_tqdm(iterable, use_tqdm=unit.is_animate_diff_batch):
        from tqdm import tqdm
        return tqdm(iterable) if use_tqdm else iterable

    controls, hr_controls = list(zip(*[preprocess_input_image(img) for img in optional_tqdm(input_images)]))
    assert len(controls) == len(hr_controls)
    return controls, hr_controls, detected_maps


#from diffusers import DiffusionPipeline 



class OVUnet(sd_unet.SdUnet):
    def __init__(self, p: processing.StableDiffusionProcessing):
        super().__init__()
        self.model_name = p.sd_model_name
        self.process = p
        #self.configs = configs
        self.loaded_config = None
        self.lora_fused = defaultdict(bool)
        self.unet = None
        self.control_models = []
        self.control_images = []
        self.vae = None
        self.current_uc_indices = None
        self.current_c_indices = None
        self.has_controlnet = False
    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):

        print('[OV] forward called, x: min(x), max(x)', torch.min(x), torch.max(x))
        if y is not None:
            print('sd-xl detected')
            prompt = self.process.prompt
            negative_prompt = self.process.negative_prompt
            device = df_pipe._execution_device
            lora_scale = None #(
            #df_pipe.cross_attention_kwargs.get("scale", None) if df_pipe.cross_attention_kwargs is not None else None
            #)
            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
            
            num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
            
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = df_pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True, #df_pipe.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                prompt_embeds=None, #prompt_embeds,
                negative_prompt_embeds=None, #negative_prompt_embeds,
                pooled_prompt_embeds=None, #pooled_prompt_embeds,
                negative_pooled_prompt_embeds=None, #negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=None,#df_pipe.clip_skip,
            )

            
            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            #extra_step_kwargs = df_pipe.prepare_extra_step_kwargs(generator, eta)

            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            if df_pipe.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = df_pipe.text_encoder_2.config.projection_dim

            add_time_ids = df_pipe._get_add_time_ids(
                (1024, 1024),#original_size,
                (0,0),#crops_coords_top_left,
                (1024,1024), #target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            negative_target_size = kwargs.get("negative_target_size", None)
            negative_original_size = kwargs.get("negative_original_size", None)
            if negative_original_size is not None and negative_target_size is not None:
                negative_add_time_ids = self._get_add_time_ids(
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
            else:
                negative_add_time_ids = add_time_ids

            if True: #df_pipe.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
                
            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            
        else:
            print('sd 1.5 detected')
            added_cond_kwargs = {}
        
        down_block_res_samples, mid_block_res_sample = None, None 
        if self.has_controlnet:
            print('controlnet detected')
            cond_mark, self.current_uc_indices, self.current_c_indices, context = unmark_prompt_context(context)

            for i in range(len(OV_df_pipe.control_models)):
                print('control model:', OV_df_pipe.control_models[i])
                control_model = OV_df_pipe.control_models[i]
                image = OV_df_pipe.control_images[i]
                print('image min and max:', torch.min(image), torch.max(image))
                print('x min and max:', torch.min(x), torch.max(x)) #  tensor(-13.3997) tensor(11.1092)
                print('context min and max:', torch.min(context), torch.max(context),'shape:', context.shape ) #  tensor(-1024.) tensor(1024.)  [2, 78, 768])
                control_model = torch.compile(OV_df_pipe.controlnet, backend = 'openvino', options = opt)  # ControlNetModel.from_single_file('./extensions/sd-webui-controlnet/models/control_v11p_sd15_canny_fp16.safetensors', local_files_only=True)
                down_samples, mid_sample = control_model(
                    x,
                    timesteps,
                    encoder_hidden_states=context,
                    controlnet_cond=image,
                    conditioning_scale=1.0,
                    guess_mode = False,
                    return_dict=False,
                    #y=y
                )
                # merge samples
                if i == 0:
                    down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
                else:
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                    ]
                    mid_block_res_sample += mid_sample
            ''''
            [diffusers ref]
            image min and max:  tensor(0.) tensor(1.)
            control model input  min and max:  tensor(-3.3987) tensor(2.9278)
            controlnet_prompt_embeds, min and max: tensor(-28.0912) tensor(33.0632)
            cond_scale: 1.0
            guess_mode: False
            min_block sample min and max: tensor(-18.6162) tensor(23.7768)
            '''
            print('mid_block_res_sample min and max:', torch.min(mid_block_res_sample), torch.max(mid_block_res_sample))
            print([d.shape for d in down_block_res_samples], mid_block_res_sample.shape)
            print('check diffusers  controlnet output')
        else:
            print('no controlnet detected')
        
        noise_pred = OV_df_pipe.unet(
                x,
                timesteps,
                context,
                class_labels = y, # sd-xl class_labels
                #timestep_cond=timestep_cond,
                #cross_attention_kwargs=({}),
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample, 
                #return_dict=False,
                ).sample

        return noise_pred
    
    def apply_loras(self, refit_dict: dict):
        if not self.refitted_keys.issubset(set(refit_dict.keys())):
            # Need to ensure that weights that have been modified before and are not present anymore are reset.
            self.refitted_keys = set()
            self.switch_engine()

        self.engine.refit_from_dict(refit_dict, is_fp16=True)
        self.refitted_keys = set(refit_dict.keys())

    def switch_engine(self):
        self.loaded_config = self.configs[self.profile_idx]
        self.engine.reset(os.path.join(OV_MODEL_DIR, self.loaded_config["filepath"]))
        self.activate(p)
    @staticmethod 
    def prepare_image(
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):  
        from diffusers.image_processor import VaeImageProcessor
        vae_scale_factor = 2 ** (len(OV_df_pipe.vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
        control_image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        image = control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # re-load weights and and compile or recompile the model
    @staticmethod
    def activate(p, checkpoint = None):

        print('[OV]unload the already loaded unet')
        from modules.sd_models import model_data, send_model_to_trash
        
        
        global df_pipe, OV_df_pipe

        
        
        if model_data.sd_model:
            #send_model_to_trash(model_data.sd_model)
            #model_data.sd_model = None
            model_data.sd_model.state_dict().clear()
            devices.torch_gc()
            print('[OV] finished unloading model')
        
        
        if OV_df_pipe is not None:
            print('del old OV_df_pipe')
            del OV_df_pipe
            del df_pipe
            gc.collect()
            print('del old OV_df_pipe and df_pipe')

        OV_df_pipe = OVUnet(p)
        df_pipe = None
        
        # model state
        #### controlnet ####
        print(p.extra_generation_params)

        
        print('loading OV unet model')
        checkpoint_name = checkpoint or shared.opts.sd_model_checkpoint.split(" ")[0]
        checkpoint_path = os.path.join(scripts.basedir(), 'models', 'Stable-diffusion', checkpoint_name)
        checkpoint_info = CheckpointInfo(checkpoint_path)
        timer = Timer()
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
        print("OpenVINO Extension:  created model from config : " + checkpoint_config)
        if 'xl' not in checkpoint_config:
            print('load sd1 or sd2 pipeline')
            df_pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, original_config_file=checkpoint_config, use_safetensors=True, variant="fp32", dtype=torch.float32)
        else:
            print('load sd-xl pipeline')
            df_pipe = StableDiffusionXLPipeline.from_single_file(checkpoint_path, original_config_file=checkpoint_config, use_safetensors=True, variant="fp32", dtype=torch.float32)
        OV_df_pipe.unet = df_pipe.unet.to("cpu") # OV device should only be in the options for torch.compile
        OV_df_pipe.unet = torch.compile(OV_df_pipe.unet, backend="openvino", options = opt)
        OV_df_pipe.vae = df_pipe.vae.to("cpu")
        print('OpenVINO Extension: loaded unet model')
            
        OV_df_pipe.has_controlnet = False
        cn_model="None"
        control_models = []
        control_images = []
        from internal_controlnet.external_code import ControlNetUnit, Preprocessor
        for param in p.script_args: 
            if isinstance(param, ControlNetUnit): 
                if param.enabled == False: continue
                print('[OV]disable enabled units')
                param.enabled = False # disable param.enabled, controlnet extension cannot find enabled units
                control_models.append(param.model.split(' ')[0])
                # preprocess the image before, adapted from controlnet extension
                unit = param
                preprocessor = Preprocessor.get_preprocessor(unit.module)
                print('preprocessor:', preprocessor)
                
                
                if unit.ipadapter_input is not None:
                    print('ipadapter_input is not None:', unit.ipadapter_input)
                    # Use ipadapter_input from API call.
                    #assert control_model_type == ControlModelType.IPAdapter
                    controls = unit.ipadapter_input
                    hr_controls = unit.ipadapter_input
                else:
                    print('unit.ipdadapter_input is None, call default get_control')
                    controls, hr_controls, additional_maps = get_control(  # get the controlnet input? Yes, preprocess
                        p, unit, 0, ControlModelType.ControlNet, preprocessor)
                    control_images.append(controls[0])
                
        if not control_images: 
            print('[OV] NO CNET detected')
            return

        print('[OV] cnet detected')
        OV_df_pipe.has_controlnet = True
        
        OV_df_pipe.control_models = control_models
        OV_df_pipe.control_images = control_images
        
        print("model_state.control_models:", model_state.control_models)
        
        print('begin loading controlnet model(s)')
        
        if (len(model_state.control_models) > 1):
            controlnet = []
            for cn_model in model_state.control_models:
                cn_model_dir_path = os.path.join(scripts.basedir(),'extensions','sd-webui-controlnet','models')
                cn_model_path = os.path.join(cn_model_dir_path, cn_model)
                if os.path.isfile(cn_model_path + '.pt'):
                    cn_model_path = cn_model_path + '.pt'
                elif os.path.isfile(cn_model_path + '.safetensors'):
                    cn_model_path = cn_model_path + '.safetensors'
                elif os.path.isfile(cn_model_path + '.pth'):
                    cn_model_path = cn_model_path + '.pth'
                controlnet.append(ControlNetModel.from_single_file(cn_model_path, local_files_only=True))
            OV_df_pipe.controlnet = controlnet
        else:
            cn_model_dir_path = os.path.join(scripts.basedir(),'extensions','sd-webui-controlnet','models')
            cn_model_path = os.path.join(cn_model_dir_path, model_state.control_models[0])
            if os.path.isfile(cn_model_path + '.pt'):
                cn_model_path = cn_model_path + '.pt'
            elif os.path.isfile(cn_model_path + '.safetensors'):
                cn_model_path = cn_model_path + '.safetensors'
            elif os.path.isfile(cn_model_path + '.pth'):
                cn_model_path = cn_model_path + '.pth'

            controlnet = ControlNetModel.from_single_file(cn_model_path, local_files_only=False)
            OV_df_pipe.controlnet = controlnet
        
        # process image
        print('begin self.prepare_image')
        for i in range(len(OV_df_pipe.control_images)):
            OV_df_pipe.control_images[i] = OVUnet.prepare_image(OV_df_pipe.control_images[i], 512, 512, 1, 1, torch.device('cpu'), torch.float32, True, False)
        print('end self.prepare_image')
        sd_unet.current_unet = OV_df_pipe
        print('end of activate')
        

    def deactivate(self):
        print('[OV]deactivate called')
        #del self.unet


        
def align_dim_latent(x: int) -> int:
    """Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel, i.e.,
    1 latent unit == 8 pixel unit."""
    return (x // 8) * 8

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

    
def prepare_mask(
    mask: Image.Image, p: processing.StableDiffusionProcessing
) -> Image.Image:
    """
    Prepare an image mask for the inpainting process.

    This function takes as input a PIL Image object and an instance of the 
    StableDiffusionProcessing class, and performs the following steps to prepare the mask:

    1. Convert the mask to grayscale (mode "L").
    2. If the 'inpainting_mask_invert' attribute of the processing instance is True,
       invert the mask colors.
    3. If the 'mask_blur' attribute of the processing instance is greater than 0,
       apply a Gaussian blur to the mask with a radius equal to 'mask_blur'.

    Args:
        mask (Image.Image): The input mask as a PIL Image object.
        p (processing.StableDiffusionProcessing): An instance of the StableDiffusionProcessing class 
                                                   containing the processing parameters.

    Returns:
        mask (Image.Image): The prepared mask as a PIL Image object.
    """
    mask = mask.convert("L")
    if getattr(p, "inpainting_mask_invert", False):
        mask = ImageOps.invert(mask)

    if hasattr(p, 'mask_blur_x'):
        if getattr(p, "mask_blur_x", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
            mask = Image.fromarray(np_mask)
        if getattr(p, "mask_blur_y", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
            mask = Image.fromarray(np_mask)
    else:
        if getattr(p, "mask_blur", 0) > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    return mask

def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]

def nake_nms(x):
    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    y = np.zeros_like(x)
    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
    return y

def lvmin_thin(x, prunings=True):
    y = x
    for i in range(32):
        y, is_done = thin_one_time(y, lvmin_kernels)
        if is_done:
            break
    if prunings:
        y, _ = thin_one_time(y, lvmin_prunings)
    return y


def get_pytorch_control(x: np.ndarray) -> torch.Tensor:
    # A very safe method to make sure that Apple/Mac works
    y = x

    # below is very boring but do not change these. If you change these Apple or Mac may fail.
    y = torch.from_numpy(y)
    y = y.float() / 255.0
    y = rearrange(y, 'h w c -> 1 c h w')
    y = y.clone()
    y = y.to(devices.get_device_for("controlnet"))
    y = y.clone()
    return y


opt = dict() # openvino pytorch compile options
class Script(scripts.Script):
    def title(self):
        return "stable-diffusion-webui-extension-openvino"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        core = Core()

        def get_config_list():
            config_dir_list = os.listdir(os.path.join(os.getcwd(), 'configs'))
            config_list = []
            config_list.append("None")
            for file in config_dir_list:
                if file.endswith('.yaml'):
                    config_list.append(file)
            return config_list
        def get_vae_list():
            vae_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'VAE'))
            vae_list = []
            vae_list.append("None")
            vae_list.append("Disable-VAE-Acceleration")
            for file in vae_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    vae_list.append(file)
            return vae_list
        def get_refiner_list():
            refiner_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'Stable-diffusion'))
            refiner_list = []
            refiner_list.append("None")
            for file in refiner_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    refiner_list.append(file)
            return refiner_list
        
        enable_ov_extension_status = gr.Textbox(label="enable", interactive=False, visible=False)
        openvino_device_status = gr.Textbox(label="device", interactive=False, visible=False)
        enable_caching_status = gr.Textbox(label="cache", interactive=False, visible=False) 
        
        def enable_ov_extension_handler(status):
            print(f'change enable to {status}')
            model_state.enable_ov_extension = status

        def openvino_device_handler(status):
            print(f'change enable to {status}')
            model_state.device = status

        def enable_caching_handler(status):
            print(f'change enable to {status}')
            model_state.enable_caching = status

            
        with gr.Accordion('OV Extension', open=False):
            
            enable_ov_extension= gr.Checkbox(label="Enable OpenVINO acceleration", value=False)
            openvino_device = gr.Dropdown(label="Select a device", choices=list(core.available_devices), value = 'CPU')
            enable_caching = gr.Checkbox(label="Cache the compiled models", value=False)

            enable_ov_extension.change(fn=enable_ov_extension_handler, inputs=enable_ov_extension, outputs=enable_ov_extension_status)
            openvino_device.change(fn=openvino_device_handler, inputs=openvino_device, outputs=openvino_device_status)
            enable_caching.change(fn=enable_caching_handler, inputs=enable_caching, outputs=enable_caching_status)
        

            
            

        
        def enable_change(choice):
                if choice:
                    processing._process_images = processing.process_images
                    print("enable vo extension")
                    processing.process_images = self.run
                else:
                    if hasattr(processing, '_process_images'):
                        processing.process_images = processing._process_images
                    print('disable ov extension')
        
        
        def device_change(choice):
            if (model_state.device == choice):
                return gr.update(value="Device selected is " + choice, visible=True)
            else:
                model_state.device = choice
                model_state.recompile = 1
                return gr.update(value="Device changed to " + choice + ". Model will be re-compiled", visible=True)
        #openvino_device.change(device_change, openvino_device, warmup_status)
        def vae_change(choice):
            if (model_state.vae_ckpt == choice):
                return gr.update(value="vae_ckpt selected is " + choice, visible=True)
            else:
                model_state.vae_ckpt = choice
                model_state.recompile = 1
                return gr.update(value="Custom VAE changed to " + choice + ". Model will be re-compiled", visible=True)
        #vae_ckpt.change(vae_change, vae_ckpt, vae_status)
        def refiner_ckpt_change(choice):
            if (model_state.refiner_ckpt == choice):
                return gr.update(value="Custom Refiner selected is " + choice, visible=True)
            else:
                model_state.refiner_ckpt = choice
        #refiner_ckpt.change(refiner_ckpt_change, refiner_ckpt)
        return [enable_ov_extension, openvino_device, enable_caching]
    
    
    @staticmethod
    def try_crop_image_with_a1111_mask(
        p: StableDiffusionProcessing,
        unit,#: ControlNetUnit,
        input_image: np.ndarray,
        resize_mode,#: ResizeMode,
    ) -> np.ndarray:
        """
        Crop ControlNet input image based on A1111 inpaint mask given.
        This logic is crutial in upscale scripts, as they use A1111 mask + inpaint_full_res
        to crop tiles.
        """
        # Note: The method determining whether the active script is an upscale script is purely
        # based on `extra_generation_params` these scripts attach on `p`, and subject to change
        # in the future.
        # TODO: Change this to a more robust condition once A1111 offers a way to verify script name.
        is_upscale_script = any("upscale" in k.lower() and "Hires" not in k for k in getattr(p, "extra_generation_params", {}).keys())
        print(f"is_upscale_script={is_upscale_script}")
        # Note: `inpaint_full_res` is "inpaint area" on UI. The flag is `True` when "Only masked"
        # option is selected.
        a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
        is_only_masked_inpaint = (
            issubclass(type(p), StableDiffusionProcessingImg2Img) and
            p.inpaint_full_res and
            a1111_mask_image is not None
        )
        if (
            'reference' not in unit.module
            and is_only_masked_inpaint
            and (is_upscale_script or unit.inpaint_crop_input_image)
        ):
            print("Crop input image based on A1111 mask.")
            input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
            input_image = [Image.fromarray(x) for x in input_image]

            mask = prepare_mask(a1111_mask_image, p)

            crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

            input_image = [
                images.resize_image(resize_mode.int_value(), i, mask.width, mask.height)
                for i in input_image
            ]

            input_image = [x.crop(crop_region) for x in input_image]
            input_image = [
                images.resize_image(ResizeMode.OUTER_FIT.int_value(), x, p.width, p.height)
                for x in input_image
            ]

            input_image = [np.asarray(x)[:, :, 0] for x in input_image]
            input_image = np.stack(input_image, axis=2)
        return input_image
    
    @staticmethod
    def detectmap_proc(detected_map, module, resize_mode, h, w):

        if 'inpaint' in module:
            detected_map = detected_map.astype(np.float32)
        else:
            detected_map = HWC3(detected_map)

        def safe_numpy(x):
            # A very safe method to make sure that Apple/Mac works
            y = x

            # below is very boring but do not change these. If you change these Apple or Mac may fail.
            y = y.copy()
            y = np.ascontiguousarray(y)
            y = y.copy()
            return y

        def high_quality_resize(x, size):
            # Written by lvmin
            # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

            inpaint_mask = None
            if x.ndim == 3 and x.shape[2] == 4:
                inpaint_mask = x[:, :, 3]
                x = x[:, :, 0:3]

            if x.shape[0] != size[1] or x.shape[1] != size[0]:
                new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
                new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
                unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
                is_one_pixel_edge = False
                is_binary = False
                if unique_color_count == 2:
                    is_binary = np.min(x) < 16 and np.max(x) > 240
                    if is_binary:
                        xc = x
                        xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                        xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                        one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                        all_edge_count = np.where(x > 127)[0].shape[0]
                        is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

                if 2 < unique_color_count < 200:
                    interpolation = cv2.INTER_NEAREST
                elif new_size_is_smaller:
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

                y = cv2.resize(x, size, interpolation=interpolation)
                if inpaint_mask is not None:
                    inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

                if is_binary:
                    y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
                    if is_one_pixel_edge:
                        y = nake_nms(y)
                        _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        y = lvmin_thin(y, prunings=new_size_is_bigger)
                    else:
                        _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    y = np.stack([y] * 3, axis=2)
            else:
                y = x

            if inpaint_mask is not None:
                inpaint_mask = (inpaint_mask > 127).astype(np.float32) * 255.0
                inpaint_mask = inpaint_mask[:, :, None].clip(0, 255).astype(np.uint8)
                y = np.concatenate([y, inpaint_mask], axis=2)

            return y

        if resize_mode == ResizeMode.RESIZE:
            detected_map = high_quality_resize(detected_map, (w, h))
            detected_map = safe_numpy(detected_map)
            return get_pytorch_control(detected_map), detected_map

        old_h, old_w, _ = detected_map.shape
        old_w = float(old_w)
        old_h = float(old_h)
        k0 = float(h) / old_h
        k1 = float(w) / old_w

        safeint = lambda x: int(np.round(x))

        if resize_mode == ResizeMode.OUTER_FIT:
            k = min(k0, k1)
            borders = np.concatenate([detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0)
            high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
            if len(high_quality_border_color) == 4:
                # Inpaint hijack
                high_quality_border_color[3] = 255
            high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
            detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
            new_h, new_w, _ = detected_map.shape
            pad_h = max(0, (h - new_h) // 2)
            pad_w = max(0, (w - new_w) // 2)
            high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = detected_map
            detected_map = high_quality_background
            detected_map = safe_numpy(detected_map)
            return get_pytorch_control(detected_map), detected_map
        else:
            k = max(k0, k1)
            detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
            new_h, new_w, _ = detected_map.shape
            pad_h = max(0, (new_h - h) // 2)
            pad_w = max(0, (new_w - w) // 2)
            detected_map = detected_map[pad_h:pad_h+h, pad_w:pad_w+w]
            detected_map = safe_numpy(detected_map)
            return get_pytorch_control(detected_map), detected_map

    @staticmethod
    def get_target_dimensions(p: StableDiffusionProcessing) -> Tuple[int, int, int, int]:
        """Returns (h, w, hr_h, hr_w)."""
        h = align_dim_latent(p.height)
        w = align_dim_latent(p.width)

        high_res_fix = (
            isinstance(p, StableDiffusionProcessingTxt2Img)
            and getattr(p, 'enable_hr', False)
        )
        if high_res_fix:
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                hr_y = int(p.height * p.hr_scale)
                hr_x = int(p.width * p.hr_scale)
            else:
                hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
            hr_y = align_dim_latent(hr_y)
            hr_x = align_dim_latent(hr_x)
        else:
            hr_y = h
            hr_x = w

        return h, w, hr_y, hr_x

    @staticmethod
    def get_remote_call(p, attribute, default=None, idx=0, strict=False, force=False):
        if not force and not shared.opts.data.get("control_net_allow_script_control", False):
            return default

        def get_element(obj, strict=False):
            if not isinstance(obj, list):
                return obj if not strict or idx == 0 else None
            elif idx < len(obj):
                return obj[idx]
            else:
                return None

        attribute_value = get_element(getattr(p, attribute, None), strict)
        return attribute_value if attribute_value is not None else default

    @staticmethod
    def choose_input_image(
            p: processing.StableDiffusionProcessing,
            unit,#: ControlNetUnit,
            idx: int
        ) -> Tuple[np.ndarray, ResizeMode]:
        """ Choose input image from following sources with descending priority:
         - p.image_control: [Deprecated] Lagacy way to pass image to controlnet.
         - p.control_net_input_image: [Deprecated] Lagacy way to pass image to controlnet.
         - unit.image: ControlNet unit input image.
         - p.init_images: A1111 img2img input image.

        Returns:
            - The input image in ndarray form.
            - The resize mode.
        """
        def from_rgba_to_input(img: np.ndarray) -> np.ndarray:
            if (
                shared.opts.data.get("controlnet_ignore_noninpaint_mask", False) or
                (img[:, :, 3] <= 5).all() or
                (img[:, :, 3] >= 250).all()
            ):
                # Take RGB
                return img[:, :, :3]
            print("Canvas scribble mode. Using mask scribble as input.")
            return HWC3(img[:, :, 3])

        # 4 input image sources.
        p_image_control = getattr(p, "image_control", None)
        #p_input_image = Script.get_remote_call(p, "control_net_input_image", None, idx)
        image = unit.get_input_images_rgba()
        a1111_image = getattr(p, "init_images", [None])[0]
        print('image, a111-iamge:', image, a1111_image) # good image, None

        resize_mode = unit.resize_mode

        '''
        if batch_hijack.instance.is_batch and p_image_control is not None:
            logger.warning("Warn: Using legacy field 'p.image_control'.")
            input_image = HWC3(np.asarray(p_image_control))
        elif p_input_image is not None:
            logger.warning("Warn: Using legacy field 'p.controlnet_input_image'")
            if isinstance(p_input_image, dict) and "mask" in p_input_image and "image" in p_input_image:
                color = HWC3(np.asarray(p_input_image['image']))
                alpha = np.asarray(p_input_image['mask'])[..., None]
                input_image = np.concatenate([color, alpha], axis=2)
            else:
                input_image = HWC3(np.asarray(p_input_image))
        '''
        if image is not None:
            print('unit.get_input_images_rgba() is not none')
            assert isinstance(image, list)
            # Inpaint mask or CLIP mask.
            if unit.is_inpaint or unit.uses_clip:
                # RGBA
                input_image = image
            else:
                # RGB
                input_image = [from_rgba_to_input(img) for img in image]

            if len(input_image) == 1:
                input_image = input_image[0]
        elif a1111_image is not None:
            print(' a1111_image is not none')
            input_image = HWC3(np.asarray(a1111_image))
            a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
            assert a1111_i2i_resize_mode is not None
            resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)

            a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
            if unit.is_inpaint:
                if a1111_mask_image is not None:
                    a1111_mask = np.array(prepare_mask(a1111_mask_image, p))
                    assert a1111_mask.ndim == 2
                    assert a1111_mask.shape[0] == input_image.shape[0]
                    assert a1111_mask.shape[1] == input_image.shape[1]
                    input_image = np.concatenate([input_image[:, :, 0:3], a1111_mask[:, :, None]], axis=2)
                else:
                    input_image = np.concatenate([
                        input_image[:, :, 0:3],
                        np.zeros_like(input_image, dtype=np.uint8)[:, :, 0:1],
                    ], axis=2)
        else:
            # No input image detected.
            #if batch_hijack.instance.is_batch:
            #    shared.state.interrupted = True
            raise ValueError("controlnet is enabled but no input image is given")

        assert isinstance(input_image, (np.ndarray, list))
        return input_image, resize_mode

    
    def after_extra_networks_activate(self, p,  *args, **kwargs):
        
        # https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras
        if not model_state.enable_ov_extension: 
            print('[OV] after_extra_networks_activate: not enabled, return ')
            return
        
        names = []
        scales = []
        
        for lora_param in p.extra_network_data['lora']:
            name = lora_param.positional[0]
            scale = float(lora_param.positional[1])
            names.append(name)
            scales.append(scale)
        names.sort()
        
        if OV_df_pipe.lora_fused[''.join(names)] == True: return
        OV_df_pipe.lora_fused[''.join(names)] = True

        global df_pipe
            
        for name in names:
            df_pipe.load_lora_weights(os.path.join(os.getcwd(), "models", "Lora"), weight_name=name + ".safetensors", adapter_name=name, low_cpu_mem_usage=True)

        df_pipe.set_adapters(names, adapter_weights=scales)
        
        df_pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
        
        df_pipe.unload_lora_weights()
    
    def build_unet(self, p):
        OVUnet.activate(p) # reload the engine
        
        
        
        

    def process(self, p, *args):
        print("[OV]ov process called")
        model_state.refiner_ckpt = p.refiner_checkpoint
        model_state.model_name = p.sd_model_name
        
        from modules import scripts
        current_extension_directory = scripts.basedir() + '/extensions/sd-webui-controlnet/scripts'
        print('current_extension_directory', current_extension_directory)
        sys.path.append(current_extension_directory)
        print('[OV]p.extra_network_data:\n' ,p.extra_network_data)
        print('[OV]p.extra_generation_params:\n', p.extra_generation_params)
        print('[OV] modules.extra_networks.extra_network_registry:', modules.extra_networks.extra_network_registry)

        enable_ov = model_state.enable_ov_extension
        openvino_device = model_state.device # device
        enable_caching = model_state.enable_caching

        if not enable_ov:
            print('ov disabled, do nothing')
            self.restore_unet(p)
            return
        
        global opt
        opt_new = dict()
        opt_new["device"] = openvino_device.upper()
        opt_new["model_caching"] = True if enable_ov and enable_caching else False
        
        dir_path = "./model_cache" 
        if enable_caching:
            opt_new["cache_dir"] = dir_path
            os.makedirs(dir_path, exist_ok=True)
            
        else:
            if "cache_dir" in opt_new: del opt_new["cache_dir"]
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' and its contents have been removed.")
            else:
                print(f"Directory '{dir_path}' does not exist.")
        
        if opt_new != opt:
            model_state.recompile = True
        else:
            model_state.recompile = False
        
        current_control_models = []
        from internal_controlnet import external_code  # noqa: F403
        from internal_controlnet.external_code import ControlNetUnit, Preprocessor
        for param in p.script_args: 
            if isinstance(param, ControlNetUnit): 
                if param.enabled == False: continue
                current_control_models.append(param.model.split(' ')[0])
        if current_control_models != model_state.control_models:
            model_state.recompile = True
            print('set recompile to true due to control models change')
            model_state.control_models = current_control_models
        
        
        
        opt = opt_new

        if model_state.recompile:
            print('[OV]recompile')
            self.build_unet(p) # rebuild unet from safe tensors
        
        self.apply_unet(p) # hook the forward function of unet, do this for every process call
        

        # to do: add feature to fallback default vae after replacement
        if loaded_vae_file is not None:
            print('hook the vae')
            global OV_df_vae
            if OV_df_vae is None: 
                OV_df_vae = AutoencoderKL.from_single_file(loaded_vae_file, local_files_only=True)
            def vaehook(img):
                print('hooked vae called')
                OV_df_vae.decode = torch.compile(OV_df_vae.decode, backend="openvino", options = opt)
                return OV_df_vae.decode(img/OV_df_vae.config.scaling_factor, return_dict = False)[0]
            shared.sd_model.decode_first_stage = vaehook
        else:
            print('no custom vae loaded')
        
        
        print('ov enabled')
        
        print("p.sd_model_name:",p.sd_model_name)

        
    def apply_unet(self, p):
        if sd_unet.current_unet is not None:
            sd_unet.current_unet.deactivate()
        
        
        
        sd_ldm = p.sd_model
        model = sd_ldm.model.diffusion_model

        model._original_forward = model.forward
        
        
        
        
        
        print('force forward ')
        model.forward = OV_df_pipe.forward
        print('finish force forward ')
    
    def restore_unet(self,p):
        if sd_unet.current_unet is not None:
            sd_unet.current_unet.deactivate()
        sd_unet.current_unet = None

        sd_ldm = p.sd_model
        model = sd_ldm.model.diffusion_model
        if hasattr(model, "_original_forward"):
            model.forward = model._original_forward 
            del model._original_forward
        
        
        print('restore unet')


def refiner_cb_fn(args):
    
    if model_state.enable_ov_extension and OV_df_pipe and OV_df_pipe.process.extra_generation_params.get('Refiner', False):
        refiner_filename = model_state.refiner_ckpt.split(' ')[0]
        print('[OV] reload refiner checkpoint:',refiner_filename)
        OVUnet.activate(OV_df_pipe.process, refiner_filename)
    elif model_state.enable_ov_extension and OV_df_pipe:
        print("[OV]re-apply main model weights")
        OVUnet.activate(OV_df_pipe.process)
        
script_callbacks.on_model_loaded(refiner_cb_fn)