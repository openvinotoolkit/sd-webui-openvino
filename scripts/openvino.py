# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: AGPL-3.0
from contextlib import closing
import cv2
import os
import logging
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
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing


import openvino
from openvino.runtime import Core, Type, PartialShape, serialize
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from scripts.global_state_openvino import model_state, pipes
df_pipe = pipes['diffusers']
ov_model = pipes['openvino']

#from scripts.utils_ov import ControlModelType
controlnet_extension_directory = scripts.basedir() + '/../sd-webui-controlnet'
sys.path.append(controlnet_extension_directory)
from scripts.utils import load_state_dict, get_state_dict
#from scripts.controlnet import Script as ControlNetScript
from scripts.utils_ov import mark_prompt_context, unmark_prompt_context, POSITIVE_MARK_TOKEN, NEGATIVE_MARK_TOKEN, MARK_EPS#, get_control

#ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging = logging.getLogger("OpenVINO")
logging.setLevel("INFO")

def cond_stage_key(self):
    return None

#sdxl invisible-watermark pixel artifact workaround
class NoWatermark:
    def apply_watermark(self, img):
        return img


def on_change(mode):
    return gr.update(visible=mode)


class OVUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str):
        self.label = f"[OV] {name}"
        self.model_name = name
        self.configs = None

    def create_unet(self):
        return OVUnet(self.model_name)



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
        logging.info('forward called')
        if y is not None:
            logging.info('sd-xl detected')
            prompt = self.process.prompt
            negative_prompt = self.process.negative_prompt
            device = df_pipe._execution_device
            lora_scale = None #(
            
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
            logging.info("standard(not xl) model detected")
            added_cond_kwargs = {}
        
        down_block_res_samples, mid_block_res_sample = None, None 
        if self.has_controlnet:
            logging.info("controlnet detected")
            cond_mark, self.current_uc_indices, self.current_c_indices, context = unmark_prompt_context(context)

            for i in range(len(pipes['openvino'].control_models)):
                logging.info(f"control model: {pipes['openvino'].control_models[i]}")
                control_model = pipes['openvino'].control_models[i]
                image = pipes['openvino'].control_images[i]
                control_model = torch.compile(pipes['openvino'].controlnet, backend = 'openvino', options = opt)  # ControlNetModel.from_single_file('./extensions/sd-webui-controlnet/models/control_v11p_sd15_canny_fp16.safetensors', local_files_only=True)
                down_samples, mid_sample = control_model(
                    x,
                    timesteps,
                    encoder_hidden_states=context,
                    controlnet_cond=image,
                    conditioning_scale=1.0,
                    guess_mode = False,
                    return_dict=False,
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
            
            logging.info(f"controlnet output shapes:{[d.shape for d in down_block_res_samples], mid_block_res_sample.shape}")
        else:
            logging.info('no controlnet detected')
        
        noise_pred = pipes['openvino'].unet(
                x,
                timesteps,
                context,
                class_labels = y, # sd-xl class_labels
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample, 
                ).sample

        return noise_pred
    
    def apply_loras(self, refit_dict: dict):
        if not self.refitted_keys.issubset(set(refit_dict.keys())):
            # TODO: Need to ensure that weights that have been modified before and are not present anymore are reset.
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
        vae_scale_factor = 2 ** (len(pipes['openvino'].vae.config.block_out_channels) - 1)
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
        logging.info('unload the already loaded unet')
        from modules.sd_models import model_data, send_model_to_trash, SdModelData
        
        if model_data.sd_model:
            model_data.sd_model.state_dict().clear()
            devices.torch_gc()
            logging.info('unloaded A1111 loaded model')

        if pipes['openvino'] is not None:
            logging.info('del old pipes[openvino]')
            del pipes['openvino']
            del pipes['diffusers']
            gc.collect()
            pipes['openvino'] = None
            pipes['diffusers'] = None
            logging.info('del old OV_df_pipe and df_pipe')

        ov_model = OVUnet(p)
        pipes['diffusers'] = None
        
        #### controlnet ####
        logging.info(p.extra_generation_params)

        
        logging.info('loading OV unet model')
        checkpoint_name = checkpoint or shared.opts.sd_model_checkpoint.split(" ")[0]
        checkpoint_path = os.path.join(scripts.basedir(), 'models', 'Stable-diffusion', checkpoint_name)
        checkpoint_info = CheckpointInfo(checkpoint_path)
        timer = Timer()
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
        logging.info("created model from config : " + checkpoint_config)
        if 'xl' not in checkpoint_config:
            logging.info('load sd pipeline')
            pipes['diffusers'] = StableDiffusionPipeline.from_single_file(checkpoint_path, original_config_file=checkpoint_config, use_safetensors=True, variant="fp32", dtype=torch.float32)
        else:
            logging.info('load sd-xl pipeline')
            pipes['diffusers'] = StableDiffusionXLPipeline.from_single_file(checkpoint_path, original_config_file=checkpoint_config, use_safetensors=True, variant="fp32", dtype=torch.float32)
        ov_model.unet = pipes['diffusers'].unet.to("cpu") # OV device should only be in the options for torch.compile
        ov_model.unet = torch.compile(ov_model.unet, backend="openvino", options = opt)
        ov_model.vae = pipes['diffusers'].vae.to("cpu")

        if loaded_vae_file is not None:
            logging.info('hook the vae')
            ov_model.vae = AutoencoderKL.from_single_file(loaded_vae_file, local_files_only=True)
        else:
            ov_model.vae = pipes['diffusers'].vae.to("cpu")
        
        ov_model.vae.decode = torch.compile(ov_model.vae.decode, backend="openvino", options = opt)
        
        
        ov_model.has_controlnet = False
        cn_model="None"
        control_models = []
        control_images = []
        from internal_controlnet.external_code import ControlNetUnit, Preprocessor
        from scripts.enums import ControlModelType
        for param in p.script_args: 
            if isinstance(param, ControlNetUnit): 
                if param.enabled == False: continue
                
                model_name = param.model.split(' ')[0] 
                control_models.append(model_name)

                cn_model_dir_path = os.path.join(scripts.basedir(),'extensions','sd-webui-controlnet','models')
                cn_model_path = os.path.join(cn_model_dir_path, model_name)
                if os.path.isfile(cn_model_path + '.pt'):
                    cn_model_path = cn_model_path + '.pt'
                elif os.path.isfile(cn_model_path + '.safetensors'):
                    cn_model_path = cn_model_path + '.safetensors'
                elif os.path.isfile(cn_model_path + '.pth'):
                    cn_model_path = cn_model_path + '.pth'
                    
                # parse controlnet type
                control_model_type = None
                logging.info('load state_dict')
                state_dict = load_state_dict(cn_model_path)
                if "lora_controlnet" in state_dict:
                    control_model_type =  ControlModelType.ControlLoRA
                elif "down_blocks.0.motion_modules.0.temporal_transformer.norm.weight" in state_dict:
                    control_model_type = ControlModelType.SparseCtrl
                elif "control_add_embedding.linear_1.bias" in state_dict: # Controlnet Union

                    control_model_type = ControlModelType.ControlNetUnion
                elif "instant_id" in cn_model_path.lower():
                    control_model_type = ControlModelType.InstantID
                else:
                    control_model_type = ControlModelType.ControlNet
                
                if control_model_type == ControlModelType.ControlNet:
                    
                    controlnet = ControlNetModel.from_single_file(cn_model_path, local_files_only=False)
                    ov_model.controlnet = controlnet
                else:
                    assert False, f"Control model type {control_model_type} is not supported."
                
                
                logging.info(' this is supported by OV, disable enabled units to avoid controlnet extension hook')
                param.enabled = False # disable param.enabled, controlnet extension cannot find enabled units
                # preprocess the image before, adapted from controlnet extension
                unit = param
                preprocessor = Preprocessor.get_preprocessor(unit.module)
                logging.info(f"preprocessor: {preprocessor}")
                
                
                if unit.ipadapter_input is not None:
                    logging.info(f"ipadapter_input is not None: {unit.ipadapter_input}")
                    # Use ipadapter_input from API call.
                    #assert control_model_type == ControlModelType.IPAdapter
                    controls = unit.ipadapter_input
                    hr_controls = unit.ipadapter_input
                else:
                    logging.info('unit.ipdadapter_input is None, call default get_control')
                    from controlnet import get_control
                    controls, hr_controls, additional_maps = get_control(  # get the controlnet input? Yes, preprocess
                        p, unit, 0, ControlModelType.ControlNet, preprocessor)
                    control_images.append(controls[0])

        
        if not control_images: 
            logging.info('NO CNET detected, unet build finished')
            pipes['openvino'] = ov_model
            return

        logging.info('controlnet detected')
        ov_model.has_controlnet = True
        ov_model.control_models = control_models
        ov_model.control_images = control_images
        
        logging.info(f"model_state.control_models: {','.join(model_state.control_models)}")
        
        logging.info('begin loading controlnet model(s)')
        
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
            ov_model.controlnet = controlnet
        else:
            cn_model_dir_path = os.path.join(scripts.basedir(),'extensions','sd-webui-controlnet','models')
            cn_model_path = os.path.join(cn_model_dir_path, model_state.control_models[0])
            if os.path.isfile(cn_model_path + '.pt'):
                cn_model_path = cn_model_path + '.pt'
            elif os.path.isfile(cn_model_path + '.safetensors'):
                cn_model_path = cn_model_path + '.safetensors'
            elif os.path.isfile(cn_model_path + '.pth'):
                cn_model_path = cn_model_path + '.pth'
        
        pipes['openvino'] = ov_model
        # process image
        for i in range(len(ov_model.control_images)):
            ov_model.control_images[i] = OVUnet.prepare_image(ov_model.control_images[i], 512, 512, 1, 1, torch.device('cpu'), torch.float32, True, False)
        sd_unet.current_unet = ov_model
        logging.info('end of activate')
        
    def deactivate(self):
        logging.info('deactivate called')
        del globals_state['openvino']
        del globals_state['diffusers']
        gc.collect()
        globals_state['openvino'] = None
        globals_state['diffusers'] = None
        
        logging.info('deactivate finished')


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
            logging.info(f'  change enable to {status}')
            model_state.enable_ov_extension = status

        def openvino_device_handler(status):
            logging.info(f'change enable to {status}')
            model_state.device = status

        def enable_caching_handler(status):
            logging.info(f'change enable to {status}')
            model_state.enable_caching = status

            
        with gr.Accordion('OV Extension', open=False):
            
            enable_ov_extension= gr.Checkbox(label="Enable OpenVINO acceleration", value=False)
            openvino_device = gr.Dropdown(label="Select a device", choices=list(core.available_devices), value = 'CPU')
            enable_caching = gr.Checkbox(label="Cache the compiled models", value=False)

            enable_ov_extension.change(fn=enable_ov_extension_handler, inputs=enable_ov_extension, outputs=enable_ov_extension_status)
            openvino_device.change(fn=openvino_device_handler, inputs=openvino_device, outputs=openvino_device_status)
            enable_caching.change(fn=enable_caching_handler, inputs=enable_caching, outputs=enable_caching_status)
        
        
        return [enable_ov_extension, openvino_device, enable_caching]
    
    
    def after_extra_networks_activate(self, p,  *args, **kwargs):
        # https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras
        if not model_state.enable_ov_extension: 
            logging.info('  after_extra_networks_activate: not enabled, return ')
            return
        
        names = []
        scales = []
        
        for lora_param in p.extra_network_data['lora']:
            name = lora_param.positional[0]
            scale = float(lora_param.positional[1])
            names.append(name)
            scales.append(scale)
            
        if not names: return
        names.sort()
        
        if pipes['openvino'] and names and pipes['openvino'].lora_fused[''.join(names)] == True: return
        pipes['openvino'].lora_fused[''.join(names)] = True

        for name in names:
            pipes['diffusers'].load_lora_weights(os.path.join(os.getcwd(), "models", "Lora"), weight_name=name + ".safetensors", adapter_name=name, low_cpu_mem_usage=True)

        pipes['diffusers'].set_adapters(names, adapter_weights=scales)
        pipes['diffusers'].fuse_lora(adapter_names=names, lora_scale=1.0)
        pipes['diffusers'].unload_lora_weights()
    
    def build_unet(self, p):
        # only called when recompile is needed
        OVUnet.activate(p) # reload the engine
        
    def process(self, p, *args):
        logging.info("ov process called")
        current_extension_directory = scripts.basedir() + '/extensions/sd-webui-controlnet/scripts'
        logging.info(f"current_extension_directory: {current_extension_directory}")
        sys.path.append(current_extension_directory)
        logging.info(f"p.extra_network_data: {p.extra_network_data}")
        logging.info(f"p.extra_generation_params: {p.extra_generation_params}")
        logging.info(f"modules.extra_networks.extra_network_registry: {modules.extra_networks.extra_network_registry}")

        enable_ov = model_state.enable_ov_extension
        openvino_device = model_state.device # device
        enable_caching = model_state.enable_caching

        if not enable_ov:
            logging.info('ov disabled, do nothing')
            self.restore_unet(p)
            self.restore_vae(p)
            return
        
        global opt
        opt_new = dict()
        opt_new["device"] = openvino_device.upper()
        opt_new["model_caching"] = True if enable_ov and enable_caching else False
        
        dir_path = "./model_cache" 
        if enable_caching:
            os.makedirs(dir_path, exist_ok=True)
            opt_new["cache_dir"] = dir_path
            
        else:
            if "cache_dir" in opt_new: del opt_new["cache_dir"]
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                logging.info(f"Directory '{dir_path}' and its contents have been removed.")
            else:
                logging.info(f"Directory '{dir_path}' does not exist.")
        
        if opt_new != opt:
            model_state.recompile = True
        
        
        current_control_models = []
        from internal_controlnet import external_code  # noqa: F403
        from internal_controlnet.external_code import ControlNetUnit, Preprocessor
        for param in p.script_args: 
            if isinstance(param, ControlNetUnit): 
                if param.enabled == False: continue
                current_control_models.append(param.model.split(' ')[0])

        if model_state.model_name != p.sd_model_name or opt_new != opt or current_control_models != model_state.control_models:
            model_state.recompile = True
            logging.info('set recompile to true')
            model_state.control_models = current_control_models
            
        model_state.refiner_ckpt = p.refiner_checkpoint
        model_state.model_name = p.sd_model_name
        opt = opt_new
        if model_state.recompile:
            logging.info('recompile')
            try:
                self.build_unet(p) # rebuild unet from safe tensors
            except Exception as e:
                logging.info(f'Error build_unet: {e}, fallback to default unet')
                return
        
        self.apply_unet(p) # hook the forward function of unet, do this for every process call
    
        self.apply_vae(p) # hook the decode_first_stage function of vae, do this for every process call 
        
        logging.info('ov enabled')
        logging.info(f"p.sd_model_name:{p.sd_model_name}")

        
    def apply_unet(self, p):
        if sd_unet.current_unet is not None and not isinstance(sd_unet.current_unet, OVUnet):
            sd_unet.current_unet.deactivate()
        sd_ldm = p.sd_model
        model = sd_ldm.model.diffusion_model
        model._original_forward = model.forward
        logging.info('force forward ')
        model.forward = pipes['openvino'].forward
        logging.info('finish force forward ')
    
    def apply_vae(self, p):
        def vaehook(img):
            logging.info('hooked vae called')
            return ov_model.vae.decode(img/OV_df_vae.config.scaling_factor, return_dict = False)[0]
        shared.sd_model._original_decode_first_stage = shared.sd_model.decode_first_stage
        shared.sd_model.decode_first_stage = vaehook
        
    def restore_unet(self,p):
        if sd_unet.current_unet is not None:
            sd_unet.current_unet.deactivate()
        sd_unet.current_unet = None

        sd_ldm = p.sd_model
        model = sd_ldm.model.diffusion_model
        if hasattr(model, "_original_forward"):
            model.forward = model._original_forward 
            del model._original_forward
        
        logging.info('restore unet')
    
    def restore_vae(self, p):
        if hasattr(shared.sd_model, "_original_decode_first_stage"):
            shared.sd_model.decode_first_stage = shared.sd_model._original_decode_first_stage
            del shared.sd_model._original_decode_first_stage
        logging.info('restore vae')

def refiner_cb_fn(args):
    if model_state.enable_ov_extension and pipes['openvino'] and pipes['openvino'].process.extra_generation_params.get('Refiner', False):
        refiner_filename = model_state.refiner_ckpt.split(' ')[0]
        logging.info(f"reload refiner checkpoint: {refiner_filename}")
        OVUnet.activate(pipes['openvino'].process, refiner_filename)
        
script_callbacks.on_model_loaded(refiner_cb_fn)