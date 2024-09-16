import torch
import sys
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing
from modules import scripts
from modules import processing, shared,  masking, devices
import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum
from PIL import Image, ImageOps
from einops import rearrange

controlnet_extension_directory = scripts.basedir() + '/extensions/sd-webui-controlnet/scripts'

POSITIVE_MARK_TOKEN = 1024
NEGATIVE_MARK_TOKEN = - POSITIVE_MARK_TOKEN
MARK_EPS = 1e-3


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
    Unknown = "Unknown"

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
        #TODO: fix multiple seeds in x y z plot or prompt matrix
        tmp_seed = int(p.all_seeds[0]) # if len(p.seed) == 1 and p.seed == -1 else max((p.seed), 0))
        tmp_subseed = int(p.all_seeds[0]) # if len(p.subseed) == 1 and p.subseed == -1 else max((p.subseed), 0))
        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
        np.random.seed(seed)
        return seed
    except Exception as e:
        logger.warning(e)
        logger.warning('Warning: Failed to use consistent random seed.')
        return None


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
    
def align_dim_latent(x: int) -> int:
    """Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel, i.e.,
    1 latent unit == 8 pixel unit."""
    return (x // 8) * 8

def get_control(
    p: StableDiffusionProcessing,
    unit, #: ControlNetUnit,
    idx: int,
    control_model_type, #: ControlModelType,
    preprocessor, #: Preprocessor,
):
    """Get input for a ControlNet unit."""
    '''
    if unit.is_animate_diff_batch:
        unit = add_animate_diff_batch_input(p, unit)
    '''

    #sys.path.append(controlnet_extension_directory)

    #from internal_controlnet import external_code  # noqa: F403

    high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)
    h, w, hr_y, hr_x = get_target_dimensions(p)
    input_image, resize_mode = choose_input_image(p, unit, 0)
    #print('input_image:', input_image) # good
    if isinstance(input_image, list):
        print('input image is list')
        assert unit.accepts_multiple_inputs or unit.is_animate_diff_batch
        input_images = input_image
    else: # Following operations are only for single input image.
        print('input image not list')
        input_image = try_crop_image_with_a1111_mask(p, unit, input_image, resize_mode)
        input_image = np.ascontiguousarray(input_image.copy()).copy() # safe numpy
        if unit.module == 'inpaint_only+lama' and resize_mode == ResizeMode.OUTER_FIT:
            # inpaint_only+lama is special and required outpaint fix
            _, input_image = detectmap_proc(input_image, unit.module, resize_mode, hr_y, hr_x)
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
            control, detected_map = detectmap_proc(detected_map, unit.module, resize_mode, h, w)
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