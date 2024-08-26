
from modules.modelloader import load_file_from_url
from modules import shared, script_callbacks, errors
#import sd_hijack_autoencoder  # noqa: F401
#import sd_hijack_ddpm_v1  # noqa: F401

import os

from modules import modelloader, errors
from modules.shared import cmd_opts, opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, path):
        self.name = "RealESRGAN[OV]"
        self.user_path = path
        super().__init__()
        #scaler_data = UpscalerData("LDSR", None, self)
        #self.scalers = [scaler_data]
        self.enable = True
        self.scalers = []
        scalers = get_realesrgan_models(self)

        local_model_paths = self.find_models(ext_filter=[".pth"])
        for scaler in scalers:
            if scaler.local_data_path.startswith("http"):
                filename = modelloader.friendly_name(scaler.local_data_path)
                local_model_candidates = [local_model for local_model in local_model_paths if local_model.endswith(f"{filename}.pth")]
                if local_model_candidates:
                    scaler.local_data_path = local_model_candidates[0]

            if scaler.name[:-5] in opts.realesrgan_enabled_models:
                self.scalers.append(scaler)

    def do_upscale(self, img, path):
        if not self.enable:
            return img

        try:
            info = self.load_model(path)
        except Exception:
            errors.report(f"Unable to load RealESRGAN model {path}", exc_info=True)
            return img

        model_descriptor = modelloader.load_spandrel_model(
            info.local_data_path,
            device=self.device,
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture="ESRGAN",  # "RealESRGAN" isn't a specific thing for Spandrel
        )
        import torch
        model = model_descriptor.model
        model = torch.compile(model, backend="openvino")

        return upscale_with_model(
            model,
            img,
            tile_size=opts.ESRGAN_tile,
            tile_overlap=opts.ESRGAN_tile_overlap,
            # TODO: `outscale`?
        )

    def load_model(self, path):
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"RealESRGAN data missing: {scaler.local_data_path}")
                return scaler
        raise ValueError(f"Unable to find model info: {path}")


def get_realesrgan_models(scaler: UpscalerRealESRGAN):
    return [
        UpscalerData(
            name="R-ESRGAN General 4xV3 [OV]",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN General WDN 4xV3 [OV]",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN AnimeVideo [OV]",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+ [OV]",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+ Anime6B [OV]",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 2x+ [OV]",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            scale=2,
            upscaler=scaler,
        ),
    ]

