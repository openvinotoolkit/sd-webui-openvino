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

pipes = {'diffusers': None, 'openvino': None}
df_pipe = None # pipeline from single file of diffusers

OV_df_pipe = None # OpenVINO diffusers pipeline

OV_df_vae = None # vae diffusers model