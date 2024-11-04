# Copyright (C) 2024-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from modules import scripts
import os

controlnet_extension_directory = scripts.basedir() + '/../sd-webui-controlnet'
is_controlnet_extension_installed = os.path.exists(controlnet_extension_directory)

if is_controlnet_extension_installed: 
    sys.path.append(controlnet_extension_directory)
    from scripts.hook import mark_prompt_context, unmark_prompt_context, POSITIVE_MARK_TOKEN, NEGATIVE_MARK_TOKEN, MARK_EPS
