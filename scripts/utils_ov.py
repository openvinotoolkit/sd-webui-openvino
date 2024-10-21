# Copyright (C) 2024-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from modules import scripts


controlnet_extension_directory = scripts.basedir() + '/../sd-webui-controlnet'
sys.path.append(controlnet_extension_directory)
from scripts.hook import mark_prompt_context, unmark_prompt_context, POSITIVE_MARK_TOKEN, NEGATIVE_MARK_TOKEN, MARK_EPS
