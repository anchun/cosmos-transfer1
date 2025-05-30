# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

import cosmos_transfer1.diffusion.config.registry as base_registry
from cosmos_transfer1.diffusion.config.transfer.conditioner import (
    CTRL_HINT_KEYS,
    BaseVideoConditionerWithCtrlConfig,
    VideoConditionerFpsSizePaddingWithCtrlConfig,
    ViewConditionedVideoConditionerFpsSizePaddingWithCtrlConfig,
)
from cosmos_transfer1.diffusion.config.transfer.net_ctrl import FADITV2EncoderConfig, FADITV2MultiCamEncoderConfig


def register_experiment_ctrlnet(cs):
    """
    transfer model related registry: controlnet architecture, hint keys, etc.
    """
    # TODO: maybe we should change the registered 'name' (faditv2_7b) here; it's the dit-encoder for net_ctrl
    # but current naming is the same as the full DiT in the main 'net' group that's defined
    # in cosmos_transfer1/diffusion/config/registry.py. Isn't an error but could be confusing.
    cs.store(group="net_ctrl", package="model.net_ctrl", name="faditv2_7b", node=FADITV2EncoderConfig)
    cs.store(group="net_ctrl", package="model.net_ctrl", name="faditv2_7b_mv", node=FADITV2MultiCamEncoderConfig)

    cs.store(group="conditioner", package="model.conditioner", name="ctrlnet", node=BaseVideoConditionerWithCtrlConfig)
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="ctrlnet_add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingWithCtrlConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="view_cond_ctrlnet_add_fps_image_size_padding_mask",
        node=ViewConditionedVideoConditionerFpsSizePaddingWithCtrlConfig,
    )
    for hint_key in CTRL_HINT_KEYS:
        cs.store(
            group="hint_key",
            package="model",
            name=hint_key,
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=False)),
        )
        cs.store(
            group="hint_key",
            package="model",
            name=f"{hint_key}_grayscale",
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=True)),
        )


def register_configs():
    cs = ConfigStore.instance()
    base_registry.register_configs()
    register_experiment_ctrlnet(cs)
