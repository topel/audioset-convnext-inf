#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from audioset_convnext_inf.pytorch.convnext import convnext_tiny, ConvNeXt
from safetensors.torch import load_model, save_model

model_fpath = "/gpfswork/rech/djl/uzj43um/audioset-convnext-inf/checkpoints/convnext_tiny_471mAP.pth"

model = ConvNeXt.from_pretrained(model_fpath, use_auth_token=None, map_location='cpu')

print(
    "# params:",
    sum(param.numel() for param in model.parameters() if param.requires_grad),
)

save_model(model, "model.safetensors")



