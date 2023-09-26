#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp

# import librosa
import numpy as np
import torch
import torchaudio

from audioset_convnext_inf.pytorch.convnext import ConvNeXt
from audioset_convnext_inf.utils.utilities import read_audioset_label_tags

# three options: 1) the ckpt is already on disk, 2) use Zenodo, 3) use the HF hub model

# Model from local disk
# model_fpath = "/gpfswork/rech/djl/uzj43um/audioset-convnext-inf/checkpoints/convnext_tiny_471mAP.pth"
# model_fpath = '/gpfswork/rech/djl/uzj43um/audioset-convnext-inf/checkpoints/convnext_tiny_465mAP_BL_AC_70kit.pth'

# Model from Zenodo
# model_fpath = "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1"

# Model from HF model.safetensors
model_fpath="topel/ConvNeXt-Tiny-AT"

AUDIO_FNAME = "f62-S-v2swA_200000_210000.wav"
AUDIO_FPATH = osp.join("/gpfswork/rech/djl/uzj43um/audioset-convnext-inf", "audio_samples", AUDIO_FNAME)

model = ConvNeXt.from_pretrained(model_fpath, use_auth_token=None, map_location='cpu')


print(
    "# params:",
    sum(param.numel() for param in model.parameters() if param.requires_grad),
)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if "cuda" in str(device):
    model = model.to(device)

sample_rate = 32000
audio_target_length = 10 * sample_rate  # 10 s

print("\nInference on " + AUDIO_FNAME + "\n")

waveform, sample_rate_ = torchaudio.load(AUDIO_FPATH)
if sample_rate_ != sample_rate:
    print("ERROR: sampling rate not 32k Hz", sample_rate_)


waveform = waveform.to(device)

# Forward
with torch.no_grad():
    model.eval()
    output = model(waveform)

logits = output["clipwise_logits"]
print("logits size:", logits.size())

# probs = torch.sigmoid(logits)
probs = output["clipwise_output"]
print("probs size:", probs.size())

current_dir=os.getcwd()
lb_to_ix, ix_to_lb, id_to_ix, ix_to_id = read_audioset_label_tags(os.path.join(current_dir, "metadata/class_labels_indices.csv"))

threshold = 0.25
sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
print("Predicted labels using activity threshold 0.25:\n")
print(sample_labels)
for l in sample_labels:
    print("%s: %.3f"%(ix_to_lb[l], probs[0,l]))

# Get audio scene embeddings
with torch.no_grad():
    model.eval()
    output = model.forward_scene_embeddings(waveform)

print("\nScene embedding, shape:", output.size())

# Get frame-level embeddings
with torch.no_grad():
    model.eval()
    output = model.forward_frame_embeddings(waveform)

print("\nFrame-level embeddings, shape:", output.size())
