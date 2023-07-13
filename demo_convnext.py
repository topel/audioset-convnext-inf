#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

import librosa
import numpy as np
import torch

from audioset_convnext_inf.pytorch.convnext import convnext_tiny

CONVNEXT_CKPT_URL = (
    "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1"
)
CONVNEXT_CKPT_FILENAME = "convnext_tiny_471mAP.pth"
AUDIO_FNAME = "f62-S-v2swA_200000_210000.wav"
AUDIO_FPATH = osp.join("audio_samples", AUDIO_FNAME)


model = convnext_tiny(
    pretrained=False,
    strict=False,
    drop_path_rate=0.0,
    after_stem_dim=[252, 56],
    use_speed_perturb=False,
)

print(
    "total_params",
    sum(param.numel() for param in model.parameters() if param.requires_grad),
)


def load_from_pretrain(model, pretrained_checkpoint_path):
    checkpoint = torch.load(pretrained_checkpoint_path)
    model.load_state_dict(checkpoint["model"])


tiny_path = osp.join(torch.hub.get_dir(), CONVNEXT_CKPT_FILENAME)
if not osp.isfile(tiny_path):
    torch.hub.download_url_to_file(CONVNEXT_CKPT_URL, tiny_path)

# tiny_path = "/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset-convnext-inf/checkpoints/convnext_tiny_471mAP.pth"
# tiny_path = '/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset-convnext-inf/checkpoints/convnext_tiny_465mAP_BL_AC_70kit.pth'
load_from_pretrain(model, tiny_path)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if "cuda" in str(device):
    model = model.to(device)

sample_rate = 32000
audio_target_length = 10 * sample_rate  # 10 s

(waveform, _) = librosa.core.load(AUDIO_FPATH, sr=sample_rate, mono=True)
# print(waveform.shape)
# waveform = pad_audio(waveform, audio_target_length)
# print(waveform.shape)

waveform = waveform[None, :]  # (1, audio_length)
waveform = torch.as_tensor(waveform).to(device)

# Forward
with torch.no_grad():
    model.eval()
    output = model(waveform)

logits = output["clipwise_logits"]
print("logits", logits.size())

probs = torch.sigmoid(logits)

print(probs)


threshold = 0.25
sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
print("\n\n" + AUDIO_FNAME + "\n\n")
print("predictions:\n\n")
print(sample_labels)
