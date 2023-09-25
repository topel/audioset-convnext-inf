#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp

# import librosa
import numpy as np
import torch
import torchaudio

from audioset_convnext_inf.pytorch.convnext import convnext_tiny

# set this to True if you need to download the checkpoint, and if your
# machine has access to the Web. Otherwise, download the ckpt
# manually, and modify model_fpath accordingly:
download_ckpt=False

model_fpath = "/gpfswork/rech/djl/uzj43um/audioset-convnext-inf/checkpoints/convnext_tiny_471mAP.pth"
# model_fpath = '/gpfswork/rech/djl/uzj43um/audioset-convnext-inf/checkpoints/convnext_tiny_465mAP_BL_AC_70kit.pth'

CONVNEXT_CKPT_URL = (
    "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1"
)
CONVNEXT_CKPT_FILENAME = "convnext_tiny_471mAP.pth"
AUDIO_FNAME = "f62-S-v2swA_200000_210000.wav"
AUDIO_FPATH = osp.join("/gpfswork/rech/djl/uzj43um/audioset-convnext-inf", "audio_samples", AUDIO_FNAME)


model = convnext_tiny(
    pretrained=False,
    strict=False,
    drop_path_rate=0.0,
    after_stem_dim=[252, 56],
    use_speed_perturb=False,
)

def load_from_pretrain(model, pretrained_checkpoint_path):
    checkpoint = torch.load(pretrained_checkpoint_path)
    model.load_state_dict(checkpoint["model"])


if download_ckpt and not osp.isfile(model_fpath):
    ckpt_dpath = osp.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dpath, exist_ok=True)
    model_fpath = osp.join(ckpt_dpath, CONVNEXT_CKPT_FILENAME)
    torch.hub.download_url_to_file(CONVNEXT_CKPT_URL, model_fpath)

load_from_pretrain(model, model_fpath)
print("Loaded ckpt from:", model_fpath)

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

# (waveform, _) = librosa.core.load(AUDIO_FPATH, sr=sample_rate, mono=True)
# # print(waveform.shape)
# # waveform = pad_audio(waveform, audio_target_length)
# # print(waveform.shape)
# waveform = waveform[None, :]  # (1, audio_length)
# waveform = torch.as_tensor(waveform).to(device)

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


threshold = 0.25
sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
print("Predicted labels using activity threshold 0.25:\n")
print(sample_labels)

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
