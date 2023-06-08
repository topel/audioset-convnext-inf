import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
# import numpy as np
# import argparse
import librosa
# import matplotlib.pyplot as plt
import torch

import glob
import pickle
import h5py

from utilities import create_folder, get_filename, pad_or_truncate, pad_audio
import config

from convnext import convnext_tiny

model = convnext_tiny(pretrained=False, strict=False, drop_path_rate=0., after_stem_dim=[252, 56], use_speed_perturb=False)

print("total_params", sum(
    param.numel() for param in model.parameters() if param.requires_grad
))


def load_from_pretrain(model, pretrained_checkpoint_path):
    checkpoint = torch.load(pretrained_checkpoint_path)
    model.load_state_dict(checkpoint['model'])

tiny_path = 'checkpoints/convnext_tiny_471mAP.pth'
load_from_pretrain(model, tiny_path)

if torch.cuda.is_available():
    device = torch.device("cuda")
else : device = torch.device("cpu")

if 'cuda' in str(device):
    model = model.to(device)

sample_rate=32000
audio_target_length = 10*sample_rate # 10 s

fpath = 'audio_sampleseval_VZHuBw-BhDg_50000_60000.wav'
(waveform, _) = librosa.core.load(fpath, sr=sample_rate, mono=True)
# print(waveform.shape)
# waveform = pad_audio(waveform, audio_target_length)
# print(waveform.shape)

waveform = waveform[None, :]    # (1, audio_length)
waveform = torch.as_tensor(waveform).to(device)

# Forward
with torch.no_grad():
    model.eval()
    output = model(waveform)

logits = output['clipwise_logits']
print("logits", logits.size())

# send logits to CPU. 
# feature_store[str(fid)] = torch.squeeze(logits).cpu()
# print(fid, logits.size())

