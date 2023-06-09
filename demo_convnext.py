import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
import config

import librosa
# import matplotlib.pyplot as plt
import torch

import numpy as np

from convnext import convnext_tiny

model = convnext_tiny(pretrained=False, strict=False, drop_path_rate=0., after_stem_dim=[252, 56], use_speed_perturb=False)

print("total_params", sum(
    param.numel() for param in model.parameters() if param.requires_grad
))


def load_from_pretrain(model, pretrained_checkpoint_path):
    checkpoint = torch.load(pretrained_checkpoint_path)
    model.load_state_dict(checkpoint['model'])

# tiny_path = '/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset-convnext-inf/checkpoints/convnext_tiny_471mAP.pth'
tiny_path = '/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset-convnext-inf/checkpoints/convnext_tiny_465mAP_BL_AC_70kit.pth'
load_from_pretrain(model, tiny_path)

if torch.cuda.is_available():
    device = torch.device("cuda")
else : device = torch.device("cpu")

if 'cuda' in str(device):
    model = model.to(device)

sample_rate=32000
audio_target_length = 10*sample_rate # 10 s

audio_name='f62-S-v2swA_200000_210000.wav'
fpath = '/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset-convnext-inf/audio_samples/' + audio_name
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

probs = torch.sigmoid(logits)

print(probs)


threshold=0.25
sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]
print("\n\n" + audio_name + "\n\n")
print("predictions:\n\n")
print(sample_labels)


