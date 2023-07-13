import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
# import numpy as np
# import argparse
import librosa
# import matplotlib.pyplot as plt
import torch

import glob
import pickle
import h5py

from utilities import create_folder, get_filename, pad_or_truncate, pad_audio
from pytorch_utils import move_data_to_device
import config

from convnext import convnext_tiny


model = convnext_tiny(pretrained=False, strict=False, drop_path_rate=0., after_stem_dim=[252, 56], use_speed_perturb=False)


print("total_params", sum(
    param.numel() for param in model.parameters() if param.requires_grad
))


def load_from_pretrain(model, pretrained_checkpoint_path):
    checkpoint = torch.load(pretrained_checkpoint_path)
    model.load_state_dict(checkpoint['model'])

tiny_path = '/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn/pretrained_models/my_models/convnext_tiny_471mAP.pth'
load_from_pretrain(model, tiny_path)

device = torch.device("cuda")
if 'cuda' in str(device):
    model.to(device)

params = {
    "dataset_dir": "/gpfsscratch/rech/owj/uzj43um/CLOTHO_v2.1",
    "audio_splits": ["development", "validation", "evaluation"]
}

with open(os.path.join(params["dataset_dir"], "audio_info.pkl"), "rb") as store:
    params["audio_fids"] = pickle.load(store)["audio_fids"]

# output_dict = {'clipwise_output': clipwise_output, 'clipwise_logits': logits}
output_file='clotho_dev_val_eval_convnextTiny_logits_torch.hdf5'

sample_rate=32000
audio_target_length = 10*sample_rate # 10 s

with h5py.File(output_file, "w") as feature_store:
    for split in params["audio_splits"]:

        subset_dir = os.path.join(params["dataset_dir"], split)
        print(subset_dir)

        nb_processed_files = 0
        
        for fpath in glob.glob("{}/*.wav".format(subset_dir)):
            try:
                fname = os.path.basename(fpath)
                fid = params["audio_fids"][split][fname]

                # print(fid)
                
                # Load audio
                (waveform, _) = librosa.core.load(fpath, sr=sample_rate, mono=True)
                # print(waveform.shape)
                # waveform = pad_audio(waveform, audio_target_length)
                # print(waveform.shape)
                
                waveform = waveform[None, :]    # (1, audio_length)
                waveform = move_data_to_device(torch.as_tensor(waveform), device)

                # Forward
                with torch.no_grad():
                    model.eval()
                    output = model(waveform)
                
                logits = output['clipwise_logits']
                # print("logits", logits.size())

                # send logits to CPU. h5py cannot store GPU tensors
                feature_store[str(fid)] = torch.squeeze(logits).cpu()
                # print(fid, logits.size())

                nb_processed_files += 1

                if nb_processed_files % 100 == 0:
                    print(" processed:", nb_processed_files)
            except:
                print("Error file: {}".format(fpath))

        print(split, nb_processed_files)
