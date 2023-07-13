import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
# sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from sklearn import metrics

from timm_weight_init import trunc_normal_

import matplotlib.pyplot as plt


from audiocaps import AudioCaps
from torch.utils.data.dataloader import DataLoader
from aac_datasets_utils import BasicCollate

from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)

from models import Cnn14


AudioCaps.FFMPEG_PATH = ""
AudioCaps.YOUTUBE_DL_PATH = ""

# AUDIO_DIR='/gpfsscratch/rech/xsl/commun/data/AUDIOCAPS_32000Hz'
ROOT_DIR='/gpfsscratch/rech/xsl/commun/data'
BASEDIR='/gpfsstore/rech/djl/uzj43um/audiocaps'


train_dataset = AudioCaps(root=ROOT_DIR, download=False, with_tags=True)
val_dataset = AudioCaps(root=ROOT_DIR, subset='val', download=False, with_tags=True)
test_dataset = AudioCaps(root=ROOT_DIR, subset='test', download=False, with_tags=True)

bs_train=64
bs_test=128
train_loader = DataLoader(train_dataset, batch_size=bs_train, collate_fn=BasicCollate(with_tags=True))
val_loader = DataLoader(val_dataset, batch_size=bs_test, collate_fn=BasicCollate(with_tags=True))
test_loader = DataLoader(test_dataset, batch_size=bs_test, collate_fn=BasicCollate(with_tags=True))


device = torch.device("cuda")


def evaluate_model(model, dataloader, subset):
    
    all_outputs, all_tags = [], []
    
    with torch.no_grad():
        for batch_data_dict in dataloader:
            # Move data to GPU
            for key in ['audio', 'tags']:
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            # eval
            model.eval()

            batch_output_dict = model(batch_data_dict['audio'], None)
            all_outputs.append(batch_output_dict['clipwise_output'].cpu().numpy())
            all_tags.append(batch_data_dict['tags'].cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    all_tags = np.concatenate(all_tags)
    
    print(all_outputs[:2])
    average_precision = metrics.average_precision_score(all_tags, all_outputs, average=None)
    return average_precision


class Transfer_Cnn(nn.Module):
    
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base=True):
        """Classifier for a new task using pretrained Cnn6 as a sub module.
        """
        super(Transfer_Cnn, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate=sample_rate, window_size=window_size, 
            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
            classes_num=classes_num)

        # Transfer to another task layer
#         self.fc_transfer = nn.Linear(512, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.named_parameters():
                if "fc_audioset" not in param[0] and "fc1" not in param[0]:
                    print(param[0])
                    param[1].requires_grad = False

#         self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, x, mixup_lambda=None):
        """x: (batch_size, data_length)
        """
#         output_dict = self.base(input, mixup_lambda)
        output_dict = self.base(x, None)
    
#         embedding = output_dict['embedding']
#         clipwise_output =  torch.sigmoid(self.fc_transfer(embedding))
#         output_dict['clipwise_output'] = clipwise_output
 
        return output_dict

sample_rate=32000
window_size=1024
hop_size=320
mel_bins=64
fmin=50
fmax=14000
classes_num=527

model = Transfer_Cnn(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num, freeze_base=True)


# pretrained_cnn6_path = '/gpfsscratch/rech/djl/uzj43um/audioset_tagging/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train//Cnn6/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=64/200000_iterations.pth'
# pretrained_cnn14_path = '/gpfsscratch/rech/djl/uzj43um/audioset_tagging/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn14/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=64/45000_iterations.pth'
pretrained_cnn14_path = '/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn/pretrained_models/Cnn14.pth'
# test mAP: 0.647
    
model.load_from_pretrain(pretrained_cnn14_path)

if 'cuda' in str(device):
        model.to(device)

print("total_params", sum(
	param.numel() for param in model.parameters() if param.requires_grad
))

# Optimizer
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
    eps=1e-08, weight_decay=0., amsgrad=True)


# Train on mini batches
criterion = torch.nn.BCELoss()

loss_list = []
for epoch in range(1,20):
    iteration = 0
    print("epoch", epoch)
    
    # Train
    model.train()

    for batch_data_dict in train_loader:
        # Move data to GPU
        for key in ['audio', 'tags']:
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)


        batch_output_dict = model(batch_data_dict['audio'], None)
        """{'clipwise_output': (batch_size, classes_num), ...}"""

         # loss
        loss = criterion(batch_output_dict['clipwise_output'], batch_data_dict['tags'])
        if iteration % 100 == 0:
            print(iteration, f"{loss.item():.5f}")
            loss_list.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += 1

#     # Stop learning
#     if iteration == stop_iteration:
#         break 
    
    subset='val'    
    mAP = evaluate_model(model, val_loader, subset)
    mAP = np.nanmean(mAP)
    print(f'{subset} - epoch {epoch} - mAP: {mAP:.4f}')
    
    subset='test'
    mAP = evaluate_model(model, test_loader, subset)
    mAP = np.nanmean(mAP)
    print(f'{subset} - epoch {epoch} - mAP: {mAP:.4f}')

    torch.save(model.state_dict(), fpath="audiocaps/model_epoch_%04d_mAP_%.4f.pt"%(epoch, mAP))

    
