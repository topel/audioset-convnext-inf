#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import argparse

import numpy as np
import torch

from audioset_convnext_inf.pytorch.evaluate import Evaluator
from audioset_convnext_inf.pytorch.convnext import convnext_tiny
from audioset_convnext_inf.utils.data_generator import (
    AudioSetDataset,
    EvaluateSampler,
    collate_fn,
)


def evaluate(args):
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

    # tiny_path = '/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn/pretrained_models/my_models/convnext_tiny_471mAP.pth'
    tiny_path = args.tiny_path

    load_from_pretrain(model, tiny_path)

    device = torch.device("cuda")
    if "cuda" in str(device):
        model.to(device)
    model.eval()

    #    h5_indexes_dir_path='/gpfsstore/rech/djl/uzj43um/audioset/hdf5s/indexes'
    h5_indexes_dir_path = args.h5_indexes_dir_path
    eval_bal_indexes_hdf5_path = os.path.join(h5_indexes_dir_path, "balanced_train.h5")
    eval_test_indexes_hdf5_path = os.path.join(h5_indexes_dir_path, "eval.h5")

    # Evaluate sampler
    batch_size = 256
    sample_rate = 32000
    num_workers = 10

    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size
    )

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size
    )

    # Data loader
    dataset = AudioSetDataset(sample_rate=sample_rate)

    eval_bal_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=eval_bal_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=eval_test_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Evaluator
    evaluator = Evaluator(model=model)

    bal_statistics = evaluator.evaluate(eval_bal_loader)
    test_statistics = evaluator.evaluate(eval_test_loader)

    bal_mAP = np.mean(bal_statistics["average_precision"])
    test_mAP = np.mean(test_statistics["average_precision"])
    bal_macro_AUC = np.mean(bal_statistics["auc"])
    test_macro_AUC = np.mean(test_statistics["auc"])
    bal_macro_dprime = np.mean(bal_statistics["d_prime"])
    test_macro_dprime = np.mean(test_statistics["d_prime"])

    print("Validate bal mAP: {:.3f}".format(bal_mAP))
    print("Validate bal AUC: {:.3f}".format(bal_macro_AUC))
    print("Validate bal d-prime: {:.3f}".format(bal_macro_dprime))
    print("Validate test mAP: {:.3f}".format(test_mAP))
    print("Validate test AUC: {:.3f}".format(test_macro_AUC))
    print("Validate test d-prime: {:.3f}".format(test_macro_dprime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on AudioSet, the balanced and the test set."
    )

    parser.add_argument("--tiny_path", type=str, required=True)
    parser.add_argument("--h5_indexes_dir_path", type=str, required=True)

    args = parser.parse_args()

    evaluate(args)
