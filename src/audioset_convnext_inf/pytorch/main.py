#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data

from torch.nn.parallel import DistributedDataParallel as DDP

from audioset_convnext_inf.utils.utilities import (
    create_folder,
    get_filename,
    create_logging,
    Mixup,
    StatisticsContainer,
)

from audioset_convnext_inf.pytorch.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_atto,
    convnext_femto,
    convnext_pico,
    convnext_nano,
)
from audioset_convnext_inf.pytorch.models import (
    Cnn14,
    Cnn14Next,
    Cnn14Deformable,
    Cnn14_no_specaug,
    Cnn14_no_dropout,
    Cnn14Sep,
    Cnn14SepPW,
    Cnn6,
    Cnn6Sobel,
    Cnn6SobelG,
    Cnn6SobelLearnable,
    Cnn6SobelG2,
    Cnn6Deformable,
    Cnn6Next,
    Cnn6NextNoStem,
    Cnn6NextDCLS,
    Cnn6NextNoLastPool,
    Cnn6Next11,
    Cnn6Next13,
    Cnn6Next11NoStem,
    Cnn6Next13NoStem,
    Cnn6NextConvPool,
    Cnn6NextConvPoolGroup1,
    convnext_cnn6,
    Cnn7Next,
    Cnn8NextNoStemNoFC1,
    convnext_cnn10,
    Cnn10,
    Cnn10Next,
    Cnn10Next11,
    Cnn10NextNoStem,
    Cnn10NextDropPath,
    ResNet22,
    ResNet38,
    ResNet54,
    Cnn14_emb512,
    Cnn14_emb128,
    Cnn14_emb32,
    MobileNetV1,
    MobileNetV2,
    LeeNet11,
    LeeNet24,
    DaiNet19,
    Res1dNet31,
    Res1dNet51,
    Wavegram_Cnn14,
    Wavegram_Logmel_Cnn14,
    Wavegram_Logmel128_Cnn14,
    Cnn14_16k,
    Cnn14_8k,
    Cnn14_mel32,
    Cnn14_mel128,
    Cnn14_mixup_time_domain,
    Cnn14_DecisionLevelMax,
    Cnn14_DecisionLevelAtt,
)
from audioset_convnext_inf.pytorch.evaluate import Evaluator
from audioset_convnext_inf.pytorch.losses import get_loss_func
from audioset_convnext_inf.pytorch.pytorch_utils import (
    move_data_to_device,
    count_parameters,
    count_flops,
    do_mixup,
    custom_weight_decay,
)

from audioset_convnext_inf.utils.data_generator import (
    AudioSetDataset,
    TrainSampler,
    BalancedTrainSampler,
    AlternateTrainSampler,
    EvaluateSampler,
    collate_fn,
)
from audioset_convnext_inf.utils import config

import idr_torch
import wandb


def train(args):
    """Train AudioSet tagging model.

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
      seed: int
      deformable_blocks: list of int, default to None
    """

    # Arugments & parameters
    workspace = args.workspace
    dataspace = args.dataspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    use_sobel = args.use_sobel
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = (
        torch.device("cuda")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    seed = args.seed
    filename = args.filename
    deformable_blocks = args.deformable_blocks
    drop_path_rate = args.drop_path_rate
    after_stem_dim = args.after_stem_dim
    use_speed_perturb = args.use_speed_perturb
    use_pydub_augment = args.use_pydub_augment
    use_roll_augment = args.use_roll_augment

    in_22k = args.in_22k

    dcls_kernel_size = args.dcls_kernel_size
    dcls_kernel_count = args.dcls_kernel_count

    num_workers = args.num_workers
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    weight_decay = args.weight_decay
    use_wd_scheduler = args.use_wd_scheduler
    # if args.weight_decay_end is None:
    #     args.weight_decay_end = args.weight_decay
    # wd_schedule_values = utils.cosine_scheduler(
    #     args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    # print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    print("weight_decay WD:", weight_decay)

    black_list_csv = args.black_list_csv
    # Paths
    # black_list_csv = None
    # black_list_csv = "/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn/metadata/black_list/audiocaps_train_val_test.csv"
    print("black_list_csv", black_list_csv)

    wandb.login(key="cfa500f72c83d0e45e69c8e1ad4a0d1e735bb5b2")

    if "ConvNext6" in model_type or "ConvNext10" in model_type:
        model_type += "_stem4422_drop20"

    if (
        "Tiny" in model_type
        or "Small" in model_type
        or "Base" in model_type
        or "Nano" in model_type
    ):
        model_type += "_pretrained_True_dp_%.1f_wd_%.2f_wdSched_%s_stemDim_%s" % (
            drop_path_rate,
            weight_decay,
            use_wd_scheduler,
            "_".join([str(el) for el in after_stem_dim]),
        )

        if in_22k:
            model_type += "_in22k"
            print("Pretrained ConNeXt: 22k")

        if use_speed_perturb:
            model_type += "_speedTrue"
        else:
            model_type += "_speedFalse"

        if use_pydub_augment:
            model_type += "_gainTrue"
        else:
            model_type += "_gainFalse"

    if use_sobel:
        model_type += "_sobel"

    print("using Speed Perturbation:", use_speed_perturb)
    print("using pydub augment:", use_pydub_augment)
    print("using roll augment:", use_roll_augment)
    print("using Sobel:", use_sobel)

    if deformable_blocks is not None:
        wandb.config = {
            "model": model_type,
            "per_device_bs": batch_size,
            "grad_acc_steps": 1,
            "steps": early_stop,
            "warmup_steps": 0,
            "lr": learning_rate,
            "deformable_blocks": deformable_blocks,
            "seed": seed,
        }
    else:
        wandb.config = {
            "model": model_type,
            "per_device_bs": batch_size,
            "grad_acc_steps": 1,
            "steps": early_stop,
            "warmup_steps": 0,
            "lr": learning_rate,
            "deformable_blocks": "none",
            "seed": seed,
        }

    wandb_run_name = "%s_8GPU_seed%d_bs%d_gradacc%d_lr%.6f_maxStep%d" % (
        model_type,
        seed,
        batch_size,
        1,
        learning_rate,
        early_stop,
    )

    if deformable_blocks is not None:
        wandb_run_name += "_deformable_" + "_".join(
            [str(el) for el in deformable_blocks]
        )

    if dcls_kernel_size is not None:
        wandb_run_name += (
            "_ks_" + str(dcls_kernel_size) + "_kc_" + str(dcls_kernel_count)
        )

    if black_list_csv is not None:
        wandb_run_name += "_blacklist"

    print("wandb RUN name:", wandb_run_name)

    # Only log on the master GPU
    if idr_torch.rank == 0:
        print("wandb init on rank 0")
        run = wandb.init(project="audioset", entity="dcls", name=wandb_run_name)
        run.define_metric("steps")
        run.define_metric("training_loss", step_metric="steps")
        run.define_metric("balanced_train_mAP", step_metric="steps")
        run.define_metric("test_mAP", step_metric="steps")
        run.define_metric("balanced_train_AUC", step_metric="steps")
        run.define_metric("test_AUC", step_metric="steps")
        run.define_metric("balanced_train_dprime", step_metric="steps")
        run.define_metric("test_macro_dprime", step_metric="steps")
        run.define_metric("lr", step_metric="steps")
        run.define_metric("weight_decay", step_metric="steps")
    else:
        run = None
    do_log = run is not None

    train_indexes_hdf5_path = os.path.join(
        dataspace, "hdf5s", "indexes", "{}.h5".format(data_type)
    )

    eval_bal_indexes_hdf5_path = os.path.join(
        dataspace, "hdf5s", "indexes", "balanced_train.h5"
    )

    eval_test_indexes_hdf5_path = os.path.join(dataspace, "hdf5s", "indexes", "eval.h5")

    if deformable_blocks is not None:
        checkpoints_dir = os.path.join(
            workspace,
            "checkpoints",
            filename,
            "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
                sample_rate, window_size, hop_size, mel_bins, fmin, fmax
            ),
            "data_type={}".format(data_type),
            model_type,
            "deformation_blocks={}".format(
                "_".join([str(el) for el in deformable_blocks])
            ),
            "loss_type={}".format(loss_type),
            "balanced={}".format(balanced),
            "augmentation={}".format(augmentation),
            "batch_size={}".format(batch_size),
        )
    elif dcls_kernel_size is not None:
        checkpoints_dir = os.path.join(
            workspace,
            "checkpoints",
            filename,
            "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
                sample_rate, window_size, hop_size, mel_bins, fmin, fmax
            ),
            "data_type={}".format(data_type),
            model_type,
            "dcls_kernel_size={}".format(dcls_kernel_size),
            "dcls_kernel_count={}".format(dcls_kernel_count),
            "loss_type={}".format(loss_type),
            "balanced={}".format(balanced),
            "augmentation={}".format(augmentation),
            "batch_size={}".format(batch_size),
        )

    else:
        checkpoints_dir = os.path.join(
            workspace,
            "checkpoints",
            filename,
            "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
                sample_rate, window_size, hop_size, mel_bins, fmin, fmax
            ),
            "data_type={}".format(data_type),
            model_type,
            "black_list_{}".format(black_list_csv),
            "loss_type={}".format(loss_type),
            "balanced={}".format(balanced),
            "augmentation={}".format(augmentation),
            "batch_size={}".format(batch_size),
        )

    if not os.path.exists(checkpoints_dir):
        create_folder(checkpoints_dir)

    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        model_type,
        "black_list_{}".format(black_list_csv),
        "loss_type={}".format(loss_type),
        "balanced={}".format(balanced),
        "augmentation={}".format(augmentation),
        "batch_size={}".format(batch_size),
        "statistics.pkl",
    )
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        model_type,
        "black_list_{}".format(black_list_csv),
        "loss_type={}".format(loss_type),
        "balanced={}".format(balanced),
        "Augmentation={}".format(augmentation),
        "Batch_size={}".format(batch_size),
    )

    create_logging(logs_dir, filemode="w")
    logging.info(args)

    if "cuda" in str(device):
        logging.info("Using GPU.")
        # device = 'cuda'
        torch.cuda.set_device(idr_torch.local_rank)
        device = torch.device("cuda")
        # each GPU should have its own seed
        seed = args.seed + idr_torch.rank
        print("Seed:", seed, "on GPU rank", idr_torch.local_rank)
    else:
        logging.info("Using CPU. Set --cuda flag to use GPU.")
        device = "cpu"

    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    # Model
    print("Model:", model_type)
    if "Deformable" in model_type:
        Model = eval(model_type)
        model = Model(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            deformable=deformable_blocks,
        )
    elif "ConvNextTiny" in model_type:
        if use_sobel:
            model = convnext_tiny(
                pretrained=True,
                strict=False,
                in_22k=in_22k,
                use_sobel=use_sobel,
                drop_path_rate=drop_path_rate,
                after_stem_dim=after_stem_dim,
                use_speed_perturb=use_speed_perturb,
                use_pydub_augment=use_pydub_augment,
                use_roll_augment=use_roll_augment,
            )
        else:
            model = convnext_tiny(
                pretrained=True,
                strict=False,
                in_22k=in_22k,
                drop_path_rate=drop_path_rate,
                after_stem_dim=after_stem_dim,
                use_speed_perturb=use_speed_perturb,
                use_pydub_augment=use_pydub_augment,
                use_roll_augment=use_roll_augment,
            )
        print("Pretrained: True")
    elif "ConvNextSmall" in model_type:
        model = convnext_small(
            pretrained=True,
            strict=False,
            in_22k=in_22k,
            drop_path_rate=drop_path_rate,
            after_stem_dim=after_stem_dim,
            use_speed_perturb=use_speed_perturb,
            use_pydub_augment=use_pydub_augment,
            use_roll_augment=use_roll_augment,
        )
        print("Pretrained: True")
    elif "ConvNextBase" in model_type:
        model = convnext_base(
            pretrained=True,
            strict=False,
            in_22k=in_22k,
            drop_path_rate=drop_path_rate,
            after_stem_dim=after_stem_dim,
            use_speed_perturb=use_speed_perturb,
            use_pydub_augment=use_pydub_augment,
            use_roll_augment=use_roll_augment,
        )
        print("Pretrained: True")
    elif "ConvNextAtto" in model_type:
        model = convnext_atto(pretrained=True, strict=False, drop_path_rate=0.1)
    elif "ConvNextFemto" in model_type:
        model = convnext_femto(pretrained=True, strict=False, drop_path_rate=0.1)
    elif "ConvNextPico" in model_type:
        model = convnext_pico(pretrained=True, strict=False, drop_path_rate=0.1)
    elif "ConvNextNano" in model_type:
        print("Pretrained: True")
        model = convnext_nano(
            pretrained=True,
            strict=False,
            drop_path_rate=drop_path_rate,
            after_stem_dim=after_stem_dim,
            use_speed_perturb=use_speed_perturb,
            use_pydub_augment=use_pydub_augment,
            use_roll_augment=use_roll_augment,
        )
    elif "ConvNext6" in model_type:
        model = convnext_cnn6(drop_path_rate=drop_path_rate)
    elif "ConvNext10" in model_type:
        model = convnext_cnn10(drop_path_rate=drop_path_rate)
    elif "Cnn6NextDCLS" in model_type:
        Model = eval(model_type)
        model = Model(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            dcls_kernel_size=dcls_kernel_size,
            classes_num=classes_num,
        )
    elif "SobelG" in model_type:
        Model = eval(model_type)
        model = Model(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            device=device,
        )
    else:
        Model = eval(model_type)
        model = Model(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
        )

    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    if idr_torch.rank == 0:
        print("Parameters num: {}".format(params_num))
        print(model)

    logging.info("Parameters num: {}".format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))

    # Dataset will be used by DataLoader later. Dataset takes a meta as input
    # and return a waveform and a target.
    dataset = AudioSetDataset(sample_rate=sample_rate)

    # Train sampler
    if balanced == "none":
        Sampler = TrainSampler
    elif balanced == "balanced":
        Sampler = BalancedTrainSampler
    elif balanced == "alternate":
        Sampler = AlternateTrainSampler

    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path,
        batch_size=batch_size * 2 if "mixup" in augmentation else batch_size,
        black_list_csv=black_list_csv,
        random_seed=seed,
    )

    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size
    )

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size
    )

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

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

    if "mixup" in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.0)

    # Evaluator
    evaluator = Evaluator(model=model)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    train_bgn_time = time.time()

    # Resume training
    print("resume_iteration", resume_iteration)

    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(
            checkpoints_dir, "{}_iterations.pth".format(resume_iteration)
        )

        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint["iteration"]
        print("Loading checkpoint {}".format(resume_checkpoint_path))
        logging.info("Loading checkpoint {}".format(resume_checkpoint_path))

    else:
        iteration = 0

    # Parallel
    print("GPU number: {}".format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)
    if "cuda" in str(device):
        model.to(device)
    model = DDP(model, device_ids=[idr_torch.local_rank])
    # , find_unused_parameters=True

    # Optimizer
    if weight_decay > 0:
        print("Using AdamW with WD")
        params = custom_weight_decay(model, wd=weight_decay)
        optimizer = optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=75000
    )
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=early_stop)

    wd_schedule_values = None
    if use_wd_scheduler:

        def wd_scheduler(
            base_value,
            final_value,
            min_value,
            niter_per_ep,
            cooldown_steps=0,
            constant_cooldown=False,
        ):
            print("Set cooldown_steps steps = %d" % cooldown_steps)
            iters = np.arange(cooldown_steps)
            if constant_cooldown:
                schedule = np.ones(len(iters)) * base_value
            else:
                schedule = np.array(
                    [
                        min_value
                        + 0.5
                        * (base_value - min_value)
                        * (1 + math.cos(math.pi * i / (len(iters))))
                        for i in iters
                    ]
                )

            start_warmup_value = schedule[-1]
            warmup_schedule = np.linspace(
                start_warmup_value, final_value, niter_per_ep - cooldown_steps
            )
            schedule = np.concatenate((schedule, warmup_schedule))

            assert len(schedule) == niter_per_ep
            return schedule

        weight_decay_end = 2 * weight_decay
        weight_decay_min_value = weight_decay / 5
        wd_schedule_values = wd_scheduler(
            base_value=weight_decay,
            final_value=weight_decay_end,
            min_value=weight_decay_min_value,
            niter_per_ep=early_stop,
            cooldown_steps=int(0.3 * early_stop),
            constant_cooldown=True,
        )
        print(
            "Max WD = %.7f, Min WD = %.7f constant_cooldown=True"
            % (max(wd_schedule_values), min(wd_schedule_values))
        )

    if resume_iteration > 0:
        train_sampler.load_state_dict(checkpoint["sampler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    time1 = time.time()

    for batch_data_dict in train_loader:
        """batch_data_dict: {
        'audio_name': (batch_size [*2 if mixup],),
        'waveform': (batch_size [*2 if mixup], clip_samples),
        'target': (batch_size [*2 if mixup], classes_num),
        (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        if do_log and idr_torch.rank == 0:
            run.log({"steps": iteration}, commit=False)

        # check if GPUs receive different audios! OK
        # if idr_torch.rank == 0:
        #     print("idr_torch.rank: ", idr_torch.rank)
        #     print(batch_data_dict['audio_name'])

        # if idr_torch.rank == 1:
        #     print("idr_torch.rank: ", idr_torch.rank)
        #     print(batch_data_dict['audio_name'])

        # # Evaluate
        # if (iteration % 10 == 0 and iteration > resume_iteration) or (iteration == 0):
        if (
            iteration % 5000 == 0 and iteration > resume_iteration
        ) or iteration == early_stop:
            train_fin_time = time.time()

            if do_log and iteration > resume_iteration and idr_torch.rank == 0:
                run.log({"training_loss": loss.item(), "steps": iteration})
                print(
                    "--- Iteration: {}, train time: {:.3f} s / 2000 iteration ---".format(
                        iteration, time.time() - time1
                    )
                )
                time1 = time.time()

            bal_statistics = evaluator.evaluate(eval_bal_loader)
            test_statistics = evaluator.evaluate(eval_test_loader)

            bal_mAP = np.mean(bal_statistics["average_precision"])
            test_mAP = np.mean(test_statistics["average_precision"])
            bal_macro_AUC = np.mean(bal_statistics["auc"])
            test_macro_AUC = np.mean(test_statistics["auc"])
            bal_macro_dprime = np.mean(bal_statistics["d_prime"])
            test_macro_dprime = np.mean(test_statistics["d_prime"])

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            if idr_torch.rank == 0:
                statistics_container.append(iteration, bal_statistics, data_type="bal")
                statistics_container.append(
                    iteration, test_statistics, data_type="test"
                )
                statistics_container.dump()

                logging.info("Validate bal mAP: {:.3f}".format(bal_mAP))
                logging.info("Validate bal AUC: {:.3f}".format(bal_macro_AUC))
                logging.info("Validate bal d-prime: {:.3f}".format(bal_macro_dprime))
                logging.info("Validate test mAP: {:.3f}".format(test_mAP))
                logging.info("Validate test AUC: {:.3f}".format(test_macro_AUC))
                logging.info("Validate test d-prime: {:.3f}".format(test_macro_dprime))

                logging.info(
                    "iteration: {}, train time: {:.3f} s, validate time: {:.3f} s"
                    "".format(iteration, train_time, validate_time)
                )

                logging.info("------------------------------------")

            train_bgn_time = time.time()

            val_logs = {
                "balanced_train_mAP": bal_mAP,
                "test_mAP": test_mAP,
                "balanced_train_AUC": bal_macro_AUC,
                "test_AUC": test_macro_AUC,
                "balanced_train_dprime": bal_macro_dprime,
                "test_macro_dprime": test_macro_dprime,
                "steps": iteration,
            }
            if do_log and idr_torch.rank == 0:
                run.log(val_logs)

            # Save model
            # if iteration % 20 == 0 :
            # if iteration % 5000 == 0 :
            # bal_statistics = evaluator.evaluate(eval_bal_loader)
            # test_statistics = evaluator.evaluate(eval_test_loader)

            # bal_mAP = np.mean(bal_statistics['average_precision'])
            # test_mAP = np.mean(test_statistics['average_precision'])
            # bal_macro_AUC = np.mean(bal_statistics['auc'])
            # test_macro_AUC = np.mean(test_statistics['auc'])
            # bal_macro_dprime = np.mean(bal_statistics['d_prime'])
            # test_macro_dprime = np.mean(test_statistics['d_prime'])

            if idr_torch.rank == 0:
                checkpoint = {
                    "balanced_train_mAP": bal_mAP,
                    "test_mAP": test_mAP,
                    "balanced_train_AUC": bal_macro_AUC,
                    "test_AUC": test_macro_AUC,
                    "balanced_train_dprime": bal_macro_dprime,
                    "test_macro_dprime": test_macro_dprime,
                    "iteration": iteration,
                    # 'train_last_minibatch_loss': loss.item(),
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "sampler": train_sampler.state_dict(),
                }

                checkpoint_path = os.path.join(
                    checkpoints_dir, "{}_iterations.pth".format(iteration)
                )
                torch.save(checkpoint, checkpoint_path)
                logging.info("Model saved to {}".format(checkpoint_path))
                del checkpoint

        # Mixup lambda
        if "mixup" in augmentation:
            batch_data_dict["mixup_lambda"] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict["waveform"])
            )

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Forward
        model.train()
        optimizer.zero_grad()

        if "mixup" in augmentation:
            batch_output_dict = model(
                batch_data_dict["waveform"], batch_data_dict["mixup_lambda"]
            )
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {
                "target": do_mixup(
                    batch_data_dict["target"], batch_data_dict["mixup_lambda"]
                )
            }
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict["waveform"], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {"target": batch_data_dict["target"]}
            """{'target': (batch_size, classes_num)}"""

        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        # print(loss.item())

        # update weight decay according to the scheduled values
        if wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[iteration]

        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()
        if do_log and idr_torch.rank == 0:
            if wd_schedule_values is not None:
                run.log(
                    {
                        "lr": current_lr[0],
                        "weight_decay": param_group["weight_decay"],
                        "steps": iteration,
                    }
                )
            else:
                # print("LR =", current_lr[0])
                run.log({"lr": current_lr[0], "steps": iteration})

        if "DCLS" in model_type:
            if hasattr(model, "module"):
                # print("1 - clamping")
                model.module.clamp_parameters()
            else:
                # print("2 - clamping")
                model.clamp_parameters()

        if "SobelLearnable" in model_type:
            if hasattr(model, "module"):
                # print("1 - clamping")
                model.module.clamp_w()
            else:
                # print("2 - clamping")
                model.clamp_w()

        # if do_log and iteration % 1000 == 0:
        #     run.log({"Rank 0/training_loss": loss.item()})

        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of parser. ")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--workspace", type=str, required=True)
    parser_train.add_argument("--dataspace", type=str, required=True)
    parser_train.add_argument(
        "--data_type",
        type=str,
        default="full_train",
        choices=["balanced_train", "full_train"],
    )
    parser_train.add_argument("--sample_rate", type=int, default=32000)
    parser_train.add_argument("--window_size", type=int, default=1024)
    parser_train.add_argument("--hop_size", type=int, default=320)
    parser_train.add_argument("--mel_bins", type=int, default=64)
    parser_train.add_argument("--fmin", type=int, default=50)
    parser_train.add_argument("--fmax", type=int, default=14000)
    parser_train.add_argument("--model_type", type=str, required=True)
    parser_train.add_argument("--drop_path_rate", type=float, default=0)
    parser_train.add_argument("--after_stem_dim", nargs="*", type=int, default=56)

    parser_train.add_argument(
        "--loss_type", type=str, default="clip_bce", choices=["clip_bce"]
    )
    parser_train.add_argument(
        "--balanced",
        type=str,
        default="balanced",
        choices=["none", "balanced", "alternate"],
    )
    parser_train.add_argument(
        "--augmentation", type=str, default="mixup", choices=["none", "mixup"]
    )
    parser_train.add_argument("--use_speed_perturb", action="store_true", default=False)
    parser_train.add_argument("--use_pydub_augment", action="store_true", default=False)
    parser_train.add_argument("--use_roll_augment", action="store_true", default=False)

    parser_train.add_argument("--use_sobel", action="store_true", default=False)

    parser_train.add_argument("--in_22k", action="store_true", default=False)
    parser_train.add_argument("--use_wd_scheduler", action="store_true", default=False)

    parser_train.add_argument("--batch_size", type=int, default=32)
    parser_train.add_argument("--num_workers", type=int, default=3)

    parser_train.add_argument("--learning_rate", type=float, default=1e-3)
    parser_train.add_argument("--resume_iteration", type=int, default=0)
    parser_train.add_argument("--early_stop", type=int, default=75000)
    parser_train.add_argument("--cuda", action="store_true", default=False)
    parser_train.add_argument("--seed", type=int, default=1978)

    parser_train.add_argument("--deformable_blocks", nargs="*", type=int, default=None)
    parser_train.add_argument("--dcls_kernel_size", type=int, default=None)
    parser_train.add_argument("--dcls_kernel_count", type=int, default=None)

    parser_train.add_argument("--black_list_csv", type=str, default=None)

    parser_train.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight decay (default: 0.0)"
    )

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=idr_torch.size,
        rank=idr_torch.rank,
    )

    if args.mode == "train":
        train(args)

    else:
        raise Exception("Error argument!")
