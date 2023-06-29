# Adapting a ConvNeXt model to audio classification on AudioSet

In this work, we adapted the computer vision architecture ConvNeXt (Tiny) to perform audio tagging on AudioSet. 

In this repo, we provide the PyTorch code to do inference with our best checkpoint, trained on the AudioSet dev subset. We do not provide code to train our models, but it is heavily based on [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://github.com/qiuqiangkong/audioset_tagging_cnn), thanks a lot to Qiuqiang Kong and colleagues for their amazing open-source work! We are currently working on modernizing the code base, and hopefully we'll release it some day.

## Install instructions

```conda env create -f environment.yml```

There are many packages listed in environment.yml that are not actually needed to use our ConvNeXt model. 

The most important ones are:

- pytorch 1.11.0 but more recent 1.* versions should work (maybe 2.* also), 
- torchlibrosa, needed to generate log mel spectrograms just as in PANN. 

## Get a checkpoint

Create a `checkpoints` directory, in which a checkpoint should be added.

A checkpoint is available on Zenodo: https://zenodo.org/record/8020843

Get the `convnext_tiny_471mAP.pth` one to do audio tagging and embedding extraction.

The following results were obtained on the AudioSet test set:

| mAP     | 0.471 |
|---------|-------|
| AUC     | 0.973 |
| d-prime | 3.071 |

A second checkpoint is also available, in case you are interested in doing experiments on the AudioCaps dataset (audio captioning and audio-text retrieval).

## Audio tagging demo

The script `demo_convnext.py` provides an example of how to do audio tagging on a single audio file, provided in the `audio_samples` directory.

It will give the following output:

```f62-S-v2swA_200000_210000.wav
predictions:
[  0 137 138 139 151 506]
```

You can associate the corresponding tag names to these indexes thank to the file `metadata/class_labels_indices.csv`:

`[  0 137 138 139 151 506] Speech; Music; Musical instrument; Plucked string instrument; Ukulele; Inside, small room`

When the ground-truth is for this recording, as given in `audio_samples/f62-S-v2swA_200000_210000_labels.txt`:

`[  0 137 151] Speech; Music; Ukulele; `

## Evaluate a checkpoint on the balanced and the test subsets of AudioSet

You can retrieve the results afore-mentioned with this script: `evaluate_convnext_on_audioset.py`

The sbatch script is provided: `scripts/5_evaluate_convnext_on_audioset.sbatch`

It loads the checkpoint and run it on a single GPU. It should take a few minutes to run and get the metric results in the log file. 

## Citation

If you find this work useful, please consider citing our paper, to be presented at INTERSPEECH 2023:

**Pellegrini, T., Khalfaoui-Hassani, I., Labbé, E., & Masquelier, T. (2023). Adapting a ConvNeXt model to audio classification on AudioSet. arXiv preprint [arXiv:2306.00830](https://arxiv.org/abs/2306.00830).**

```
@misc{pellegrini2023adapting,
      title={{Adapting a ConvNeXt model to audio classification on AudioSet}}, 
      author={Thomas Pellegrini and Ismail Khalfaoui-Hassani and Etienne Labbé and Timothée Masquelier},
      year={2023},
      eprint={2306.00830},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
