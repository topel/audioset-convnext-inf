#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pickle

import numpy as np

from scipy import stats

import csv
import json

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd, exist_ok=True)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split("/")[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def read_metadata(csv_path, audio_dir, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, "r") as fr:
        lines = fr.readlines()
        lines = lines[3:]  # Remove heads

    # first, count the audio names only of existing files on disk only

    audios_num = 0
    for n, line in enumerate(lines):
        items = line.split(", ")
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        # audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        audio_name = "{}_{}_{}.flac".format(
            items[0], items[1].replace(".", ""), items[2].replace(".", "")
        )
        audio_name = audio_name.replace("_0000_", "_0_")

        if os.path.exists(os.path.join(audio_dir, audio_name)):
            audios_num += 1

    print("CSV audio files: %d" % (len(lines)))
    print("Existing audio files: %d" % audios_num)

    # audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=bool)
    audio_names = []

    n = 0
    for line in lines:
        items = line.split(", ")
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        # audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        audio_name = "{}_{}_{}.flac".format(
            items[0], items[1].replace(".", ""), items[2].replace(".", "")
        )
        audio_name = audio_name.replace("_0000_", "_0_")

        if not os.path.exists(os.path.join(audio_dir, audio_name)):
            continue

        label_ids = items[3].split('"')[1].split(",")

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1
        n += 1

    meta_dict = {"audio_name": np.array(audio_names), "target": targets}
    return meta_dict


def read_audioset_ontology(id_to_ix):
    with open('../metadata/audioset_ontology.json', 'r') as f:
        data = json.load(f)

    # Output: {'name': 'Bob', 'languages': ['English', 'French']}                                                                                             
    sentences = []
    for el in data:
        print(el.keys())
        id = el['id']
        if id in id_to_ix:
            name = el['name']
            desc = el['description']
            # if '(' in desc:                                                                                                                                 
                # print(name, '---', desc)                                                                                                                    
            # print(id_to_ix[id], name, '---', )                                                                                                              

            # sent = name                                                                                                                                     
            # sent = name + ', ' + desc.replace('(', '').replace(')', '').lower()                                                                             
            # sent = desc.replace('(', '').replace(')', '').lower()                                                                                           
            # sentences.append(sent)                                                                                                                          
            sentences.append(desc)
            # print(sent)                                                                                                                                     
            # break                                                                                                                                           
    return sentences


def original_read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, "r") as fr:
        lines = fr.readlines()
        lines = lines[3:]  # Remove heads

    # Thomas Pellegrini: added 02/12/2022
    # check if the audio files indeed exist, otherwise remove from list

    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=bool)
    audio_names = []

    for n, line in enumerate(lines):
        items = line.split(", ")
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        audio_name = "{}_{}_{}.flac".format(
            items[0], items[1].replace(".", ""), items[2].replace(".", "")
        )  # Audios are started with an extra 'Y' when downloading
        audio_name = audio_name.replace("_0000_", "_0_")

        label_ids = items[3].split('"')[1].split(",")

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1

    meta_dict = {"audio_name": np.array(audio_names), "target": targets}
    return meta_dict

def read_audioset_label_tags(class_labels_indices_csv):
    with open(class_labels_indices_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)

    labels = []
    ids = []    # Each label has a unique id such as "/m/068hy"                                                                                               
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)

    classes_num = len(labels)

    lb_to_ix = {label : i for i, label in enumerate(labels)}
    ix_to_lb = {i : label for i, label in enumerate(labels)}

    id_to_ix = {id : i for i, id in enumerate(ids)}
    ix_to_id = {i : id for i, id in enumerate(ids)}

    return lb_to_ix, ix_to_lb, id_to_ix, ix_to_id



def float32_to_int16(x):
    # assert np.max(np.abs(x)) <= 1.5
    x = np.clip(x, -1, 1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0:audio_length]


def pad_audio(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x


def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator."""
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1.0 - lam)

        return np.array(mixup_lambdas)


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations."""
        self.statistics_path = statistics_path

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"bal": [], "test": []}

    def append(self, iteration, statistics, data_type):
        statistics["iteration"] = iteration
        self.statistics_dict[data_type].append(statistics)

    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, "wb"))
        logging.info("    Dump statistics to {}".format(self.statistics_path))
        logging.info("    Dump statistics to {}".format(self.backup_statistics_path))

    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, "rb"))

        resume_statistics_dict = {"bal": [], "test": []}

        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics["iteration"] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)

        self.statistics_dict = resume_statistics_dict
