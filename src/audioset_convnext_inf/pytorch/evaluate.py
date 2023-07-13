#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt

from scipy.stats import norm
from sklearn import metrics

from audioset_convnext_inf.pytorch.pytorch_utils import forward


class Evaluator(object):
    def __init__(self, model, use_torchaudio=False):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        self.use_torchaudio = use_torchaudio

    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict,
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model,
            generator=data_loader,
            use_torchaudio=self.use_torchaudio,
            return_target=True,
        )

        clipwise_output = output_dict["clipwise_output"]  # (audios_num, classes_num)
        target = output_dict["target"]  # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None
        )

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)

        # d' = √2F^−1(ROCAUC) where F^−1 is the inverse CDF for a unit Gaussian
        per_class_d_prime = sqrt(2) * norm.ppf(auc)
        # macro_avg_d_prime = np.mean(per_class_d_prime)

        statistics = {
            "average_precision": average_precision,
            "auc": auc,
            "d_prime": per_class_d_prime,
        }

        return statistics
