# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic Inc. All rights reserved.
# Authors: kingqluo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from entity.base_metric import BaseMetric
from entity import MetricResult


class AUC(BaseMetric):

    def __init__(self, label_name):
        self._label_name = label_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        if np.sum(label) == 0 or np.sum(label) == label.size:
            return MetricResult(result=float('nan'))
        else:
            auc = roc_auc_score(y_true=label, y_score=predict)
            return MetricResult(result=auc, meta={'#': predict.size})

    @property
    def required_label_names(self):
        return [self._label_name]
    def __repr__(self):
        print("go")