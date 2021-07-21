# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from __future__ import annotations

import abc
from typing import List

from entity.entity import Entity


class Model(Entity):
    def __init__(self,
                 name: str,
                 model_path: str,
                 task_type: str,
                 train_flag: bool,
                 supervised_feature_config: List[int] = None,
                 preprocessing_feature_config_path: str = None
                 ):

        self._model_path = model_path
        self._task_type = task_type
        self._train_flag = train_flag
        self._train_finished = False

        self._model_param_dict = {}

        self._supervised_feature_config = supervised_feature_config
        self._preprocessing_feature_config_path = preprocessing_feature_config_path
        self._feature_dict = {}

        super(Model, self).__init__(
            name=name,
        )

    @abc.abstractmethod
    def train(self, **entity):
        pass

    @abc.abstractmethod
    def predict(self, **entity):
        pass

    @abc.abstractmethod
    def eval(self, **entity):
        pass

    @abc.abstractmethod
    def get_train_metric(self):
        pass

    @abc.abstractmethod
    def get_train_loss(self):
        pass

    @abc.abstractmethod
    def get_val_loss(self):
        pass

    def update_params(self, **params):
        self._model_param_dict.update(params)

    @abc.abstractmethod
    def preprocess(self):
        """
        This method is used to implement Normalization, Standardization, which need self._train_flag parameters.
        :return: None
        """
        if self._train_flag:
            self._train_preprocess()
        else:
            self._predict_process()

    @abc.abstractmethod
    def _train_preprocess(self):
        pass

    @abc.abstractmethod
    def _predict_process(self):
        pass
