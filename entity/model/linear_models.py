# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

import os
import shelve

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor

from entity.model.model import Model
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from utils.bunch import Bunch
from utils.common_component import mkdir


class GaussLinearModels(Model):
    def __init__(self, **params):
        super(GaussLinearModels, self).__init__(params["name"], params["model_path"], params["task_type"],
                                                params["train_flag"])
        self.file_name = self._name

        self._linear_model = None
        self._val_metrics = None

        # lr.Dataset
        self.lr_train = None
        self.lr_eval = None
        self.lr_test = None

        self._model_param_dict = {}

    def __repr__(self):
        pass

    def load_data(self, dataset: BaseDataset, val_dataset: BaseDataset = None):
        """

        :param val_dataset:
        :param dataset:
        :return: lgb.Dataset
        """
        # dataset is a bunch object, including data, target, feature_names, target_names, generated_feature_names.
        if self._train_flag:
            assert val_dataset is not None
            dataset = dataset.get_dataset()
            val_dataset = val_dataset.get_dataset()

            self._check_bunch(dataset=dataset)
            self._check_bunch(dataset=val_dataset)

            train_data = [dataset.data.values, dataset.target.values]
            validation_set = [val_dataset.data.values, val_dataset.target.values]

            return train_data, validation_set
        else:
            assert val_dataset is None

            dataset = dataset.get_dataset()
            self._check_bunch(dataset=dataset)
            return dataset.data.values

    @classmethod
    def _check_bunch(cls, dataset: Bunch):
        keys = ["data", "target", "feature_names", "target_names", "generated_feature_names"]
        for key in dataset.keys():
            assert key in keys

    def train(self, dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        assert self._train_flag is True

        self.lr_train, self.lr_eval = self.load_data(dataset=dataset, val_dataset=val_dataset)

        if self._model_param_dict is not None:
            params = self._model_param_dict

            if self._name == "lr":
                params["loss"] = "log"

            if self._task_type == "classification":
                self._linear_model = SGDClassifier(loss=params["loss"], penalty=params["penalty"],
                                                   alpha=params["alpha"], l1_ratio=params["l1-ratio"],
                                                   learning_rate=params["learning_rate"], eta0=params["eta0"],
                                                   early_stopping=params["early_stopping"],
                                                   class_weight=params["weight"],
                                                   n_jobs=-1, max_iter=10000)

                self._linear_model.fit(X=self.lr_train[0], y=self.lr_train[1])

            elif self._task_type == "regression":
                self._linear_model = SGDRegressor(loss=params["loss"], penalty=params["penalty"],
                                                  alpha=params["alpha"], l1_ratio=params["l1-ratio"],
                                                  learning_rate=params["learning_rate"], eta0=params["eta0"],
                                                  early_stopping=params["early_stopping"],
                                                  class_weight=params["weight"],
                                                  n_jobs=-1, max_iter=10000)

                self._linear_model.fit(X=self.lr_train[0], y=self.lr_train[1])

        else:
            raise ValueError("Model parameters is None.")

    def predict(self, test_dataset: BaseDataset):
        assert self._train_flag is False

        self.lr_test = self.load_data(dataset=test_dataset)
        assert os.path.isfile(self._model_path + "/" + self.file_name)

        with shelve.open(filename=os.path.join(self._model_path, self.file_name)) as shelve_open:
            self._linear_model = shelve_open[self.file_name]

        inference_result = self._linear_model.predict(self.lr_test)
        inference_result = pd.DataFrame({"result": inference_result})

        return inference_result

    def preprocess(self):
        pass

    def _train_preprocess(self):
        pass

    def _predict_process(self):
        pass

    def eval(self, metrics: BaseMetric, **entity):
        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        y_pred = self._linear_model.predict(self.lr_eval[0])

        assert isinstance(y_pred, np.ndarray)
        assert isinstance(self.lr_eval[1], np.ndarray)

        metrics.evaluate(predict=y_pred, labels_map=self.lr_eval[1])
        metrics = metrics.metrics_result
        assert isinstance(metrics, MetricResult)
        self._val_metrics = metrics

        return metrics

    def get_train_loss(self):
        pass

    def get_val_loss(self):
        pass

    def get_train_metric(self):
        pass

    @property
    def val_metrics(self):
        return self._val_metrics

    def model_save(self, model_path=None):
        if model_path is not None:
            self._model_path = model_path

        assert self._model_path is not None
        assert self._linear_model is not None

        try:
            assert os.path.isdir(self._model_path)
        except AssertionError:
            mkdir(self._model_path)
        with shelve.open(filename=os.path.join(self._model_path, self.file_name)) as shelve_open:
            shelve_open[self.file_name] = self._linear_model

    def update_params(self, **params):
        self._model_param_dict.update(params)

    def set_weight(self):
        pass