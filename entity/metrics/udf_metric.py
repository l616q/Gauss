"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic Inc. All rights reserved.
Authors: Lab
This file contains user-defined metric entities,
and these entities is only used for binary classification task.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

from entity.metrics.base_metric import BaseMetric
from entity.metrics.base_metric import MetricResult
from utils.constant_values import ConstantValues


class AUC(BaseMetric):
    """
    Binary classification task.
    """
    def __init__(self, **params):
        super().__init__(name=params["name"],
                         optimize_mode="maximize")

        self.__callback_func = params[ConstantValues.callback_func]
        self._metric_result = None

        message = "Create AUC object successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)

    def __repr__(self):
        print("AUC is running!")

    def evaluate(self, predict: np.ndarray, labels_map: dict):
        """
        :param predict: np.ndarray object, (n_sample,)
        :param labels_map: key: label name, str, value: np.ndarray object, (n_samples,)
        :return: MetricResult object
        """
        try:
            assert self._label_name is not None, "Value: label name can not be None."
            assert self._label_name in labels_map, \
                "Label name: {} does not exist in labels_map: {}".format(
                    self._label_name, labels_map.keys()
                )
        except AssertionError:
            message = "Parameters is not complete."
            self.__callback_func(type_name="entity_configure",
                                 object_name="metric",
                                 success_flag=False,
                                 message=message)
            raise ValueError(message)

        label = labels_map[self._label_name]
        if np.sum(label) == 0 or np.sum(label) == label.shape[0]:
            self._metric_result = MetricResult(name=self._name,
                                               metric_name=self._name,
                                               result=float('nan'),
                                               optimize_mode=self._optimize_mode)
        else:
            auc = roc_auc_score(y_true=label, y_score=predict)
            self._metric_result = MetricResult(name=self._name,
                                               metric_name=self._name,
                                               result=auc,
                                               meta={'#': predict.size},
                                               optimize_mode=self._optimize_mode)

        message = "Calculate AUC Value successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)

        return self._metric_result

    @property
    def required_label_names(self):
        return [self._label_name]

    @property
    def metric_result(self):
        assert self._metric_result is not None
        return self._metric_result


class BinaryF1(BaseMetric):
    """
    Binary classification task.
    """
    def __init__(self, **params):
        super().__init__(name=params["name"],
                         optimize_mode="maximize")

        self.__callback_func = params[ConstantValues.callback_func]
        self._metric_result = None
        self._threshold = 0.5

        message = "Create BinaryF1 object successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)

    def __repr__(self):
        print("F1 is running!")

    def evaluate(self, predict: np.ndarray, labels_map: dict):
        """
        :param predict: np.ndarray object, (n_sample)
        :param labels_map: np.ndarray object, (n_samples)
        :return: MetricResult object
        """
        try:
            assert self._label_name is not None, "Value: label name can not be None."
            assert self._label_name in labels_map, \
                "Label name: {} does not exist in labels_map: {}".format(
                    self._label_name, labels_map.keys()
                )
        except AssertionError:
            message = "Parameters is not complete."
            self.__callback_func(type_name="entity_configure",
                                 object_name="metric",
                                 success_flag=False,
                                 message=message)
            raise ValueError(message)

        label = labels_map[self._label_name]
        if np.sum(label) == 0 or np.sum(label) == label.shape[0]:
            self._metric_result = MetricResult(
                name=self._name,
                metric_name=self._name,
                result=float('nan'),
                optimize_mode=self._optimize_mode)
        else:
            predict_label = np.round(predict)

            f1 = f1_score(y_true=label, y_pred=predict_label)
            self._metric_result = MetricResult(
                name=self._name,
                metric_name=self._name,
                result=f1,
                meta={'#': predict.size},
                optimize_mode=self._optimize_mode)

        message = "Calculate BinaryF1 Value successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)

        return self._metric_result

    @property
    def required_label_names(self):
        return [self._label_name]

    @property
    def metric_result(self):
        assert self._metric_result is not None
        return self._metric_result


class MulticlassF1(BaseMetric):
    """
    multiclass classification task.
    """
    def __init__(self, **params):
        super().__init__(name=params["name"],
                         optimize_mode="maximize")

        self.__callback_func = params[ConstantValues.callback_func]

        self._metric_result = None
        self._threshold = 0.5

        message = "Create MulticlassF1 object successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)

    def __repr__(self):
        print("F1 is running!")

    def evaluate(self, predict: np.ndarray, labels_map: dict):
        """
        :param predict: np.ndarray object, (n_sample)
        :param labels_map: np.ndarray object, (n_samples)
        :return: MetricResult object
        """
        try:
            assert self._label_name is not None, "Value: label name can not be None."
            assert self._label_name in labels_map, \
                "Label name: {} does not exist in labels_map: {}".format(
                    self._label_name, labels_map.keys()
                )
        except AssertionError:
            message = "Parameters is not complete."
            self.__callback_func(type_name="entity_configure",
                                 object_name="metric",
                                 success_flag=False,
                                 message=message)
            raise ValueError(message)

        label = labels_map[self._label_name]
        predict_label = [result.tolist().index(max(result)) for result in predict]

        f1 = f1_score(y_true=label, y_pred=predict_label, average="macro")
        self._metric_result = MetricResult(
            name=self._name,
            metric_name=self._name,
            result=f1,
            meta={'#': predict.size},
            optimize_mode=self._optimize_mode)

        message = "Calculate MulticlassF1 Value successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)
        return self._metric_result

    @property
    def required_label_names(self):
        return [self._label_name]

    @property
    def metric_result(self):
        assert self._metric_result is not None
        return self._metric_result


class MSE(BaseMetric):
    """
    regression task.
    """
    def __init__(self, **params):
        super().__init__(name=params["name"],
                         optimize_mode="minimize")

        self.__callback_func = params[ConstantValues.callback_func]

        self._metric_result = None
        self._threshold = 0.5

        message = "Create MSE object successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)

    def __repr__(self):
        print("MSE is running!")

    def evaluate(self, predict: np.ndarray, labels_map: dict):
        """
        :param predict: np.ndarray object, (n_sample)
        :param labels_map: np.ndarray object, (n_samples)
        :return: MetricResult object
        """
        try:
            assert self._label_name is not None, "Value: label name can not be None."
            assert self._label_name in labels_map, \
                "Label name: {} does not exist in labels_map: {}".format(
                    self._label_name, labels_map.keys()
                )
        except AssertionError:
            message = "Parameters is not complete."
            self.__callback_func(type_name="entity_configure",
                                 object_name="metric",
                                 success_flag=False,
                                 message=message)
            raise ValueError(message)

        label = labels_map[self._label_name]

        mse = mean_squared_error(y_true=label, y_pred=predict)
        self._metric_result = MetricResult(
            name=self._name,
            metric_name=self._name,
            result=mse,
            meta={'#': predict.size},
            optimize_mode=self._optimize_mode)

        message = "Calculate MSE Value successfully."
        self.__callback_func(type_name="entity_configure",
                             object_name="metric",
                             success_flag=True,
                             message=message)

        return self._metric_result

    @property
    def required_label_names(self):
        return [self._label_name]

    @property
    def metric_result(self):
        assert self._metric_result is not None
        return self._metric_result
