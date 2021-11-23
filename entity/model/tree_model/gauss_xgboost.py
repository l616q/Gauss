"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
GBDT model instances, containing xgboost, xgboost and catboost.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

import os.path
import operator

import numpy as np
import pandas as pd
from scipy import special

import xgboost as xgb

from entity.model.model import ModelWrapper
from entity.model.package_dataset import PackageDataset
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from entity.losses.base_loss import LossResult

from utils.base import get_current_memory_gb
from utils.base import mkdir
from utils.bunch import Bunch
from utils.constant_values import ConstantValues
from utils.yaml_exec import yaml_write
from utils.Logger import logger


class GaussXgboost(ModelWrapper):
    """
    xgboost object.
    """
    def __init__(self, **params):
        assert params[ConstantValues.train_flag] in [ConstantValues.train,
                                                     ConstantValues.increment,
                                                     ConstantValues.inference]
        if params[ConstantValues.train_flag] == ConstantValues.train:
            super().__init__(
                name=params[ConstantValues.name],
                model_root_path=params[ConstantValues.model_root_path],
                init_model_root=params[ConstantValues.init_model_root],
                task_name=params[ConstantValues.task_name],
                train_flag=params[ConstantValues.train_flag],
                metric_eval_used_flag=params[ConstantValues.metric_eval_used_flag]
            )
        elif params[ConstantValues.train_flag] == ConstantValues.increment:
            super().__init__(
                name=params[ConstantValues.name],
                model_root_path=params[ConstantValues.model_root_path],
                task_name=params[ConstantValues.task_name],
                train_flag=params[ConstantValues.train_flag],
                decay_rate=params[ConstantValues.decay_rate]
            )
        else:
            assert params[ConstantValues.train_flag] == ConstantValues.inference
            super().__init__(
                name=params[ConstantValues.name],
                model_root_path=params[ConstantValues.model_root_path],
                increment_flag=params[ConstantValues.increment_flag],
                infer_result_type=params[ConstantValues.infer_result_type],
                task_name=params[ConstantValues.task_name],
                train_flag=params[ConstantValues.train_flag]
            )

        self._model_file_name = self._name + ".txt"
        self._model_config_file_name = self._name + ".yaml"
        self._feature_config_file_name = self._name + ".yaml"

        self._loss_function = None
        self._eval_function = None

        self.count = 0

    def __repr__(self):
        pass

    @PackageDataset
    def __load_data(self, **kwargs):
        """
        :param dataset:
        :return: xgb.Dataset
        """
        dataset_bunch = kwargs.get("dataset")
        train_flag = kwargs.get("train_flag")

        # dataset is a BaseDataset object, you can use get_dataset() method to get a Bunch object,
        # including data, target, feature_names, target_names, generated_feature_names.
        assert isinstance(dataset_bunch.data, pd.DataFrame)
        if train_flag == ConstantValues.train or train_flag == ConstantValues.increment:
            data_shape = dataset_bunch.data.shape
            label_shape = dataset_bunch.target.shape
            logger.info("Data shape: {}, label shape: {}".format(data_shape, label_shape))
            assert data_shape[0] == label_shape[0], "Data shape is inconsistent with label shape."

            if dataset_bunch.dataset_weight is not None:
                weight = dataset_bunch.dataset_weight
            else:
                weight = None

            if isinstance(weight, pd.DataFrame):
                weight = weight.values.flatten()

            xgb_data = xgb.DMatrix(
                data=dataset_bunch.data,
                label=dataset_bunch.target,
                weight=weight,
                enable_categorical=True,
                silent=True
            )

            logger.info(
                "Method load_data() has finished, with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            return xgb_data, dataset_bunch
        else:
            xgb_data = xgb.DMatrix(
                data=dataset_bunch.data,
                label=dataset_bunch.target,
                silent=True
            )
        return xgb_data, dataset_bunch

    def _initialize_model(self):
        pass

    def _binary_train(self,
                      train_dataset: BaseDataset,
                      val_dataset: BaseDataset,
                      **entity):
        """
        This method is used to train xgboost
        model in binary classification.
        :param train_dataset:
        :param val_dataset:
        :param entity:
        :return: None
        """
        assert self._train_flag == ConstantValues.train
        assert self._task_name == ConstantValues.binary_classification

        init_model_path = self._init_model_root
        if init_model_path:
            assert os.path.isfile(init_model_path), \
                "Value: init_model_path({}) is not a valid model path.".format(
                    init_model_path)

        params = self._model_params
        params["objective"] = "binary:logistic"

        if entity["loss"] is not None:
            self._loss_function = entity["loss"].loss_fn
            obj_function = self._loss_func
        else:
            obj_function = None

        train_target_names = train_dataset.get_dataset().target_names
        eval_target_names = val_dataset.get_dataset().target_names

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        # One label learning is achieved now, multi_label
        # learning will be supported in future.
        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        train_label_set = pd.unique(train_dataset.get_dataset().target[self._target_names])
        eval_label_set = pd.unique(val_dataset.get_dataset().target[self._target_names])
        train_label_num = len(train_label_set)
        eval_label_num = len(eval_label_set)

        assert train_label_num == eval_label_num and train_label_num == 2, \
            "Set of train label is: {}, length: {}, validation label is {}, length is {}, " \
            "and binary classification can not be used.".format(
                train_label_set, train_label_num, eval_label_set, eval_label_num
            )

        if self._metric_eval_used_flag and entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            params["eval_metric"] = "logloss"
            eval_function = None

        logger.info(
            "Construct xgboost training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        xgb_train = xgb_entity.xgb_train
        xgb_eval = xgb_entity.xgb_eval

        logger.info(
            "Set preprocessing parameters for xgboost, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        if self._model_params is not None:

            self._model_config = {
                "Name": self._name,
                "Normalization": False,
                "Standardization": False,
                "OnehotEncoding": False,
                "ModelParameters": self._model_params
            }

            logger.info(
                "Start training xgboost model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping")
            assert isinstance(xgb_train, xgb.DMatrix)

            self._model = xgb.train(
                params=params,
                dtrain=xgb_train,
                xgb_model=init_model_path,
                num_boost_round=num_boost_round,
                evals=[(xgb_eval, "eval_set")],
                early_stopping_rounds=early_stopping_rounds,
                obj=obj_function,
                feval=eval_function,
                verbose_eval=0,
            )

            logger.info(
                "Training xgboost model finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

        else:
            raise ValueError("Model parameters is None.")
        self.count += 1

    def _multiclass_train(self,
                          train_dataset: BaseDataset,
                          val_dataset: BaseDataset,
                          **entity):
        assert self._train_flag == ConstantValues.train
        assert self._task_name == ConstantValues.multiclass_classification

        init_model_path = self._init_model_root

        params = self._model_params
        params["objective"] = "multi:softmax"

        if entity["loss"] is not None:
            self._loss_function = entity["loss"].loss_fn
            obj_function = self._loss_func
        else:
            obj_function = None

        train_target_names = train_dataset.get_dataset().target_names
        eval_target_names = val_dataset.get_dataset().target_names

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        # One label learning is achieved now, multi_label
        # learning will be supported in future.
        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        train_label_set = pd.unique(train_dataset.get_dataset().target[self._target_names])
        eval_label_set = pd.unique(val_dataset.get_dataset().target[self._target_names])
        train_label_num = len(train_label_set)
        eval_label_num = len(eval_label_set)

        params["num_class"] = train_label_num

        assert train_label_num == eval_label_num and train_label_num > 2 and eval_label_num > 2, \
            "Set of train label is: {}, length: {}, validation label is {}, length is {}, " \
            "and multiclass classification can not be used.".format(
                train_label_set, train_label_num, eval_label_set, eval_label_num
            )

        if self._metric_eval_used_flag and entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            params["eval_metric"] = "mlogloss"
            eval_function = None

        logger.info(
            "Construct xgboost training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        xgb_train = xgb_entity.xgb_train
        xgb_eval = xgb_entity.xgb_eval

        logger.info(
            "Set preprocessing parameters for xgboost, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        if self._model_params is not None:
            self._model_config = {
                "Name": self.name,
                "Normalization": False,
                "Standardization": False,
                "OnehotEncoding": False,
                "ModelParameters": self._model_params
            }

            logger.info(
                "Training xgboost model with params: {}".format(params)
            )
            logger.info(
                "Start training xgboost model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping")

            self._model = xgb.train(
                params=params,
                dtrain=xgb_train,
                xgb_model=init_model_path,
                num_boost_round=num_boost_round,
                evals=[(xgb_eval, "eval_set")],
                early_stopping_rounds=early_stopping_rounds,
                obj=obj_function,
                feval=eval_function,
                verbose_eval=0,
            )

            logger.info(
                "Training xgboost model finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

        else:
            raise ValueError("Model parameters is None.")
        self.count += 1

    def _regression_train(self,
                          train_dataset: BaseDataset,
                          val_dataset: BaseDataset,
                          **entity):
        assert self._task_name == ConstantValues.regression
        assert self._train_flag == ConstantValues.train

        init_model_path = self._init_model_root

        params = self._model_params
        params["objective"] = "reg:squarederror"

        if entity["loss"] is not None:
            self._loss_function = entity["loss"].loss_fn
            obj_function = self._loss_func
        else:
            obj_function = None

        train_target_names = train_dataset.get_dataset().target_names
        eval_target_names = val_dataset.get_dataset().target_names

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        if self._metric_eval_used_flag and entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            params["eval_metric"] = "rmse"
            eval_function = None

        logger.info(
            "Construct xgboost training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        xgb_train = xgb_entity.xgb_train
        xgb_eval = xgb_entity.xgb_eval

        logger.info(
            "Set preprocessing parameters for xgboost, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        if self._model_params is not None:

            self._model_config = {
                "Name": self.name,
                "Normalization": False,
                "Standardization": False,
                "OnehotEncoding": False,
                "ModelParameters": self._model_params
            }

            logger.info(
                "Start training xgboost model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping")

            self._model = xgb.train(
                params=params,
                dtrain=xgb_train,
                xgb_model=init_model_path,
                num_boost_round=num_boost_round,
                evals=[(xgb_eval, "eval_set")],
                early_stopping_rounds=early_stopping_rounds,
                obj=obj_function,
                feval=eval_function,
                verbose_eval=0,
            )

            params["num_boost_round"] = num_boost_round
            params["early_stopping_rounds"] = early_stopping_rounds

            logger.info(
                "Training xgboost model finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

        else:
            raise ValueError("Model parameters is None.")
        self.count += 1

    def _binary_increment(self, train_dataset: BaseDataset, **entity):
        """
        This method is used to train xgboost (booster)
        model in binary classification.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        params = self._model_params
        params["objective"] = "binary:logistic"

        init_model_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(init_model_path)

        assert self._train_flag == ConstantValues.increment
        assert self._task_name == ConstantValues.binary_classification

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=None,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        xgb_train = xgb_entity.xgb_train

        params.update({'process_type': 'update',
                       'updater': 'refresh',
                       'refresh_leaf': True})

        num_boost_round = params.pop("num_boost_round")
        params.pop("early_stopping")

        self._model = xgb.train(
            params=params,
            dtrain=xgb_train,
            xgb_model=init_model_path,
            num_boost_round=num_boost_round,
            verbose_eval=0,
        )

    def _multiclass_increment(self, train_dataset: BaseDataset, **entity):
        """
        This method is used to train xgboost (booster)
        model in multiclass classification.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        params = self._model_params
        params["objective"] = "multi:softmax"

        init_model_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(init_model_path)

        assert self._train_flag == ConstantValues.increment
        assert self._task_name == ConstantValues.multiclass_classification

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=None,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        xgb_train = xgb_entity.xgb_train

        params.update({'process_type': 'update',
                       'updater': 'refresh',
                       'refresh_leaf': True})

        num_boost_round = params.pop("num_boost_round")
        params.pop("early_stopping")

        self._model = xgb.train(
            params=params,
            dtrain=xgb_train,
            xgb_model=init_model_path,
            num_boost_round=num_boost_round,
            verbose_eval=0,
        )

    def _regression_increment(self, train_dataset: BaseDataset, **entity):
        """
        This method is used to train xgboost (booster)
        model in regression.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        params = self._model_params
        params["objective"] = "reg:squarederror"

        init_model_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(init_model_path)

        assert self._train_flag == ConstantValues.increment
        assert self._task_name == ConstantValues.regression

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=None,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        xgb_train = xgb_entity.xgb_train

        params.update({'process_type': 'update',
                       'updater': 'refresh',
                       'refresh_leaf': True})

        num_boost_round = params.pop("num_boost_round")
        params.pop("early_stopping")

        self._model = xgb.train(
            params=params,
            dtrain=xgb_train,
            xgb_model=init_model_path,
            num_boost_round=num_boost_round,
            verbose_eval=0,
        )

    def _predict_prob(self, infer_dataset: BaseDataset, **entity):
        assert self._train_flag == ConstantValues.inference
        model_file_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(model_file_path)

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                infer_dataset=infer_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        infer_dataset = xgb_entity.infer_dataset
        xgb_infer = xgb_entity.xgb_infer
        assert "data" in infer_dataset

        self._model = xgb.Booster(
            model_file=model_file_path
        )

        inference_result = self._model.predict(data=xgb_infer, output_margin=False)
        inference_result = pd.DataFrame({"result": inference_result})
        return inference_result

    def _predict_logit(self, infer_dataset: BaseDataset, **entity):
        assert self._train_flag == ConstantValues.inference

        model_file_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(model_file_path)

        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                infer_dataset=infer_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        infer_dataset = xgb_entity.infer_dataset
        xgb_infer = xgb_entity.xgb_infer
        assert "data" in infer_dataset

        self._model = xgb.Booster(
            model_file=model_file_path
        )

        inference_result = self._model.predict(data=xgb_infer, output_margin=True)
        inference_result = pd.DataFrame({"result": inference_result})
        return inference_result

    def _train_preprocess(self):
        pass

    def _predict_preprocess(self):
        pass

    def _eval(self,
              train_dataset: BaseDataset,
              val_dataset: BaseDataset,
              metric: BaseMetric,
              **entity):
        """
        Evaluating
        :param val_dataset: BaseDataset object, used to get validation metric and loss.
        :param train_dataset: BaseDataset object, used to get training metric and loss.
        :param metric: BaseMetric object, used to calculate metric.
        :return: None
        """
        logger.info(
            "Starting evaluation, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        assert "data" in train_dataset.get_dataset() and "target" in train_dataset.get_dataset()
        assert "data" in val_dataset.get_dataset() and "target" in val_dataset.get_dataset()

        # 去除label name
        xgb_entity = self.__xgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        train_dataset = xgb_entity.train_dataset
        xgb_train = xgb_entity.xgb_train
        eval_dataset = xgb_entity.eval_dataset
        xgb_eval = xgb_entity.xgb_eval

        train_data, train_label, train_target_names = train_dataset.data, train_dataset.target, train_dataset.target_names
        eval_data, eval_label, eval_target_names = eval_dataset.data, eval_dataset.target, eval_dataset.target_names

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        # 默认生成的为预测值的概率值，传入metric之后再处理.
        logger.info(
            "Starting predicting, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        # 默认生成的为预测值的概率值，传入metric之后再处理.
        val_y_pred = self._model.predict(
            xgb_eval
        )

        train_y_pred = self._model.predict(xgb_train)

        assert isinstance(val_y_pred, np.ndarray)
        assert isinstance(train_y_pred, np.ndarray)

        train_label = self.__generate_labels_map(
            target=train_label,
            target_names=train_target_names)

        eval_label = self.__generate_labels_map(
            target=eval_label,
            target_names=eval_target_names)

        metric.label_name = self._target_names

        metric.evaluate(predict=val_y_pred, labels_map=eval_label)
        val_metric_result = metric.metric_result

        metric.evaluate(predict=train_y_pred, labels_map=train_label)
        train_metric_result = metric.metric_result

        assert isinstance(val_metric_result, MetricResult)
        assert isinstance(train_metric_result, MetricResult)

        self._val_metric_result = val_metric_result
        self._train_metric_result = train_metric_result

        logger.info("train_metric: %s, val_metric: %s",
                    self._train_metric_result.result,
                    self._val_metric_result.result)

    @classmethod
    def __generate_labels_map(cls, target, target_names):
        assert isinstance(target, pd.DataFrame)
        assert isinstance(target_names, list)

        labels_map = {}
        for feature in target_names:
            labels_map[feature] = target[feature]
        return labels_map

    def model_save(self):
        assert self._model_save_root is not None
        assert self._model is not None

        try:
            assert os.path.isdir(self._model_save_root)

        except AssertionError:
            mkdir(self._model_save_root)

        self._model.save_model(
            os.path.join(
                self._model_save_root,
                self._model_file_name
            )
        )

        yaml_write(yaml_dict=self._model_config,
                   yaml_file=os.path.join(
                       self._model_config_root,
                       self._model_config_file_name
                   )
                   )

        yaml_write(yaml_dict={"features": self._feature_list},
                   yaml_file=os.path.join(
                       self._feature_config_root,
                       self._feature_config_file_name
                   )
                   )

    def _update_best(self):
        """
        Do not need to operate.
        :return: None
        """

    def _set_best(self):
        """
        Do not need to operate.
        :return: None
        """

    def _loss_func(self, preds, train_data):
        assert self._loss_function is not None
        preds = special.expit(preds)
        loss = self._loss_function
        label = train_data.get_label()
        loss_result = loss(score=preds, label=label)

        assert isinstance(loss_result, LossResult)
        return loss_result.grad, loss_result.hess

    def _eval_func(self, preds, train_data):
        assert self._eval_function is not None
        label_map = {self._target_names: train_data.get_label()}
        metric_result = self._eval_function(predict=preds, labels_map=label_map)

        assert isinstance(metric_result, MetricResult)
        assert metric_result.optimize_mode in ["maximize", "minimize"]
        is_higher_better = True if metric_result.optimize_mode == "max            inference_result = self._model.predict(data=infer_dataset.data, raw_score=False)imize" else False
        return metric_result.metric_name, float(metric_result.result), is_higher_better

    def __xgb_preprocessing(self, **params):
        xgb_entity = Bunch()
        if ConstantValues.train_dataset in params and params[ConstantValues.train_dataset]:
            params["dataset"] = params.pop("train_dataset")
            xgb_train, train_dataset = self.__load_data(**params)
            xgb_entity.xgb_train = xgb_train
            xgb_entity.train_dataset = train_dataset

        if ConstantValues.val_dataset in params and params[ConstantValues.val_dataset]:
            params["dataset"] = params.pop("val_dataset")
            xgb_eval, eval_dataset = self.__load_data(**params)
            xgb_entity.xgb_eval = xgb_eval
            xgb_entity.eval_dataset = eval_dataset

        if ConstantValues.infer_dataset in params and params[ConstantValues.infer_dataset]:
            params["dataset"] = params.pop("infer_dataset")
            xgb_infer, infer_dataset = self.__load_data(**params)
            xgb_entity.xgb_infer = xgb_infer
            xgb_entity.infer_dataset = infer_dataset
        return xgb_entity
