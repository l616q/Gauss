"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
GBDT model instances, containing xgboost, xgboost and catboost.
"""
from Gauss.entity.dataset.base_dataset import BaseDataset
from Gauss.entity.metrics.base_metric import BaseMetric
from Gauss.entity.model.model import ModelWrapper


class GaussCatboost(ModelWrapper):
    def __init__(self, **params):
        super().__init__(**params)

    def _initialize_model(self):
        pass

    def _update_best(self):
        pass

    def _binary_train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        pass

    def _multiclass_train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        pass

    def _regression_train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        pass

    def _binary_increment(self, train_dataset: BaseDataset, **entity):
        pass

    def _multiclass_increment(self, train_dataset: BaseDataset, **entity):
        pass

    def _regression_increment(self, train_dataset: BaseDataset, **entity):
        pass

    def _predict_prob(self, infer_dataset: BaseDataset, **entity):
        pass

    def _predict_logit(self, infer_dataset: BaseDataset, **entity):
        pass

    def _eval(self, train_dataset: BaseDataset, val_dataset: BaseDataset, metric: BaseMetric, **entity):
        pass

    def model_save(self):
        pass

    def _set_best(self):
        pass

    def _train_preprocess(self):
        pass

    def _predict_preprocess(self):
        pass

    def _loss_func(self, *params):
        pass

    def _eval_func(self, *params):
        pass
