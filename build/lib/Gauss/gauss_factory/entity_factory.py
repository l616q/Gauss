"""-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab"""
from Gauss.gauss_factory.abstarct_guass import AbstractGauss

from Gauss.entity.dataset.plain_dataset import PlaintextDataset
from Gauss.entity.feature_configuration.feature_config import FeatureConf
from Gauss.entity.model.tree_model.gauss_lightgbm import GaussLightgbm
from Gauss.entity.model.tree_model.gauss_xgboost import GaussXgboost
from Gauss.entity.model.tree_model.guass_catboost import GaussCatboost
from Gauss.entity.metrics.udf_metric import AUC
from Gauss.entity.metrics.udf_metric import BinaryF1
from Gauss.entity.metrics.udf_metric import MulticlassF1
from Gauss.entity.metrics.udf_metric import MSE
from Gauss.entity.losses.udf_loss import MSELoss
from Gauss.entity.losses.udf_loss import BinaryLogLoss

"""
This class will be used in local_pipeline.
"""
class EntityFactory(AbstractGauss):
    def get_entity(self, entity_name: str, **params):
        if entity_name is None:
            return None
        if entity_name.lower() == "plaindataset":
            # parameters: name: str, task_type: str, data_pair: Bunch, data_path: str, target_name: str, memory_only: bool
            return PlaintextDataset(**params)
        elif entity_name.lower() == "feature_configure":
            # parameters: name: str, file_path: str
            return FeatureConf(**params)
        elif entity_name.lower() == "auc":
            # parameters: name: str, label_name: str
            return AUC(**params)
        elif entity_name.lower() == "binary_f1":
            # parameters: name: str, label_name: str
            return BinaryF1(**params)
        elif entity_name.lower() == "multiclass_f1":
            return MulticlassF1(**params)
        elif entity_name.lower() == "mse":
            return MSE(**params)
        elif entity_name.lower() == "mse_loss":
            return MSELoss(**params)
        elif entity_name.lower() == "binary_logloss":
            return BinaryLogLoss(**params)
        elif entity_name.lower() == "lightgbm":
            # parameters: name: str, label_name: str
            return GaussLightgbm(**params)
        elif entity_name.lower() == "xgboost":
            return GaussXgboost(**params)
        elif entity_name.lower() == "catboost":
            return GaussCatboost(**params)
        else:
            raise ValueError("Entity factory can not construct entity by name: %s.", entity_name)

    def get_component(self, component_name: str):
        return None
