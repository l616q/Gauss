# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

class UdfModelingTree(Object):
   
    def __init__(self,
                 name: str,
                 work_root: str,
                 task_type: str,
                 metric_name: str,
                 train_data_path: str,
                 val_data_path: str=None,
                 feature_configue_path: str=None,
                 dataset_type: str="plain",
                 type_inference: str="plain",
                 data_clear: str="plain",
                 data_clear_flag: []=[True,False],
                 feature_generator: str="featuretools",
                 feature_generator_flag: bool=True,
                 unsupervised_feature_selector: str="unsupervised",
                 unsupervised_feature_generator_flag: bool=True,
                 supervised_feature_selector: str="supervised",
                 supervised_feature_selector_flag: bool=True,
                 model_zoo: list=["xgb", "lightgbm", "cgb", "lr_lightgbm", "dnn"],
                 auto_ml: str="plain"
                 ):  
        self.name = name
        self.work_root = work_root
        self.dataset_type=dataset_type
        self.type_inference=type_inference
        self.data_clear=data_clear
        self.data_clear_flag = data_clear_flag
        self.feature_generator=feature_generator
        self.feature_generator_flag = feature_generator_flag
        self.unsupervised_feature_selector = unsupervised_feature_selector
        self.unsupervised_feature_selector_flag = unsupervised_feature_selector_flag 
        self.supervised_feature_selector=supervised_feature_selector
        self.supervised_feature_selector_flag = supervised_feature_selector_flag
        self.model_zoo=model_zoo
        self.auto_ml=auto_ml
        self.need_data_clear = False

   