# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os

import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from gauss.auto_ml.base_auto_ml import BaseAutoMl
from entity.base_dataset import BaseDataset
from entity.base_metric import BaseMetric
from entity.model import Model


class HyperOptAutoMl(BaseAutoMl):
    def __init__(self, name, train_flag, enable, opt_mode=None):
        super(BaseAutoMl, self).__init__(name=name, train_flag=train_flag, enable=enable)
        opt_pool={}
        if(opt_mode==None):
        else:
            assert(opt_mode in ("aa", "bb", "cc"))
        


    def _train_run(self, **entity):
        model = entity["Model"]
        train_set = entity["train_set"]
        val_set = entity["val_set"]
         


    def _predict_run(self, **entity):
        if self.model_name == "tree_model":
            assert "dataset" in entity.keys()
            self._clean(dataset=entity["dataset"])