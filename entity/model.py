# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab


from entity import Entity
from entity import BaseDataset
from entity import Entity

class Model(Entity):
    def __init__(self,
                 name: str,
                 model_path: str,
                 task_type:  str,
                 train_flag: bool
                 ):
        self._model_path = file_path
        self._train_flag = train_flag
        self._train_finished = False
        super(Entity, self).__init__(
            name = name,
        )
    @abc.abstractmethod
    def train(self, train: BaseDataset, val: BaseDataset=None):
        pass
    @abc.abstractmethod
    def predict(self, test: BaseDataset):
        pass
    # @abc.abstractmethod
    # def eval(self, test: BaseDataset):
    #     pass
    # @abc.abstractmethod
    # def get_train_metric(self):
    # @abc.abstractmethod
    # def get_train_loss(self):
    # @abc.abstractmethod
    # def get_val_loss(self):
