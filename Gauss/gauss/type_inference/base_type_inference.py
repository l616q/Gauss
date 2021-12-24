"""-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab"""
import abc

from Gauss.gauss.component import Component
from Gauss.entity.dataset.base_dataset import BaseDataset


class BaseTypeInference(Component):
    def __init__(self,
                 name: str,
                 train_flag: str,
                 task_name: str,
                 source_file_path: str = None,
                 final_file_path: str = None
                 ):
        """
        BaseTypeInference object.
        :param name: type inference name
        :param task_name: str object, optional: binary_classification, multiclass_classification, regression
        :param train_flag: str object, optional: train, increment, inference
        :param source_file_path: input feature configure file path
        :param final_file_path: output feature configure file path
        """
        super(BaseTypeInference, self).__init__(
            name=name,
            train_flag=train_flag,
            enable=True,
            task_name=task_name,
            source_file_path=source_file_path,
            final_file_path=final_file_path
        )
        self._update_flag = False

    @abc.abstractmethod
    def _dtype_inference(self, dataset: BaseDataset):
        pass

    @abc.abstractmethod
    def _ftype_inference(self, dataset: BaseDataset):
        pass

    @abc.abstractmethod
    def _check_target_columns(self, target: BaseDataset):
        pass

    def _train_run(self, **entity):
        pass

    def _predict_run(self, **entity):
        pass

    def _increment_run(self, **entity):
        pass
