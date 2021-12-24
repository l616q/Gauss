"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import abc

from Gauss.gauss.component import Component


class BaseFeatureSelector(Component):
    """
    BaseFeatureSelector object.
    """

    def __init__(self,
                 name: str,
                 train_flag: str,
                 enable: bool,
                 task_name: str,
                 source_file_path: str = None,
                 final_file_path: str = None):
        super().__init__(
            name=name,
            train_flag=train_flag,
            enable=enable,
            task_name=task_name,
            source_file_path=source_file_path,
            final_file_path=final_file_path
        )

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass

    @abc.abstractmethod
    def _increment_run(self, **entity):
        pass
