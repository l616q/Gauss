"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import abc
from gauss.component import Component


class BaseAutoML(Component):
    def __init__(self,
                 name: str,
                 train_flag: str,
                 enable: bool,
                 task_name: str,
                 opt_model_names: list = None,
                 source_file_path: str = None,
                 final_file_path: str = None):
        """

        :param name:
        :param train_flag:
        :param enable:
        :param task_name:
        :param opt_model_names:
        :param source_file_path:
        :param final_file_path:
        """
        super().__init__(
            name=name,
            train_flag=train_flag,
            enable=enable,
            task_name=task_name,
            source_file_path=source_file_path,
            final_file_path=final_file_path
        )

        self._opt_model_names = opt_model_names

        self._train_method_count = 0
        self._trial_count = 0
        self._algorithm_method_count = 0

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass

    @abc.abstractmethod
    def _increment_run(self, **entity):
        pass

    @property
    def train_method_count(self):
        return self._train_method_count

    @property
    def algorithm_method_count(self):
        return self._algorithm_method_count

    @property
    def trial_count(self):
        return self._trial_count
