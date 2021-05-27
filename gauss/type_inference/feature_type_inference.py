# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab

from gauss.entity import BaseTypeInference

class BaseTypeInference(BaseTypeInference):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 source_file_path="null",
                 target_file_path: str,
                 target_file_prefix="target"):
        super(BaseTypeInference, self).__init__(
             name = name,
             train_flag = train_flag,
             source_file_path = source_file_path,
             target_file_path = target_file_path,
             target_file_prefix = target_file_prefix
        )
    def _train_run(self,**entity):
        assert entity.has_key("dataset")
        