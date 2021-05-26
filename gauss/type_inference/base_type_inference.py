# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Callable, List

class BaseTypeInference(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 source_file_path: str,
                 target_file_path: str):
        self._source_file_path = source_file_path
        self._target_file_path = target_file_path
        self._target_file_prefix = target_file_prefix
        self._update_flag = False
        super(Component, self).__init__(
            name = name,
            train_flag = train_flag
        )