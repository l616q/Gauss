# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab


from collections import defaultdict

class FeatureItemConf(object):

    def __init__(self, name, dtype, start_postion, range, 
                 default_value=None, ftype="numerical"):
        assert dtype in ("int64", "float32", "string", "bool", "date")
        self._name = name
        self._dtype = dtype
        self._start_postion = start_postion
        self._range = range
        if default_value is None:
            if _dtype == "string" or _dtype == "date":
                self._default_value = "UNK"
            else:
                self._default_value = 0
        else:
            self._default_value = default_value
        self._ftype = ftype
    def reset_feature_type(ftype):
        self._ftype = _ftype
class FeatureConf(object):
    def __init__(self, name, file_path):
        self._name = name
        self._file_path = file_path
        self._dict = {}
    def parse():
    def reset_feature_type(key, ftype):
    def set_feature_file(file_path):
        self._file_path = file_path


        
