# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab

from collections import defaultdict
from entity import Entity

class FeatureItemConf(object):

    def __init__(self, name, index, size=1, dtype="float32", 
                 default_value=None, ftype="numerical"):
        assert dtype in ("int64", "float32", "string", "bool", "date")
        assert ftype in ("numerical", "catagory")
        self.name = name
        self.dtype = dtype
        self.index = index
        self.size = size
        if default_value is None:
            if dtype == "string" or dtype == "date":
                self.default_value = "UNK"
            else:
                self.default_value = 0
        else:
            self.default_value = default_value
        self.ftype = ftype
class FeatureConf(Entity):
    def __init__(self, name, file_path):
        self._file_path = file_path
        self._feature_dict = {}
        super(Entity, self).__init__(
            name = name,
        )
    def parse():
    def reset_feature_type(key, ftype):
        assert(_feature_dict.has_key("key"))
        assert(ftype in ("numerical", "catagory"))


        
