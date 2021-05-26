# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab


from entity import Entity

class BaseDataset(Entity):
    def __init__(self,
                 name: str,
                 data_path: str,
                 target_name="target_name",
                 memory_only=True):
        self._data_path = data_path
        self._target_name = target_name
        self._memory_only = memory_only
        self._colum_size = 0
        self._row_size = 0
        self._default_print_size = 100
        super(Entity, self).__init__(
            name = name,
        )
    @abc.abstractmethod
    def load_data(self)
        pass
    @property
    def colum_size(self):
        return self._colum_size
    @property
    def row_size(self):
        return self._row_size
    @property
    def target_name(self):
        return self._target_name
    
    
    
