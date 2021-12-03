"""-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab"""
import copy
import shelve

import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from gauss.data_clear.base_data_clear import BaseDataClear
from entity.dataset.base_dataset import BaseDataset

from utils.base import get_current_memory_gb
from utils.bunch import Bunch
from utils.yaml_exec import yaml_read
from utils.Logger import logger
from utils.constant_values import ConstantValues


# 需要传入三个参数， 数模型的数据/非数模型的数据， yaml文件， base dataset
class PlainDataClear(BaseDataClear):
    def __init__(self, **params):
        """Construct a PlainDataClear.

        :param name: The name of the Component.
        :param strategy_dict: strategy for missing value. You can use 'mean', 'median', 'most_frequent' and 'constant',
        and if 'constant' is used, an efficient fill_value must be given.you can use two strategy_dict formats, for example:
        1 > {"model": {"name": "ftype"}, "category": {"name": 'most_frequent'}, "numerical": {"name": "mean"}, "bool": {"name": "most_frequent"}, "datetime": {"name": "most_frequent"}}
        2 > {"model": {"name": "feature"}, "feature 1": {"name": 'most_frequent'}, "feature 2": {"name": 'constant', "fill_value": 0}}
        But you can just use one of them, PlainDataClear object will use strict coding check programming.
        """
        super(PlainDataClear, self).__init__(name=params[ConstantValues.name],
                                             train_flag=params[ConstantValues.train_flag],
                                             enable=params[ConstantValues.enable],
                                             task_name=params[ConstantValues.task_name],
                                             source_file_path=params[ConstantValues.source_file_path],
                                             final_file_path=params[ConstantValues.final_file_path])

        self.__callback_func = params[ConstantValues.callback_func]

        self._data_clear_configure_path = params["data_clear_configure_path"]
        self._strategy_dict = params["strategy_dict"]
        self._missing_values = np.nan

        self._default_cat_impute_model = SimpleImputer(missing_values=self._missing_values,
                                                       strategy="most_frequent")
        self._default_num_impute_model = SimpleImputer(missing_values=self._missing_values,
                                                       strategy="mean")

        self._impute_models = {}
        self._already_data_clear = None

    def _train_run(self, **entity):
        logger.info("Data clear component flag: " + str(self._enable))
        if self._enable is True:
            self._already_data_clear = True
            assert ConstantValues.train_dataset in entity.keys()
            logger.info("Running clean() method and clearing, with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            self._clean(dataset=entity["train_dataset"])
            self.__check_dtype(dataset=entity["train_dataset"])
        else:
            self._already_data_clear = False
        logger.info("Data clearing feature configuration is generating, with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self.final_configure_generation()

        logger.info("Data clearing impute models serializing, with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self._data_clear_serialize()

        message = "Plain data clear executes successfully."
        self.__callback_func(type_name="component_configure",
                             object_name="data_clear",
                             success_flag=True,
                             message=message)

    def __check_dtype(self, dataset: BaseDataset):
        feature_conf = yaml_read(self._source_file_path)
        data = dataset.get_dataset().data
        for item in feature_conf.keys():
            item_configure = Bunch(**feature_conf[item])
            if "category" in item_configure["ftype"] or "datetime" in item_configure["ftype"]:
                data[item_configure.name] = data[item_configure["name"]].astype("category")
            else:
                if "int" in item_configure["dtype"]:
                    data[item_configure.name] = data[item_configure["name"]].astype("int64")
                elif "float" in item_configure["dtype"]:
                    data[item_configure.name] = data[item_configure["name"]].astype("float64")
                else:
                    message = "Feature: {} is not category ftype, but its dtype is numerical.".format(
                            item_configure.name)
                    self.__callback_func(type_name="component_configure",
                                         object_name="data_clear",
                                         success_flag=False,
                                         message=message)
                    raise ValueError(message)

    def _increment_run(self, **entity):
        assert ConstantValues.increment_dataset in entity.keys()
        dataset = entity[ConstantValues.increment_dataset]

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        feature_names = dataset.get_dataset().feature_names
        feature_conf = yaml_read(self._source_file_path)
        self._aberrant_modify(data=data)

        if self._enable is True:
            with shelve.open(self._data_clear_configure_path) as shelve_open:
                dc_model_list = shelve_open['impute_models']

            for col in feature_names:
                item_conf = feature_conf[col]
                if dc_model_list.get(col):
                    item_data = np.array(data[col]).reshape(-1, 1)

                    if "int" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.int64))
                    elif "float" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.float64))
                    else:
                        dc_model_list.get(col).fit(item_data)

                    item_data = dc_model_list.get(col).transform(item_data)
                    data[col] = item_data.reshape(1, -1).squeeze(axis=0)
        self.__check_dtype(dataset=dataset)

        message = "Plain data clear executes successfully."
        self.__callback_func(type_name="component_configure",
                             object_name="data_clear",
                             success_flag=True,
                             message=message)

    def _predict_run(self, **entity):
        assert "infer_dataset" in entity.keys()
        dataset = entity["infer_dataset"]

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        feature_names = dataset.get_dataset().feature_names
        feature_conf = yaml_read(self._source_file_path)
        self._aberrant_modify(data=data)

        if self._enable is True:
            with shelve.open(self._data_clear_configure_path) as shelve_open:
                dc_model_list = shelve_open['impute_models']

            for col in feature_names:
                item_conf = feature_conf[col]
                if dc_model_list.get(col):
                    item_data = np.array(data[col]).reshape(-1, 1)

                    if "int" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.int64))
                    elif "float" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.float64))
                    else:
                        dc_model_list.get(col).fit(item_data)

                    item_data = dc_model_list.get(col).transform(item_data)
                    data[col] = item_data.reshape(1, -1).squeeze(axis=0)
        self.__check_dtype(dataset=dataset)

        message = "Plain data clear executes successfully."
        self.__callback_func(type_name="component_configure",
                             object_name="data_clear",
                             success_flag=True,
                             message=message)

    def _clean(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        feature_names = dataset.get_dataset().feature_names

        assert isinstance(data, pd.DataFrame)
        self.__clear_dataframe(dataset=dataset)
        feature_conf = yaml_read(self._source_file_path)

        for feature in feature_names:
            item_data = np.array(data[feature])
            item_conf = feature_conf[feature]

            if self._strategy_dict is not None:
                if self._strategy_dict["model"]["name"] == "ftype":
                    impute_model = SimpleImputer(missing_values=self._missing_values,
                                                 strategy=self._strategy_dict[item_conf['ftype']]["name"],
                                                 fill_value=self._strategy_dict[item_conf['ftype']].get("fill_value"),
                                                 add_indicator=False)
                else:
                    assert self._strategy_dict["model"]["name"] == "feature"
                    impute_model = SimpleImputer(missing_values=self._missing_values,
                                                 strategy=self._strategy_dict[feature]["name"],
                                                 fill_value=self._strategy_dict[feature].get("fill_value"),
                                                 add_indicator=False)
            else:
                if item_conf['ftype'] == "numerical":
                    impute_model = copy.deepcopy(self._default_num_impute_model)
                else:
                    assert item_conf['ftype'] in ["category", "bool", "datetime"]
                    impute_model = copy.deepcopy(self._default_cat_impute_model)

            item_data = item_data.reshape(-1, 1)

            # This block is used to avoid some warning in special running environment.
            if "int" in str(item_data.dtype):
                impute_model.fit(item_data.astype("int64"))
            elif "float" in str(item_data.dtype):
                impute_model.fit(item_data.astype("float64"))
            else:
                impute_model.fit(item_data)

            item_data = impute_model.transform(item_data)
            item_data = item_data.reshape(1, -1).squeeze(axis=0)

            self._impute_models[feature] = impute_model
            data[feature] = item_data

    def __clear_dataframe(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        feature_names = dataset.get_dataset().feature_names

        assert isinstance(data, pd.DataFrame)
        self._aberrant_modify(data=data)

        feature_conf = yaml_read(self._source_file_path)

        def convert_int(x):
            if isinstance(x, int):
                return x
            elif isinstance(x, float):
                if not np.isnan(x):
                    return int(x)
                else:
                    return x
            elif isinstance(x, str):
                if len(x) > 0:
                    try:
                        return int(x)
                    except ValueError:
                        return int(x[0])
                else:
                    return x
            else:
                message = "Value: {} can not be converted to a int value."
                self.__callback_func(type_name="component_configure",
                                     object_name="data_clear",
                                     success_flag=False,
                                     message=message)
                raise ValueError(message)

        def convert_float(x):
            if isinstance(x, float):
                return x
            elif isinstance(x, str):
                return float(x)
            else:
                message = "Value: {} can not be converted to a float value."
                self.__callback_func(type_name="component_configure",
                                     object_name="data_clear",
                                     success_flag=False,
                                     message=message)
                raise ValueError(message)

        def convert_string(x):
            if isinstance(x, str):
                return x
            else:
                return str(x)

        for feature in feature_names:
            # feature configuration, dict type
            item_conf = feature_conf[feature]
            if "int" in item_conf["dtype"]:
                data[feature] = data[feature].map(convert_int)
            elif "float" in item_conf["dtype"]:
                data[feature] = data[feature].map(convert_float)
            else:
                data[feature] = data[feature].map(convert_string)

    def _aberrant_modify(self, data: pd.DataFrame):
        feature_conf = yaml_read(self._source_file_path)
        for col in data.columns:
            dtype = feature_conf[col]["dtype"]
            check_nan = [self._type_check(item, dtype) for item in data[col]]
            if not all(check_nan):
                data[col] = data[col].where(check_nan)

    @classmethod
    def _type_check(cls, item, dtype):
        """
        this method is used to infer if a type of an object is int, float or string based on TypeInference object.
        :param item:
        :param dtype: dtype of a feature in feature configure file.
        :return: bool
        """
        assert dtype in ["int64", "float64", "string"]

        # When dtype is int, np.nan or string item can exist.
        if dtype == "int64":
            if dtype == "int64":
                try:
                    int(item)
                    return True
                except ValueError:
                    try:
                        float(item)
                        value = float(item)
                        if abs(value - round(value)) < 0.00001:
                            return True
                        else:
                            return False
                    except ValueError:
                        return False

        if dtype == "float64":
            try:
                float(item)
                return True
            except ValueError:
                return False
        return True

    def _data_clear_serialize(self):
        # 序列化label encoding模型字典
        with shelve.open(self._data_clear_configure_path) as shelve_open:
            shelve_open['impute_models'] = self._impute_models

    def final_configure_generation(self):
        feature_conf = yaml_read(yaml_file=self._source_file_path)

        with open(self._final_file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(feature_conf, yaml_file)

    @property
    def already_data_clear(self):
        return self._already_data_clear
