# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from __future__ import annotations

import os
import csv

import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

from utils.bunch import Bunch
from entity.dataset.base_dataset import BaseDataset
from utils.Logger import logger
from utils.reduce_data import reduce_data


class PlaintextDataset(BaseDataset):
    """loading raw data to PlaintextDataset object.
    Further operation could be applied directly to current dataSet after the construction function
    used.
    Dataset can split by call `split()` function when need a validation set, and `union()` will merge
    train set and validation set in vertical.
    Features of current dataset can be eliminated by `feature_choose()` function, both index and
    feature name are accepted.
    """

    def __init__(self, **params):
        """
        Two kinds of raw data supported:
            1. The first is read file whose format is an option in `.csv`, `.txt`,and
        `.libsvm`, data after processing will wrapped into a `Bunch` object which contains
        `data` and `target`, meanwhile `feature_names` and `target_name` also can be a
        content when exist.
            2. Pass a `data_pair` wrapped by Bunch to the construct function, `data` and
        `target` must provided at least.

        ======
        :param name: A string to represent module's name.
        :param data_path:  A string or `dict`; if string, it can be directly used by load data
            function, otherwise `train_dataset` and `val_dataset` must be the key of the dict.
        :param data_pair: default is None, must be filled if data_path not applied.
        :param task_type: A string which is an option between `classification` or `regression`.
        :param target_name: A `list` containing label names in string format.
        :param memory_only: a boolean value, true for memory, false for others, default True.
        """
        for item in ["name", "task_type", "data_pair", "data_path", "target_name", "memory_only"]:
            if params.get(item) is None:
                params[item] = None

        super(PlaintextDataset, self).__init__(params["name"], params["data_path"], params["task_type"],
                                               params["target_name"], params["memory_only"])

        if not params["data_path"] and not params["data_pair"]:
            raise AttributeError("data_path or data_pair must provided.")

        self._data_pair = params["data_pair"]
        self.type_doc = None
        # mark start point of validation set in all dataset, if just one data file offers, start point will calculate
        # by train_test_split = 0.3, and if train data file and validation file offer, start point will calculate
        # by the length of validation dataset.
        self._val_start = None
        # This value is a bool value, and true means plaindataset has missing values and need to clear.
        self._need_data_clear = False

        assert params["data_path"] is not None or params["data_pair"] is not None
        if params["data_path"] is not None and isinstance(params["data_path"], str):
            self._bunch = self.load_data()

        # if params["data_path] is a dict, it's format must be
        # {"train_dataset": "./t_data.csv", "val_dataset": "./v_data.csv"}
        elif isinstance(params["data_path"], dict):
            if not (self._data_path.get("train_dataset") and self._data_path.get("val_dataset")):
                raise ValueError(
                    "data_path must include `train_dataset` and `val_dataset` when pass a dictionary."
                )
            self._bunch = Bunch(train_dataset=self.load_data(data_path=params["data_path"]["train_dataset"]),
                                val_dataset=self.load_data(data_path=params["data_path"]["val_dataset"]))

        else:
            self._bunch = self._data_pair

    def __repr__(self):
        data = self._bunch.data
        target = self._bunch.target
        df = pd.concat((data, target), axis=1)
        return str(df.head()) + str(df.info())

    def get_dataset(self):
        return self._bunch

    def set_dataset(self, data_pair):
        self._bunch = data_pair
        return self

    def load_data(self, data_path=None):
        if data_path is not None:
            self._data_path = data_path

        if not os.path.isfile(self._data_path):
            raise TypeError("<{path}> is not a valid file.".format(
                path=self._data_path
            ))

        self.type_doc = self._data_path.split(".")[-1]

        if self.type_doc not in ["csv", "libsvm", "txt"]:
            raise TypeError(
                "Unsupported file, excepted option in `csv`, `libsvm`, `txt`, <{type_doc}> received.".format(type_doc=self.type_doc)
            )

        data = None
        target = None
        feature_names = None
        target_name = None

        if self.type_doc == "csv":
            try:
                data, target, feature_names, target_name = self.load_mixed_csv()
            except IOError:
                logger.info("File path does not exist.")
            finally:
                logger.info(".csv file has been converted to Bunch object.")

            self._bunch = Bunch(data=data,
                                target=target,
                                target_names=target_name,
                                feature_names=feature_names)

        elif self.type_doc == 'libsvm':
            data, target = self.load_libsvm()
            _, data, target = self._convert_data_dataframe(data=data,
                                                           target=target)
            self._bunch = Bunch(
                data=data,
                target=target
            )

        elif self.type_doc == 'txt':
            data, target = self.load_txt()
            _, data, target = self._convert_data_dataframe(data=data,
                                                           target=target)
            self._bunch = Bunch(
                data=data,
                target=target
            )
        else:
            raise TypeError("File type can not be accepted.")
        return self._bunch

    def load_mixed_csv(self):
        target = None
        target_name = None

        data = reduce_data(data_path=self._data_path)

        feature_names = data.columns
        self._row_size = data.shape[0]

        if self._target_name is not None:
            target = data[self._target_name]
            data = data.drop(self._target_name, axis=1)
            feature_names = data.columns
            target_name = self._target_name
            self._column_size = data.shape[1] + target.shape[1]
        return data, target, feature_names, target_name

    def load_csv(self):
        """Loads data from csv_file_name.

        Returns
        -------
        data : Numpy array
            A 2D array with each row representing one sample and each column
            representing the features of a given sample.

        target : Numpy array
            A 1D array holding target variables for all the samples in `data.
            For example target[0] is the target variable for data[0].

        target_names : Numpy array
            A 1D array containing the names of the classifications. For example
            target_names[0] is the name of the target[0] class.
        """
        with open(self._data_path, 'r') as csv_file:

            data_file = csv.reader(csv_file)
            feature_names = next(data_file)
            target_location = -1

            try:
                target_location = feature_names.index(self._target_name)
                target_name = feature_names.pop(target_location)
            except IndexError:
                logger.info("Label is not exist.")
            assert target_name == self._target_name

            self._row_size = n_samples = self.wc_count() - 1
            self._column_size = n_features = len(feature_names)

            data = np.empty((n_samples, n_features))
            target = np.empty((n_samples,), dtype=int)

            for index, row in enumerate(data_file):
                label = row.pop(target_location)
                data[index] = np.asarray(row, dtype=np.float64)
                target[index] = np.asarray(label, dtype=int)

        return data, target, feature_names, self._target_name

    def load_libsvm(self):
        data, target = load_svmlight_file(self._data_path)
        data = data.toarray()
        self._column_size = len(data[0]) + 1
        self._row_size = len(data)
        return data, target

    def load_txt(self):
        target_index = 0
        data = []
        target = []

        with open(self._data_path, 'r') as file:
            lines = file.readlines()

            for index, line_content in enumerate(lines):
                data_index = []
                line_content = line_content.split(' ')

                for column, item in enumerate(line_content):
                    if column != target_index:
                        data_index.append(item)
                    else:
                        target.append(item)

                data_index = list(map(np.float64, data_index))
                data.append(data_index)

            data = np.asarray(data, dtype=np.float64)
            target = list(map(int, target))
            target = np.asarray(target, dtype=int)

            self._column_size = len(data[0]) + 1
            self._row_size = len(data)

            return data, target

    def wc_count(self):
        import subprocess
        out = subprocess.getoutput("wc -l %s" % self._data_path)
        return int(out.split()[0])

    def get_column_size(self):
        return self._column_size

    def get_row_size(self):
        return self._row_size

    def get_target_name(self):
        return self._target_name

    @classmethod
    def _convert_data_dataframe(cls, data, target,
                                feature_names=None, target_names=None):

        data_df = pd.DataFrame(data, columns=feature_names)
        target_df = pd.DataFrame(target, columns=target_names)
        combined_df = pd.concat([data_df, target_df], axis=1)

        return combined_df, data_df, target_df

    def feature_choose(self, feature_list):
        try:
            data = self._bunch.data.iloc[:, feature_list]
        except IndexError:
            logger.info("index method used.")
            data = self._bunch.data.loc[:, feature_list]
        return data

    # dataset is a PlainDataset object
    def union(self, val_dataset: PlaintextDataset):
        """ This method is used for concatenating train dataset and validation dataset.Merge train set and
        validation set in vertical, this procedure will operated on train set.
        example:
            trainSet = PlaintextDataset(...)
            validationSet = trainSet.split()
            trainSet.union(validationSet)

        :return: Plaindataset
        """
        self._val_start = self._bunch.target.shape[0]

        assert self._bunch.data.shape[1] == val_dataset.get_dataset().data.shape[1]
        self._bunch.data = pd.concat([self._bunch.data, val_dataset.get_dataset().data], axis=0)

        assert self._bunch.target.shape[1] == val_dataset.get_dataset().target.shape[1]
        self._bunch.target = pd.concat([self._bunch.target, val_dataset.get_dataset().target], axis=0)

        self._bunch.data = self._bunch.data.reset_index(drop=True)
        self._bunch.target = self._bunch.target.reset_index(drop=True)

        if self._bunch.get("feature_names") is not None and val_dataset.get_dataset().get("feature_names") is not None:
            for item in self._bunch.feature_names:
                assert item in val_dataset.get_dataset().feature_names

        if self._bunch.get("target_names") is not None and val_dataset.get_dataset().get("target_names") is not None:
            for item in self._bunch.target_names:
                assert item in val_dataset.get_dataset().target_names

    def split(self, val_start: float = 0.8):
        """split a validation set from train set.
        :param val_start: ration of sample count as validation set.
        :return: plaindataset object containing validation set.
        """
        assert self._bunch is not None

        for key in self._bunch.keys():
            assert key in ["data", "target", "feature_names", "target_names", "generated_feature_names"]

        if self._val_start is None:
            self._val_start = int(val_start * self._bunch.data.shape[0])

        val_data = self._bunch.data.iloc[self._val_start:, :]
        self._bunch.data = self._bunch.data.iloc[:self._val_start, :]

        val_target = self._bunch.target.iloc[self._val_start:]
        self._bunch.target = self._bunch.target.iloc[:self._val_start]

        self._bunch.data.reset_index(drop=True, inplace=True)
        self._bunch.target.reset_index(drop=True, inplace=True)

        val_data = val_data.reset_index(drop=True)
        val_target = val_target.reset_index(drop=True)

        data_pair = Bunch(data=val_data, target=val_target)

        if "feature_names" in self._bunch.keys():
            data_pair.target_names = self._bunch.target_names
            data_pair.feature_names = self._bunch.feature_names

        return PlaintextDataset(name="train_data",
                                task_type="train",
                                data_pair=data_pair)

    @property
    def need_data_clear(self):
        return self._need_data_clear

    @need_data_clear.setter
    def need_data_clear(self, data_clear: bool):
        self._need_data_clear = data_clear
