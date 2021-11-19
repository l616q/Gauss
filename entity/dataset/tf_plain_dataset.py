# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from entity.dataset.base_dataset import BaseDataset
from entity.dataset.plain_dataset import PlaintextDataset

pd.options.mode.chained_assignment = None  # default = "warn"


class TFPlainDataset(BaseDataset):
    """Gauss_nn Dataset wrapper 
    
    This class is aiming to convert Plain dataset generated by Gauss system
    in memory to tf.train.Dataset which will be used in Gauss_nn system under
    the instruction of feature config file. To use the dataset, user need to 
    build the dataset by call the build() function, then call the init() to 
    pass a value to batch_size defined as a tf.placeholder to activate tf.Ops. 
    Finally, all hyper-parameters could be updated by update_dataset_parameters().
    """

    def __init__(self, **params):
        """
        :param dataset: PlainDataset, dataset generated by Gauss system.
        :param feature_config: feature configs yaml file for data map. 
        :param file_repeat: bool, if dataset need repeat.
        :param file_repeat_count: int, repeat count times, activate when file_repeat is True.
        :param shuffle_buffer_size: int, count of elements will be filled in buffer.
        :param prefetch_buffer_size: int, count of elements will be prefetched from dataset.
        :param drop_remainder: bool, drop last element when drop_remainder is True and number of elements
                                can not divide batch_size evenly; keep if True.
        """
        if not isinstance(params["dataset"], PlaintextDataset):
            raise TypeError("dataset must be a instance of PlainDataset.")

        super(TFPlainDataset, self).__init__(
            name=params["name"],
            data_path=None,
            task_name=params["task_name"],
            train_flag=params["train_flag"],
            memory_only=params["memory_only"] \
                if params.get("memory_only") else True
            )
        
        self._target_names = params["target_names"]
        self._df_dataset = self._v_stack(params["dataset"])

        self._file_repeat = params["file_repeat"] \
            if params.get("file_repeat") else False
        self._file_repeat_count = params["file_repeat_count"] \
            if params.get("file_repeat_count") and params["_file_repeat"] else 1
        self._shuffle_buffer_size = params["shuffle_buffer_size"] \
            if params.get("shuffle_buffer_size") else 100000
        self._prefetch_buffer_size = params["prefetch_buffer_size"] \
            if params.get("prefetch_buffer_size") else 1
        self._drop_remainder = params["drop_remainder"] \
            if params.get("drop_remainder") else True

        # hyper-parameters
        self._batch_size = tf.compat.v1.placeholder(dtype=tf.int64, shape=())
        self._default_batch_size = 8
        self._default_label_class_count = 1


    def __repr__(self):
        if not hasattr(self, "_selected_features"):
            return str(self._df_dataset.head())
        else:
            return str(self._df_dataset[self._selected_features].head())

    def update_features(self, features: list, cate_fea):
        """update private attribute selected_features, which are actually
        needed features in current trail.
        """
        # if self._target_names:
        #     self._selected_features = features + self._target_names
        # else:
        self._selected_features = features
        self._categorical_features = cate_fea

    def bulid(self):
        """build the dataset could be used in the tensorflow work flow.
        
        this function include 3 procedures: 
        1. build dataset from pd.DataFrame,
        meanwile, dataset based operations like shuffle, batch, or other any
        tf operations will be applied here. 
        2. transfer well processed tf dataset to iterator.
        3. build the static graph to return data batch by batch.
        """
        self._dataset = self._build_dataset()
        self._iterator = tf.compat.v1.data.make_initializable_iterator(self._dataset)
        # self._iterator = self._dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next()

    def update_dataset_parameters(self, batch_size, label_class_count):
        # TODO: update hyper parameters which will be used in nn further.
        """update hyper-parameters generated by auto ml search space
        """
        if not hasattr(self, "_next_batch"):
            raise AttributeError("dataset has not been constructed to batches.")
        # if not params.get("batch_size"):
        #     raise TypeError("update_dataset_parameters missing 1 required key word parameter: batch_size.")
        
        self._default_batch_size = batch_size
        self._default_label_class_count = label_class_count
        
    def init(self,sess):
        """initialize current iterated tf.Dataset.
        
        activate tf.Operations defined yet by feed batch_size to iterated tf.Dataset,
        batched dataset will be applied to followed steps. 
        """
        if not hasattr(self, "_iterator"):
            raise AttributeError("dataset has not been iterated, use build() before init().")
        self._iterator.initializer.run(session=sess, feed_dict={self._batch_size: self._default_batch_size})
            
    @property 
    def target_names(self):
        return self._target_names

    @property
    def batch_size(self):
        return self._default_batch_size

    @property
    def next_batch(self):
        if not hasattr(self, "_next_batch"):
            raise AttributeError("tf dataset has not been built yet, call `build()` function before use next_batch")
        return self._next_batch

    @property
    def shape(self):
        ori_shape = self._df_dataset.shape
        return (ori_shape[0], ori_shape[1] + 1)

    @property
    def info(self):
        return self._df_dataset.info()

    def _build_dataset(self):
        dataset = self._filter_feature(self._df_dataset)
        dataset = self._dtype_cast(dataset)
        dataset = self._build_raw_dataset(dataset)
        dataset = self._repeat_dataset(dataset)
        dataset = self._shuffle_and_batch(dataset)
        dataset = self._apply_prefetch(dataset)
        return dataset

    def _filter_feature(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """return a smaller DataFrame depends on private attribute selected_features.
        """
        dataset = dataset.loc[:, self._selected_features]
        return dataset

    def _dtype_cast(self, dataset):
        for name in self._categorical_features:
            dataset.loc[:, name] = dataset.loc[:, name].apply(lambda x: int(x))
        return dataset

    def _build_raw_dataset(self, dataset) -> tf.data.Dataset:
        """load pd.DataFrame data to tf.data.Dataset.

        :return : a `tf.data.Dataset` object contains whole PlainDataset contents.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self._dim_expand(dataset))
        return dataset

    def _repeat_dataset(self, dataset):
        """repeat current dataset count times if repeat applied"""
        if self._repeat_dataset:
            dataset = dataset.repeat(self._file_repeat_count)
        return dataset

    def _shuffle_and_batch(self, dataset):
        """randomly sample data from dataset and batch to batch_size."""
        dataset = dataset.shuffle(self._shuffle_buffer_size)
        dataset = dataset.batch(self._batch_size, self._drop_remainder)
        return dataset

    def _apply_prefetch(self, dataset):
        """prefetch for acceleration"""
        dataset = dataset.prefetch(self._prefetch_buffer_size)
        return dataset

    def _v_stack(self, dataset: PlaintextDataset) -> pd.DataFrame:
        X = dataset.get_dataset().data
        y = dataset.get_dataset().target
        dataset = pd.concat((X, y), axis=1)
        return dataset

    def _dim_expand(self, dataset: pd.DataFrame) -> dict:
        """normalize data dimensions for pipeline.

        transfer python list to numpy array first, then expend scaler dimensions
        to convert them as vectors.
        :return: python dictionary, keye column name in DataFrame,
            value is numpy array shaped as (?, 1)
        """
        dataset = dataset.to_dict("list")
        dataset = {k: np.array(v).reshape(-1, 1) for k, v in dataset.items()}
        return dataset

    def feature_choose(self):
        pass

    def get_dataset(self):
        pass

    def load_data(self):
        pass

    def split(self):
        pass

    def union(self):
        pass

    def set_dataset(self):
        pass
