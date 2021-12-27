"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
Abstract object for pipelines.
"""
from __future__ import annotations

import abc

from gauss_factory.gauss_factory_producer import GaussFactoryProducer

from utils.bunch import Bunch
from utils.constant_values import ConstantValues


class BaseModelingGraph:
    """
    BaseModelingGraph object.
    """
    def __init__(self, **params):
        """
        This object is base class for all pipelines,
        all pipeline object except preprocessing chain and core chain should inherit it.
        :param name: string object, pipeline name.
        :param work_root: the name of root folder.
        :param task_name: string object, task name.
        :param metric_name: string object, name of the target metric object.
        :param train_data_path: string object, path of train dataset.
        :param val_data_path: string object, path of validation dataset.
        :param target_names: list object, dataset label names or index.
        :param feature_configure_path: path of user-define feature configure file.
        :param dataset_name: name of dataset object.
        :param type_inference_name: name of type inference component.
        :param data_clear_name: name of data clear component.
        :param feature_generator_name: name of feature generator component.
        :param unsupervised_feature_selector_name: name of unsupervised feature selector component.
        :param supervised_feature_selector_name: name of supervised feature selector component.
        :param auto_ml_name: name of auto ml component.
        :param opt_model_names: list object, names of hyperparameter algorithms.
        :param auto_ml_path: root path of auto ml configure files.
        :param selector_configure_path: root path of supervised feature selector.
        """
        assert params["opt_model_names"] is not None

        self._attributes_names = Bunch(
            name=params[ConstantValues.name],
            task_name=params[ConstantValues.task_name],
            target_names=params[ConstantValues.target_names],
            metric_eval_used_flag=params[ConstantValues.metric_eval_used_flag]
        )

        self._work_paths = Bunch(
            work_root=params[ConstantValues.work_root],
            train_data_path=params[ConstantValues.train_data_path],
            val_data_path=params[ConstantValues.val_data_path],
            feature_configure_path=params[ConstantValues.feature_configure_path],
            auto_ml_path=params[ConstantValues.auto_ml_path],
            selector_configure_path=params[ConstantValues.selector_configure_path],
            improved_selector_configure_path=params[ConstantValues.improved_selector_configure_path],
            init_model_root=params[ConstantValues.init_model_root]
        )

        self._entity_names = Bunch(
            dataset_name=params[ConstantValues.dataset_name],
            metric_name=params[ConstantValues.metric_name],
            loss_name=params[ConstantValues.loss_name],
            feature_configure_name=params[ConstantValues.feature_configure_name]
        )

        self._component_names = Bunch(
            type_inference_name=params[ConstantValues.type_inference_name],
            data_clear_name=params[ConstantValues.data_clear_name],
            label_encoder_name=params[ConstantValues.label_encoder_name],
            feature_generator_name=params[ConstantValues.feature_generator_name],
            unsupervised_feature_selector_name=params[ConstantValues.unsupervised_feature_selector_name],
            supervised_feature_selector_name=params[ConstantValues.supervised_feature_selector_name],
            improved_supervised_feature_selector_name=params[ConstantValues.improved_supervised_feature_selector_name],
            auto_ml_name=params[ConstantValues.auto_ml_name]
        )

        self._global_values = Bunch(
            dataset_weight_dict=params[ConstantValues.dataset_weight_dict],
            use_weight_flag=params[ConstantValues.use_weight_flag],
            weight_column_name=params[ConstantValues.weight_column_name],
            train_column_name_flag=params[ConstantValues.train_column_name_flag],
            val_column_name_flag=params[ConstantValues.val_column_name_flag],
            data_file_type=params[ConstantValues.data_file_type],
            selector_trial_num=params["selector_trial_num"],
            auto_ml_trial_num=params["auto_ml_trial_num"],
            opt_model_names=params["opt_model_names"],
            supervised_selector_mode=params["supervised_selector_mode"],
            feature_model_trial=params["feature_model_trial"],
            supervised_selector_model_names=params["supervised_selector_model_names"]
        )

        self._flag_dict = Bunch(
            data_clear_flag=params["data_clear_flag"],
            label_encoder_flag=params["label_encoder_flag"],
            feature_generator_flag=params["feature_generator_flag"],
            unsupervised_feature_selector_flag=params["unsupervised_feature_selector_flag"],
            supervised_feature_selector_flag=params["supervised_feature_selector_flag"]
        )

        self._already_data_clear = None
        self._model_need_clear_flag = params[ConstantValues.model_need_clear_flag]
        self._pipeline_configure = None

    @abc.abstractmethod
    def _run_route(self, **params):
        pass

    @classmethod
    def _create_component(cls, component_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(component_name=component_name, **params)

    @classmethod
    def _create_entity(cls, entity_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)

    def run(self):
        """
        Start training model with pipeline.
        :return:
        """
        self._run()
        self._set_pipeline_config()

    @abc.abstractmethod
    def _run(self):
        pass

    @abc.abstractmethod
    def _set_pipeline_config(self):
        pass

    @abc.abstractmethod
    def _find_best_result(self, train_results):
        pass

    @property
    def pipeline_configure(self):
        """
        This method is used to get pipeline configure.
        :return: dict
        """
        return self._pipeline_configure

    def __init_report_configure(self):
        self.__report_configure = Bunch(
            entity_configure=Bunch(
                dataset=Bunch(),
                feature_conf=Bunch(),
                loss=Bunch(),
                metric=Bunch(),
                model=Bunch()
            ),
            component_configure=Bunch(
                type_inference=Bunch(),
                data_clear=Bunch(),
                label_encode=Bunch(),
                feature_generation=Bunch(),
                unsupervised_feature_selector=Bunch(),
                supervised_feature_selector=Bunch()
            ),
            main_pipeline=Bunch(),
            success_flag=None
        )

    def report_configure(self):
        return self.__report_configure

    @classmethod
    def __replace_type(cls, report_configure: dict):
        """
        This method will replace folder name in a json dict by recursion method.
        :param report_configure: Bunch object.
        :return: dict object.
        """
        if isinstance(report_configure, Bunch):
            report_configure = dict(report_configure)

        for key in report_configure.keys():
            if isinstance(report_configure[key], Bunch):
                report_configure[key] = dict(report_configure[key])
                cls.__replace_type(
                    report_configure=report_configure[key]
                )
        return report_configure

    @classmethod
    def __restore_type(cls, report_configure: dict):
        """
        This method will replace folder name in a json dict by recursion method.
        :param report_configure: Bunch object.
        :return: dict object.
        """
        if not isinstance(report_configure, Bunch) and isinstance(report_configure, dict):
            report_configure = Bunch(**report_configure)

        for key in report_configure.keys():
            if not isinstance(report_configure[key], Bunch) and isinstance(report_configure[key], dict):
                report_configure[key] = Bunch(**report_configure[key])
                cls.__restore_type(
                    report_configure=report_configure[key]
                )
        return report_configure
