"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
inference pipeline.
"""
from __future__ import annotations

from os.path import join

import pandas as pd

from Gauss.pipeline.local_pipeline.core_chain import CoreRoute
from Gauss.pipeline.local_pipeline.mapping import EnvironmentConfigure
from Gauss.pipeline.local_pipeline.preprocess_chain import PreprocessRoute
from Gauss.utils.bunch import Bunch
from Gauss.utils.constant_values import ConstantValues


class Inference:
    """
    Inference object
    """
    def __init__(self, **params):
        self.__name = params[ConstantValues.name]
        self.__model_name = params[ConstantValues.model_name]
        self.__work_root = params[ConstantValues.work_root]

        self.__task_name = params[ConstantValues.task_name]
        self.__infer_result_type = params[ConstantValues.infer_result_type]
        self.__metric_name = params[ConstantValues.metric_name]
        self.__feature_configure_name = params[ConstantValues.feature_configure_name]
        self.__data_file_type = params[ConstantValues.data_file_type]
        self.__inference_column_name_flag = params[ConstantValues.inference_column_name_flag]
        self.__inference_data_path = params[ConstantValues.inference_data_path]

        self.__dataset_name = params[ConstantValues.dataset_name]
        self.__target_names = params[ConstantValues.target_names]

        self.__type_inference_name = params[ConstantValues.type_inference_name]
        self.__data_clear_name = params[ConstantValues.data_clear_name]
        self.__label_encoder_name = params[ConstantValues.label_encoder_name]
        self.__feature_generator_name = params[ConstantValues.feature_generator_name]
        self.__unsupervised_feature_selector_name = params[ConstantValues.unsupervised_feature_selector_name]
        self.__supervised_feature_selector_name = params[ConstantValues.supervised_feature_selector_name]

        self.__data_clear_flag = params[ConstantValues.data_clear_flag]
        self.__label_encoder_flag = params[ConstantValues.label_encoder_flag]
        self.__feature_generator_flag = params[ConstantValues.feature_generator_flag]
        self.__unsupervised_feature_selector_flag = params[ConstantValues.unsupervised_feature_selector_flag]
        self.supervised_feature_selector_flag = params[ConstantValues.supervised_feature_selector_flag]

        self.__final_file_path = params[ConstantValues.model][self.__model_name][ConstantValues.final_file_path]
        self.__work_model_root = params[ConstantValues.model][self.__model_name][ConstantValues.work_model_root]
        self.__increment_flag = params[ConstantValues.model][self.__model_name][ConstantValues.increment_flag]
        assert isinstance(self.__increment_flag, bool)

        self.__out_put_path = params[ConstantValues.out_put_path]

    def output_result(self, predict_result: pd.DataFrame):
        """
        Write inference result to csv file.
        :param predict_result:
        :return: None
        """
        assert isinstance(predict_result, pd.DataFrame)
        predict_result.to_csv(join(self.__out_put_path, "result.csv"), index=False)

    def online_run(self, dataframe):
        """
        online inference
        :param dataframe:
        :return: None
        """
        raise NotImplementedError("This method will be finished in future edition.")

    def offline_run(self):
        """
        offline inference
        :return:
        """
        dispatch_work_root = join(self.__work_root, self.__model_name)
        work_feature_root = join(dispatch_work_root, ConstantValues.feature)

        feature_dict = EnvironmentConfigure.feature_dict()
        feature_dict = {"user_feature_path": join(work_feature_root,
                                                  feature_dict.user_feature),

                        "type_inference_feature_path": join(
                            work_feature_root,
                            feature_dict.type_inference_feature),

                        "data_clear_feature_path": join(
                            work_feature_root,
                            feature_dict.data_clear_feature),

                        "feature_generator_feature_path": join(
                            work_feature_root,
                            feature_dict.feature_generator_feature),

                        "unsupervised_feature_path": join(
                            work_feature_root,
                            feature_dict.unsupervised_feature),

                        "supervised_feature_path": join(
                            work_feature_root,
                            feature_dict.supervised_feature),

                        "label_encoding_models_path": join(
                            work_feature_root,
                            feature_dict.label_encoding_path),

                        "impute_models_path": join(
                            work_feature_root,
                            feature_dict.impute_path),

                        "label_encoder_feature_path": join(
                            work_feature_root,
                            feature_dict.label_encoder_feature)
                        }

        preprocessing_params = Bunch(
            name="PreprocessRoute",
            feature_path_dict=feature_dict,
            task_name=self.__task_name,
            train_flag=ConstantValues.inference,
            train_data_path=None,
            val_data_path=None,
            inference_data_path=self.__inference_data_path,
            inference_column_name_flag=self.__inference_column_name_flag,
            data_file_type=self.__data_file_type,
            dataset_name=self.__dataset_name,
            type_inference_name=self.__type_inference_name,
            data_clear_name=self.__data_clear_name,
            data_clear_flag=self.__data_clear_flag,
            label_encoder_name=self.__label_encoder_name,
            label_encoder_flag=self.__label_encoder_flag,
            feature_generator_name=self.__feature_generator_name,
            feature_generator_flag=self.__feature_generator_flag,
            unsupervised_feature_selector_name=self.__unsupervised_feature_selector_name,
            unsupervised_feature_selector_flag=self.__unsupervised_feature_selector_flag
        )
        preprocess_chain = PreprocessRoute(**preprocessing_params)

        entity_dict = preprocess_chain.run()

        assert ConstantValues.infer_dataset in entity_dict

        core_params = Bunch(
            name="core_route",
            train_flag=ConstantValues.inference,
            task_name=self.__task_name,
            model_root_path=self.__work_model_root,
            target_feature_configure_path=self.__final_file_path,
            pre_feature_configure_path=None,
            infer_result_type=self.__infer_result_type,
            model_name=self.__model_name,
            increment_flag=self.__increment_flag,
            feature_configure_name=self.__feature_configure_name,
            label_encoding_path=feature_dict[ConstantValues.label_encoding_models_path],
            metrics_name=self.__metric_name,
            supervised_feature_selector_flag=self.supervised_feature_selector_flag
        )
        core_chain = CoreRoute(**core_params)

        core_chain.run(**entity_dict)
        predict_result = core_chain.result
        self.output_result(predict_result=predict_result)

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
