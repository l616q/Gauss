"""-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab"""
from __future__ import annotations

import shutil
from os.path import join
import itertools

from pipeline.local_pipeline.core_chain import CoreRoute
from pipeline.local_pipeline.preprocess_chain import PreprocessRoute
from pipeline.local_pipeline.mapping import EnvironmentConfigure
from pipeline.local_pipeline.base_modeling_graph import BaseModelingGraph
from utils.bunch import Bunch

from utils.check_dataset import check_data
from utils.yaml_exec import yaml_write
from utils.exception import PipeLineLogicError, NoResultReturnException
from utils.constant_values import ConstantValues


# This class is used to train model in fast module.
class AutoModelingGraph(BaseModelingGraph):
    def __init__(self, name: str, user_configure: dict, system_configure: dict):
        """
        Create fast auto modeling graph pipeline.
        :param name: pipeline name.
        :param user_configure: user configure dict.
        :param system_configure: system configure dict.
        """
        if user_configure[ConstantValues.model_zoo] is None:
            user_configure[ConstantValues.model_zoo] = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if user_configure[ConstantValues.supervised_feature_selector_flag] is None:
            user_configure[ConstantValues.supervised_feature_selector_flag] = [True, False]

        if user_configure[ConstantValues.unsupervised_feature_selector_flag] is None:
            user_configure[ConstantValues.unsupervised_feature_selector_flag] = [True, False]

        if user_configure[ConstantValues.feature_generator_flag] is None:
            user_configure[ConstantValues.feature_generator_flag] = [True, False]

        if user_configure[ConstantValues.data_clear_flag] is None:
            user_configure[ConstantValues.data_clear_flag] = [True, False]

        super().__init__(
            name=name,
            metric_eval_used_flag=user_configure[ConstantValues.metric_eval_used_flag],
            data_file_type=user_configure[ConstantValues.data_file_type],
            work_root=user_configure[ConstantValues.work_root],
            task_name=user_configure[ConstantValues.task_name],
            dataset_weight_dict=user_configure[ConstantValues.dataset_weight_dict],
            train_column_name_flag=user_configure[ConstantValues.train_column_name_flag],
            val_column_name_flag=user_configure[ConstantValues.val_column_name_flag],
            weight_column_name=user_configure[ConstantValues.weight_column_name],
            use_weight_flag=user_configure[ConstantValues.use_weight_flag],
            metric_name=user_configure[ConstantValues.metric_name],
            init_model_root=user_configure[ConstantValues.init_model_root],
            loss_name=user_configure[ConstantValues.loss_name],
            train_data_path=user_configure[ConstantValues.train_data_path],
            val_data_path=user_configure[ConstantValues.val_data_path],
            target_names=user_configure[ConstantValues.target_names],
            model_need_clear_flag=system_configure[ConstantValues.model_need_clear_flag],
            feature_configure_path=user_configure[ConstantValues.feature_configure_path],
            feature_configure_name=system_configure[
                ConstantValues.feature_configure_name],
            dataset_name=system_configure[ConstantValues.dataset_name],
            type_inference_name=system_configure[ConstantValues.type_inference_name],
            label_encoder_name=system_configure[ConstantValues.label_encoder_name],
            label_encoder_flag=system_configure[ConstantValues.label_encoder_flag],
            data_clear_name=system_configure[ConstantValues.data_clear_name],
            data_clear_flag=user_configure[ConstantValues.data_clear_flag],
            feature_generator_name=system_configure[
                ConstantValues.feature_generator_name],
            feature_generator_flag=user_configure[ConstantValues.feature_generator_flag],
            unsupervised_feature_selector_name=system_configure[
                ConstantValues.unsupervised_feature_selector_name],
            unsupervised_feature_selector_flag=user_configure[
                ConstantValues.unsupervised_feature_selector_flag],
            supervised_selector_mode=user_configure[
                ConstantValues.supervised_selector_mode],
            supervised_feature_selector_name=system_configure[
                ConstantValues.supervised_feature_selector_name],
            improved_supervised_feature_selector_name=system_configure[
                ConstantValues.improved_supervised_feature_selector_name],
            supervised_feature_selector_flag=user_configure[
                ConstantValues.supervised_feature_selector_flag],
            supervised_selector_model_names=system_configure[
                ConstantValues.supervised_selector_model_names],
            improved_selector_configure_path=system_configure[
                ConstantValues.improved_selector_configure_path],
            feature_model_trial=system_configure[ConstantValues.feature_model_trial],
            selector_trial_num=system_configure[ConstantValues.selector_trial_num],
            auto_ml_name=system_configure[ConstantValues.auto_ml_name],
            auto_ml_trial_num=system_configure[ConstantValues.auto_ml_trial_num],
            opt_model_names=system_configure[ConstantValues.opt_model_names],
            auto_ml_path=system_configure[ConstantValues.auto_ml_path],
            selector_configure_path=system_configure[ConstantValues.selector_configure_path])

        self._model_zoo = user_configure[ConstantValues.model_zoo]
        self._label_switch_type = user_configure[ConstantValues.label_switch_type]

        self.__pipeline_configure = \
            {ConstantValues.work_root:
                 self._work_paths[ConstantValues.work_root],
             ConstantValues.data_clear_flag:
                 self._flag_dict[ConstantValues.data_clear_flag],
             ConstantValues.data_clear_name:
                 self._component_names[ConstantValues.data_clear_name],
             ConstantValues.feature_generator_flag:
                 self._flag_dict[ConstantValues.feature_generator_flag],
             ConstantValues.feature_generator_name:
                 self._component_names[ConstantValues.feature_generator_name],
             ConstantValues.unsupervised_feature_selector_flag:
                 self._flag_dict[ConstantValues.unsupervised_feature_selector_flag],
             ConstantValues.unsupervised_feature_selector_name:
                 self._component_names[ConstantValues.unsupervised_feature_selector_name],
             ConstantValues.supervised_feature_selector_flag:
                 self._flag_dict[ConstantValues.supervised_feature_selector_flag],
             ConstantValues.supervised_feature_selector_name:
                 self._component_names[ConstantValues.supervised_feature_selector_name],
             ConstantValues.metric_name:
                 self._entity_names[ConstantValues.metric_name],
             ConstantValues.task_name:
                 self._attributes_names[ConstantValues.task_name],
             ConstantValues.target_names:
                 self._attributes_names[ConstantValues.target_names],
             ConstantValues.dataset_name:
                 self._entity_names[ConstantValues.dataset_name],
             ConstantValues.type_inference_name:
                 self._component_names[ConstantValues.type_inference_name],
             ConstantValues.model_zoo: self._model_zoo
             }
        self.__report_configure = None

    def _run_route(self, **params):
        self.__init_report_configure()

        data_clear_flag = params[ConstantValues.data_clear_flag]
        if not isinstance(data_clear_flag, bool):
            raise TypeError(
                "Value: data clear flag should be type of bool, "
                "but get {} instead.".format(
                    data_clear_flag)
            )

        feature_generator_flag = params[ConstantValues.feature_generator_flag]
        if not isinstance(feature_generator_flag, bool):
            raise TypeError(
                "Value: feature generator flag should be type of bool, "
                "but get {} instead.".format(
                    feature_generator_flag)
            )

        unsupervised_feature_selector_flag = params[ConstantValues.unsupervised_feature_selector_flag]
        if not isinstance(unsupervised_feature_selector_flag, bool):
            raise TypeError(
                "Value: unsupervised feature selector flag should be type of bool, "
                "but get {} instead.".format(
                    unsupervised_feature_selector_flag)
            )

        supervised_feature_selector_flag = params[ConstantValues.supervised_feature_selector_flag]
        if not isinstance(supervised_feature_selector_flag, bool):
            raise TypeError(
                "Value: supervised feature selector flag should be type of bool, "
                "but get {} instead.".format(
                    supervised_feature_selector_flag)
            )

        supervised_selector_model_names = params[ConstantValues.supervised_selector_model_names]
        if not isinstance(supervised_selector_model_names, str):
            raise TypeError(
                "Value: data clear flag should be type of bool, "
                "but get {} instead.".format(
                    data_clear_flag)
            )

        opt_model_names = params[ConstantValues.opt_model_names]
        if not isinstance(opt_model_names, str):
            raise TypeError(
                "Value: opt model names should be type of bool, "
                "but get {} instead.".format(
                    opt_model_names)
            )

        model_name = params[ConstantValues.model_name]
        if not isinstance(model_name, str):
            raise TypeError(
                "Value: model name should be type of bool, "
                "but get {} instead.".format(
                    model_name)
            )

        folder_name = "_".join([str(data_clear_flag),
                                str(feature_generator_flag),
                                str(unsupervised_feature_selector_flag),
                                str(supervised_feature_selector_flag),
                                str(supervised_selector_model_names),
                                opt_model_names,
                                model_name])

        supervised_selector_model_names = [supervised_selector_model_names]
        opt_model_names = [opt_model_names]

        dispatch_model_root = join(self._work_paths[ConstantValues.work_root], folder_name)
        work_feature_root = join(dispatch_model_root, ConstantValues.feature)
        feature_dict = EnvironmentConfigure.feature_dict()

        feature_dict = \
            {ConstantValues.user_feature_path: self._work_paths[ConstantValues.feature_configure_path],
             ConstantValues.type_inference_feature_path: join(
                 work_feature_root,
                 feature_dict.type_inference_feature),

             ConstantValues.data_clear_feature_path: join(
                 work_feature_root,
                 feature_dict.data_clear_feature),

             ConstantValues.feature_generator_feature_path: join(
                 work_feature_root,
                 feature_dict.feature_generator_feature),

             ConstantValues.unsupervised_feature_path: join(
                 work_feature_root,
                 feature_dict.unsupervised_feature),

             ConstantValues.supervised_feature_path: join(
                 work_feature_root,
                 feature_dict.supervised_feature),

             ConstantValues.label_encoding_models_path: join(
                 work_feature_root,
                 feature_dict.label_encoding_path),

             ConstantValues.impute_models_path: join(
                 work_feature_root,
                 feature_dict.impute_path),

             ConstantValues.label_encoder_feature_path: join(
                 work_feature_root,
                 feature_dict.label_encoder_feature),

             ConstantValues.success_file_path : join(
                 dispatch_model_root,
                 feature_dict.success_file_name)
             }

        work_model_root = join(
            dispatch_model_root,
            ConstantValues.model
        )

        entity_dict = None
        feature_configure_root = join(work_model_root, ConstantValues.feature_configure)
        feature_dict[ConstantValues.final_feature_configure] = join(
            feature_configure_root,
            EnvironmentConfigure.feature_dict().final_feature_configure
        )

        preprocess_chain = PreprocessRoute(
            name=ConstantValues.PreprocessRoute,
            feature_path_dict=feature_dict,
            train_column_name_flag=self._global_values[ConstantValues.train_column_name_flag],
            val_column_name_flag=self._global_values[ConstantValues.val_column_name_flag],
            data_file_type=self._global_values[ConstantValues.data_file_type],
            task_name=self._attributes_names[ConstantValues.task_name],
            train_flag=ConstantValues.train,
            label_switch_type=self._label_switch_type,
            use_weight_flag=self._global_values[ConstantValues.use_weight_flag],
            dataset_weight_dict=self._global_values[ConstantValues.dataset_weight_dict],
            weight_column_name=self._global_values[ConstantValues.weight_column_name],
            train_data_path=self._work_paths[ConstantValues.train_data_path],
            val_data_path=self._work_paths[ConstantValues.val_data_path],
            inference_data_path=None,
            target_names=self._attributes_names[ConstantValues.target_names],
            dataset_name=self._entity_names[ConstantValues.dataset_name],
            type_inference_name=self._component_names[ConstantValues.type_inference_name],
            data_clear_name=self._component_names[ConstantValues.data_clear_name],
            data_clear_flag=data_clear_flag,
            label_encoder_name=self._component_names[ConstantValues.label_encoder_name],
            label_encoder_flag=self._flag_dict[ConstantValues.label_encoder_flag],
            feature_generator_name=self._component_names[ConstantValues.feature_generator_name],
            feature_generator_flag=feature_generator_flag,
            unsupervised_feature_selector_name=self._component_names[
                ConstantValues.unsupervised_feature_selector_name],
            unsupervised_feature_selector_flag=unsupervised_feature_selector_flag,
            report_configure=self.__report_configure
        )

        try:
            entity_dict = preprocess_chain.run()
        except PipeLineLogicError:
            self.__report_configure.main_pipeline.info = "Error: pipeline logic error happens in preprocessing chain."
            self.__report_configure.main_pipeline.success_flag = False
        except (ValueError,
                TypeError,
                RuntimeError,
                IndexError,
                IOError,
                BaseException):
            report_configure = preprocess_chain.report_configure
            self.__report_configure.main_pipeline.info = "General error happens in preprocessing chain."
            self.__report_configure.main_pipeline.success_flag = False
            return report_configure
        finally:
            if not isinstance(entity_dict, dict):
                self.__report_configure.main_pipeline.info = "Get illegal type of entity dict in preprocessing chain."
                self.__report_configure.main_pipeline.success_flag = False

        self._already_data_clear = preprocess_chain.already_data_clear

        assert params.get(ConstantValues.model_name) is not None
        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
        if check_data(already_data_clear=self._already_data_clear,
                      model_need_clear_flag=self._model_need_clear_flag.get(model_name)) is not True:
            return None

        assert ConstantValues.train_dataset in entity_dict and ConstantValues.val_dataset in entity_dict

        core_chain = CoreRoute(
            name=ConstantValues.CoreRoute,
            train_flag=ConstantValues.train,
            model_root_path=work_model_root,
            target_feature_configure_path=feature_dict[ConstantValues.final_feature_configure],
            pre_feature_configure_path=feature_dict[ConstantValues.unsupervised_feature_path],
            model_name=model_name,
            init_model_root=self._work_paths[ConstantValues.init_model_root],
            metric_eval_used_flag=self._attributes_names.metric_eval_used_flag,
            feature_configure_name=self._entity_names[ConstantValues.feature_configure_name],
            metric_name=self._entity_names[ConstantValues.metric_name],
            loss_name=self._entity_names[ConstantValues.loss_name],
            task_name=self._attributes_names[ConstantValues.task_name],
            supervised_selector_name=self._component_names[ConstantValues.supervised_feature_selector_name],
            feature_selector_model_names=supervised_selector_model_names,
            selector_trial_num=self._global_values[ConstantValues.selector_trial_num],
            supervised_feature_selector_flag=supervised_feature_selector_flag,
            supervised_selector_mode=self._global_values[ConstantValues.supervised_selector_mode],
            improved_supervised_selector_name=self._component_names[
                ConstantValues.improved_supervised_feature_selector_name],
            improved_selector_configure_path=self._work_paths[ConstantValues.improved_selector_configure_path],
            feature_model_trial=self._global_values[ConstantValues.feature_model_trial],
            auto_ml_name=self._component_names[ConstantValues.auto_ml_name],
            auto_ml_trial_num=self._global_values[ConstantValues.auto_ml_trial_num],
            auto_ml_path=self._work_paths[ConstantValues.auto_ml_path],
            opt_model_names=opt_model_names,
            selector_configure_path=self._work_paths[ConstantValues.selector_configure_path]
        )

        try:
            core_chain.run(**entity_dict)
        except PipeLineLogicError:
            self.__report_configure.main_pipeline.info = "Error: pipeline logic error happens in core chain."
            self.__report_configure.main_pipeline.success_flag = False
        except (ValueError,
                TypeError,
                RuntimeError,
                IndexError,
                IOError,
                BaseException):
            report_configure = preprocess_chain.report_configure
            self.__report_configure.main_pipeline.info = "General error happens in core chain."
            self.__report_configure.main_pipeline.success_flag = False
            return report_configure

        local_metric = core_chain.optimal_metric

        assert local_metric is not None
        self.__report_configure.global_configure.success_flag = True

        yaml_write(yaml_dict={},
                   yaml_file=feature_dict[ConstantValues.success_file_path])

        return {"work_model_root": work_model_root,
                "model_name": params.get(ConstantValues.model_name),
                "increment_flag": False,
                "metric_result": local_metric,
                "final_file_path":
                    feature_dict[ConstantValues.final_feature_configure],
                "dispatch_id": folder_name}

    def _run(self):
        train_results = []
        routes = self.__generate_route()

        for params in routes:
            data_clear_flag, feature_generator_flag, unsupervised_feature_selector_flag, supervised_feature_selector_flag, supervised_selector_model_names, opt_model_names, model_name = params

            local_result = self._run_route(
                data_clear_flag=data_clear_flag,
                feature_generator_flag=feature_generator_flag,
                unsupervised_feature_selector_flag=unsupervised_feature_selector_flag,
                supervised_feature_selector_flag=supervised_feature_selector_flag,
                supervised_selector_model_names=supervised_selector_model_names,
                opt_model_names=opt_model_names,
                model_name=model_name)

            train_results.append(local_result)

        self._find_best_result(train_results=train_results)

    def _find_best_result(self, train_results):
        best_result = {}
        best_id = []
        if len(train_results) == 0:
            raise NoResultReturnException("No model is trained successfully.")

        for result in train_results:
            model_name = result.get(ConstantValues.model_name)

            if best_result.get(model_name) is None:
                best_result[model_name] = result
            else:
                if result.get(ConstantValues.metric_result) is not None:
                    if best_result.get(model_name).get(ConstantValues.metric_result).__cmp__(
                            result.get(ConstantValues.metric_result)) < 0:
                        best_result[model_name] = result

        for model_name in best_result.keys():
            best_id.append(best_result[model_name]["dispatch_id"])

        routes = self.__generate_route()

        for params in routes:
            data_clear_flag, feature_generator_flag, unsupervised_feature_selector_flag, supervised_feature_selector_flag, supervised_selector_model_names, opt_model_names, model_name = params

            folder_name = "_".join([str(data_clear_flag),
                                    str(feature_generator_flag),
                                    str(unsupervised_feature_selector_flag),
                                    str(supervised_feature_selector_flag),
                                    str(supervised_selector_model_names),
                                    opt_model_names,
                                    model_name])
            if folder_name not in best_id:
                self.__delete_generated_folder(folder_name)

        # this code will change metric result object to metric value.
        for result in train_results:
            result[ConstantValues.optimize_mode] = result.get(ConstantValues.metric_result).optimize_mode
            result[ConstantValues.metric_result] = float(result.get(ConstantValues.metric_result).result)

        model_dict = {ConstantValues.model: best_result}
        self.__pipeline_configure.update(model_dict)

    def _set_pipeline_config(self):
        feature_dict = EnvironmentConfigure.feature_dict()
        yaml_dict = {}

        if self.__pipeline_configure is not None:
            yaml_dict.update(self.__pipeline_configure)

        yaml_write(yaml_dict=yaml_dict,
                   yaml_file=join(self._work_paths["work_root"], feature_dict.pipeline_configure))
        yaml_write(yaml_dict={},
                   yaml_file=join(self._work_paths["work_root"], feature_dict.success_file_name))

    def __delete_generated_folder(self, temp_id):
        folder_path = join(self._work_paths["work_root"], temp_id)
        shutil.rmtree(folder_path)

    @property
    def pipeline_configure(self):
        """
        property method
        :return: A dict of udf model graph configuration.
        """
        if self.__pipeline_configure is None:
            raise RuntimeError("This pipeline has not start.")
        return self.__pipeline_configure

    @classmethod
    def build_pipeline(cls, name: str, **params):
        if name == ConstantValues.PreprocessRoute:
            return PreprocessRoute(**params)
        if name == ConstantValues.CoreRoute:
            return CoreRoute(**params)

    def __generate_route(self):
        supervised_selector_model_names = self._global_values[ConstantValues.supervised_selector_model_names]
        if not isinstance(supervised_selector_model_names, list):
            raise TypeError(
                "Value: supervised selector model names should be type of list, "
                "but get {} instead.".format(
                    supervised_selector_model_names)
            )

        opt_model_names = self._global_values[ConstantValues.opt_model_names]
        if not isinstance(opt_model_names, list):
            raise TypeError(
                "Value: opt model names should be type of list, "
                "but get {} instead.".format(
                    opt_model_names)
            )

        data_clear_flag = self._flag_dict[ConstantValues.data_clear_flag]
        if not isinstance(data_clear_flag, list):
            raise TypeError(
                "Value: data clear flag should be type of list, "
                "but get {} instead.".format(
                    data_clear_flag)
            )

        feature_generator_flag = self._flag_dict[ConstantValues.feature_generator_flag]
        if not isinstance(feature_generator_flag, list):
            raise TypeError(
                "Value: feature generator flag should be type of list, "
                "but get {} instead.".format(
                    feature_generator_flag)
            )

        unsupervised_feature_selector_flag = self._flag_dict[ConstantValues.unsupervised_feature_selector_flag]
        if not isinstance(unsupervised_feature_selector_flag, list):
            raise TypeError(
                "Value: unsupervised feature selector flag should be type of list, "
                "but get {} instead.".format(
                    unsupervised_feature_selector_flag)
            )

        supervised_feature_selector_flag = self._flag_dict[ConstantValues.supervised_feature_selector_flag]
        if not isinstance(supervised_feature_selector_flag, list):
            raise TypeError(
                "Value: supervised feature selector flag should be type of list, "
                "but get {} instead.".format(
                    supervised_feature_selector_flag)
            )

        model_zoo = self._model_zoo
        if not isinstance(model_zoo, list):
            raise TypeError(
                "Value: model zoo should be type of list, "
                "but get {} instead.".format(
                    model_zoo)
            )

        routes = itertools.product(
            data_clear_flag,
            feature_generator_flag,
            unsupervised_feature_selector_flag,
            supervised_feature_selector_flag,
            supervised_selector_model_names,
            opt_model_names,
            model_zoo)
        return routes

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
            global_configure=Bunch(
                main_pipeline=Bunch(),
                success_flag=False
            )
        )

    def report_configure(self):
        return self.__report_configure
