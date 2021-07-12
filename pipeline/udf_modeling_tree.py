# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

from pipeline.core_chain import CoreRoute
from pipeline.preprocess_chain import PreprocessRoute

from gauss_factory.gauss_factory_producer import GaussFactoryProducer

from utils.common_component import yaml_write, mkdir


# pipeline defined by user.
class UdfModelingTree(object):
    def __init__(self,
                 name: str,
                 work_root: str,
                 task_type: str,
                 metric_name: str,
                 label_name: [],
                 train_data_path: str,
                 val_data_path: str = None,
                 target_names=None,
                 feature_configure_path: str = None,
                 dataset_type: str = "plain",
                 type_inference: str = "plain",
                 data_clear: str = "plain",
                 data_clear_flag=None,
                 feature_generator: str = "featuretools",
                 feature_generator_flag=None,
                 unsupervised_feature_selector: str = "unsupervised",
                 unsupervised_feature_selector_flag=None,
                 supervised_feature_selector: str = "supervised",
                 supervised_feature_selector_flag=None,
                 model_zoo=None,
                 auto_ml: str = "plain"
                 ):
        """
        :param name:
        :param work_root:
        :param task_type:
        :param metric_name:
        :param label_name:
        :param train_data_path:
        :param val_data_path:
        :param feature_configure_path:
        :param dataset_type:
        :param type_inference:
        :param data_clear:
        :param data_clear_flag:
        :param feature_generator:
        :param feature_generator_flag:
        :param unsupervised_feature_selector:
        :param unsupervised_feature_selector_flag:
        :param supervised_feature_selector:
        :param supervised_feature_selector_flag:
        :param model_zoo:
        :param auto_ml:
        """

        if model_zoo is None:
            model_zoo = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if supervised_feature_selector_flag is None:
            supervised_feature_selector_flag = [True, False]

        if unsupervised_feature_selector_flag is None:
            unsupervised_feature_selector_flag = [True, False]

        if feature_generator_flag is None:
            feature_generator_flag = [True, False]

        if data_clear_flag is None:
            data_clear_flag = [True, False]

        self.name = name
        self.work_root = work_root
        self.task_type = task_type
        self.metric_name = metric_name
        self.label_name = label_name
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.target_names = target_names
        self.feature_configure_path = feature_configure_path
        self.dataset_type = dataset_type
        self.type_inference = type_inference
        self.data_clear = data_clear
        self.data_clear_flag = data_clear_flag
        self.feature_generator = feature_generator
        self.feature_generator_flag = feature_generator_flag
        self.unsupervised_feature_selector = unsupervised_feature_selector
        self.unsupervised_feature_selector_flag = unsupervised_feature_selector_flag
        self.supervised_feature_selector = supervised_feature_selector
        self.supervised_feature_selector_flag = supervised_feature_selector_flag
        self.model_zoo = model_zoo
        self.auto_ml = auto_ml
        self.need_data_clear = False
        self.best_model = None
        self.best_metric = None
        self.best_result_root = None

    def run_route(self,
                  folder_prefix_str,
                  data_clear_flag,
                  feature_generator_flag,
                  unsupervised_feature_selector_flag,
                  supervised_feature_selector_flag,
                  model_name,
                  auto_ml_path,
                  selector_config_path):

        work_root = self.work_root + "/" + folder_prefix_str
        pipeline_configure_path = work_root + "/" + "pipeline.configure"

        pipeline_configure = {"data_clear_flag": data_clear_flag,
                              "feature_generator_flag": feature_generator_flag,
                              "unsupervised_feature_selector_flag": unsupervised_feature_selector_flag,
                              "supervised_feature_selector_flag": supervised_feature_selector_flag}

        yaml_write(yaml_file=pipeline_configure_path, yaml_dict=pipeline_configure)

        work_feature_root = work_root + "/feature"
        work_model_root = work_root + "/model"
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_config_root"

        mkdir(work_root)
        mkdir(work_feature_root)
        mkdir(work_model_root)
        mkdir(model_save_root)
        mkdir(model_config_root)

        feature_dict = {"user_feature": self.feature_configure_path,
                        "type_inference_feature": work_feature_root + "/." + "type_inference_feature.yaml",
                        "data_clear_feature": work_feature_root + "/." + "data_clear_feature.yaml",
                        "feature_generator_feature": work_feature_root + "/." + "feature_generator_feature.yaml",
                        "unsupervised_feature": work_feature_root + "/." + "unsupervised_feature.yaml",
                        "supervised_feature": work_feature_root + "/." + "supervise_feature.yaml"}

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=True,
                                           train_data_path=self.train_data_path,
                                           val_data_path=self.val_data_path,
                                           test_data_path=None,
                                           target_names=self.target_names,
                                           type_inference_name="typeinference",
                                           data_clear_name="plaindataclear",
                                           data_clear_flag=data_clear_flag,
                                           feature_generator_name="featuretools",
                                           feature_generator_flag=feature_generator_flag,
                                           feature_selector_name="unsupervised",
                                           feature_selector_flag=unsupervised_feature_selector_flag)

        entity_dict = preprocess_chain.run()
        self.need_data_clear = preprocess_chain.need_data_clear

        self.check_data(need_data_clear=self.need_data_clear, model_name=model_name)

        assert "dataset" in entity_dict and "val_dataset" in entity_dict
        work_model_root = work_root + "/model/" + model_name + "/"
        model_save_root = work_model_root + "/model_save"
        # model_config_root = work_model_root + "/model_config"

        core_chain = CoreRoute(name="core_route",
                               train_flag=True,
                               model_save_root=model_save_root,
                               target_feature_configure_path=feature_dict["supervised_feature"],
                               pre_feature_configure_path=feature_dict["unsupervised_feature"],
                               model_name=model_name,
                               label_encoding_path=feature_dict["label_encoding_path"],
                               model_type="tree_model",
                               metrics_name=self.metric_name,
                               task_type=self.task_type,
                               feature_selector_name="feature_selector",
                               feature_selector_flag=supervised_feature_selector_flag,
                               auto_ml_type="auto_ml",
                               auto_ml_path=auto_ml_path,
                               selector_config_path=selector_config_path)

        local_model = core_chain.run(**entity_dict)
        local_metric = local_model.get_val_metric()
        return local_model, local_metric, work_root, model_name

    # local_best_model, local_best_metric, local_best_work_root, local_best_model_name
    def update_best(self, *params):
        if params[0] is None or self.compare(params[1], self.best_metric) < 0:
            self.best_model = params[0]
            self.best_metric = params[1]
            self.best_result_root = params[2]

    @classmethod
    def compare(cls, local_best_metric, best_metric):
        return best_metric - local_best_metric

    @classmethod
    def check_data(cls, need_data_clear, model_name):
        assert isinstance(need_data_clear, bool)
        assert isinstance(model_name, str)

        if not need_data_clear:
            if model_name not in ["lightgbm", "xgboost", "catboost"]:
                raise ValueError("This model need data clear algorithms.")

    @classmethod
    def create_component(cls, component_name: str, **params):

        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(component_name=component_name, **params)

    @classmethod
    def create_entity(cls, entity_name: str, **params):

        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)

    def run(self):

        for data_clear in self.data_clear_flag:
            for feature_generator in self.feature_generator_flag:

                for unsupervised_feature_sel in self.unsupervised_feature_selector_flag:

                    for supervise_feature_sel in self.supervised_feature_selector_flag:

                        for model in self.model_zoo:
                            prefix = str(data_clear) + "_" + str(feature_generator) + "_" + str(
                                unsupervised_feature_sel) + "_" + str(supervise_feature_sel)

                            self.update_best(self.run_route(folder_prefix_str=prefix,
                                                            data_clear_flag=data_clear,
                                                            feature_generator_flag=feature_generator,
                                                            unsupervised_feature_selector_flag=unsupervised_feature_sel,
                                                            supervised_feature_selector_flag=supervise_feature_sel,
                                                            model_name=[model],
                                                            auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                                                            selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config"))
