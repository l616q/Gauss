# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
import argparse

from local_pipeline.auto_modeling_graph import AutoModelingGraph
from local_pipeline.udf_modeling_graph import UdfModelingGraph
from local_pipeline.multiprocess_udf_graph import MultiprocessUdfModelingGraph
from utils.common_component import yaml_read

# --------------- this block just for test ---------------
from local_pipeline.mapping import EnvironmentConfigure
from utils.bunch import Bunch
from utils.common_component import yaml_write
from utils.Logger import logger

user_feature = "/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml"
environ_configure = EnvironmentConfigure(work_root="/home/liangqian/PycharmProjects/Gauss/experiments",
                                         user_feature="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml")

pipeline_dict = Bunch()
# ["udf", "auto", "multi_udf"]
pipeline_dict.mode = "multi_udf"
pipeline_dict.work_root = environ_configure.work_root
pipeline_dict.task_type = "classification"
pipeline_dict.metric_name = "auc"
pipeline_dict.train_data_path = "/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical_train_realdata.csv"
pipeline_dict.val_data_path = None
pipeline_dict.target_names = ["deposit"]
pipeline_dict.feature_configure_path = environ_configure.user_feature_path
pipeline_dict.dataset_type = "plaindataset"
pipeline_dict.model_zoo = ["lightgbm"]
pipeline_dict.data_clear_flag = False
pipeline_dict.feature_generator_flag = False
pipeline_dict.unsupervised_feature_selector_flag = False
pipeline_dict.supervised_feature_selector_flag = True
config_path = environ_configure.work_root + "/train_user_config.yaml"
yaml_write(yaml_dict=dict(pipeline_dict), yaml_file=config_path)
# --------------- test block end ---------------


def main(config=config_path):
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    pipeline_configure.system_config_root = "/home/liangqian/PycharmProjects/Gauss/configure_files"
    auto_ml_path = pipeline_configure.system_config_root + "/" + "automl_params"
    selector_config_path = pipeline_configure.system_config_root + "/" + "selector_params"
    system_config = yaml_read(pipeline_configure.system_config_root + "/" + "system_config/system_config.yaml")
    system_config = Bunch(**system_config)

    pipeline_configure.update(system_config)

    if pipeline_configure.mode == "auto":
        auto_model_tree = AutoModelingGraph(name="auto",
                                            work_root=pipeline_configure.work_root,
                                            task_type=pipeline_configure.task_type,
                                            metric_name=pipeline_configure.metric_name,
                                            train_data_path=pipeline_configure.train_data_path,
                                            val_data_path=pipeline_configure.val_data_path,
                                            feature_configure_path=pipeline_configure.feature_configure_path,
                                            target_names=pipeline_configure.target_names,
                                            dataset_type=pipeline_configure.dataset_type,
                                            type_inference=pipeline_configure.type_inference,
                                            data_clear=pipeline_configure.data_clear,
                                            feature_generator=pipeline_configure.feature_generator,
                                            unsupervised_feature_selector=pipeline_configure.unsupervised_feature_selector,
                                            supervised_feature_selector=pipeline_configure.supervised_feature_selector,
                                            auto_ml=pipeline_configure.auto_ml,
                                            opt_model_names=pipeline_configure.opt_model_names)

        auto_model_tree.run()

    elif pipeline_configure.mode == "udf":
        udf_model_tree = UdfModelingGraph(name="udf",
                                          work_root=pipeline_configure.work_root,
                                          task_type=pipeline_configure.task_type,
                                          metric_name=pipeline_configure.metric_name,
                                          target_names=pipeline_configure.target_names,
                                          train_data_path=pipeline_configure.train_data_path,
                                          val_data_path=pipeline_configure.val_data_path,
                                          feature_configure_path=pipeline_configure.feature_configure_path,
                                          dataset_type=pipeline_configure.dataset_type,
                                          type_inference=pipeline_configure.type_inference,
                                          data_clear=pipeline_configure.data_clear,
                                          data_clear_flag=pipeline_configure.data_clear_flag,
                                          feature_generator=pipeline_configure.feature_generator,
                                          feature_generator_flag=pipeline_configure.feature_generator_flag,
                                          unsupervised_feature_selector=pipeline_configure.unsupervised_feature_selector,
                                          unsupervised_feature_selector_flag=pipeline_configure.unsupervised_feature_selector_flag,
                                          supervised_feature_selector=pipeline_configure.supervised_feature_selector,
                                          supervised_feature_selector_flag=pipeline_configure.supervised_feature_selector_flag,
                                          model_zoo=pipeline_configure.model_zoo,
                                          auto_ml=pipeline_configure.auto_ml,
                                          opt_model_names=pipeline_configure.opt_model_names,
                                          auto_ml_path=auto_ml_path,
                                          selector_config_path=selector_config_path)

        udf_model_tree.run()
    elif pipeline_configure.mode == "multi_udf":

        pipeline_configure.dataset_type = "multiprocess_" + pipeline_dict.dataset_type
        pipeline_configure.model_zoo = ["multiprocess_" + model_name for model_name in pipeline_configure.model_zoo]
        multi_udf_model_tree = MultiprocessUdfModelingGraph(name="udf",
                                                            work_root=pipeline_configure.work_root,
                                                            task_type=pipeline_configure.task_type,
                                                            metric_name=pipeline_configure.metric_name,
                                                            target_names=pipeline_configure.target_names,
                                                            train_data_path=pipeline_configure.train_data_path,
                                                            val_data_path=pipeline_configure.val_data_path,
                                                            feature_configure_path=pipeline_configure.feature_configure_path,
                                                            dataset_type=pipeline_configure.dataset_type,
                                                            type_inference=pipeline_configure.type_inference,
                                                            data_clear=pipeline_configure.data_clear,
                                                            data_clear_flag=pipeline_configure.data_clear_flag,
                                                            feature_generator=pipeline_configure.feature_generator,
                                                            feature_generator_flag=pipeline_configure.feature_generator_flag,
                                                            unsupervised_feature_selector=pipeline_configure.unsupervised_feature_selector,
                                                            unsupervised_feature_selector_flag=pipeline_configure.unsupervised_feature_selector_flag,
                                                            supervised_feature_selector=pipeline_configure.supervised_feature_selector,
                                                            supervised_feature_selector_flag=pipeline_configure.supervised_feature_selector_flag,
                                                            model_zoo=pipeline_configure.model_zoo,
                                                            supervised_selector_names=pipeline_configure.supervised_selector_names,
                                                            auto_ml=pipeline_configure.auto_ml,
                                                            opt_model_names=pipeline_configure.opt_model_names,
                                                            auto_ml_path=auto_ml_path,
                                                            selector_config_path=selector_config_path)

        multi_udf_model_tree.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")

    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()

    logger.info(environ_configure.work_root)
