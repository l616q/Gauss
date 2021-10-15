"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
import os
import argparse

from local_pipeline.singleprocess.incremental_modeling_graph import IncrementModelingGraph
from utils.yaml_exec import yaml_read

# --------------- this block just for test ---------------
from local_pipeline.pipeline_utils.mapping import EnvironmentConfigure
from utils.bunch import Bunch
from utils.yaml_exec import yaml_write
from utils.Logger import logger

from utils.copy_folder import copy_folder
from utils.reconstruct_folder import reconstruct_folder

user_feature = None
environ_configure = EnvironmentConfigure(work_root="/home/liangqian/Gauss/experiments",
                                         user_feature=None)

pipeline_dict = Bunch()
# new work root for increment task
pipeline_dict.work_root = environ_configure.work_root
logger.info("work_root: %s", pipeline_dict.work_root)
# optional: ["libsvm", "txt", "csv"]
pipeline_dict.data_file_type = "libsvm"
# increment dataset
pipeline_dict.init_work_root = "/home/liangqian/Gauss/experiments/C1Ackz"
pipeline_dict.train_data_path = "/home/liangqian/文档/公开数据集/a9a/a9a.t"

# user must set a specific model for increment
pipeline_dict.model_zoo = ["lightgbm"]
pipeline_dict.init_folder_name = os.path.join(pipeline_dict.init_work_root).split("/")[-1]
copy_folder(source_path=pipeline_dict.init_work_root, target_path=pipeline_dict.work_root)
config_path = environ_configure.work_root + "/train_user_config.yaml"
reconstruct_folder(folder=pipeline_dict.work_root, init_prefix=pipeline_dict.init_folder_name)
config_dict = yaml_read(yaml_file=config_path)
config_dict.update(pipeline_dict)
yaml_write(yaml_dict=dict(config_dict), yaml_file=config_path)
# --------------- test block end ---------------


def main(config=config_path):
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    pipeline_configure.system_configure_root = "/home/liangqian/Gauss/configure_files"
    pipeline_configure.auto_ml_path = pipeline_configure.system_configure_root + "/" + "automl_params"
    pipeline_configure.selector_configure_path = pipeline_configure.system_configure_root + "/" + "selector_params"
    system_config = yaml_read(pipeline_configure.system_configure_root + "/" + "system_config/system_config.yaml")
    system_config = Bunch(**system_config)

    pipeline_configure.update(system_config)

    model_graph = IncrementModelingGraph(name="increment", **pipeline_configure)

    model_graph.run()


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