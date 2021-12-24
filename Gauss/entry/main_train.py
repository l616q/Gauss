"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
import argparse

from Gauss.pipeline.local_pipeline.auto_modeling_graph import AutoModelingGraph
from Gauss.pipeline.local_pipeline.udf_modeling_graph import UdfModelingGraph

from Gauss.utils.yaml_exec import yaml_read
from Gauss.utils.bunch import Bunch


def main(user_configure_path=None, system_configure_path=None):
    user_configure = yaml_read(user_configure_path)
    user_configure = Bunch(**user_configure)

    system_configure = yaml_read(system_configure_path)
    system_configure = Bunch(**system_configure)

    user_configure.val_column_name_flag = True

    if user_configure.mode == "auto":
        model_graph = AutoModelingGraph(name="auto",
                                        user_configure=user_configure,
                                        system_configure=system_configure)

        model_graph.run()
    elif user_configure.mode == "udf":
        model_graph = UdfModelingGraph(name="udf",
                                       user_configure=user_configure,
                                       system_configure=system_configure)

        model_graph.run()
    else:
        raise ValueError("Value: pipeline_configure.mode is illegal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")

    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
