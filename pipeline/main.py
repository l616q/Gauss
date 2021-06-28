
import argparse
import AutoModelingTree



def main(config="./config.yaml"):
     conf = Config.read(config)
     auto_model_tree = AutoModelingTree("auto",
                                       conf.work_root,
                                       conf.task_type,
                                       conf.metric_name,
                                       conf.train_data_path,
                                       conf.val_data_path,
                                       conf.feature_configue_path,
                                       conf.dataset_type,
                                       conf.type_inference,
                                       conf.data_clear,
                                       conf.feature_generator,
                                       conf.unsupervised_feature_selector,
                                       conf.supervised_feature_selector,
                                       conf.auto_ml)
     auto_model_tree.run()
if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")
   
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
