class SupervisedFeatureSelector(BaseFeatureSelector):

    def __init__(self, name, train_flag, enable, feature_config_path, label_encoding_configure_path):
        super(SupervisedFeatureSelector, self).__init__(name=name,
                                                        train_flag=train_flag,
                                                        enable=enable,
                                                        feature_configure_path=feature_config_path)

        
        self.hyperopt_ml = HyperOptAutoMl("HyperOptAutoMl", train_flag, True, opt_mode=["a","b","c"])
        def _train_run(self, **entity):
                hyperopt_ml.run(entity)

        
    def _predict_run(self, **entity):