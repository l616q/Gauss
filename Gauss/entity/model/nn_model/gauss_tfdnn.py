# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab
#
import os
import gc
import copy
import shutil

from Gauss.entity.model.model import ModelWrapper
from Gauss.entity.dataset.tf_plain_dataset import TFPlainDataset
from Gauss.entity.feature_configuration.feature_config import FeatureConf
from Gauss.core.tfdnn.trainers.trainer import (
    Trainer,
    IncrementalTrainer
)
from Gauss.core.tfdnn.evaluators.evaluator import Evaluator
from Gauss.core.tfdnn.evaluators.predictor import Predictor
from Gauss.core.tfdnn.transforms.numerical_transform import (
    ClsNumericalTransform,
    RegNumericalTransform
)
from Gauss.core.tfdnn.transforms.categorical_transform import CategoricalTransform
from Gauss.core.tfdnn.statistics_gens.dataset_statistics_gen import DatasetStatisticsGen
from Gauss.core.tfdnn.statistics_gens.external_statistics_gen import ExternalStatisticsGen
from Gauss.utils.base import mkdir
from Gauss.utils.feature_name_exec import generate_feature_list
from Gauss.core.tfdnn.factory.network_factory import NetworkFactory
from Gauss.core.tfdnn.factory.loss_factory import LossFunctionFactory


class GaussNN(ModelWrapper):
    """Multi layer perceptron neural network wrapper.

    Model wrapper wrapped a neural network model which can be used in training,
    increment training, and predicting. 

    Parameters:
    --------------
    model_root_path:
    init_model_root:
    metric_eval_used_flag:
    use_weight_flag:
    loss_func: str, loss function name using in current training.
    TODO: fill description.
    """

    def __init__(self, **params):
        name_dict = {
            "binary_classification": "dnn_binary_cls",
            "multiclass_classification": "dnn_multi_cls",
            "regression": "dnn_reg"
            } 
        super(GaussNN, self).__init__(
            name=name_dict[params["task_name"]],
            model_root_path=params["model_root_path"],
            task_name=params["task_name"],
            train_flag=params["train_flag"],
            init_model_root=params["init_model_root"],
            metric_eval_used_flag=params["metric_eval_used_flag"],
            # use_weight_flag=params["use_weight_flag"],
            decay_rate=params["decay_rate"] if params.get("decay_rate")
                else None
        )

        # file name defintion
        self.model_file_name = self._model_root_path + "/" + self.name + ".txt"
        self.model_config_file_name = self._model_config_root + \
            "/" + self.name + ".model_conf.yaml"
        self.feature_config_file_name = self._feature_config_root + \
            "/" + self.name + ".final.yaml"

        # model component saved path
        self._save_statistics_dir = os.path.join(
            self._model_root_path, "statistics")
        self._save_checkpoints_dir = os.path.join(
            self._model_root_path, "checkpoint")
        self._restore_checkpoints_dir = os.path.join(
            self._model_root_path, "restore_checkpoint")
        self._save_tensorboard_logdir = os.path.join(
            self._model_root_path, "tensorboard_logdir")

        self._model_params = {}
        self._categorical_features = None
        self._best_categorical_features = None
        self._numerical_features = None
        self._best_numerical_features = None
        self._best_metric_result = None

        # network components
        self._statistics_gen = None
        self._statistics = None
        self._transform1 = None
        self._transform2 = None
        self._network = None
        self._evaluator = None
        self._trainer = None

        self._create_folders()

    def __repr__(self):
        pass
    

    @property
    def val_metric(self):
        return self._metric.metric_result

    @property
    def val_best_metric_result(self):
        return self._metric.metric_result

    def update_feature_conf(self, feature_conf):
        """Select features using in current model before 'dataset.build()'.

        Select features using in current model and classify them 
        to categorical or numerical. 'feature_conf' is the description
        object of all features, property 'used' in conf will decide whether
        to keep the feature or not, and features will be classified by it's 
        `dtype` mentioned in 'feature_conf'.
        """

        self._feature_list = generate_feature_list(feature_conf=feature_conf)
        self._categorical_features, self._numerical_features = \
            self._cate_num_split(feature_conf=feature_conf)

    def _cate_num_split(self, feature_conf):
        if isinstance(feature_conf, FeatureConf):
            configs = feature_conf.feature_dict
        elif isinstance(feature_conf, dict):
            configs = feature_conf
        else:
            raise AttributeError(
                "feature_conf can only support `dict` or `FeatureConf`."
            )
        categorical_features, numerical_features = [], []
        for name, info in configs.items():
            if name in self._feature_list:
                if info["ftype"] == "numerical":
                    numerical_features.append(name)
                else:
                    categorical_features.append(name)

        return categorical_features, numerical_features

    # Normal training
    def _binary_train(self, train_dataset, val_dataset, **entity):
        """binary classification trainer"""

        self._train_init(train_dataset, val_dataset, **entity)
        self._trainer.run()

    def _multiclass_train(self, train_dataset, val_dataset, **entity):
        """multi classification trainer"""

        self._train_init(train_dataset, val_dataset, **entity)
        self._trainer.run()

    def _train_init(self, train_dataset, val_dataset, **entity):
        """Initialize modules using in training a model and build 
        'Calculation Graph' by tensorflow.

        A neural network training model includes seperated modules,
        'DatasetStatisticsGen', 'CategoricalTransform', 'NumericalTransform', 
        'LossFunction', 'MlpNetwork', 'Evaluator', and 'Trainer', all above parts
        collaborate and construct a whole training procedure. 

        Parameters:
        --------------
        dataset: PlaintextDataset, dataset for training.
        val_dataset: PlaintextDataset, dataset for validate current model.
        metrics: Metrics, judgement scores for evaluate a model.
        """
        self._reset_tf_graph()

        # Phase 1. Load and transform Dataset -----------------------
        train_dataset = self.preprocess(train_dataset)
        val_dataset = self.preprocess(val_dataset)


        train_dataset.update_features(
            self._feature_list, self._categorical_features)
        val_dataset.update_features(
            self._feature_list, self._categorical_features)

        train_dataset.build()
        val_dataset.build()

        # TODO: fix the label_class_count
        train_dataset.update_dataset_parameters(
            batch_size = self._model_params["batch_size"],
            label_class_count = 2
            )
        val_dataset.update_dataset_parameters(
            self._model_params["batch_size"],
            label_class_count = 2
            )

        # Phase 2. Create Feature Statistics, and Run -----------------------
        statistics_gen = DatasetStatisticsGen(
            dataset=train_dataset,
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features
        )
        self._statistics = statistics_gen.run()
        # Phase 3. Create Transform and Network, and Run -----------------------
        self._transform1 = CategoricalTransform(
            statistics=self._statistics,
            feature_names=self._categorical_features,
            embed_size=self._model_params["embed_size"],
        )
        self._transform2 = ClsNumericalTransform(
            statistics=self._statistics,
            feature_names=self._numerical_features
        )
        Loss = LossFunctionFactory.get_loss_function(
            func_name=self._model_params["loss_name"]
            )
        Network = NetworkFactory.get_network(task_name=self._task_name)
        self._network = Network(
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features,
            task_name=self._task_name,
            activation=self._model_params["activation"],
            hidden_sizes=self._model_params["hidden_sizes"],
            loss=Loss(label_name=train_dataset.target_names)
        )
        # Phase 4. Create Evaluator and Trainer ----------------------------
        self._metric = entity["metric"]
        metrics_wrapper = {
            self._metric._name: self._metric
        }
        self._evaluator = Evaluator(
            dataset=val_dataset,
            transform_functions=[
                self._transform1.transform_fn, self._transform2.transform_fn],
            eval_fn=self._network.eval_fn,
            metrics=metrics_wrapper,
        )

        self._trainer = Trainer(
            dataset=train_dataset,
            transform_functions=[
                self._transform1.transform_fn, self._transform2.transform_fn],
            train_fn=self._network.train_fn,
            validate_steps=self._model_params["validate_steps"],
            log_steps=self._model_params["log_steps"],
            learning_rate=self._model_params["learning_rate"],
            optimizer_type=self._model_params["optimizer_type"],
            train_epochs=self._model_params["train_epochs"],
            early_stop=self._model_params["early_stop"],
            evaluator=self._evaluator,
            save_checkpoints_dir=self._save_checkpoints_dir,
            tensorboard_logdir=self._save_tensorboard_logdir
        )

    def increment_init(self, train_dataset, val_dataset, **entity):

        # Phase 1. Load and transform Dataset -----------------------
        train_dataset = self.preprocess(train_dataset)
        val_dataset = self.preprocess(val_dataset)
        train_dataset.update_features(
            self._feature_list, self._categorical_features)
        val_dataset.update_features(
            self._feature_list, self._categorical_features)
        train_dataset.build()
        val_dataset.build()
        # Phase 2. Load Feature Statistics, and Run -----------------------
        statistic_gen = ExternalStatisticsGen(
            filepath=os.path.join(self._save_statistics_dir, "statistics.pkl")
        )
        res = os.path.join(self._save_statistics_dir, "statistics.pkl")
        statistics = statistic_gen.run()
        # Phase 3. Create Transform and Network, and Run -----------------------
        self._transform1 = CategoricalTransform(
            statistics=statistics,
            feature_names=self._categorical_features,
            embed_size=self._model_params["embed_size"],
        )
        self._transform2 = RegNumericalTransform(
            statistics=statistics,
            feature_names=self._numerical_features
        )
        Loss = LossFunctionFactory.get_loss_function(func_name=self._loss_name)
        Network = NetworkFactory.get_network(task_name=self._task_name)
        self._network = Network(
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features,
            task_name=self._task_name,
            activation=self._model_params["activation"],
            hidden_sizes=self._model_params["hidden_sizes"],
            loss=Loss(label_name=train_dataset.target_names)
        )
        # Phase 4. Create Evaluator and Trainer ----------------------------
        self._metric = entity["metric"]
        metrics_wrapper = {
            self._metric.name: self._metric
        }
        self._trainer = IncrementalTrainer(
            dataset=train_dataset,
            transform_functions=[
                self._transform1.transform_fn, self._transform2.transform_fn],
            train_fn=self._network.train_fn,
            log_steps=self._model_params["log_steps"],
            learning_rate=self._model_params["learning_rate"],
            optimizer_type=self._model_params["optimizer_type"],
            train_epochs=1,
            save_checkpoints_dir=self._save_checkpoints_dir,
            tensorboard_logdir=self._save_tensorboard_logdir
        )

    def inference_init(self, **entity):
        #TODO: sycn with super class
        """Initialize calculation graph and load 'tf.Variables' to graph for
        prediction mission.
        Activate only in predict mission. Data statistic information from 
        current best performance existed model will be loaded. And Checkpoint 
        of same model will be load to 'tf.Graph'
        """
        assert(entity.get("val_dataset"))

        # Phase 1. Load and transform Dataset -----------------------
        dataset = self.preprocess(entity["val_dataset"])
        dataset.update_features(self._feature_list, self._categorical_features)
        dataset.build()

        # Phase 2. Load Feature Statistics, and Run -----------------------
        statistic_gen = ExternalStatisticsGen(
            filepath=os.path.join(self._save_statistics_dir, "statistics.pkl")
        )
        statistics = statistic_gen.run()

        # Phase 3. Create Transform and Network, and Run -----------------------
        self._transform1 = CategoricalTransform(
            statistics=statistics,
            feature_names=self._categorical_features,
            embed_size=self._model_params["embed_size"],
        )
        self._transform2 = RegNumericalTransform(
            statistics=statistics,
            feature_names=self._numerical_features
        )
        Network = NetworkFactory.get_network(task_name=self._task_name)
        self._network = Network(
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features,
            task_name=self._task_name,
            hidden_sizes=self._model_params["hidden_sizes"],
        )
        # Phase 4. Create Predictor ----------------------------
        if not self._train_flag:
            self._evaluator = Predictor(
                dataset=dataset,
                transform_functions=[
                    self._transform1.transform_fn, self._transform2.transform_fn],
                eval_fn=self._network.eval_fn,
                restore_checkpoints_dir=self._restore_checkpoints_dir,
            )


    def increment(self, **entity):
        self.increment_init(**entity)
        self._trainer.run()

    def predict(self, **entity):
        self.inference_init(**entity)
        predict = self._inference_evaluator.run()
        return predict

    def _eval(self, train_dataset, val_dataset, metric, **entity):
        if self._train_flag == "train":
            pass
        else:
            self.inference_init(**entity)
            self._inference_evaluator.run()

    def update_best_model(self):
        assert self._trainer is not None
        
        self.metric_history.append(self._metric.metric_result.result)

        if self._best_metric_result is None or \
                (self._metric.metric_result.result > self._best_metric_result):

            self._update_checkpoint()
            self._update_statistics()
            # TODO: update tensorboard
            self._best_model_params = copy.deepcopy(self._model_params)
            self._best_metric_result = self._metric.metric_result.result
            self._best_feature_list = copy.deepcopy(self._feature_list)
            self._best_categorical_features = copy.deepcopy(
                self._categorical_features)
            self._best_numerical_features = copy.deepcopy(
                self._numerical_features)

    def set_best_model(self):
        self._model_params = copy.deepcopy(self._best_model_params)
        self._feature_list = copy.deepcopy(self._best_feature_list)
        self._categorical_features = copy.deepcopy(
            self._best_categorical_features)
        self._numerical_features = copy.deepcopy(self._best_numerical_features)

    def preprocess(self, dataset):
        dataset = TFPlainDataset(
            name="tf_dataset",
            dataset=dataset,
            task_name=self._task_name,
            train_flag=self._train_flag,
            memory_only=True,
            target_names=dataset.get_dataset().target_names
        )
        return dataset

    def _create_folders(self):
        path_attrs = [
            self._save_statistics_dir, self._save_checkpoints_dir,
            self._restore_checkpoints_dir, self._save_tensorboard_logdir,
        ]
        for path in path_attrs:
            if not os.path.isdir(path):
                mkdir(path)

    def _reset_trail(self):
        attrs = [
            "_model_params", "_feature_conf", "_feature_list", "_categorical_features"
            "_numerical_features", "_statistics_gen", "_statistics", "_transform1",
            "_transform2", "_network", "_evaluator", "_trainer"
        ]
        for attr in attrs:
            delattr(self, attr)
            gc.collect()
            setattr(self, attr, None)

    def _update_checkpoint(self):
        import tensorflow as tf

        if len(os.listdir(self._save_checkpoints_dir)) == 0:
            raise TypeError("checkpoint has not been found in folder <{dir}>".format(
                dir=self._save_checkpoints_dir
            ))
        if os.path.isdir(self._restore_checkpoints_dir):
            shutil.rmtree(self._restore_checkpoints_dir)
        os.mkdir(self._restore_checkpoints_dir)
        prefix = tf.train.latest_checkpoint(self._save_checkpoints_dir) + "*"
        os.system("cp {ckpt_dir} {target_dir}".format(
            ckpt_dir=self._save_checkpoints_dir + "checkpoint",
            target_dir=self._restore_checkpoints_dir
        ))
        os.system("cp {ckpt} {target_dir}".format(
            ckpt=prefix,
            target_dir=self._restore_checkpoints_dir
        ))

    def _update_statistics(self):
        self._statistics.save_to_file(
            filepath=os.path.join(self._save_statistics_dir, "statistics.pkl")
        )

    def initialize(self):
        self._reset_trail()

    def _reset_tf_graph(self):
        """dismiss the existed calculation graph"""

        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

    def _initialize_model(self):
        pass

    def _update_best(self):
        pass

    def _regression_train(self):
        pass

    def _binary_increment(self):
        pass

    def _multiclass_increment(self):
        pass

    def _regression_increment(self):
        pass

    def model_save(self):
        pass

    def _predict_prob(self):
        pass

    def _predict_logit(self):
        pass

    def _set_best(self):
        pass

    def _train_preprocess(self):
        pass

    def _predict_preprocess(self):
        pass

    def _loss_func(self):
        pass

    def _eval_func(self):
        pass
