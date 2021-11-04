# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

from core.tfdnn.trainers.base_trainer import BaseTrainer
from core.tfdnn.utils.earlystop import Earlystop


class Trainer(BaseTrainer):
    """Training process for normal training task.

    Parameters:
    ----------
    early_stop: bool, Wether the Early stop function activated. If True, `patience`
        and `delta` should be provided, otherwise could be None.
    patience: int, toralance of the epoch number when monitored metrics decreasing.
    delta: int, float, toralance of the increasing value monitored metrics value.
    """

    def __init__(self,
                 dataset,
                 transform_functions,
                 train_fn,
                 validate_steps,
                 log_steps,
                 learning_rate,
                 optimizer_type="lazy_adam",
                 train_epochs=100,
                 early_stop=False,
                 patience=3,
                 delta=0.001,
                 evaluator=None,
                 save_checkpoints_dir=None,
                 restore_checkpoint_dir=None,
                 validate_at_start=False,
                 tensorboard_logdir=None):

        super(Trainer, self).__init__(
            dataset=dataset,
            transform_functions=transform_functions,
            train_fn=train_fn,
            log_steps=log_steps,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            train_epochs=train_epochs,
            save_checkpoints_dir=save_checkpoints_dir,
            restore_checkpoint_dir=restore_checkpoint_dir,
            tensorboard_logdir=tensorboard_logdir
        )
        self._validate_steps = validate_steps
        self._validate_at_start = validate_at_start
        self._evaluator = evaluator
        self._early_stop = early_stop

        if self._early_stop:
            self._terminator = Earlystop(patience, delta)


    def _run_train_loop(self):
        if self._validate_at_start:
            self._validate(epoch=0, step=0)

        step = 0
        for epoch in range(self._train_epochs):
            avg_loss = 0
            counter = 0
            self._dataset.init(self._sess)
            while True:
                success, step_loss = self._train_step(epoch, step)
                avg_loss += step_loss
                counter += 1
                if not success:
                    avg_loss /= counter
                    break
                step += 1
                if step % self._validate_steps == 0:
                    eval_result = self._run_validate_loop(epoch=epoch, step=step)
            if self._early_stop:
                self._terminator(epoch, avg_loss)
                if self._terminator.flag:
                    break
            self._save_checkpoint(epoch + 1)
            
    def _run_validate_loop(self, epoch, step):
        if self._evaluator:
            eval_results = self._evaluator.run(sess=self._sess)
            if self._valid_logger:
                self._valid_logger.log_info(eval_results, epoch=epoch, step=step)


class IncrementalTrainer(BaseTrainer):
    """Training process for incremental training.

    Training epochs will be locked as a certain value no matter how many batches 
    data will be incremented.
    """

    def __init__(self,
                 dataset,
                 transform_functions,
                 train_fn,
                 log_steps,
                 learning_rate,
                 optimizer_type="lazy_adam",
                 train_epochs=5,
                 save_checkpoints_dir=None,
                 restore_checkpoint_dir=None,
                 tensorboard_logdir=None):

        super(IncrementalTrainer, self).__init__(
            dataset=dataset,
            transform_functions=transform_functions,
            train_fn=train_fn,
            log_steps=log_steps,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            train_epochs=5,
            save_checkpoints_dir=save_checkpoints_dir,
            restore_checkpoint_dir=restore_checkpoint_dir,
            tensorboard_logdir=tensorboard_logdir
        )


    def _run_train_loop(self):
        step = 0
        for epoch in range(self._train_epochs):
            avg_loss = 0
            counter = 0
            self._dataset.init(self._sess)
            while True:
                success, step_loss = self._train_step(epoch, step)
                avg_loss += step_loss
                counter += 1
                if not success:
                    avg_loss /= counter
                    break
                step += 1
            self._save_checkpoint(epoch + 1)