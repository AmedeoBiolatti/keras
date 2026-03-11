import collections
import itertools
import warnings
from functools import partial

import jax
import numpy as np

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.backend.config import is_nnx_enabled
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
from keras.src.utils.python_utils import pythonify_logs

if is_nnx_enabled():
    from flax import nnx

    jit = nnx.jit
else:
    jit = jax.jit


class JAXTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        self._jax_state_synced = True

    def compute_loss_and_updates(
            self,
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=False,
            optimizer_variables=None,
    ):
        """This method is stateless and is intended for use with jax.grad."""
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = training

        # Run stateless forward pass
        y_pred, non_trainable_variables, losses = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            return_losses=True,
            **kwargs,
        )
        if losses:
            # Make forward pass losses available to compute_loss.
            self._losses_override.clear()
            self._losses_override = losses

        loss, variables = self.stateless_compute_loss(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=training,
        )
        if losses:
            self._losses_override.clear()
        (trainable_variables, non_trainable_variables, metrics_variables) = (
            variables
        )

        # Handle loss scaling
        unscaled_loss = loss
        if training and self.optimizer is not None:
            # Scale loss with a StatelessScope, to use an update scale variable.
            mapping = list(zip(self.optimizer.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping):
                loss = self.optimizer.scale_loss(loss)
        return loss, (
            unscaled_loss,
            y_pred,
            non_trainable_variables,
            metrics_variables,
        )

    def _update_metrics_variables(
            self, metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
    ):
        with backend.StatelessScope(
                state_mapping=[
                    (ref_v, v)
                    for ref_v, v in zip(self.metrics_variables, metrics_variables)
                ]
        ) as scope:
            self._loss_tracker.update_state(
                unscaled_loss,
                sample_weight=next(
                    i for i in tree.flatten(x) if i is not None
                ).shape[0],
            )
            logs = self.compute_metrics(x, y, y_pred, sample_weight)

        new_metrics_variables = []
        for ref_v in self.metrics_variables:
            new_v = scope.get_current_value(ref_v)
            if new_v is None:
                new_v = ref_v.value
            new_metrics_variables.append(new_v)
        return logs, new_metrics_variables

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )
        (loss, aux), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=True,
            optimizer_variables=optimizer_variables,
        )
        (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
            aux
        )

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        logs, metrics_variables = self._update_metrics_variables(
            metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
        )

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        return logs, state

    def test_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        loss, aux = self.compute_loss_and_updates(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=False,
        )
        (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
            aux
        )

        logs, metrics_variables = self._update_metrics_variables(
            metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
        )

        state = (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        )
        return logs, state

    def predict_step(self, state, data):
        trainable_variables, non_trainable_variables = state
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = False

        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        outputs, non_trainable_variables = self.stateless_call(
            trainable_variables, non_trainable_variables, x, **kwargs
        )
        return outputs, non_trainable_variables

    def _make_function(
            self,
            step_function,
            concatenate_outputs=False,
            donate_argnums=0,
            out_shardings=None
    ):
        def _leading_dim(batch):
            leaves = jax.tree.leaves(batch)
            if not leaves:
                raise ValueError("Batch pytree has no leaves.")
            return leaves[0].shape[0]

        def _concat_outputs(first, second):
            return tree.map_structure(
                lambda x, y: jax.numpy.concatenate((x, y), axis=0),
                first,
                second,
            )

        def _flatten_scan_outputs(outputs):
            return tree.map_structure(
                lambda x: x.reshape((-1,) + x.shape[2:]), outputs
            )

        step_function_raw = step_function
        remainder_warning_emitted = False

        if self.steps_per_execution > 1:
            def scan_body_with_outputs(state, data):
                outputs, state = step_function_raw(state, data)
                return state, outputs

            def scan_body_no_outputs(state, data):
                _outputs, state = step_function_raw(state, data)
                return state, None

            # stack_and_scan
            def multistep_function(state, stacked_data):
                if concatenate_outputs:
                    state, outputs = jax.lax.scan(
                        scan_body_with_outputs,
                        init=state,
                        xs=stacked_data
                    )
                    return _flatten_scan_outputs(outputs), state
                else:
                    # n = _leading_dim(stacked_data)
                    # if n == 1:
                    #     last = jax.tree.map(lambda xi: xi[-1], stacked_data)
                    #     state, outputs = scan_body_with_outputs(state, last)
                    #     return outputs, state
                    #
                    # prefix = jax.tree.map(lambda xi: xi[:-1], stacked_data)
                    # last = jax.tree.map(lambda xi: xi[-1], stacked_data)
                    #
                    # state, _ = jax.lax.scan(
                    #     lambda s, d: (scan_body_no_outputs(s, d)[0], None),
                    #     init=state,
                    #     xs=prefix,
                    # )
                    # state, outputs = scan_body_with_outputs(state, last)
                    state, outputs = jax.lax.scan(
                        scan_body_with_outputs,
                        init=state,
                        xs=stacked_data,
                        unroll=2
                    )
                    outputs = jax.tree.map(lambda xi: xi[-1], outputs)
                    return outputs, state

            if not self.run_eagerly and self.jit_compile:
                step_function = jit(
                    step_function,
                    donate_argnums=donate_argnums,
                    out_shardings=out_shardings
                )
                multistep_function = jit(
                    multistep_function,
                    donate_argnums=donate_argnums,
                    out_shardings=out_shardings
                )

            def iterator_step(state, iterator):
                nonlocal remainder_warning_emitted
                # Stacked epoch iterator path used in fit().
                if (
                        isinstance(iterator, (tuple, list))
                        and len(iterator) == 2
                ):
                    stacked_batches, remainder_batch = iterator

                    outputs = None
                    if stacked_batches is not None:
                        with jax.profiler.TraceAnnotation(
                            "stacked.iterator_step.multistep_function"
                        ):
                            outputs, state = multistep_function(
                                state, stacked_batches
                            )
                    if remainder_batch is not None:
                        # In SPE>1 mode, a smaller remainder batch at epoch end
                        # often triggers a one-time JIT compile for step_function.
                        if (
                                not remainder_warning_emitted
                                and stacked_batches is not None
                        ):
                            leaves = jax.tree.leaves(stacked_batches)
                            full_batch_dim = (
                                leaves[0].shape[1]
                                if leaves and leaves[0].ndim >= 2
                                else None
                            )
                            remainder_dim = _leading_dim(remainder_batch)
                            if (
                                    full_batch_dim is not None
                                    and remainder_dim != full_batch_dim
                            ):
                                warnings.warn(
                                    "JAX fit(): remainder batch size "
                                    f"{remainder_dim} differs from full batch "
                                    f"size {full_batch_dim} with "
                                    "steps_per_execution>1. This can trigger a "
                                    "one-time JIT compile near epoch end."
                                )
                                remainder_warning_emitted = True
                        with jax.profiler.TraceAnnotation(
                            "stacked.iterator_step.remainder_step_function"
                        ):
                            remainder_outputs, state = step_function(
                                state, remainder_batch
                            )
                        if concatenate_outputs and outputs is not None:
                            outputs = _concat_outputs(
                                outputs, remainder_outputs
                            )
                        else:
                            outputs = remainder_outputs

                    return outputs, state

                # Standard iterator path used in evaluate/predict/*_on_batch.
                has_step = False
                outputs = None
                for _, data in zip(range(self.steps_per_execution), iterator):
                    has_step = True
                    step_outputs, state = step_function(state, data)
                    if not concatenate_outputs:
                        outputs = step_outputs
                    elif outputs is None:
                        outputs = step_outputs
                    else:
                        outputs = _concat_outputs(outputs, step_outputs)
                if not has_step:
                    raise StopIteration
                return outputs, state

        else:
            if not self.run_eagerly and self.jit_compile:
                step_function = jit(
                    step_function,
                    donate_argnums=donate_argnums,
                    out_shardings=out_shardings
                )

            def iterator_step(state, iterator):
                return step_function(state, next(iterator))

        return iterator_step

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return

        out_shardings = None
        if distribution_lib.distribution() is not None:
            state_shardings = self._get_state_sharding_spec()
            out_shardings = (None, state_shardings)
        if is_nnx_enabled():
            step_fn = lambda state, data: type(self).train_step(
                self, state, data
            )
        else:
            step_fn = self.train_step

        step_function = self._make_function(step_fn, donate_argnums=0, out_shardings=out_shardings)
        self.train_function = step_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return
        if not self.run_eagerly and self.jit_compile:
            out_shardings = None
            if distribution_lib.distribution() is not None:
                (
                    trainable_shardings,
                    non_trainable_shardings,
                    _,  # optimizer_shardings
                    metrics_shardings,
                ) = self._get_state_sharding_spec()
                state_shardings = (
                    trainable_shardings,
                    non_trainable_shardings,
                    metrics_shardings,
                )
                out_shardings = (None, state_shardings)
            if is_nnx_enabled():
                step_fn = lambda state, data: type(self).test_step(
                    self, state, data
                )
            else:
                step_fn = self.test_step
            test_step = step_fn
        else:
            out_shardings = None
            test_step = self.test_step

        step_function = self._make_function(test_step, donate_argnums=0, out_shardings=out_shardings)

        self.test_function = step_function

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def predict_step(state, data):
            outputs, non_trainable_variables = self.predict_step(state, data)
            return outputs, (state[0], non_trainable_variables)

        if not self.run_eagerly and self.jit_compile:
            out_shardings = None
            if distribution_lib.distribution() is not None:
                (
                    trainable_shardings,
                    non_trainable_shardings,
                    _,  # optimizer_shardings
                    _,  # metrics_shardings
                ) = self._get_state_sharding_spec()
                state_shardings = (
                    trainable_shardings,
                    non_trainable_shardings,
                )
                out_shardings = (None, state_shardings)
        else:
            out_shardings = None

        _step_function = self._make_function(
            predict_step, concatenate_outputs=True, donate_argnums=0, out_shardings=out_shardings
        )

        def step_function(state, iterator):
            outputs, state = _step_function(state, iterator)
            return outputs, state

        self.predict_function = step_function

    @traceback_utils.filter_traceback
    def fit(
            self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose="auto",
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
    ):
        self._assert_compile_called("fit")
        # Possibly cap epochs for debugging runs.
        max_epochs = config.max_epochs()
        if max_epochs and max_epochs < epochs:
            warnings.warn("Limiting epochs to %d" % max_epochs)
            epochs = max_epochs
        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        stacked = (self.steps_per_execution > 1)
        # Create an iterator that yields batches for one epoch.
        epoch_iterator_cls = JAXEpochStackedIteratorV2 if stacked else JAXEpochIterator
        epoch_iterator = epoch_iterator_cls(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        self._symbolic_build(data_batch=epoch_iterator.get_batch())
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_train_function()
        self.stop_training = False
        training_logs = {}
        training_finished = False
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        try:
            for epoch in range(initial_epoch, epochs):
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)

                self._jax_state_synced = True
                with epoch_iterator.catch_stop_iteration():
                    for begin_step, end_step, iterator in epoch_iterator:
                        # Callbacks
                        with jax.profiler.TraceAnnotation(
                            "fit.on_train_batch_begin"
                        ):
                            callbacks.on_train_batch_begin(begin_step)

                        # Train step
                        if self._jax_state_synced:
                            # The state may have been synced by a callback.
                            state = self._get_jax_state(
                                trainable_variables=True,
                                non_trainable_variables=True,
                                optimizer_variables=True,
                                metrics_variables=True,
                                purge_model_variables=True,
                            )
                            self._jax_state_synced = False

                        with jax.profiler.TraceAnnotation(
                            "fit.train_function_call"
                        ):
                            logs, state = self.train_function(state, iterator)
                        (
                            trainable_variables,
                            non_trainable_variables,
                            optimizer_variables,
                            metrics_variables,
                        ) = state

                        # Setting _jax_state enables callbacks to force a state
                        # sync if they need to.
                        self._jax_state = {
                            "trainable_variables": trainable_variables,
                            "non_trainable_variables": non_trainable_variables,
                            "optimizer_variables": optimizer_variables,
                            "metrics_variables": metrics_variables,
                        }
                        # Dispatch callbacks. This takes care of async dispatch.
                        with jax.profiler.TraceAnnotation(
                            "fit.on_train_batch_end"
                        ):
                            callbacks.on_train_batch_end(end_step, logs)

                        if self.stop_training:
                            # Stop training if a callback has set
                            # this flag in on_(train_)batch_end.
                            break

                # Reattach state to the model
                # (if not already done by a callback).
                # NOTE: doing this after each step would be a big performance
                # bottleneck.
                self.jax_state_sync()

                # Override with model metrics instead of last step logs if
                # needed.
                epoch_logs = dict(self._get_metrics_result_or_logs(logs))

                # Run validation.
                if validation_data is not None and self._should_eval(
                        epoch, validation_freq
                ):
                    # Create JAXEpochIterator for evaluation and cache it.
                    if getattr(self, "_eval_epoch_iterator", None) is None:
                        self._eval_epoch_iterator = JAXEpochIterator(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_execution=self.steps_per_execution,
                            steps_per_epoch=validation_steps,
                            shuffle=False,
                        )
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        return_dict=True,
                        _use_cached_eval_dataset=True,
                    )
                    val_logs = {
                        f"val_{name}": val for name, val in val_logs.items()
                    }
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break
            training_finished = True

        finally:
            self.jax_state_sync()
            if (
                    isinstance(self.optimizer, optimizers_module.Optimizer)
                    and epochs > 0
            ):
                self.optimizer.finalize_variable_values(self.trainable_weights)

            # If _eval_epoch_iterator exists, delete it after all epochs
            # are done.
            if getattr(self, "_eval_epoch_iterator", None) is not None:
                del self._eval_epoch_iterator
            if training_finished:
                callbacks.on_train_end(logs=training_logs)
            self._jax_state = None
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(
            self,
            x=None,
            y=None,
            batch_size=None,
            verbose="auto",
            sample_weight=None,
            steps=None,
            callbacks=None,
            return_dict=False,
            **kwargs,
    ):
        self._assert_compile_called("evaluate")
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of
            # input/target data.
            epoch_iterator = JAXEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        self._symbolic_build(data_batch=epoch_iterator.get_batch())
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()

        self._jax_state_synced = True
        with epoch_iterator.catch_stop_iteration():
            for begin_step, end_step, iterator in epoch_iterator:
                callbacks.on_test_batch_begin(begin_step)

                if self._jax_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_jax_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        metrics_variables=True,
                        purge_model_variables=True,
                    )
                    self._jax_state_synced = False

                logs, state = self.test_function(state, iterator)
                (
                    trainable_variables,
                    non_trainable_variables,
                    metrics_variables,
                ) = state

                # Setting _jax_state enables callbacks to force a state sync
                # if they need to.
                self._jax_state = {
                    # I wouldn't recommend modifying non-trainable model state
                    # during evaluate(), but it's allowed.
                    "trainable_variables": trainable_variables,
                    "non_trainable_variables": non_trainable_variables,
                    "metrics_variables": metrics_variables,
                }

                # Dispatch callbacks. This takes care of async dispatch.
                callbacks.on_test_batch_end(end_step, logs)

                if self.stop_evaluating:
                    break

        # Reattach state back to model (if not already done by a callback).
        self.jax_state_sync()

        logs = pythonify_logs(self._get_metrics_result_or_logs(logs))
        callbacks.on_test_end(logs)
        self._jax_state = None

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
            self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = JAXEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        if not all(layer.built for layer in self._flatten_layers()):
            # Build the model on one batch of data.
            x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(
                epoch_iterator.get_batch()
            )
            if is_nnx_enabled():
                self(x)
            else:
                with backend.StatelessScope():
                    self(x)
        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        self._jax_state_synced = True
        outputs = None
        non_trainable_variables = None
        with epoch_iterator.catch_stop_iteration():
            for begin_step, end_step, iterator in epoch_iterator:
                callbacks.on_predict_batch_begin(begin_step)
                if self._jax_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_jax_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        purge_model_variables=True,
                    )
                    self._jax_state_synced = False
                batch_outputs, state = self.predict_function(state, iterator)
                (
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                self._jax_state = {
                    "trainable_variables": trainable_variables,
                    # I wouldn't recommend modifying non-trainable model state
                    # during predict(), but it's allowed.
                    "non_trainable_variables": non_trainable_variables,
                }
                outputs = append_to_outputs(batch_outputs, outputs)

                # Dispatch callbacks. This takes care of async dispatch.
                callbacks.on_predict_batch_end(
                    end_step, {"outputs": batch_outputs}
                )

                if self.stop_predicting:
                    break

        self.jax_state_sync()
        callbacks.on_predict_end()
        self._jax_state = None
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

    def train_on_batch(
            self,
            x,
            y=None,
            sample_weight=None,
            class_weight=None,
            return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        def data():
            yield _distribute_data((x, y, sample_weight))

        # Maybe build model
        self._symbolic_build(data_batch=next(data()))
        self.make_train_function()

        # Train step
        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            optimizer_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._jax_state_synced = False
        logs, state = self.train_function(state, data())

        # State sync
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "optimizer_variables": optimizer_variables,
            "metrics_variables": metrics_variables,
        }
        self.jax_state_sync()

        # Format return values
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
            self,
            x,
            y=None,
            sample_weight=None,
            return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")

        def data():
            yield _distribute_data((x, y, sample_weight))

        # Maybe build model
        self._symbolic_build(data_batch=next(data()))
        self.make_test_function()

        # Test step
        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._jax_state_synced = False
        logs, state = self.test_function(state, data())

        # State sync
        trainable_variables, non_trainable_variables, metrics_variables = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "metrics_variables": metrics_variables,
        }
        self.jax_state_sync()

        # Format return values.
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        if not all(layer.built for layer in self._flatten_layers()):
            # Build model
            with backend.StatelessScope():
                self(x)
        self.make_predict_function()

        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=False,
            purge_model_variables=False,
        )
        self._jax_state_synced = False

        def data():
            yield (x,)

        batch_outputs, state = self.predict_function(state, data())
        trainable_variables, non_trainable_variables = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
        }
        self.jax_state_sync()
        batch_outputs = tree.map_structure(lambda x: np.array(x), batch_outputs)
        return batch_outputs

    def jax_state_sync(self):
        if not getattr(self, "_jax_state", None) or self._jax_state_synced:
            return

        trainable_variables = self._jax_state.get("trainable_variables", None)
        non_trainable_variables = self._jax_state.get(
            "non_trainable_variables", None
        )
        optimizer_variables = self._jax_state.get("optimizer_variables", None)
        metrics_variables = self._jax_state.get("metrics_variables", None)
        if trainable_variables:
            for ref_v, v in zip(self.trainable_variables, trainable_variables):
                ref_v.assign(v)
        if non_trainable_variables:
            for ref_v, v in zip(
                    self.non_trainable_variables, non_trainable_variables
            ):
                ref_v.assign(v)
        if optimizer_variables:
            for ref_v, v in zip(self.optimizer.variables, optimizer_variables):
                ref_v.assign(v)
        if metrics_variables:
            for ref_v, v in zip(self.metrics_variables, metrics_variables):
                ref_v.assign(v)
        self._jax_state_synced = True

    def _get_state_sharding_spec(self):
        trainable_shardings = [
            v.value.sharding for v in self.trainable_variables
        ]
        non_trainable_shardings = [
            v.value.sharding for v in self.non_trainable_variables
        ]
        if hasattr(self, "optimizer") and self.optimizer is not None:
            optimizer_shardings = [
                v.value.sharding for v in self.optimizer.variables
            ]
        else:
            optimizer_shardings = []
        metrics_shardings = [v.value.sharding for v in self.metrics_variables]
        return (
            trainable_shardings,
            non_trainable_shardings,
            optimizer_shardings,
            metrics_shardings,
        )

    def _purge_model_variables(
            self,
            trainable_variables=False,
            non_trainable_variables=False,
            optimizer_variables=False,
            metrics_variables=False,
    ):
        """Remove all the model variable for memory saving.

        During JAX training, since the training function is stateless, we have
        to pass in and get the model weights over and over, during which the
        copy of the weights that attached to the Variable are still and
        occupying extra memory. We remove those variable to save memory (for
        better memory utilization) at the beginning of the epoch, and reattach
        the value back to variables at the end of the epoch, via
        `jax_state_sync()`.
        """
        if trainable_variables:
            for v in self.trainable_variables:
                v._value = None
        if non_trainable_variables:
            for v in self.non_trainable_variables:
                v._value = None
        if optimizer_variables:
            for v in self.optimizer.variables:
                v._value = None
        if metrics_variables:
            for v in self.metrics_variables:
                v._value = None

    def _get_jax_state(
            self,
            trainable_variables=False,
            non_trainable_variables=False,
            optimizer_variables=False,
            metrics_variables=False,
            purge_model_variables=False,
    ):
        state = []
        if trainable_variables:
            state.append([v.value for v in self.trainable_variables])
        if non_trainable_variables:
            state.append([v.value for v in self.non_trainable_variables])
        if optimizer_variables:
            state.append([v.value for v in self.optimizer.variables])
        if metrics_variables:
            state.append([v.value for v in self.metrics_variables])
        if purge_model_variables:
            self._purge_model_variables(
                trainable_variables=trainable_variables,
                non_trainable_variables=non_trainable_variables,
                optimizer_variables=optimizer_variables,
                metrics_variables=metrics_variables,
            )
        return tuple(state)


def _distribute_data(data, layouts=None):
    distribution = distribution_lib.distribution()

    if distribution is not None:
        if layouts is None:
            layouts = tree.map_structure(
                lambda d: distribution.get_data_layout(d.shape),
                data,
            )
        jax_dist_data_input = partial(
            jax_distribution_lib.distribute_data_input,
            batch_dim_name=distribution.batch_dim_name,
        )
        return tree.map_structure(jax_dist_data_input, data, layouts)

    return tree.map_structure(jax.device_put, data)


class JAXEpochIterator(EpochIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = self.steps_per_execution + 1

    def __next__(self):
        return next(self._epoch_iterator)

    def get_batch(self):
        return next(iter(self._get_iterator()))

    def _get_iterator(self):
        distribution = distribution_lib.distribution()
        if distribution is not None:
            return self._get_distributed_iterator(distribution)
        if self.data_adapter.builtin_prefetch:
            return self.data_adapter.get_jax_iterator()
        else:
            return self._prefetch_numpy_iterator(
                self.data_adapter.get_jax_iterator()
            )

    def _get_distributed_iterator(self, distribution):
        """Lazily compute layouts to reduce host to device transfer latency."""
        layouts = None
        for data in self.data_adapter.get_jax_iterator():
            if layouts is None:
                layouts = tree.map_structure(
                    lambda d: distribution.get_data_layout(
                        d.shape
                    ).backend_layout,
                    data,
                )
            yield _distribute_data(data, layouts)

    def _prefetch_numpy_iterator(self, numpy_iterator):
        """Shard and prefetch batches on device.

        Most of the implementation has been borrowed from
        `flax.jax_utils.prefetch_to_device`

        This utility takes an iterator and returns a new iterator which fills an
        on device prefetch buffer. Eager prefetching can improve the performance
        of training loops significantly by overlapping compute and data
        transfer.
        """
        queue = collections.deque()

        # If you're training on GPUs, 2 is generally the best choice because
        # this guarantees that you can overlap a training step on GPU with a
        # data prefetch step on CPU.
        def enqueue(n=2):
            for data in itertools.islice(numpy_iterator, n):
                queue.append(_distribute_data(data))

        enqueue(n=self.n)
        while queue:
            yield queue.popleft()
            enqueue(1)


class JAXEpochStackedIterator(JAXEpochIterator):
    """JAX epoch iterator that groups batches on the host before device transfer.

    Each iteration returns:
        (begin_step, end_step, (stacked_batch, remainder_batch))

    where:
      - stacked_batch is a pytree stacked on axis 0 with shape [k, batch, ...]
        for 1 <= k <= steps_per_execution, or None.
      - remainder_batch is the final partial batch if encountered, else None.

    Unlike the previous version, this iterator:
      1. reads raw batches from the data adapter,
      2. groups and stacks them on the host,
      3. transfers the grouped object to device once,
      4. prefetches grouped items.
    """

    def __init__(self, *args, **kwargs):
        self._configured_batch_size = kwargs.get("batch_size", None)
        self.batch_size = self._configured_batch_size
        super().__init__(*args, **kwargs)
        self._group_prefetch_n = 2
        self._layout_cache = {}
        self._queue_end = object()
        self._prefetch_queue = None
        self._producer_thread = None
        self._producer_stop_event = None
        self._producer_error = None

    def reset(self):
        self._stop_producer()
        self.batch_size = self._configured_batch_size
        self._layout_cache = {}
        self._prefetch_queue = None
        self._producer_thread = None
        self._producer_stop_event = None
        self._producer_error = None
        super().reset()

    def get_batch(self):
        # Keep symbolic build behavior simple: return one regular batch.
        batch = next(iter(self.data_adapter.get_jax_iterator()))
        return self._distribute_tree(batch)

    def _get_iterator(self):
        # Important: bypass parent per-batch device prefetch.
        # We want raw batches here so we can stack first, then device_put once.
        return self.data_adapter.get_jax_iterator()

    def __next__(self):
        self._maybe_start_producer()
        item = self._prefetch_queue.get()
        if item is self._queue_end:
            if self._producer_error is not None:
                self._clear_producer_state()
                raise self._producer_error
            self._clear_producer_state()
            raise StopIteration
        return item

    def _maybe_start_producer(self):
        if (
                self._producer_thread is not None
                and self._producer_thread.is_alive()
        ):
            return
        if self._producer_thread is not None and not self._producer_thread.is_alive():
            self._clear_producer_state()

        import queue
        import threading

        self._producer_stop_event = threading.Event()
        self._prefetch_queue = queue.Queue(maxsize=self._group_prefetch_n)
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="jax_epoch_stacked_prefetch",
            daemon=True,
        )
        self._producer_thread.start()

    def _put_with_backpressure(self, item):
        import queue

        while not self._producer_stop_event.is_set():
            try:
                self._prefetch_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _wait_for_queue_slot(self):
        import time

        while not self._producer_stop_event.is_set():
            if not self._prefetch_queue.full():
                return True
            time.sleep(0.001)
        return False

    def _producer_loop(self):
        try:
            while not self._producer_stop_event.is_set():
                try:
                    begin_step, end_step, iterator = next(self._epoch_iterator)
                except StopIteration:
                    break

                target_steps = end_step - begin_step + 1
                stacked_host, remainder_host = self._collect_group(
                    iterator, target_steps
                )

                actual_steps = self._count_steps(stacked_host, remainder_host)
                if actual_steps == 0:
                    continue

                # Avoid staging another on-device group when the queue is full.
                # This reduces allocator pressure (memfree/memcpy sync points).
                if not self._wait_for_queue_slot():
                    return

                stacked_batch = self._distribute_tree(stacked_host)
                remainder_batch = self._distribute_tree(remainder_host)

                actual_end_step = begin_step + actual_steps - 1
                item = (
                    begin_step,
                    actual_end_step,
                    (stacked_batch, remainder_batch),
                )
                if not self._put_with_backpressure(item):
                    return
        except Exception as e:
            self._producer_error = e
        finally:
            self._put_with_backpressure(self._queue_end)

    def _stop_producer(self):
        if self._producer_stop_event is not None:
            self._producer_stop_event.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=1.0)
            # Avoid clearing shared state while producer thread may still run.
            if self._producer_thread.is_alive():
                return
        self._clear_producer_state()

    def _clear_producer_state(self):
        self._prefetch_queue = None
        self._producer_thread = None
        self._producer_stop_event = None

    def __del__(self):
        try:
            self._stop_producer()
        except Exception:
            # Best-effort cleanup during object finalization.
            pass

    def _collect_group(self, iterator, target_steps):
        full_batches = []
        remainder_batch = None

        for _ in range(target_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch_dim = self._leading_dim(batch)

            # Infer full batch size from the first observed batch if needed.
            if self.batch_size is None:
                self.batch_size = batch_dim

            # Partial batch -> keep separate and stop grouping.
            if batch_dim < self.batch_size:
                remainder_batch = batch
                break

            full_batches.append(batch)

        stacked_batch = None
        if full_batches:
            stacked_batch = self._stack_batches(full_batches)

        return stacked_batch, remainder_batch

    def _count_steps(self, stacked_batch, remainder_batch):
        n = 0
        if stacked_batch is not None:
            n += self._leading_dim(stacked_batch)
        if remainder_batch is not None:
            n += 1
        return n

    def _leading_dim(self, batch):
        for leaf in tree.flatten(batch):
            if leaf is not None:
                return leaf.shape[0]
        raise ValueError("Batch pytree has no non-None leaves.")

    def _shape_signature(self, batch):
        if batch is None:
            return None
        return tuple(
            None if leaf is None else tuple(leaf.shape)
            for leaf in tree.flatten(batch)
        )

    def _stack_batches(self, batches):
        """Stack a list of single batches on axis 0.

        Prefer NumPy stacking for host arrays. If the adapter already yields JAX
        arrays, fall back to jax.numpy.stack to avoid host round-trips.
        """
        leaves = [leaf for leaf in tree.flatten(batches[0]) if leaf is not None]
        use_jax_stack = any(isinstance(leaf, jax.Array) for leaf in leaves)
        stack_fn = jax.numpy.stack if use_jax_stack else np.stack

        def _stack_or_none(*xs):
            if xs[0] is None:
                return None
            return stack_fn(xs, axis=0)

        return tree.map_structure(_stack_or_none, *batches)

    def _distribute_tree(self, batch):
        if batch is None:
            return None

        distribution = distribution_lib.distribution()
        if distribution is None:
            return tree.map_structure(
                lambda x: None if x is None else jax.device_put(x),
                batch,
            )

        # Cache layouts by shape signature to avoid recomputing them.
        signature = self._shape_signature(batch)
        layouts = self._layout_cache.get(signature, None)
        if layouts is None:
            layouts = tree.map_structure(
                lambda d: None
                if d is None
                else distribution.get_data_layout(d.shape).backend_layout,
                batch,
            )
            self._layout_cache[signature] = layouts

        return _distribute_data(batch, layouts)


class JAXEpochStackedIteratorV2(JAXEpochStackedIterator):
    """Stacked iterator variant with host+device two-stage prefetch.

    Stage 1 (background thread): collect and stack groups on host.
    Stage 2 (main thread): move stacked groups to device into a tiny queue.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._host_prefetch_n = 4
        self._device_prefetch_n = self._group_prefetch_n
        self._stack_slot_pool = {}
        self._host_queue = None
        self._device_queue = None
        self._host_end = object()
        self._device_end = object()
        self._host_producer_thread = None
        self._host_producer_stop_event = None
        self._host_producer_error = None
        self._device_producer_thread = None
        self._device_producer_stop_event = None
        self._device_producer_error = None
        self._host_ended = False
        self._device_ended = False
        self._prewarm_done = False

    def reset(self):
        self._stop_device_producer()
        self._stop_host_producer()
        self._clear_v2_state()
        self._stack_slot_pool = {}
        super().reset()

    def __next__(self):
        self._maybe_start_producers()
        self._prewarm_queues()
        return self._get_device_item()

    def _maybe_start_producers(self):
        self._maybe_start_host_producer()
        self._maybe_start_device_producer()

    def _maybe_start_host_producer(self):
        if (
                self._host_producer_thread is not None
                and self._host_producer_thread.is_alive()
        ):
            return
        if self._host_producer_thread is not None:
            return

        import queue
        import threading

        self._host_ended = False
        self._host_producer_error = None
        self._host_producer_stop_event = threading.Event()
        self._host_queue = queue.Queue(maxsize=self._host_prefetch_n)
        self._host_producer_thread = threading.Thread(
            target=self._host_producer_loop,
            name="jax_epoch_stacked_v2_host_prefetch",
            daemon=True,
        )
        self._host_producer_thread.start()

    def _maybe_start_device_producer(self):
        if (
                self._device_producer_thread is not None
                and self._device_producer_thread.is_alive()
        ):
            return
        if self._device_producer_thread is not None:
            return

        import queue
        import threading

        self._device_ended = False
        self._device_producer_error = None
        self._device_producer_stop_event = threading.Event()
        self._device_queue = queue.Queue(maxsize=self._device_prefetch_n)
        self._device_producer_thread = threading.Thread(
            target=self._device_producer_loop,
            name="jax_epoch_stacked_v2_device_prefetch",
            daemon=True,
        )
        self._device_producer_thread.start()

    def _put_host_item(self, item):
        import queue

        with jax.profiler.TraceAnnotation("stacked_v2.host_queue_put"):
            while not self._host_producer_stop_event.is_set():
                try:
                    self._host_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
        return False

    def _host_producer_loop(self):
        try:
            while not self._host_producer_stop_event.is_set():
                with jax.profiler.TraceAnnotation(
                    "stacked_v2.host_next_epoch_item"
                ):
                    try:
                        begin_step, end_step, iterator = next(self._epoch_iterator)
                    except StopIteration:
                        break

                target_steps = end_step - begin_step + 1
                with jax.profiler.TraceAnnotation(
                    "stacked_v2.host_collect_group"
                ):
                    stacked_host, remainder_host, slot_handle = (
                        self._collect_group_v2(
                        iterator, target_steps
                        )
                    )

                actual_steps = self._count_steps(stacked_host, remainder_host)
                if actual_steps == 0:
                    continue

                actual_end_step = begin_step + actual_steps - 1
                item = (
                    begin_step,
                    actual_end_step,
                    (stacked_host, remainder_host, slot_handle),
                )
                if not self._put_host_item(item):
                    return
        except Exception as e:
            self._host_producer_error = e
        finally:
            self._put_host_item(self._host_end)

    def _get_host_item(self):
        import queue

        with jax.profiler.TraceAnnotation("stacked_v2.host_queue_get"):
            while True:
                if self._device_producer_stop_event.is_set():
                    return None
                try:
                    return self._host_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

    def _put_device_item(self, item):
        import queue

        with jax.profiler.TraceAnnotation("stacked_v2.device_queue_put"):
            while not self._device_producer_stop_event.is_set():
                try:
                    self._device_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
        return False

    def _device_producer_loop(self):
        try:
            while not self._device_producer_stop_event.is_set():
                item = self._get_host_item()
                if item is None:
                    return
                if item is self._host_end:
                    self._host_ended = True
                    if self._host_producer_error is not None:
                        raise self._host_producer_error
                    break

                begin_step, end_step, (stacked_host, remainder_host, slot_handle) = (
                    item
                )
                with jax.profiler.TraceAnnotation("stacked_v2.device_put_pair"):
                    try:
                        stacked_batch, remainder_batch = self._distribute_tree(
                            (stacked_host, remainder_host)
                        )
                    finally:
                        self._release_stack_slot(slot_handle)

                if not self._put_device_item(
                    (begin_step, end_step, (stacked_batch, remainder_batch))
                ):
                    return
        except Exception as e:
            self._device_producer_error = e
        finally:
            self._put_device_item(self._device_end)

    def _prewarm_queues(self):
        if self._prewarm_done:
            return
        import time

        # Fill device queue once at startup to avoid first-step bubbles.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._device_ended:
                break
            if self._device_queue.qsize() >= self._device_prefetch_n:
                break
            if self._device_queue.qsize() > 0 and self._host_queue.empty():
                break
            time.sleep(0.001)
        self._prewarm_done = True

    def _get_device_item(self):
        import queue

        with jax.profiler.TraceAnnotation("stacked_v2.device_queue_get"):
            while True:
                if self._device_producer_stop_event.is_set():
                    raise StopIteration
                try:
                    item = self._device_queue.get(timeout=0.1)
                except queue.Empty:
                    if (
                            self._device_producer_thread is not None
                            and not self._device_producer_thread.is_alive()
                    ):
                        self._device_ended = True
                        self._stop_device_producer()
                        self._stop_host_producer()
                        self._clear_v2_state()
                        if self._device_producer_error is not None:
                            raise self._device_producer_error
                        raise StopIteration
                    continue
                if item is self._device_end:
                    self._device_ended = True
                    self._stop_device_producer()
                    self._stop_host_producer()
                    self._clear_v2_state()
                    if self._device_producer_error is not None:
                        raise self._device_producer_error
                    raise StopIteration
                return item

    def _stop_host_producer(self):
        if self._host_producer_stop_event is not None:
            self._host_producer_stop_event.set()
        if self._host_producer_thread is not None:
            self._host_producer_thread.join(timeout=1.0)
            if self._host_producer_thread.is_alive():
                return
        self._host_producer_thread = None
        self._host_producer_stop_event = None

    def _stop_device_producer(self):
        if self._device_producer_stop_event is not None:
            self._device_producer_stop_event.set()
        if self._device_producer_thread is not None:
            self._device_producer_thread.join(timeout=1.0)
            if self._device_producer_thread.is_alive():
                return
        self._device_producer_thread = None
        self._device_producer_stop_event = None

    def _clear_v2_state(self):
        self._host_queue = None
        self._device_queue = None
        self._host_ended = False
        self._device_ended = False
        self._host_producer_error = None
        self._device_producer_error = None
        self._host_producer_thread = None
        self._host_producer_stop_event = None
        self._device_producer_thread = None
        self._device_producer_stop_event = None
        self._prewarm_done = False

    def __del__(self):
        try:
            self._stop_device_producer()
            self._stop_host_producer()
        except Exception:
            pass

    def _collect_group_v2(self, iterator, target_steps):
        import queue

        full_count = 0
        remainder_batch = None
        full_batches = []
        stacked_slot = None
        slot_handle = None
        slot_spec = None
        slot_buffers = None

        for _ in range(target_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch_dim = self._leading_dim(batch)
            if self.batch_size is None:
                self.batch_size = batch_dim
            if batch_dim < self.batch_size:
                remainder_batch = batch
                break

            if slot_spec is None:
                slot_spec = self._get_or_create_stack_slot_spec(batch)
                if slot_spec is not None:
                    while not self._host_producer_stop_event.is_set():
                        try:
                            slot_index = slot_spec["free_slots"].get(timeout=0.1)
                            break
                        except queue.Empty:
                            continue
                    else:
                        break
                    slot_handle = (slot_spec["signature"], slot_index)
                    slot_buffers = slot_spec["buffers"][slot_index]
                    stacked_slot = [None] * slot_spec["num_flat_leaves"]

            if slot_spec is None:
                full_batches.append(batch)
                full_count += 1
                continue

            flat_batch = tree.flatten(batch)
            for i, leaf_index in enumerate(slot_spec["leaf_indices"]):
                np.copyto(slot_buffers[i][full_count], flat_batch[leaf_index])
            full_count += 1

        stacked_batch = None
        if full_count > 0:
            if slot_spec is None:
                stacked_batch = self._stack_batches(full_batches)
            else:
                for i, leaf_index in enumerate(slot_spec["leaf_indices"]):
                    stacked_slot[leaf_index] = slot_buffers[i][:full_count]
                stacked_batch = tree.pack_sequence_as(
                    slot_spec["template"], stacked_slot
                )

        if full_count == 0 and slot_handle is not None:
            self._release_stack_slot(slot_handle)
            slot_handle = None

        return stacked_batch, remainder_batch, slot_handle

    def _get_or_create_stack_slot_spec(self, batch):
        import queue

        leaves = tree.flatten(batch)
        if any(isinstance(leaf, jax.Array) for leaf in leaves if leaf is not None):
            return None

        signature = self._shape_signature(batch)
        spec = self._stack_slot_pool.get(signature, None)
        if spec is not None:
            return spec

        leaf_indices = [
            i for i, leaf in enumerate(leaves) if leaf is not None
        ]
        slot_count = max(self._host_prefetch_n + 1, 2)
        buffers = []
        for _ in range(slot_count):
            slot = []
            for idx in leaf_indices:
                leaf = leaves[idx]
                slot.append(
                    np.empty(
                        (self.steps_per_execution,) + leaf.shape,
                        dtype=leaf.dtype,
                    )
                )
            buffers.append(slot)

        free_slots = queue.Queue(maxsize=slot_count)
        for i in range(slot_count):
            free_slots.put(i)

        spec = {
            "signature": signature,
            "template": batch,
            "num_flat_leaves": len(leaves),
            "leaf_indices": leaf_indices,
            "buffers": buffers,
            "free_slots": free_slots,
        }
        self._stack_slot_pool[signature] = spec
        return spec

    def _release_stack_slot(self, slot_handle):
        if slot_handle is None:
            return
        signature, slot_index = slot_handle
        spec = self._stack_slot_pool.get(signature, None)
        if spec is None:
            return
        spec["free_slots"].put(slot_index)
