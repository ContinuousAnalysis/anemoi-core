# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from collections.abc import Callable
from collections.abc import Mapping

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class RolloutConfig:
    """Rollout configuration for autoregressive training."""

    def __init__(self, start: int = 1, epoch_increment: int = 0, maximum: int = 1) -> None:
        """Initialize rollout configuration."""
        self.start = start
        self.epoch_increment = epoch_increment
        self.maximum = maximum
        self.step = self.start

    def should_increase(self, current_epoch: int) -> bool:
        """Check if rollout should be increased at the end of the current epoch."""
        return self.epoch_increment > 0 and current_epoch % self.epoch_increment == 0

    def increase(self) -> None:
        """Increase the rollout window by one step."""
        if self.step < self.maximum:
            self.step += 1
            LOGGER.info("Rollout window length has been increased to %d.", self.step)


class Forecaster(BaseTask):
    """Forecasting task implementation.

    Builds input and output offsets from ``multistep_input``,
    ``multistep_output`` and a ``timestep`` string (e.g. ``"6H"``).

    For rollout training the ``offset`` property extends the output
    offsets up to ``rollout_max`` steps so the datamodule loads enough
    time steps, while ``steps`` only iterates over the current
    ``rollout`` value which grows via ``on_train_epoch_end``.
    """

    name: str = "forecaster"

    def __init__(
        self,
        multistep_input: int,
        multistep_output: int,
        timestep: str,
        rollout: dict | None = None,
        validation_rollout: int = 1,
        rollout_forcing_policy: str = "last_available",
        dataset_time_offsets: object | None = None,
        **kwargs,
    ) -> None:

        self.timestep = frequency_to_timedelta(timestep)
        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output
        self.rollout = RolloutConfig(**(rollout or {}))
        self.validation_rollout = validation_rollout
        if rollout_forcing_policy not in {"last_available", "exact"}:
            msg = (
                f"Unsupported sparse rollout forcing policy '{rollout_forcing_policy}'. "
                "Expected 'last_available' or 'exact'."
            )
            raise ValueError(msg)
        self.rollout_forcing_policy = rollout_forcing_policy
        self.dataset_time_offsets = dataset_time_offsets
        self.dataset_relative_time_indices: dict[str, list[int]] = {}
        self.dataset_time_maps: dict[str, dict[int, int]] = {}
        self._rollout_sampler_warning_keys: set[tuple[str, int, int]] = set()

        if len(kwargs) > 0:
            LOGGER.warning(
                "The following extra parameters were provided to %s but will be ignored: %s",
                self.__class__.__name__,
                kwargs,
            )

        # Input: e.g. multistep_input=2, timestep=6H     ->  [-6H, 0H]
        input_offsets = [-1 * i * self.timestep for i in range(multistep_input)]
        # Outputs: e.g. multistep_output=1, timestep=6H  -> [[6H], [12H], [18H], ...] up to rollout.maximum
        output_offsets = [(i + 1) * self.timestep for i in range(multistep_output)]
        super().__init__(input_offsets=input_offsets, output_offsets=output_offsets)
        self._plot_adapter = ForecasterPlotAdapter(self)

    def steps(self, mode: str = "training") -> tuple[dict[str, int], ...]:
        """Return the current steps configuration based on the rollout step."""
        max_rollout = self.validation_rollout if mode == "validation" else self.rollout.step
        return tuple({"rollout_step": i} for i in range(max_rollout))

    def get_metric_name(self, rollout_step: int = 0, **_kwargs) -> str:
        """Get the metric name for the current step."""
        return f"_rstep{rollout_step}"

    @property
    def _step_shift(self) -> datetime.timedelta:
        """Time shift between consecutive rollout steps."""
        return self.timestep * self.num_output_steps

    def _compute_rollout_offsets(self, rollout_step: int) -> list[datetime.timedelta]:
        """Compute the full list of offsets needed for the current rollout configuration."""
        all_offsets = set(self._input_offsets)
        for step in range(rollout_step):
            shift = self._step_shift * step
            for o in self._output_offsets:
                all_offsets.add(o + shift)
        return sorted(all_offsets)

    def get_offsets(self, mode: str | None = None) -> list[datetime.timedelta]:
        if mode == "training":
            rollout_step = self.rollout.maximum
        elif mode == "validation":
            rollout_step = self.validation_rollout
        else:
            LOGGER.debug(
                "Unknown mode '%s' for %s.get_offsets(), defaulting to training rollout.",
                mode,
                self.__class__.__name__,
            )
            rollout_step = max(self.rollout.maximum, self.validation_rollout)

        return self._compute_rollout_offsets(rollout_step)

    def get_output_offsets(self, rollout_step: int = 0, mode: str = "training", **_kwargs) -> list[datetime.timedelta]:
        """Return output offsets shifted by ``rollout_step``."""
        rollout_step = rollout_step if mode == "training" else self.validation_rollout
        shift = self._step_shift * rollout_step
        return sorted(o + shift for o in self._output_offsets)

    def fill_metadata(self, md_dict: dict) -> None:
        """Persist per-dataset timing metadata for rollout."""
        super().fill_metadata(md_dict)
        metadata_inference = md_dict.get("metadata_inference", {})
        dataset_names = metadata_inference.get("dataset_names", []) if isinstance(metadata_inference, Mapping) else []
        if len(dataset_names) == 0:
            return

        relative_by_dataset = self._resolve_relative_time_metadata(metadata_inference, dataset_names)
        fallback_relative_indices = list(range(max(self.get_batch_output_indices(rollout_step=self.num_steps - 1)) + 1))
        self.dataset_relative_time_indices = {
            dataset_name: relative_by_dataset.get(dataset_name, fallback_relative_indices)
            for dataset_name in dataset_names
        }
        self.dataset_time_maps = {
            dataset_name: {int(relative_time): batch_idx for batch_idx, relative_time in enumerate(relative_times)}
            for dataset_name, relative_times in self.dataset_relative_time_indices.items()
        }

    def _resolve_relative_time_metadata(
        self,
        metadata_inference: Mapping,
        dataset_names: list[str],
    ) -> dict[str, list[int]]:
        """Choose the richest per-dataset time window exposed by the datamodule metadata."""
        relative_by_dataset: dict[str, list[int]] = {}
        keys = (
            "relative_date_indices_validation_by_dataset",
            "relative_date_indices_training_by_dataset",
        )

        for dataset_name in dataset_names:
            dataset_meta = metadata_inference.get(dataset_name, {})
            timesteps_meta = dataset_meta.get("timesteps", {}) if isinstance(dataset_meta, Mapping) else {}

            chosen: list[int] | None = None
            for key in keys:
                raw_relative = timesteps_meta.get(key, None)
                if not isinstance(raw_relative, Mapping):
                    continue
                raw_values = raw_relative.get(dataset_name, None)
                if raw_values is None:
                    continue
                candidate = [int(value) for value in raw_values]
                if chosen is None or max(candidate, default=-1) > max(chosen, default=-1):
                    chosen = candidate

            if chosen is not None:
                relative_by_dataset[dataset_name] = chosen

        return relative_by_dataset

    def _sample_batch_position(self, *, dataset_name: str, relative_time: int) -> int:
        time_map = self.dataset_time_maps.get(dataset_name, {})
        exact_idx = time_map.get(int(relative_time), None)
        if exact_idx is not None:
            return int(exact_idx)

        available_times = sorted(int(value) for value in time_map)
        if not available_times:
            msg = f"Dataset '{dataset_name}' has no available relative times for sparse rollout."
            raise ValueError(msg)

        if self.rollout_forcing_policy == "last_available":
            candidate_times = [value for value in available_times if value <= int(relative_time)]
            if not candidate_times:
                msg = (
                    f"Dataset '{dataset_name}' has no forcing/boundary time at or before relative time "
                    f"{relative_time}. Available times: {available_times}"
                )
                raise ValueError(msg)
            sampled_time = candidate_times[-1]
        else:
            msg = (
                f"Dataset '{dataset_name}' is missing exact relative time {relative_time}. "
                f"Available times: {available_times}"
            )
            raise ValueError(msg)

        warning_key = (dataset_name, int(relative_time), int(sampled_time))
        if warning_key not in self._rollout_sampler_warning_keys:
            LOGGER.info(
                "Sparse rollout dataset=%s requested_time=%s sampled_time=%s policy=%s",
                dataset_name,
                relative_time,
                sampled_time,
                self.rollout_forcing_policy,
            )
            self._rollout_sampler_warning_keys.add(warning_key)

        return int(time_map[sampled_time])

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        if len(self.dataset_time_maps) == 0:
            return super().get_inputs(batch, data_indices)

        requested_relative_times = self.get_batch_input_indices()
        x = {}
        for dataset_name, dataset_batch in batch.items():
            input_positions = [
                self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
                for relative_time in requested_relative_times
            ]
            input_index = torch.tensor(input_positions, device=dataset_batch.device, dtype=torch.long)
            x_time = dataset_batch.index_select(1, input_index)
            x[dataset_name] = x_time[..., data_indices[dataset_name].data.input.full]
        return x

    def get_targets(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        if len(self.dataset_time_maps) == 0:
            return super().get_targets(batch, **kwargs)

        requested_relative_times = self.get_batch_output_indices(rollout_step=kwargs.get("rollout_step", 0))
        y = {}
        for dataset_name, dataset_batch in batch.items():
            target_positions = [
                self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
                for relative_time in requested_relative_times
            ]
            target_index = torch.tensor(target_positions, device=dataset_batch.device, dtype=torch.long)
            y[dataset_name] = dataset_batch.index_select(1, target_index)
        return y

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
        output_mask: object | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Advance a single dataset's input state for the next rollout step.

        Supports model outputs shaped like ``(B, T, E, G, V)``.
        """
        keep_steps = min(self.num_input_steps, self.num_output_steps)

        x = x.roll(-keep_steps, dims=1)

        # Compute batch indices for the output offsets of this rollout step
        output_batch_indices = self.get_batch_output_indices(rollout_step=rollout_step)

        for i in range(keep_steps):
            # Get prognostic variables
            x[:, -(i + 1), ..., data_indices.model.input.prognostic] = y_pred[
                :,
                -(i + 1),
                ...,
                data_indices.model.output.prognostic,
            ]

            batch_time_index = output_batch_indices[-(i + 1)]
            true_state = batch[:, batch_time_index]

            if output_mask is not None and true_state.shape[1] == 1 and x[:, -(i + 1)].shape[1] != 1:
                true_state = true_state.expand(-1, x[:, -(i + 1)].shape[1], -1, -1)

            x[:, -(i + 1)] = output_mask.rollout_boundary(
                x[:, -(i + 1)],
                true_state,
                data_indices,
                grid_shard_slice=grid_shard_slice,
            )

            # get new "constants" needed for time-varying fields
            x[:, -(i + 1), ..., data_indices.model.input.forcing] = batch[
                :,
                batch_time_index,
                ...,
                data_indices.data.input.forcing,
            ]
        return x

    def _build_rollout_input_step(
        self,
        *,
        dataset_name: str,
        dataset_batch: torch.Tensor,
        y_pred_full: dict[str, torch.Tensor],
        relative_time: int,
        rollout_step: int,
        data_indices: dict[str, IndexCollection],
        output_mask: dict[str, object] | None,
        grid_shard_slice: dict[str, slice | None] | None,
    ) -> torch.Tensor:
        batch_position = self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
        x_step = dataset_batch[
            :,
            batch_position,
            ...,
            data_indices[dataset_name].data.input.full,
        ].clone()

        ensemble_size = y_pred_full[dataset_name].shape[2] if dataset_name in y_pred_full else x_step.shape[1]
        if x_step.shape[1] == 1 and ensemble_size != 1:
            x_step = x_step.expand(-1, ensemble_size, -1, -1).clone()

        pred_start = self.num_input_timesteps + rollout_step * self.num_output_timesteps
        pred_end = pred_start + self.num_output_timesteps - 1
        if pred_start <= int(relative_time) <= pred_end and dataset_name in y_pred_full:
            pred_position = int(relative_time - pred_start)
            x_step[..., data_indices[dataset_name].model.input.prognostic] = y_pred_full[dataset_name][
                :,
                pred_position,
                ...,
                data_indices[dataset_name].model.output.prognostic,
            ]

        dataset_output_mask = None if output_mask is None else output_mask[dataset_name]
        if dataset_output_mask is not None:
            true_state = dataset_batch[:, batch_position]
            if true_state.shape[1] == 1 and x_step.shape[1] != 1:
                true_state = true_state.expand(-1, x_step.shape[1], -1, -1)
            x_step = dataset_output_mask.rollout_boundary(
                x_step,
                true_state,
                data_indices[dataset_name],
                grid_shard_slice=None if grid_shard_slice is None else grid_shard_slice[dataset_name],
            )

        forcing = dataset_batch[
            :,
            batch_position,
            ...,
            data_indices[dataset_name].data.input.forcing,
        ]
        if forcing.shape[1] == 1 and x_step.shape[1] != 1:
            forcing = forcing.expand(-1, x_step.shape[1], -1, -1)
        x_step[..., data_indices[dataset_name].model.input.forcing] = forcing
        return x_step

    def advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int = 0,
        data_indices: dict[str, IndexCollection] | None = None,
        output_mask: dict[str, object] | None = None,
        grid_shard_slice: dict[str, slice | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance the input state for the next rollout step."""
        if len(self.dataset_time_maps) == 0:
            for dataset_name in x:
                x[dataset_name] = self._advance_dataset_input(
                    x[dataset_name],
                    y_pred[dataset_name],
                    batch[dataset_name],
                    rollout_step=rollout_step,
                    data_indices=data_indices[dataset_name],
                    output_mask=None if output_mask is None else output_mask[dataset_name],
                    grid_shard_slice=None if grid_shard_slice is None else grid_shard_slice[dataset_name],
                )
            return x

        next_input_relative_times = [
            int(relative_time + (rollout_step + 1) * self.num_output_timesteps)
            for relative_time in self.get_batch_input_indices()
        ]
        next_x: dict[str, torch.Tensor] = {}
        for dataset_name in x:
            next_steps = [
                self._build_rollout_input_step(
                    dataset_name=dataset_name,
                    dataset_batch=batch[dataset_name],
                    y_pred_full=y_pred,
                    relative_time=relative_time,
                    rollout_step=rollout_step,
                    data_indices=data_indices,
                    output_mask=output_mask,
                    grid_shard_slice=grid_shard_slice,
                )
                for relative_time in next_input_relative_times
            ]
            next_x[dataset_name] = torch.stack(next_steps, dim=1)
        return next_x

    def log_extra(self, logger: Callable, logger_enabled: bool) -> None:
        """Log any task-specific information."""
        logger(
            "rollout",
            float(self.rollout.step),
            on_step=True,
            logger=logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )

    def on_train_epoch_end(self, current_epoch: int) -> None:
        if self.rollout.should_increase(current_epoch):
            self.rollout.increase()

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return frequency_to_string(self.timestep)
