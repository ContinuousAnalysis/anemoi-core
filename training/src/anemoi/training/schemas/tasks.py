# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Annotated
from typing import Literal

from pydantic import Discriminator
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from anemoi.utils.schemas import BaseModel


class RolloutSchema(BaseModel):
    """Rollout configuration for task."""

    start: NonNegativeInt = Field(example=1)
    "Number of rollouts to start with."
    epoch_increment: NonNegativeInt = Field(example=0)
    "Number of epochs to increment the rollout."
    maximum: NonNegativeInt = Field(example=1)
    "Maximum number of rollouts."


class ForecasterSchema(BaseModel):
    """Configuration for forecasting tasks."""

    target_: Literal["anemoi.training.tasks.Forecaster"] = Field(..., alias="_target_")
    "Task class path for the forecasting task."
    multistep_input: PositiveInt = Field(example=2)
    "Number of input timesteps provided to the model."
    multistep_output: PositiveInt = Field(example=1)
    "Number of output timesteps the model should predict."
    timestep: str = Field(example="6H")
    "Timestep string (e.g. '6H') defining the frequency of the input and output steps."
    rollout: RolloutSchema = Field(...)
    "Rollout configuration for autoregressive training."
    validation_rollout: NonNegativeInt = Field(example=[0, 6, 12])
    "Number of rollouts to use for validation."
    rollout_forcing_policy: Literal["last_available", "exact"] = Field(default="last_available")
    "How sparse rollout fills missing coarse timesteps during autoregressive updates."
    dataset_time_offsets: SparseDatasetTimeOffsetsConfigSchema | dict[str, SparseDatasetTimeOffsetsSchema] | None = (
        Field(default=None)
    )
    "Optional per-dataset sparse input and target offsets for mixed-frequency loading."


class SparseDatasetTimeOffsetsSchema(BaseModel):
    """Per-dataset sparse offsets for mixed-frequency loading."""

    input_offsets: list[int | str] = Field(examples=[[0]])
    "Offsets to load as model inputs for this dataset."
    target_offsets: list[int | str] = Field(default_factory=list, examples=[["5m", "10m"]])
    "Offsets to load as model targets for this dataset."


class SparseDatasetTimeOffsetsConfigSchema(BaseModel):
    """Sparse offsets keyed by dataset name."""

    datasets: dict[str, SparseDatasetTimeOffsetsSchema]
    "Per-dataset sparse offsets."


class AutoencoderTaskSchema(BaseModel):
    """Configuration for autoencoding tasks."""

    target_: Literal["anemoi.training.tasks.Autoencoder"] = Field(..., alias="_target_")
    "Task class path for the autoencoding task."


class TemporalDownscalerSchema(BaseModel):
    """Configuration for temporal downscaling task."""

    target_: Literal["anemoi.training.tasks.TemporalDownscaler"] = Field(..., alias="_target_")
    "Task class path for the temporal downscaling task."
    input_timestep: str = Field(example="6H")
    "Input data timestep as a duration string (e.g. '6H')."
    output_timestep: str = Field(example="1H")
    "Desired output timestep as a duration string (e.g. '1H')."
    output_left_boundary: bool = Field(example=False)
    "Whether to include the left boundary in the output."
    output_right_boundary: bool = Field(example=False)
    "Whether to include the right boundary in the output."


TaskSchema = Annotated[
    ForecasterSchema | AutoencoderTaskSchema | TemporalDownscalerSchema,
    Discriminator("target_"),
]
