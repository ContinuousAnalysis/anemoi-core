# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf
from pytest_mock import MockFixture

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_timedelta


class FakeDatasetReader:
    def __init__(self, *, dataset_name: str, frequency: str, start: str, stop: str) -> None:
        self.data = dataset_name
        self.frequency = frequency
        self.dates = np.arange(
            np.datetime64(start),
            np.datetime64(stop),
            np.timedelta64(int(frequency_to_timedelta(frequency).total_seconds() // 60), "m"),
        )
        self.missing = set()
        self.has_trajectories = False
        self.statistics = {}
        self.metadata = {}
        self.supporting_arrays = {}
        self.variables = ["forcing_var", "prog_var"]
        self.name_to_index = {"forcing_var": 0, "prog_var": 1}
        self.resolution = "test"

    def get_sample(self, *args: Any, **kwargs: Any) -> None:
        msg = "FakeDatasetReader is only used for datamodule timing tests."
        raise NotImplementedError(msg)


def get_reader_dataset_config(dataset_cfg: Any) -> dict[str, Any]:
    if hasattr(dataset_cfg, "dataset_config"):
        return dict(dataset_cfg.dataset_config)
    if isinstance(dataset_cfg, dict):
        return dict(dataset_cfg["dataset_config"])
    msg = f"Unsupported dataset config type: {type(dataset_cfg)!r}"
    raise TypeError(msg)


def make_multidataset_cfg(*, meps_frequency: str, radar_frequency: str, time_index_mode: str) -> Any:
    return OmegaConf.create(
        {
            "data": {
                "timestep": "5m",
                "datasets": {
                    "meps": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                    "nordic_radar": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.Forecaster",
                "multistep_input": 1,
                "multistep_output": 1,
                "timestep": "5m",
                "rollout": {"start": 1, "epoch_increment": 0, "maximum": 1},
                "validation_rollout": 1,
            },
            "dataloader": {
                "pin_memory": False,
                "debug": {"time_index_mode": time_index_mode},
                "training": {
                    "datasets": {
                        "meps": {
                            "dataset_config": {"dataset": "meps_source", "frequency": meps_frequency},
                            "end": "2020-01-02",
                        },
                        "nordic_radar": {
                            "dataset_config": {"dataset": "radar_source", "frequency": radar_frequency},
                            "end": "2020-01-02",
                        },
                    },
                },
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
            },
            "training": {},
        },
    )


def test_datamodule_relative_date_indices_follow_task_config_for_sparse_forecaster() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "timestep": "5m",
                "datasets": {
                    "meps": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                    "nordic_radar": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.Forecaster",
                "multistep_input": 1,
                "multistep_output": 1,
                "timestep": "5m",
                "rollout": {"start": 1, "epoch_increment": 0, "maximum": 3},
                "validation_rollout": 1,
            },
            "dataloader": {
                "pin_memory": False,
                "training": {
                    "datasets": {
                        "meps": {"dataset_config": {"dataset": "meps_source", "frequency": "1h"}, "end": "2020-01-02"},
                        "nordic_radar": {
                            "dataset_config": {"dataset": "radar_source", "frequency": "5m"},
                            "end": "2020-01-02",
                        },
                    },
                },
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
            },
            "training": {},
        },
    )

    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 3},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    assert datamodule.relative_date_indices() == [0, 1, 2, 3]


def test_datamodule_relative_date_indices_include_sparse_dataset_time_offsets() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "timestep": "5m",
                "datasets": {
                    "meps": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                    "nordic_radar": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.Forecaster",
                "multistep_input": 1,
                "multistep_output": 1,
                "timestep": "5m",
                "rollout": {"start": 1, "epoch_increment": 0, "maximum": 1},
                "validation_rollout": 1,
                "dataset_time_offsets": {
                    "datasets": {
                        "meps": {"input_offsets": [0], "target_offsets": []},
                        "nordic_radar": {"input_offsets": [0], "target_offsets": ["15m"]},
                    },
                },
            },
            "dataloader": {
                "pin_memory": False,
                "training": {
                    "datasets": {
                        "meps": {"dataset_config": {"dataset": "meps_source", "frequency": "1h"}, "end": "2020-01-02"},
                        "nordic_radar": {
                            "dataset_config": {"dataset": "radar_source", "frequency": "5m"},
                            "end": "2020-01-02",
                        },
                    },
                },
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
            },
            "training": {},
        },
    )

    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
        dataset_time_offsets=cfg.task.dataset_time_offsets,
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    assert datamodule.relative_date_indices() == [0, 3]


def test_datamodule_timestep_falls_back_to_task_when_data_timestep_is_missing() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "frequency": "6h",
                "datasets": {
                    "data": {"forcing": [], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.Forecaster",
                "multistep_input": 2,
                "multistep_output": 1,
                "timestep": "6h",
                "rollout": {"start": 1, "epoch_increment": 0, "maximum": 1},
                "validation_rollout": 1,
            },
            "dataloader": {
                "pin_memory": False,
                "training": {
                    "datasets": {
                        "data": {"dataset_config": {"dataset": "source", "frequency": "6h"}, "end": "2020-01-02"},
                    },
                },
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
            },
            "training": {},
        },
    )

    task = Forecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    assert datamodule.config_timestep == "6h"
    assert datamodule._lead_time_for_step(1) == "6h"


def test_datamodule_dense_mode_keeps_aligned_frequencies_without_reopening(
    mocker: MockFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="5m", radar_frequency="5m", time_index_mode="dense")
    created_frequencies: list[str] = []

    def _create_dataset(dataset_cfg: Any, **_kwargs: Any) -> FakeDatasetReader:
        dataset_config = get_reader_dataset_config(dataset_cfg)
        frequency = dataset_config.get("interpolate_frequency", dataset_config.get("frequency"))
        created_frequencies.append(frequency)
        return FakeDatasetReader(
            dataset_name=dataset_config["dataset"],
            frequency=frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        )

    mocker.patch("anemoi.training.data.datamodule.create_dataset", side_effect=_create_dataset)
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    with caplog.at_level(logging.WARNING, logger="anemoi.training.data.datamodule"):
        _ = datamodule.ds_train

    assert created_frequencies == ["5m", "5m"]
    assert "interpolate_frequency" not in caplog.text


def test_datamodule_dense_mode_interpolates_coarser_datasets(
    mocker: MockFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="1h", radar_frequency="5m", time_index_mode="dense")
    created_configs: list[dict[str, Any]] = []

    def _create_dataset(dataset_cfg: Any, **_kwargs: Any) -> FakeDatasetReader:
        dataset_config = get_reader_dataset_config(dataset_cfg)
        created_configs.append(dataset_config)
        frequency = dataset_config.get("interpolate_frequency", dataset_config.get("frequency"))
        return FakeDatasetReader(
            dataset_name=dataset_config["dataset"],
            frequency=frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        )

    mocker.patch("anemoi.training.data.datamodule.create_dataset", side_effect=_create_dataset)
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    with caplog.at_level(logging.WARNING, logger="anemoi.training.data.datamodule"):
        ds_train = datamodule.ds_train

    assert created_configs == [
        {"dataset": "meps_source", "frequency": "1h"},
        {"dataset": "radar_source", "frequency": "5m"},
        {"dataset": "meps_source", "interpolate_frequency": "5m"},
    ]
    assert frequency_to_seconds(ds_train.frequencies["meps"]) == frequency_to_seconds("5m")
    assert frequency_to_seconds(ds_train.frequencies["nordic_radar"]) == frequency_to_seconds("5m")
    assert "interpolate_frequency=5m" in caplog.text
    assert "meps" in caplog.text
