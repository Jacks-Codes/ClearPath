#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from .constants import (
    DEFAULT_DOCUMENTS_SAMPLING_TYPE,
    DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT,
    DEFAULT_SAMPLE_SIZE_LIMIT,
    DEFAULT_SAMPLING_TYPE,
)

__all__ = [
    "ExperimentIterableDataset",
    "DEFAULT_SAMPLING_TYPE",
    "DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT",
    "DEFAULT_DOCUMENTS_SAMPLING_TYPE",
    "DEFAULT_SAMPLE_SIZE_LIMIT",
]

from ibm_watsonx_ai.data_loaders.datasets.tabular import TabularIterableDataset

ExperimentIterableDataset = TabularIterableDataset
