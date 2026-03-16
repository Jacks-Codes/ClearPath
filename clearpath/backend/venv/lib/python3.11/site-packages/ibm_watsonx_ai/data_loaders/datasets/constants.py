#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.utils.autoai.enums import (
    DocumentsSamplingTypes,
    SamplingTypes,
)

DEFAULT_SAMPLE_SIZE_LIMIT = (
    1073741824  # 1GB in Bytes is verified later by _set_sample_size_limit
)
DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT = 104857600  # 100MB in bytes
DEFAULT_SAMPLING_TYPE = SamplingTypes.FIRST_VALUES
DEFAULT_DOCUMENTS_SAMPLING_TYPE = DocumentsSamplingTypes.RANDOM
