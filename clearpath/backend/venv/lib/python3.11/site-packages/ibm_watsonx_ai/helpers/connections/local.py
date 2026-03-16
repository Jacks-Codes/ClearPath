#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


import sys
from pathlib import Path
from typing import Iterator

import pandas as pd


class LocalBatchReader:
    """LocalBatchReader is designed to"""

    def __init__(self, file_path: str | Path, batch_size: int = 1073741824 // 10):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        self.file_path = file_path
        self.batch_size = batch_size  # default 100 MB
        self.row_size = 0

        self._determine_row_size()

    def _determine_row_size(self) -> None:
        data_row = (
            pd.read_excel(self.file_path, nrows=1)
            if self.file_path.suffix == ".xlsx" or self.file_path.suffix == ".xls"
            else pd.read_csv(self.file_path, chunksize=1)
        )
        self.row_size = sys.getsizeof(data_row)

    def _calculate_chunk_size(self) -> int:
        return self.batch_size // self.row_size

    def __iter__(self) -> Iterator[pd.DataFrame]:
        x = self._calculate_chunk_size()
        if self.file_path.suffix == ".xlsx" or self.file_path.suffix == ".xls":
            yield pd.read_excel(self.file_path, nrows=x)
        else:
            yield from pd.read_csv(self.file_path, chunksize=x)
