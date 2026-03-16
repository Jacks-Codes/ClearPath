#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import importlib.metadata
import importlib.util

from ibm_watsonx_ai.libs.repo import base_constants as const
from ibm_watsonx_ai.libs.repo.util.base_singleton import BaseSingleton


class LibraryChecker(BaseSingleton):
    def __init__(self):
        self.supported_libs = [
            const.PYSPARK,
            const.SCIKIT,
            const.PANDAS,
            const.XGBOOST,
            const.MLPIPELINE,
            const.IBMSPARKPIPELINE,
            const.TENSORFLOW,
        ]
        self.installed_libs = {
            const.PYSPARK: False,
            const.SCIKIT: False,
            const.PANDAS: False,
            const.XGBOOST: False,
            const.MLPIPELINE: False,
            const.TENSORFLOW: False,
            const.IBMSPARKPIPELINE: False,
        }

        atleast_one_lib_installed = self._check_if_lib_installed(self.supported_libs)

        if not atleast_one_lib_installed:
            supported_lib_str = self.supported_libs[0]
            lib_num = len(self.supported_libs)
            for i in range(1, lib_num - 1):
                supported_lib_str += ", " + self.supported_libs[i]
            supported_lib_str += " and " + self.supported_libs[lib_num - 1]
            raise ImportError(
                "The system lacks installations of "
                + supported_lib_str
                + ". At least one of the libraries is required for the repository-client to be used"
            )

    def _check_if_lib_installed(self, lib_names):
        import sys

        atleast_one_lib_installed = False
        for name in lib_names:
            is_found = importlib.util.find_spec(name) is not None
            self.installed_libs[name] = is_found
            if is_found:
                atleast_one_lib_installed = True

        if self.installed_libs[const.SCIKIT]:
            if importlib.metadata.version(
                "scikit-learn"
            ) == "0.23.0" and sys.version_info <= (3, 7):
                raise Exception(
                    " Scikit learn version 0.23.0 is not supported, Please downgrade scikit version to a lower version and re-try. "
                )

        return atleast_one_lib_installed

    def check_lib(self, lib_name):
        lib_display_names = {
            const.PYSPARK: const.DISPLAY_PYSPARK,
            const.SCIKIT: const.DISPLAY_SCIKIT,
            const.PANDAS: const.DISPLAY_PANDAS,
            const.XGBOOST: const.DISPLAY_XGBOOST,
            const.MLPIPELINE: const.DISPLAY_MLPIPELINE,
            const.TENSORFLOW: const.DISPLAY_TENSORFLOW,
            const.IBMSPARKPIPELINE: const.DISPLAY_IBMSPARKPIPELINE,
        }
        if not self.installed_libs[lib_name]:
            raise NameError(
                "{} Library is not installed. Please install it and execute the command".format(
                    lib_display_names[lib_name]
                )
            )
