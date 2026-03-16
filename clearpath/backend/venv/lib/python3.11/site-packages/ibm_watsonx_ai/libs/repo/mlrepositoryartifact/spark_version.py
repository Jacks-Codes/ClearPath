#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.libs.repo.base_constants import *
from ibm_watsonx_ai.libs.repo.util.library_imports import LibraryChecker

lib_checker = LibraryChecker()

class SparkVersion(object):
    @staticmethod
    def significant():
        lib_checker.check_lib(PYSPARK)
        from pyspark import SparkConf, SparkContext

        conf = SparkConf()
        sc = SparkContext.getOrCreate(conf=conf)
        version_parts = sc.version.split('.')
        spark_version = version_parts[0]+'.' + version_parts[1]
        return format(spark_version)
