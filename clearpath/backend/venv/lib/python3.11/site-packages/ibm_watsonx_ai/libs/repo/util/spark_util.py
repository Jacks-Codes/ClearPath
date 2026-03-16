#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.libs.repo.base_constants import *

from .library_imports import LibraryChecker

lib_checker = LibraryChecker()


class SparkUtil(object):
    DEFAULT_LABEL_COL = 'label'

    @staticmethod
    def get_label_col(spark_artifact):
        lib_checker.check_lib(PYSPARK)

        from pyspark.ml import Pipeline, PipelineModel

        if isinstance(spark_artifact, PipelineModel):
            pipeline = Pipeline(stages=spark_artifact.stages)
            return SparkUtil.get_label_col_from__stages(pipeline.getStages())
        elif isinstance(spark_artifact, Pipeline):
            return SparkUtil.get_label_col_from__stages(spark_artifact.getStages())
        else:
            return SparkUtil.DEFAULT_LABEL_COL

    @staticmethod
    def get_label_col_from__stages(stages):
        lib_checker.check_lib(PYSPARK)
        label = SparkUtil._get_label_col_from_python_stages(stages)

        if label == SparkUtil.DEFAULT_LABEL_COL:
            label = SparkUtil._get_label_col_from_java_stages(stages)

        return label

    @staticmethod
    def _get_label_col_from_python_stages(stages):
        try:
            label_col = stages[-1].getLabelCol()
        except Exception:
            label_col = SparkUtil.DEFAULT_LABEL_COL

        reversed_stages = stages[:]
        reversed_stages.reverse()

        for stage in reversed_stages[1:]:
            try:
                if stage.getOutputCol() == label_col:
                    label_col = stage.getInputCol()
            except Exception:
                pass

        return label_col

    @staticmethod
    def _get_label_col_from_java_stages(stages):
        try:
            label_col = stages[-1]._call_java("getLabelCol")
        except Exception:
            label_col = SparkUtil.DEFAULT_LABEL_COL

        reversed_stages = stages[:]
        reversed_stages.reverse()

        for stage in reversed_stages[1:]:
            try:
                if stage._call_java("getOutputCol") == label_col:
                    label_col = stage._call_java("getInputCol")
            except Exception:
                pass

        return label_col
