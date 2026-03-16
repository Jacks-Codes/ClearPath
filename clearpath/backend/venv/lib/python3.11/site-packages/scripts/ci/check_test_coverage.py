#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from pathlib import Path
import re
from typing import Generic, TypeVar, TypedDict
import coverage


T = TypeVar("T", int, float)


class CoverageValues(TypedDict, Generic[T]):
    """
    Coverage values for a single pattern.
    """

    statement: T
    branch: T


class MissingCoverageError(Exception):
    """
    Exception raised when pattern coverage is insufficient.
    """

    def __init__(
        self,
        pattern: re.Pattern[str],
        coverage_type: str,
        coverage_value: float,
        required_coverage_value: float,
    ):
        coverage_type = coverage_type.capitalize()

        super().__init__(
            f"{coverage_type} coverage of {pattern.pattern} didn't meet the requirements. "
            f"Calculated coverage: {coverage_value * 100:.2f}%, "
            f"Required coverage: {required_coverage_value * 100:.2f}%"
        )


def get_coverage_count(
    coverage_patterns: list[re.Pattern[str]],
) -> tuple[
    dict[re.Pattern[str], CoverageValues[int]],
    dict[re.Pattern[str], CoverageValues[int]],
]:
    """
    Get the covered lines and branches count for the given coverage patterns.
    """

    cov = coverage.Coverage(Path(__file__).parents[2] / ".coverage")
    cov.load()

    data = cov.get_data()

    covered_count: dict[re.Pattern[str], CoverageValues[int]] = {
        pattern: CoverageValues(statement=0, branch=0) for pattern in coverage_patterns
    }

    total_count: dict[re.Pattern[str], CoverageValues[int]] = {
        pattern: CoverageValues(statement=0, branch=0) for pattern in coverage_patterns
    }

    for file_path in data.measured_files():
        if "python/ibm_watsonx_ai" not in file_path:
            continue

        _, statements, missing, _ = cov.analysis(file_path)
        branch_stats = cov.branch_stats(file_path)

        statement_count = len(statements)
        covered_statement_count = statement_count - len(missing)

        branch_count = sum(x[0] for x in branch_stats.values())
        covered_branch_count = sum(x[1] for x in branch_stats.values())

        for pattern in coverage_patterns:
            if pattern.match(file_path) is None:
                continue

            total_count[pattern]["statement"] += statement_count
            covered_count[pattern]["statement"] += covered_statement_count

            total_count[pattern]["branch"] += branch_count
            covered_count[pattern]["branch"] += covered_branch_count

    return covered_count, total_count


def raise_missing_coverage_errors(
    coverage_values: dict[re.Pattern[str], CoverageValues],
    required_coverage: dict[re.Pattern[str], CoverageValues],
) -> None:
    """
    Raise exception group if the coverage is insufficient.
    """

    coverage_errors: list[MissingCoverageError] = [
        MissingCoverageError(
            pattern,
            coverage_type,
            coverage_values[pattern][coverage_type],
            required_coverage[pattern][coverage_type],
        )
        for pattern in required_coverage
        for coverage_type in ["statement", "branch"]
        if coverage_values[pattern][coverage_type]
        < required_coverage[pattern][coverage_type]
    ]

    if coverage_errors:
        raise ExceptionGroup("Insufficient test coverage detected", coverage_errors)


def print_coverage(
    coverage_values: dict[re.Pattern[str], CoverageValues[float]],
) -> None:
    """
    Print the calculated coverage.
    """

    print("Calculated coverage:")
    for pattern, coverage_value in coverage_values.items():
        print(
            f"{pattern.pattern} statement coverage: {coverage_value['statement'] * 100:.2f}%"
        )
        print(
            f"{pattern.pattern} branch coverage: {coverage_value['branch'] * 100:.2f}%"
        )


def check_test_coverage(
    required_coverage: dict[re.Pattern[str], CoverageValues],
) -> None:
    """
    Check the test coverage for the given patterns. The coverage is calculated as ratio of
    the covered lines or branches to the total lines or branches. Coverage data is obtained
    from the `.coverage` file.

    Raises `ExceptionGroup` of `MissingCoverageError` exceptions if the coverage is insufficient.
    """

    covered_count, total_count = get_coverage_count(list(required_coverage))

    coverage_values: dict[re.Pattern[str], CoverageValues[float]] = {
        pattern: CoverageValues(
            statement=covered_count[pattern]["statement"]
            / total_count[pattern]["statement"],
            branch=covered_count[pattern]["branch"] / total_count[pattern]["branch"],
        )
        for pattern in required_coverage
    }

    print_coverage(coverage_values)

    raise_missing_coverage_errors(coverage_values, required_coverage)


if __name__ == "__main__":
    # Entry order matters - coverage counts towards the first matched pattern
    required_coverage_values: dict[re.Pattern[str], CoverageValues[float]] = {
        re.compile(r".*/python/ibm_watsonx_ai/ibm_watsonx_ai/.*"): CoverageValues(
            statement=0.67, branch=0.45
        ),
        re.compile(r".*/python/ibm_watsonx_ai/tests/.*"): CoverageValues(
            statement=1.0, branch=1.0
        ),
    }

    check_test_coverage(required_coverage_values)
