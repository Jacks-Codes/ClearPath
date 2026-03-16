#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from itertools import groupby
import os
import re
import subprocess
from typing import NamedTuple


VERSION_REGEX = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


class Version(NamedTuple):
    """
    A version number, consisting of major, minor, and patch components.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def from_tag(cls, tag: str) -> Version:
        """
        Creates `Version` instance from version tag.
        Raises `ValueError` if the tag does not match the expected format.

        >>> Version.from_tag("v1.2.3")
        Version(1, 2, 3)
        >>> Version.from_tag("my_tag")
        ValueError: Invalid version format: my_tag
        """

        if not (m := VERSION_REGEX.match(tag)):
            raise ValueError(f"Invalid version format: {tag}")

        major, minor, patch = map(int, m.groups())
        return cls(major, minor, patch)

    def __bool__(self) -> bool:
        return bool(self.major or self.minor or self.patch)

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    def to_major_minor(self) -> tuple[int, int]:
        """
        Returns a tuple of major and minor components.

        >>> Version(1, 2, 3).to_major_minor()
        (1, 2)
        """
        return (self.major, self.minor)


_MAX_VERSION_FROM_ENV = (
    Version.from_tag(os.environ["DOCUMENTATION_MAX_TAG"])
    if "DOCUMENTATION_MAX_TAG" in os.environ
    else None
)


def get_versions_from_git_tags(
    max_version: Version | None = _MAX_VERSION_FROM_ENV,
) -> list[Version]:
    """
    Returns a list of all git tags that match the version format.
    Skips versions before `v1.1.0` and after `max_version` if specified.

    >>> get_versions_from_git_tags()
    [Version(1, 2, 3), Version(1, 2, 4), Version(1, 2, 5)]
    """

    try:
        output = subprocess.check_output(
            ["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"], text=True
        )
    except subprocess.CalledProcessError:
        return []

    tags = map(lambda x: x.strip(), output.splitlines())
    version_tags = filter(VERSION_REGEX.match, tags)
    versions = map(Version.from_tag, version_tags)
    filtered_versions = filter(lambda x: x >= Version(1, 1, 0), versions)

    if max_version:
        filtered_versions = filter(lambda x: x <= max_version, filtered_versions)

    return list(filtered_versions)


def get_latest_version_by_major_minor(
    versions: list[Version],
) -> dict[tuple[int, int], Version]:
    """
    Returns a dictionary of the latest version for each major.minor version.

    >>> versions = [
        Version(0, 1, 0),
        Version(1, 2, 0),
        Version(1, 2, 1),
        Version(1, 2, 2),
        Version(1, 3, 0)
    ]
    >>> get_latest_version_by_major_minor(versions, Version(1, 2, 2))
    {
        (0, 1): Version(0, 1, 0),
        (1, 2): Version(1, 2, 2),
        (1, 3): Version(1, 3, 0),
    }
    """

    return {
        major_minor: max(versions_by_major_minor)
        for major_minor, versions_by_major_minor in groupby(
            versions, lambda x: x.to_major_minor()
        )
    }
