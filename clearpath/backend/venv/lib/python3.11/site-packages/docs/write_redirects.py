#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from pathlib import Path
import argparse
import sys
from docs.utils import (
    VERSION_REGEX,
    get_latest_version_by_major_minor,
    get_versions_from_git_tags,
)


REDIRECT_HTML = """<!doctype html>
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}">
<p>Redirecting to <a href="{target}">{target}</a>…</p>
"""


def write_redirect_files(source_path: Path, target_path: Path):
    """
    Creates redirect files for each html file in the target directory.
    """

    print(source_path, target_path)
    for html_file in target_path.rglob("*.html"):
        relative_file_path = html_file.relative_to(target_path)

        target = f"{target_path.name}/{relative_file_path.as_posix()}"
        if VERSION_REGEX.match(source_path.name):
            target = f"../{target}"

        source_file_path = source_path / relative_file_path
        source_file_path.parent.mkdir(parents=True, exist_ok=True)
        source_file_path.write_text(
            REDIRECT_HTML.format(target=target), encoding="utf-8"
        )

        print(
            f"Redirect {source_file_path} -> {target_path / relative_file_path.as_posix()}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML redirects to latest version."
    )
    parser.add_argument(
        "--out",
        default="docs/_build/html",
        help='Output directory for sphinx-multiversion (e.g. "docs/_build/html")',
    )
    args = parser.parse_args()

    output_directory = Path(args.out)
    if not output_directory.exists():
        print(f"Output directory does not exist: {output_directory}")
        sys.exit(1)

    versions = get_versions_from_git_tags()
    latest_version_by_major_minor = get_latest_version_by_major_minor(versions)
    latest_version = str(max(latest_version_by_major_minor.values()))

    # Redirect non-versioned URLs to the latest version
    write_redirect_files(
        output_directory,
        output_directory / latest_version,
    )

    for version in versions:
        if version.to_major_minor() not in latest_version_by_major_minor:
            continue

        latest_version_for_major_minor = latest_version_by_major_minor[
            version.to_major_minor()
        ]

        if version == latest_version_for_major_minor:
            continue

        write_redirect_files(
            output_directory / str(version),
            output_directory / str(latest_version_for_major_minor),
        )


if __name__ == "__main__":
    main()
