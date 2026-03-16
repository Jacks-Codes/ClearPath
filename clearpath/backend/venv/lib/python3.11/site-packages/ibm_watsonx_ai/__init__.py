#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import APIClient  # noqa: F401
    from .credentials import Credentials  # noqa: F401
    from .utils.enums import AssetDuplicateAction  # noqa: F401

package_name = __name__.replace("_", "-")
__version__ = version(package_name)

__all__ = ["APIClient", "AssetDuplicateAction", "Credentials", "package_name"]


def __getattr__(name: str) -> Any:
    if name == "APIClient":
        from .client import APIClient  # noqa: F401

        APIClient.version = __version__

        return APIClient

    if name == "Credentials":
        from .credentials import Credentials  # noqa: F401

        return Credentials

    if name == "AssetDuplicateAction":
        from .utils.enums import AssetDuplicateAction  # noqa: F401

        return AssetDuplicateAction

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
