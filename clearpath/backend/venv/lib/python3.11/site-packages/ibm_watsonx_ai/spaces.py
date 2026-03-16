#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    List,
    Literal,
    cast,
    overload,
)

from cachetools import TTLCache, cached

from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import (
    MemberMetaNames,
    SpacesMetaNames,
)
from ibm_watsonx_ai.service_instance import ServiceInstance
from ibm_watsonx_ai.utils import (
    StatusLogger,
    print_text_header_h1,
    print_text_header_h2,
)
from ibm_watsonx_ai.utils.deployment.errors import PromotionFailed
from ibm_watsonx_ai.wml_client_error import (
    MultipleResourceIdByNameFound,
    ResourceIdByNameNotFound,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from httpx import Response
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient


class Spaces(WMLResource):
    """Store and manage spaces."""

    ConfigurationMetaNames = SpacesMetaNames()
    """MetaNames for spaces creation."""

    MemberMetaNames = MemberMetaNames()
    """MetaNames for space members creation."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

    def _get_resources(
        self, url: str, op_name: str, params: dict | None = None
    ) -> dict:
        if params is not None and "limit" in params.keys():
            if params["limit"] < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif params["limit"] > 1000:
                raise WMLClientError("Limit cannot be larger than 1000.")

        if params is not None and len(params) > 0:
            response_get = self._client.httpx_client.get(
                url=url, headers=self._client._get_headers(), params=params
            )

            return self._handle_response(200, op_name, response_get)
        else:
            resources = []

            while True:
                response_get = self._client.httpx_client.get(
                    url=url, headers=self._client._get_headers()
                )

                result = self._handle_response(200, op_name, response_get)
                resources.extend(result["resources"])

                if "next" not in result:
                    break
                else:
                    url = self._credentials.url + result["next"]["href"]
                    if "start=invalid" in url:
                        break
            return {"resources": resources}

    def _connection_validation(self) -> None:
        """Trial connection to validate authorization.

        :raises WMLClientError: raised when connection is unauthorized
        """
        href = self._client._href_definitions.get_platform_spaces_href()
        response_get = self._client.httpx_client.get(
            url=href, headers=self._client._get_headers(), params={"limit": 1}
        )
        if response_get.status_code == 401:
            raise WMLClientError(
                Messages.get_message(
                    response_get.json()["errors"][0]["message"],
                    message_id="invalid_authorization",
                )
            )

    @staticmethod
    def _get_state(space_details: dict) -> str:
        """Get the status state from the space details.

        :param space_details: metadata of the stored space
        :type space_details: dict

        :return: status state of the stored space
        :rtype: str
        """
        Spaces._validate_type(space_details, "space_details", dict, True)

        return Spaces._get_required_element_from_dict(
            space_details, "space_details", ["entity", "status", "state"], str
        )

    def _validate_store_meta_props(self, meta_props: dict) -> None:
        Spaces._validate_type(meta_props, "meta_props", dict, True)

        if (
            any(prop in meta_props for prop in {"compute", "storage"})
            and self._client.ICP_PLATFORM_SPACES
        ):
            raise WMLClientError(
                "'STORAGE' and 'COMPUTE' meta props are not applicable on "
                "IBM Cloud Pak® for Data. If using any of these, remove and retry"
            )

        if "storage" not in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("'STORAGE' is mandatory for cloud")

        if "compute" in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            if any(prop not in meta_props["compute"] for prop in {"name", "crn"}):
                raise WMLClientError("'name' and 'crn' is mandatory for 'COMPUTE'")

            if "type" not in meta_props["compute"]:
                meta_props["compute"]["type"] = "machine_learning"

        if "stage" in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            if not isinstance(meta_props["stage"]["production"], bool):
                raise WMLClientError("'production' for 'STAGE' must be boolean")

    def _prepare_store_payload(self, meta_props: dict) -> dict[str, Any]:
        space_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        if self._client.CLOUD_PLATFORM_SPACES and "compute" in meta_props:
            space_meta["compute"] = [space_meta["compute"]]

        return space_meta

    def _init_service_instance_if_compute(self, space_details: dict) -> None:
        """Initialize `ServiceInstance` if "compute" is present in `space_details`."""

        # Cloud Convergence: Set self._client.credentials.instance_id to instance_id
        # during client.set.default_space since that's where space is associated with client
        # and also in client.set.default_project
        if (
            self._client.CLOUD_PLATFORM_SPACES
            and "compute" in space_details["entity"].keys()
        ):
            instance_id = space_details["entity"]["compute"][0]["guid"]
            self._client.service_instance = ServiceInstance(self._client)
            self._client.service_instance._instance_id = instance_id

    @staticmethod
    def _print_background_mode_info(_async: bool = False) -> None:
        """Print message for storing in background mode."""
        msg = (
            "Space has been created. However some background setup activities might still be on-going. "
            "Check for 'status' field in the response. It has to show 'active' before space can be used. "
            "If it's not 'active', you can monitor the state with a call to client.spaces.{0}(space_id). "
            "Alternatively, use background_mode=False when calling client.spaces.{1}()."
        ).format(
            "aget_details" if _async else "get_details", "astore" if _async else "store"
        )
        print(msg)

    @staticmethod
    def _print_store_result(state: str, space_id: str, space_details: dict) -> None:
        """Print storing result and return `space_details` if the process finished successfully.
        Otherwise, raise `WMLClientError`.
        """
        if "active" in state:
            print_text_header_h2(
                "\nCreating space  '{}' finished successfully.".format(space_id)
            )
        else:
            raise WMLClientError(
                f"Space {space_id} creation failed with status: {space_details['entity']['status']}"
            )

    def store(self, meta_props: dict, background_mode: bool = True) -> dict:
        """Create a space. The instance associated with the space via COMPUTE will be used for billing purposes on
        the cloud. Note that STORAGE and COMPUTE are applicable only for cloud.

        :param meta_props: metadata of the space configuration. To see available meta names, use:

            .. code-block:: python

                client.spaces.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param background_mode: indicator if store() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: metadata of the stored space
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: "my_space",
                client.spaces.ConfigurationMetaNames.DESCRIPTION: "spaces",
                client.spaces.ConfigurationMetaNames.STORAGE: {
                    "resource_crn": "provide crn of the COS storage"
                },
                client.spaces.ConfigurationMetaNames.COMPUTE: {
                    "name": "test_instance",
                    "crn": "provide crn of the instance",
                },
                client.spaces.ConfigurationMetaNames.STAGE: {
                    "production": True,
                    "name": "stage_name",
                },
                client.spaces.ConfigurationMetaNames.TAGS: [
                    "sample_tag_1",
                    "sample_tag_2",
                ],
                client.spaces.ConfigurationMetaNames.TYPE: "cpd",
            }
            spaces_details = client.spaces.store(meta_props=metadata)
        """
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        self._validate_store_meta_props(meta_props)
        payload = self._prepare_store_payload(meta_props)

        creation_response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_platform_spaces_href(),
            headers=self._client._get_headers(),
            json=payload,
        )

        spaces_details = self._handle_response(
            202, "creating new spaces", creation_response, _silent_response_logging=True
        )

        self._init_service_instance_if_compute(spaces_details)

        if background_mode:
            self._print_background_mode_info()
            return spaces_details

        # note: monitor space status (state)
        space_id = self.get_id(spaces_details)
        print_text_header_h1(
            "Synchronous space creation with id: '{}' started".format(space_id)
        )

        state = self._get_state(spaces_details)

        with StatusLogger(state) as status_logger:
            while state not in {
                "failed",
                "error",
                "completed",
                "canceled",
                "active",
            }:
                time.sleep(10)
                spaces_details = self.get_details(space_id)
                state = self._get_state(spaces_details)
                status_logger.log_state(state)
        # --- end note

        self._print_store_result(state, space_id, spaces_details)

        return spaces_details

    async def astore(self, meta_props: dict, background_mode: bool = True) -> dict:
        """Create a space asynchronously. The instance associated with the space via COMPUTE will be used for billing purposes on
        the cloud. Note that STORAGE and COMPUTE are applicable only for cloud.

        :param meta_props: metadata of the space configuration. To see available meta names, use:

            .. code-block:: python

                client.spaces.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param background_mode: indicator if astore() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: metadata of the stored space
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: "my_space",
                client.spaces.ConfigurationMetaNames.DESCRIPTION: "spaces",
                client.spaces.ConfigurationMetaNames.STORAGE: {
                    "resource_crn": "provide crn of the COS storage"
                },
                client.spaces.ConfigurationMetaNames.COMPUTE: {
                    "name": "test_instance",
                    "crn": "provide crn of the instance",
                },
                client.spaces.ConfigurationMetaNames.STAGE: {
                    "production": True,
                    "name": "stage_name",
                },
                client.spaces.ConfigurationMetaNames.TAGS: [
                    "sample_tag_1",
                    "sample_tag_2",
                ],
                client.spaces.ConfigurationMetaNames.TYPE: "cpd",
            }
            spaces_details = await client.spaces.astore(meta_props=metadata)
        """
        self._validate_store_meta_props(meta_props)
        payload = self._prepare_store_payload(meta_props)

        creation_response = await self._client.async_httpx_client.post(
            url=self._client._href_definitions.get_platform_spaces_href(),
            headers=await self._client._aget_headers(),
            json=payload,
        )

        spaces_details = self._handle_response(
            202, "creating new spaces", creation_response, _silent_response_logging=True
        )

        self._init_service_instance_if_compute(spaces_details)

        if background_mode:
            self._print_background_mode_info(_async=True)
            return spaces_details

        # note: monitor space status (state)
        space_id = self.get_id(spaces_details)
        print_text_header_h1(
            "Synchronous space creation with id: '{}' started".format(space_id)
        )

        state = self._get_state(spaces_details)

        with StatusLogger(state) as status_logger:
            while state not in {
                "failed",
                "error",
                "completed",
                "canceled",
                "active",
            }:
                await asyncio.sleep(10)
                spaces_details = await self.aget_details(space_id)
                state = self._get_state(spaces_details)
                status_logger.log_state(state)
        # --- end note

        self._print_store_result(state, space_id, spaces_details)

        return spaces_details

    @staticmethod
    def get_id(space_details: dict) -> str:
        """Get the space_id from the space details.

        :param space_details: metadata of the stored space
        :type space_details: dict

        :return: ID of the stored space
        :rtype: str

        **Example:**

        .. code-block:: python

            space_details = client.spaces.store(meta_props)
            space_id = client.spaces.get_id(space_details)
        """
        Spaces._validate_type(space_details, "space_details", dict, True)

        return Spaces._get_required_element_from_dict(
            space_details, "space_details", ["metadata", "id"], str
        )

    def get_id_by_name(self, space_name: str) -> str:
        """Get the ID of a stored space by name.

        :param space_name: name of the stored space
        :type space_name: str

        :return: ID of the stored space
        :rtype: str

        **Example:**

        .. code-block:: python

            space_id = client.spaces.get_id_by_name(space_name)

        """
        Spaces._validate_type(space_name, "space_name", str, True)

        details = self.get_details(space_name=space_name)

        if len(details["resources"]) > 1:
            raise MultipleResourceIdByNameFound(space_name, "space")
        elif len(details["resources"]) == 0:
            raise ResourceIdByNameNotFound(space_name, "space")

        return details["resources"][0]["metadata"]["id"]

    @staticmethod
    def get_uid(space_details: dict) -> str:
        """Get the unique ID of the space.

         *Deprecated:* Use ``get_id(space_details)`` instead.

         :param space_details: metadata of the space
         :type space_details: dict

         :return: unique ID of the space
         :rtype: str

        **Example:**

        .. code-block:: python

            space_details = client.spaces.store(meta_props)
            space_uid = client.spaces.get_uid(space_details)

        """
        return Spaces.get_id(space_details)

    def delete(self, space_id: str) -> Literal["SUCCESS"]:
        """Delete a stored space.

        :param space_id: ID of the space
        :type space_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        :raises WMLClientError: if deletion failed

        **Example:**

        .. code-block:: python

            client.spaces.delete(space_id)
        """
        Spaces._validate_type(space_id, "space_id", str, True)

        space_endpoint = self._client._href_definitions.get_platform_space_href(
            space_id
        )

        response_delete = self._client.httpx_client.delete(
            url=space_endpoint, headers=self._client._get_headers()
        )

        return cast(
            Literal["SUCCESS"],
            self._handle_response(202, "space deletion", response_delete, False),
        )

    async def adelete(self, space_id: str) -> Literal["SUCCESS"]:
        """Delete a stored space asynchronously.

        :param space_id: ID of the space
        :type space_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        :raises WMLClientError: if deletion failed

        **Example:**

        .. code-block:: python

            await client.spaces.adelete(space_id)
        """
        Spaces._validate_type(space_id, "space_id", str, True)

        space_endpoint = self._client._href_definitions.get_platform_space_href(
            space_id
        )

        response_delete = await self._client.async_httpx_client.delete(
            url=space_endpoint, headers=await self._client._aget_headers()
        )

        return cast(
            Literal["SUCCESS"],
            self._handle_response(202, "space deletion", response_delete, False),
        )

    @staticmethod
    def _get_details_prepare_query_params(
        space_id: str | None, space_name: str | None, include: str | None
    ) -> dict:
        """Helper method for `(a)get_details` methods.
        Validate `space_id` type. Return query parameters for a request to get space details.
        """
        Spaces._validate_type(space_id, "space_id", str, False)

        query_params = {}
        if include:
            query_params["include"] = include

        if space_name and space_id is None:
            query_params["name"] = space_name

        return query_params

    @overload
    def get_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        *,
        asynchronous: Literal[True],
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> Generator: ...

    @overload
    def get_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        asynchronous: Literal[False] = False,
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    @overload
    def get_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Generator: ...

    def get_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Generator:
        """Get metadata of stored space(s). The method uses TTL cache.

        :param space_id: ID of the space
        :type space_id: str, optional

        :param limit: applicable when `space_id` is not provided, otherwise `limit` will be ignored
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param space_name: name of the stored space, can be used only when `space_id` is None
        :type space_name: str, optional

        :return: metadata of stored space(s)
        :rtype:
            - **dict** - if space_id is not None
            - **{"resources": [dict]}** - if space_id is None

        **Example:**

        .. code-block:: python

            space_details = client.spaces.get_details(space_id)
            space_details = client.spaces.get_details(space_name)
            space_details = client.spaces.get_details(limit=100)
            space_details = client.spaces.get_details(limit=100, get_all=True)
            space_details = []
            for entry in client.spaces.get_details(
                limit=100, asynchronous=True, get_all=True
            ):
                space_details.extend(entry)

        """
        query_params = self._get_details_prepare_query_params(
            space_id, space_name, include=kwargs.get("include")
        )

        if space_id is not None:
            response_get = self._client.httpx_client.get(
                url=self._client._href_definitions.get_platform_space_href(space_id),
                headers=self._client._get_headers(),
                params=query_params,
            )

            return self._handle_response(
                200, "Get space", response_get, _silent_response_logging=True
            )

        return self._get_with_or_without_limit(
            self._client._href_definitions.get_platform_spaces_href(),
            limit,
            "spaces",
            summary=False,
            pre_defined=False,
            skip_space_project_chk=True,
            query_params=query_params,
            _async=asynchronous,
            _all=get_all,
            _silent_response_logging=True,
        )

    @overload
    async def aget_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        *,
        asynchronous: Literal[True],
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator: ...

    @overload
    async def aget_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        asynchronous: Literal[False] = False,
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    @overload
    async def aget_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator: ...

    async def aget_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator:
        """Get metadata of stored space(s) asynchronously. The method uses TTL cache.

        :param space_id: ID of the space
        :type space_id: str, optional

        :param limit: applicable when `space_id` is not provided, otherwise `limit` will be ignored
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param space_name: name of the stored space, can be used only when `space_id` is None
        :type space_name: str, optional

        :return: metadata of stored space(s)
        :rtype:
            - **dict** - if space_id is not None
            - **{"resources": [dict]}** - if space_id is None

        **Example:**

        .. code-block:: python

            space_details = await client.spaces.aget_details(space_id)
            space_details = await client.spaces.aget_details(space_name)
            space_details = await client.spaces.aget_details(limit=100)
            space_details = await client.spaces.aget_details(
                limit=100, get_all=True
            )
            space_details = []
            for entry in await client.spaces.aget_details(
                limit=100, asynchronous=True, get_all=True
            ):
                space_details.extend(entry)

        """
        query_params = self._get_details_prepare_query_params(
            space_id, space_name, include=kwargs.get("include")
        )

        if space_id is not None:
            response_get = await self._client.async_httpx_client.get(
                url=self._client._href_definitions.get_platform_space_href(space_id),
                headers=await self._client._aget_headers(),
                params=query_params,
            )

            return self._handle_response(
                200, "Get space", response_get, _silent_response_logging=True
            )

        return await self._aget_with_or_without_limit(
            self._client._href_definitions.get_platform_spaces_href(),
            limit,
            "spaces",
            summary=False,
            pre_defined=False,
            skip_space_project_chk=True,
            query_params=query_params,
            _async=asynchronous,
            _all=get_all,
            _silent_response_logging=True,
        )

    @cached(cache=TTLCache(maxsize=32, ttl=4.5 * 60))
    def _get_details(
        self,
        space_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        space_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Generator:
        """Get metadata of stored space(s) with caching. It's dedicated for internal usage."""
        return self.get_details(
            space_id=space_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            space_name=space_name,
            **kwargs,
        )

    def list(
        self,
        limit: int | None = None,
        member: str | None = None,
        roles: str | None = None,
        space_type: str | None = None,
    ) -> DataFrame:
        """List stored spaces in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param member: filters the result list, only includes spaces where the user with a matching user ID
            is a member
        :type member: str, optional

        :param roles: a list of comma-separated space roles to use to filter the query results,
            must be used in conjunction with the "member" query parameter,
            available values : `admin`, `editor`, `viewer`
        :type roles: str, optional

        :param space_type: filter spaces by their type, available types are 'wx', 'cpd', and 'wca'
        :type space_type: str, optional

        :return: pandas.DataFrame with listed spaces
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.spaces.list()
        """
        Spaces._validate_type(limit, "limit", int, False)
        href = self._client._href_definitions.get_platform_spaces_href()

        params: dict[str, Any] = {}

        if limit is not None:
            params.update({"limit": limit})

        if member is not None:
            params.update({"member": member})

        if roles is not None:
            params.update({"roles": roles})

        if space_type is not None:
            params.update({"type": space_type})

        space_resources = [
            m
            for r in self._get_with_or_without_limit(
                href,
                None,
                "spaces",
                summary=False,
                pre_defined=False,
                skip_space_project_chk=True,
                query_params=params,
                _async=True,
                _all=True,
                _silent_response_logging=True,
            )
            for m in r["resources"]
        ]

        if limit is not None:
            space_resources = space_resources[:limit]

        space_values = [
            (m["metadata"]["id"], m["entity"]["name"], m["metadata"]["created_at"])
            for m in space_resources
        ]

        table = self._list(space_values, ["ID", "NAME", "CREATED"], limit)
        return table

    def _update_validate_args(self, space_id: str, changes: dict) -> None:
        """Helper method for `(a)update` methods.
        Validate `space_id` and `changes` passed to `(a)update` method.
        Raise `WMLClientError` if forbidden keys are found in `changes`.
        """
        if (
            any(prop in changes for prop in {"compute", "storage"})
            and self._client.ICP_PLATFORM_SPACES
        ):
            raise WMLClientError(
                "'STORAGE' and 'COMPUTE' meta props are not applicable on "
                "IBM Cloud Pak® for Data. If using any of these, remove and retry"
            )

        if "storage" in changes:
            raise WMLClientError("STORAGE cannot be updated")

        self._validate_type(space_id, "space_id", str, True)
        self._validate_type(changes, "changes", dict, True)

    def _update_get_url_and_payload(
        self, space_id: str, changes: dict, details: dict
    ) -> tuple[str, List[dict]]:
        """Helper method for `(a)update` methods.
        Return URL and payload for an update space request.
        """
        if self._client.CLOUD_PLATFORM_SPACES and "compute" in changes:
            changes["compute"]["type"] = "machine_learning"
            changes["compute"] = [changes["compute"]]

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details["entity"], changes
        )

        url = self._client._href_definitions.get_platform_space_href(space_id)

        return url, patch_payload

    def update(self, space_id: str, changes: dict) -> dict:
        """Update existing space metadata. 'STORAGE' cannot be updated.
        STORAGE and COMPUTE are applicable only for cloud.

        :param space_id: ID of the space with the definition to be updated
        :type space_id: str

        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated space
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: "updated_space",
                client.spaces.ConfigurationMetaNames.COMPUTE: {
                    "name": "test_instance",
                    "crn": "v1:staging:public:pm-20-dev:us-south:a/09796a1b4cddfcc9f7fe17824a68a0f8:f1026e4b-77cf-4703-843d-c9984eac7272::",
                },
            }
            space_details = client.spaces.update(space_id, changes=metadata)
        """
        self._update_validate_args(space_id, changes)

        details = self.get_details(space_id)

        url, patch_payload = self._update_get_url_and_payload(
            space_id, changes, details
        )

        response = self._client.httpx_client.patch(
            url=url, json=patch_payload, headers=self._client._get_headers()
        )

        updated_details = self._handle_response(
            200, "spaces patch", response, _silent_response_logging=True
        )

        self._init_service_instance_if_compute(updated_details)

        return updated_details

    async def aupdate(self, space_id: str, changes: dict) -> dict:
        """Update existing space metadata asynchronously. 'STORAGE' cannot be updated.
        STORAGE and COMPUTE are applicable only for cloud.

        :param space_id: ID of the space with the definition to be updated
        :type space_id: str

        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated space
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: "updated_space",
                client.spaces.ConfigurationMetaNames.COMPUTE: {
                    "name": "test_instance",
                    "crn": "v1:staging:public:pm-20-dev:us-south:a/09796a1b4cddfcc9f7fe17824a68a0f8:f1026e4b-77cf-4703-843d-c9984eac7272::",
                },
            }
            space_details = await client.spaces.aupdate(space_id, changes=metadata)
        """
        self._update_validate_args(space_id, changes)

        details = await self.aget_details(space_id)

        url, patch_payload = self._update_get_url_and_payload(
            space_id, changes, details
        )

        response = await self._client.async_httpx_client.patch(
            url=url, json=patch_payload, headers=await self._client._aget_headers()
        )

        updated_details = self._handle_response(
            200, "spaces patch", response, _silent_response_logging=True
        )

        self._init_service_instance_if_compute(updated_details)

        return updated_details

    #######SUPPORT FOR SPACE MEMBERS

    def _prepare_create_member_payload(
        self, space_id: str, meta_props: dict
    ) -> dict[str, Any]:
        """Helper method for `(a)create_member` methods.
        Validate `space_id` and `meta_props` types.
        Return payload for a request to create member.
        """
        Spaces._validate_type(space_id, "space_id", str, True)
        Spaces._validate_type(meta_props, "meta_props", dict, True)

        meta = {}

        if "members" in meta_props:
            meta = meta_props
        elif "member" in meta_props:
            meta["members"] = [meta_props["member"]]

        return self.MemberMetaNames._generate_resource_metadata(
            meta, with_validation=True, client=self._client
        )

    def create_member(self, space_id: str, meta_props: dict) -> dict:
        """Create a member within a space.

        :param space_id: ID of the space with the definition to be updated
        :type space_id: str

        :param meta_props: metadata of the member configuration. To see available meta names, use:

            .. code-block:: python

                client.spaces.MemberMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored member
        :rtype: dict

        .. note::
            * `role` can be any one of the following: "viewer", "editor", "admin"
            * `type` can be any one of the following: "user", "service"
            * `id` can be one of the following: service-ID or IAM-userID

        **Examples**

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.MEMBERS: [
                    {"id": "IBMid-100000DK0B", "type": "user", "role": "admin"}
                ]
            }
            members_details = client.spaces.create_member(
                space_id=space_id, meta_props=metadata
            )

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.MEMBERS: [
                    {
                        "id": "iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                        "type": "service",
                        "role": "admin",
                    }
                ]
            }
            members_details = client.spaces.create_member(
                space_id=space_id, meta_props=metadata
            )
        """
        space_meta = self._prepare_create_member_payload(space_id, meta_props)

        creation_response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_platform_spaces_members_href(
                space_id
            ),
            headers=self._client._get_headers(),
            json=space_meta,
        )

        members_details = self._handle_response(
            200, "creating new members", creation_response
        )

        return members_details

    async def acreate_member(self, space_id: str, meta_props: dict) -> dict:
        """Create a member within a space asynchronously.

        :param space_id: ID of the space with the definition to be updated
        :type space_id: str

        :param meta_props: metadata of the member configuration. To see available meta names, use:

            .. code-block:: python

                client.spaces.MemberMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored member
        :rtype: dict

        .. note::
            * `role` can be any one of the following: "viewer", "editor", "admin"
            * `type` can be any one of the following: "user", "service"
            * `id` can be one of the following: service-ID or IAM-userID

        **Examples**

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.MEMBERS: [
                    {"id": "IBMid-100000DK0B", "type": "user", "role": "admin"}
                ]
            }
            members_details = await client.spaces.acreate_member(
                space_id=space_id, meta_props=metadata
            )

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.MEMBERS: [
                    {
                        "id": "iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                        "type": "service",
                        "role": "admin",
                    }
                ]
            }
            members_details = await client.spaces.acreate_member(
                space_id=space_id, meta_props=metadata
            )
        """
        space_meta = self._prepare_create_member_payload(space_id, meta_props)

        creation_response = await self._client.async_httpx_client.post(
            url=self._client._href_definitions.get_platform_spaces_members_href(
                space_id
            ),
            headers=await self._client._aget_headers(),
            json=space_meta,
        )

        members_details = self._handle_response(
            200, "creating new members", creation_response
        )

        return members_details

    def _get_url_for_space_member(self, space_id: str, member_id: str) -> str:
        """Helper method for `(a)get_member_details` and `(a)delete_member` methods.
        Validate `space_id` and `member_id` types.
        Return URL for a request to get member details or delete member.
        """
        Spaces._validate_type(space_id, "space_id", str, True)
        Spaces._validate_type(member_id, "member_id", str, True)

        return self._client._href_definitions.get_platform_spaces_member_href(
            space_id, member_id
        )

    def get_member_details(self, space_id: str, member_id: str) -> dict:
        """Get metadata of a member associated with a space.

        :param space_id: ID of that space with the definition to be updated
        :type space_id: str

        :param member_id: ID of the member
        :type member_id: str

        :return: metadata of the space member
        :rtype: dict

        **Example:**

        .. code-block:: python

            member_details = client.spaces.get_member_details(space_id, member_id)
        """
        url = self._get_url_for_space_member(space_id, member_id)

        response_get = self._client.httpx_client.get(
            url=url, headers=self._client._get_headers()
        )

        return self._handle_response(200, "Get space member", response_get)

    async def aget_member_details(self, space_id: str, member_id: str) -> dict:
        """Get metadata of a member associated with a space asynchronously.

        :param space_id: ID of that space with the definition to be updated
        :type space_id: str

        :param member_id: ID of the member
        :type member_id: str

        :return: metadata of the space member
        :rtype: dict

        **Example:**

        .. code-block:: python

            member_details = await client.spaces.aget_member_details(
                space_id, member_id
            )
        """
        url = self._get_url_for_space_member(space_id, member_id)

        response_get = await self._client.async_httpx_client.get(
            url=url, headers=await self._client._aget_headers()
        )

        return self._handle_response(200, "Get space member", response_get)

    def delete_member(self, space_id: str, member_id: str) -> Literal["SUCCESS"]:
        """Delete a member associated with a space.

        :param space_id: ID of the space
        :type space_id: str

        :param member_id: ID of the member
        :type member_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        :raises WMLClientError: if deletion failed

        **Example:**

        .. code-block:: python

            client.spaces.delete_member(space_id, member_id)
        """
        url = self._get_url_for_space_member(space_id, member_id)

        response_delete = self._client.httpx_client.delete(
            url=url, headers=self._client._get_headers()
        )

        return cast(
            Literal["SUCCESS"],
            self._handle_response(204, "space member deletion", response_delete, False),
        )

    async def adelete_member(self, space_id: str, member_id: str) -> Literal["SUCCESS"]:
        """Delete a member associated with a space asynchronously.

        :param space_id: ID of the space
        :type space_id: str

        :param member_id: ID of the member
        :type member_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        :raises WMLClientError: if deletion failed

        **Example:**

        .. code-block:: python

            await client.spaces.adelete_member(space_id, member_id)
        """
        url = self._get_url_for_space_member(space_id, member_id)

        response_delete = await self._client.async_httpx_client.delete(
            url=url, headers=await self._client._aget_headers()
        )

        return cast(
            Literal["SUCCESS"],
            self._handle_response(204, "space member deletion", response_delete, False),
        )

    def _update_member_get_url_and_payload(
        self, details: dict, space_id: str, member_id: str, changes: dict
    ) -> tuple[str, List[dict]]:
        """Helper method for `(a)update_member` methods.
        Return URL and payload for an update member request.
        """

        # The member record is a bit different from most other type of records we deal w.r.t patch
        # There is no encapsulating object for the fields. We need to be consistent with the way we
        # provide the meta in create/patch. When we give with .MEMBER, _generate_patch_payload
        # will generate with /member patch. So, separate logic for member patch inline here
        changes1 = changes["member"]

        # Union of two dictionaries. The one in changes1 will override existent ones in current meta
        details.update(changes1)

        patch_payload = [
            {"op": "replace", "path": f"/{k}", "value": details[k]}
            for k in ("role", "state")
            if k in details
        ]

        url = self._client._href_definitions.get_platform_spaces_member_href(
            space_id, member_id
        )

        return url, patch_payload

    def update_member(self, space_id: str, member_id: str, changes: dict) -> dict:
        """Update the metadata of an existing member.

        :param space_id: ID of the space
        :type space_id: str

        :param member_id: ID of the member to be updated
        :type member_id: str

        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated member
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {client.spaces.MemberMetaNames.MEMBER: {"role": "editor"}}
            member_details = client.spaces.update_member(
                space_id, member_id, changes=metadata
            )
        """
        self._validate_type(space_id, "space_id", str, True)
        self._validate_type(member_id, "member_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = self.get_member_details(space_id, member_id)

        url, patch_payload = self._update_member_get_url_and_payload(
            details, space_id, member_id, changes
        )

        response = self._client.httpx_client.patch(
            url=url, json=patch_payload, headers=self._client._get_headers()
        )

        updated_details = self._handle_response(200, "members patch", response)

        return updated_details

    async def aupdate_member(
        self, space_id: str, member_id: str, changes: dict
    ) -> dict:
        """Update the metadata of an existing member asynchronously.

        :param space_id: ID of the space
        :type space_id: str

        :param member_id: ID of the member to be updated
        :type member_id: str

        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated member
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {client.spaces.MemberMetaNames.MEMBER: {"role": "editor"}}
            member_details = await client.spaces.aupdate_member(
                space_id, member_id, changes=metadata
            )
        """
        self._validate_type(space_id, "space_id", str, True)
        self._validate_type(member_id, "member_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = await self.aget_member_details(space_id, member_id)

        url, patch_payload = self._update_member_get_url_and_payload(
            details, space_id, member_id, changes
        )

        response = await self._client.async_httpx_client.patch(
            url=url, json=patch_payload, headers=await self._client._aget_headers()
        )

        updated_details = self._handle_response(200, "members patch", response)

        return updated_details

    def list_members(
        self,
        space_id: str,
        limit: int | None = None,
        identity_type: str | None = None,
        role: str | None = None,
        state: str | None = None,
    ) -> DataFrame:
        """Print the stored members of a space in a table format.

        :param space_id: ID of the space
        :type space_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param identity_type: filter the members by type
        :type identity_type: str, optional

        :param role: filter the members by role
        :type role: str, optional

        :param state: filter the members by state
        :type state: str, optional

        :return: pandas.DataFrame with listed members
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.spaces.list_members(space_id)
        """
        self._validate_type(space_id, "space_id", str, True)

        params: dict[str, Any] = {}

        if limit is not None:
            params.update({"limit": limit})

        if identity_type is not None:
            params.update({"type": identity_type})

        if role is not None:
            params.update({"role": role})

        if state is not None:
            params.update({"state": state})

        href = self._client._href_definitions.get_platform_spaces_members_href(space_id)

        member_resources = self._get_resources(href, "space members", params)[
            "resources"
        ]

        space_values = [
            (
                (m["id"], m["type"], m["role"], m["state"])
                if "state" in m
                else (m["id"], m["type"], m["role"], None)
            )
            for m in member_resources
        ]

        table = self._list(space_values, ["ID", "TYPE", "ROLE", "STATE"], limit)
        return table

    def _promote_get_url_and_payload(
        self,
        asset_id: str,
        source_project_id: str,
        target_space_id: str,
        rev_id: str | None,
    ) -> tuple[str, dict]:
        """Helper method for `(a)promote` methods.
        Return URL and payload for a promote request.
        """
        promote_payload = {
            "spaceId": target_space_id,
            "projectId": source_project_id,
            "assetDescription": "Asset promoted by ibm_wml client",
        }

        if rev_id:
            promote_payload["revisionId"] = rev_id

        url = self._client._href_definitions.promote_asset_href(asset_id)

        return url, promote_payload

    def _promote_process_response(
        self, response: Response, source_project_id: str, target_space_id: str
    ) -> str:
        """Helper method for `(a)promote` methods.
        Handle 'promote' request response and return promoted asset id.
        """
        promotion_details = self._client.repository._handle_response(
            200, "promote asset", response
        )

        try:
            promoted_asset_id = promotion_details["promotedAsset"]["asset_id"]
        except KeyError as key_err:
            raise PromotionFailed(
                source_project_id,
                target_space_id,
                promotion_details,
                reason=str(key_err),
            )

        return promoted_asset_id

    def promote(
        self,
        asset_id: str,
        source_project_id: str,
        target_space_id: str,
        rev_id: str | None = None,
    ) -> str:
        """Promote an asset from a project to a space.

        :param asset_id: ID of the stored asset
        :type asset_id: str

        :param source_project_id: source project, from which the asset is promoted
        :type source_project_id: str

        :param target_space_id: target space, where the asset is promoted
        :type target_space_id: str

        :param rev_id: revision ID of the promoted asset
        :type rev_id: str, optional

        :return: ID of the promoted asset
        :rtype: str

        **Examples**

        .. code-block:: python

            promoted_asset_id = client.spaces.promote(
                asset_id, source_project_id=project_id, target_space_id=space_id
            )
            promoted_model_id = client.spaces.promote(
                model_id, source_project_id=project_id, target_space_id=space_id
            )
            promoted_function_id = client.spaces.promote(
                function_id, source_project_id=project_id, target_space_id=space_id
            )
            promoted_data_asset_id = client.spaces.promote(
                data_asset_id,
                source_project_id=project_id,
                target_space_id=space_id,
            )
            promoted_connection_asset_id = client.spaces.promote(
                connection_id,
                source_project_id=project_id,
                target_space_id=space_id,
            )
        """
        url, payload = self._promote_get_url_and_payload(
            asset_id, source_project_id, target_space_id, rev_id
        )

        response = self._client.httpx_client.post(
            url=url, headers=self._client._get_headers(), json=payload
        )

        return self._promote_process_response(
            response, source_project_id, target_space_id
        )

    async def apromote(
        self,
        asset_id: str,
        source_project_id: str,
        target_space_id: str,
        rev_id: str | None = None,
    ) -> str:
        """Promote an asset from a project to a space asynchronously.

        :param asset_id: ID of the stored asset
        :type asset_id: str

        :param source_project_id: source project, from which the asset is promoted
        :type source_project_id: str

        :param target_space_id: target space, where the asset is promoted
        :type target_space_id: str

        :param rev_id: revision ID of the promoted asset
        :type rev_id: str, optional

        :return: ID of the promoted asset
        :rtype: str

        **Examples**

        .. code-block:: python

            promoted_asset_id = await client.spaces.apromote(
                asset_id, source_project_id=project_id, target_space_id=space_id
            )
            promoted_model_id = await client.spaces.apromote(
                model_id, source_project_id=project_id, target_space_id=space_id
            )
            promoted_function_id = await client.spaces.apromote(
                function_id, source_project_id=project_id, target_space_id=space_id
            )
            promoted_data_asset_id = await client.spaces.apromote(
                data_asset_id,
                source_project_id=project_id,
                target_space_id=space_id,
            )
            promoted_connection_asset_id = await client.spaces.apromote(
                connection_id,
                source_project_id=project_id,
                target_space_id=space_id,
            )
        """
        url, payload = self._promote_get_url_and_payload(
            asset_id, source_project_id, target_space_id, rev_id
        )

        response = await self._client.async_httpx_client.post(
            url=url,
            headers=await self._client._aget_headers(),
            json=payload,
        )

        return self._promote_process_response(
            response, source_project_id, target_space_id
        )
