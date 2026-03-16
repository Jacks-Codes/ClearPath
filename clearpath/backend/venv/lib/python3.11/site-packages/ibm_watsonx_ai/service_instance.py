#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from warnings import warn

from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    NoWMLCredentialsProvided,
    WMLClientError,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.href_definitions import HrefDefinitions


class ServiceInstance:
    """Connect, get details, and check usage of a Watson Machine Learning service instance."""

    def __init__(self, client: APIClient) -> None:
        self._logger = logging.getLogger(__name__)
        self._client = client

        self._instance_id = self._client.credentials.instance_id

        # ml_repository_client is initialized in repo
        self._details: dict | None = None
        self._refresh_details = False

    @property
    def _credentials(self) -> Credentials:
        return self._client.credentials

    def _get_token(self) -> str:
        """Get token.

        .. deprecated:: v1.2.3
               This protected function is deprecated since v1.2.3. Use ``APIClient.token`` instead.
        """
        get_token_method_deprecated_warning = (
            "`APIClient.service_instance._get_token()` is deprecated since v1.2.3. "
            "Use ``APIClient.token`` instead."
        )
        warn(get_token_method_deprecated_warning, category=DeprecationWarning)
        return self._client.token

    @property
    def _href_definitions(self) -> HrefDefinitions:
        return self._client._href_definitions

    @property
    def instance_id(self) -> str:
        if self._instance_id is None:
            raise WMLClientError(
                (
                    "instance_id for this plan is picked up from the space or project with which "
                    "this instance_id is associated with. Set the space or project with associated "
                    "instance_id to be able to use this function"
                )
            )
        return self._instance_id

    @property
    def details(self) -> dict | None:
        details_attribute_deprecated_warning = (
            "Attribute `details` is deprecated. "
            "Please use method `get_details()` instead."
        )
        warn(details_attribute_deprecated_warning, category=DeprecationWarning)
        if self._details is None or self._refresh_details:
            # By passing instance_id as argument, we assure that proper error is raised from property `instance_id`
            # when self.instance_id is None
            self._details = self.get_details(instance_id=self.instance_id)
            self._refresh_details = False
        return self._details

    @details.setter
    def details(self, value: dict | None) -> None:
        self._details = value

    def get_instance_id(self) -> str:
        """Get the instance ID of a Watson Machine Learning service.

        :return: ID of the instance
        :rtype: str

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_instance_id()
        """
        if self._instance_id is None:
            raise WMLClientError(
                "instance_id for this plan is picked up from the space or project with which "
                "this instance_id is associated with. Set the space or project with associated "
                "instance_id to be able to use this function"
            )

        return self.instance_id

    def get_api_key(self) -> str | None:
        """Get the API key of a Watson Machine Learning service.

        :return: API key
        :rtype: str | None

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_api_key()
        """
        return self._credentials.api_key

    def get_url(self) -> str | None:
        """Get the instance URL of a Watson Machine Learning service.

        :return: URL of the instance
        :rtype: str | None

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_url()
        """
        return self._credentials.url

    def get_username(self) -> str | None:
        """Get the username for the Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.

        :return: username
        :rtype: str | None

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_username()
        """
        if self._client.ICP_PLATFORM_SPACES:
            if self._credentials.username is not None:
                return self._credentials.username
            else:
                raise WMLClientError("`username` missing in credentials.")
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_password(self) -> str | None:
        """Get the password for the Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.

        :return: password
        :rtype: str | None

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_password()
        """
        if self._client.ICP_PLATFORM_SPACES:
            if self._credentials.password is not None:
                return self._credentials.password
            else:
                raise WMLClientError("`password` missing in credentials.")
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_details(self, instance_id: str | None = None) -> dict:
        """Get information about the Watson Machine Learning instance.

        :param instance_id: ID of the instance, defaults to None
        :type instance_id: str, optional

        :return: metadata of the service instance
        :rtype: dict

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_details()

        """

        if not self._client.CLOUD_PLATFORM_SPACES:
            return {}

        if self._credentials is None:
            raise NoWMLCredentialsProvided()

        if instance_id is None:
            if self._instance_id is None:
                raise WMLClientError(
                    "Either, argument `instance_id` needs to be specified, "
                    "or param `scope_validation` needs to be set to True when initializing APIClient instance. "
                    "In the latter case, `instance_id` for this plan is picked up from the space or project which "
                    "this instance_id is associated with. Set the space or project with associated "
                    "instance_id to be able to use this function"
                )

                # /ml/v4/instances will need either space_id or project_id as mandatory params
            # We will enable this service instance class only during create space or
            # set space/project. So, space_id/project_id would have been populated at this point
            instance_id = self.instance_id
        headers = self._client._get_headers()

        del headers["User-Agent"]
        if "ML-Instance-ID" in headers:
            headers.pop("ML-Instance-ID")
        response_get_instance = self._client.httpx_client.get(
            url=self._href_definitions.get_v4_instance_id_href(instance_id),
            params=self._client._params(skip_space_project_chk=True),
            headers=headers,
        )

        if response_get_instance.status_code == 200:
            return response_get_instance.json()
        else:
            raise ApiRequestFailure(
                "Getting instance details failed.", response_get_instance
            )
