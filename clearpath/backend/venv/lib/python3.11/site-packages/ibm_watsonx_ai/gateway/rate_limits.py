#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from typing import Literal, TypedDict, cast

import pandas as pd

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_resource import WMLResource


class RateLimitSettings(TypedDict):
    """
    Model Gateway rate limit settings.

    :param amount: amount is the number of tokens refilled into the bucket each interval
    :type amount: int

    :param capacity: capacity is the maximum number of tokens (requests) the bucket can hold
    :type capacity: int

    :param duration: duration is the refill interval, formatted as a Go duration string
        (for more information please see: https://pkg.go.dev/time#ParseDuration)
    :type duration: str
    """

    amount: int
    capacity: int
    duration: str


class RateLimits(WMLResource):
    """Model Gateway rate limits class."""

    def __init__(self, api_client: APIClient):
        WMLResource.__init__(self, __name__, api_client)

    def _create(
        self,
        *,
        payload: dict,
        request: RateLimitSettings | None,
        token: RateLimitSettings | None,
    ) -> dict:
        if request:
            payload["request"] = request

        if token:
            payload["token"] = token

        response = self._client.httpx_client.post(
            self._client._href_definitions.get_gateway_rate_limits_href(),
            headers=self._client._get_headers(),
            json=payload,
        )

        return self._handle_response(201, "rate limit creation", response)

    def create_for_tenant(
        self,
        *,
        request: RateLimitSettings | None = None,
        token: RateLimitSettings | None = None,
    ) -> dict:
        """Create rate limit for tenant in Model Gateway.

        :param request: request rate limiting settings
        :type request: RateLimitSettings, optional

        :param token: token rate limiting settings
        :type token: RateLimitSettings, optional

        :returns: rate limit details
        :rtype: dict
        """

        return self._create(
            payload={"type": "tenant"},
            request=request,
            token=token,
        )

    def create_for_provider(
        self,
        provider_id: str,
        *,
        request: RateLimitSettings | None = None,
        token: RateLimitSettings | None = None,
    ) -> dict:
        """Create rate limit for provider in Model Gateway.

        :param provider_id: ID of the Model Gateway provider
        :type provider_id: str

        :param request: request rate limiting settings
        :type request: RateLimitSettings, optional

        :param token: token rate limiting settings
        :type token: RateLimitSettings, optional

        :returns: rate limit details
        :rtype: dict
        """

        return self._create(
            payload={"type": "provider", "provider_uuid": provider_id},
            request=request,
            token=token,
        )

    def create_for_model(
        self,
        model_id: str,
        *,
        request: RateLimitSettings | None = None,
        token: RateLimitSettings | None = None,
    ) -> dict:
        """Create rate limit for model in Model Gateway.

        :param model_id: ID of the Model Gateway model
        :type model_id: str

        :param request: request rate limiting settings
        :type request: RateLimitSettings, optional

        :param token: token rate limiting settings
        :type token: RateLimitSettings, optional

        :returns: rate limit details
        :rtype: dict
        """

        return self._create(
            payload={"type": "model", "model_uuid": model_id},
            request=request,
            token=token,
        )

    def _update(
        self,
        *,
        rate_limit_id: str,
        payload: dict,
        request: RateLimitSettings | None,
        token: RateLimitSettings | None,
    ) -> dict:
        if request:
            payload["request"] = request

        if token:
            payload["token"] = token

        response = self._client.httpx_client.put(
            self._client._href_definitions.get_gateway_rate_limit_href(rate_limit_id),
            headers=self._client._get_headers(),
            json=payload,
        )

        return self._handle_response(200, "rate limit update", response)

    def update_for_tenant(
        self,
        rate_limit_id: str,
        *,
        request: RateLimitSettings | None = None,
        token: RateLimitSettings | None = None,
    ) -> dict:
        """Update rate limit for tenant in Model Gateway.

        :param rate_limit_id: ID of the rate limit
        :type rate_limit_id: str

        :param request: request rate limiting settings
        :type request: RateLimitSettings, optional

        :param token: token rate limiting settings
        :type token: RateLimitSettings, optional

        :returns: rate limit details
        :rtype: dict
        """

        return self._update(
            rate_limit_id=rate_limit_id,
            payload={"type": "tenant"},
            request=request,
            token=token,
        )

    def update_for_provider(
        self,
        rate_limit_id: str,
        provider_id: str,
        *,
        request: RateLimitSettings | None = None,
        token: RateLimitSettings | None = None,
    ) -> dict:
        """Update rate limit for provider in Model Gateway.

        :param rate_limit_id: ID of the rate limit
        :type rate_limit_id: str

        :param provider_id: ID of the Model Gateway provider
        :type provider_id: str

        :param request: request rate limiting settings
        :type request: RateLimitSettings, optional

        :param token: token rate limiting settings
        :type token: RateLimitSettings, optional

        :returns: rate limit details
        :rtype: dict
        """

        return self._update(
            rate_limit_id=rate_limit_id,
            payload={"type": "provider", "provider_uuid": provider_id},
            request=request,
            token=token,
        )

    def update_for_model(
        self,
        rate_limit_id: str,
        model_id: str,
        *,
        request: RateLimitSettings | None = None,
        token: RateLimitSettings | None = None,
    ) -> dict:
        """Update rate limit for model in Model Gateway.

        :param rate_limit_id: ID of the rate limit
        :type rate_limit_id: str

        :param model_id: ID of the Model Gateway model
        :type model_id: str

        :param request: request rate limiting settings
        :type request: RateLimitSettings, optional

        :param token: token rate limiting settings
        :type token: RateLimitSettings, optional

        :returns: rate limit details
        :rtype: dict
        """

        return self._update(
            rate_limit_id=rate_limit_id,
            payload={"type": "model", "model_uuid": model_id},
            request=request,
            token=token,
        )

    def get_details(self, *, rate_limit_id: str | None = None) -> dict:
        """Get details of rate limits.
        If ``rate_limit_id`` is specified, returns details of that rate limit.

        :param rate_limit_id: ID of the rate limit
        :type rate_limit_id: str, optional

        :returns: details of rate limits or rate limit if ``rate_limit_id`` is specified
        :rtype: dict
        """

        url = (
            self._client._href_definitions.get_gateway_rate_limit_href(rate_limit_id)
            if rate_limit_id
            else self._client._href_definitions.get_gateway_rate_limits_href()
        )

        response = self._client.httpx_client.get(
            url=url, headers=self._client._get_headers()
        )

        return self._handle_response(200, "getting rate limit details", response)

    def list(self) -> pd.DataFrame:
        """List rate limits registered in Model Gateway.

        :returns: dataframe containing list results
        :rtype: pandas.DataFrame
        """

        rate_limit_details = self.get_details()["data"]

        rate_limit_values = [
            (item["uuid"], item["type"], "request" in item, "token" in item)
            for item in rate_limit_details
        ]

        table = self._list(
            rate_limit_values, ["ID", "TYPE", "FOR_REQUESTS", "FOR_TOKENS"], limit=None
        )

        return table

    def delete(self, rate_limit_id: str) -> Literal["SUCCESS"]:
        """Delete rate limit from Model Gateway.

        :param rate_limit_id: ID of the rate limit
        :type rate_limit_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        :raises WMLClientError: if deletion failed
        """

        response = self._client.httpx_client.delete(
            self._client._href_definitions.get_gateway_rate_limit_href(rate_limit_id),
            headers=self._client._get_headers(),
        )

        return cast(
            Literal["SUCCESS"],
            self._handle_response(204, "model deletion", response, json_response=False),
        )

    @staticmethod
    def get_id(rate_limit_details: dict) -> str:
        """Get rate limit ID from rate limit details.

        :param rate_limit_details: details of the rate limit
        :type rate_limit_details: dict

        :returns: ID of the rate limit
        :rtype: str
        """

        return rate_limit_details["uuid"]
