#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import json
from typing import Any, AsyncIterator, Iterator, Literal, overload

import httpx

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.gateway.models import Models
from ibm_watsonx_ai.gateway.policies import Policies
from ibm_watsonx_ai.gateway.providers import Providers
from ibm_watsonx_ai.gateway.rate_limits import RateLimits
from ibm_watsonx_ai.wml_client_error import InvalidMultipleArguments, WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource


def _streaming_create(api_client: APIClient, url: str, request_json: dict) -> Iterator:
    kw_args: dict = dict(
        method="POST",
        url=url,
        json=request_json,
        headers=api_client._get_headers(),
    )

    if hasattr(api_client.httpx_client, "post_stream"):
        stream_function = api_client.httpx_client.post_stream
    else:
        stream_function = api_client.httpx_client.stream

    with stream_function(**kw_args) as resp:
        if resp.status_code == 200:
            resp_iter = resp.iter_lines()

            for chunk in resp_iter:
                field_name, _, response = chunk.partition(":")

                if response.strip() == "[DONE]":
                    break

                if field_name == "data" and response:
                    try:
                        parsed_response = json.loads(response)
                    except json.JSONDecodeError:
                        raise Exception(f"Could not parse {response} as json")
                    yield parsed_response
        else:
            resp.read()
            raise WMLClientError(
                f"Request failed with: {resp.text} ({resp.status_code})"
            )


async def _streaming_acreate(
    api_client: APIClient, url: str, request_json: dict
) -> AsyncIterator:
    kw_args: dict = dict(
        method="POST",
        url=url,
        json=request_json,
        headers=await api_client._aget_headers(),
    )

    if hasattr(api_client.async_httpx_client, "post_stream"):
        stream_function = api_client.async_httpx_client.post_stream
    else:
        stream_function = api_client.async_httpx_client.stream

    async with stream_function(**kw_args) as resp:
        if resp.status_code == 200:
            resp_iter = resp.aiter_lines()

            async for chunk in resp_iter:
                field_name, _, response = chunk.partition(":")

                if response.strip() == "[DONE]":
                    break

                if field_name == "data" and response:
                    try:
                        parsed_response = json.loads(response)
                    except json.JSONDecodeError:
                        raise Exception(f"Could not parse {response} as json")
                    yield parsed_response

        else:
            await resp.aread()
            raise WMLClientError(
                f"Request failed with: ({resp.text} {resp.status_code})"
            )


class Gateway(WMLResource):
    """Model Gateway class."""

    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
    ):
        if credentials:
            api_client = APIClient(credentials, verify=verify)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        WMLResource.__init__(self, __name__, api_client)

        if self._client.ICP_PLATFORM_SPACES and self._client.CPD_version < 5.2:
            raise WMLClientError("AI Gateway is not supported for this release.")

        self.providers = Providers(self._client)
        self.models = Models(self._client)
        self.policies = Policies(self._client)
        self.rate_limits = RateLimits(self._client)

        # Chat completions
        class _ChatCompletions(WMLResource):
            def __init__(self, api_client: APIClient):
                WMLResource.__init__(self, __name__, api_client)

            @overload
            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> Iterator: ...

            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | Iterator | httpx.Response:
                """Generate chat completions for given model and messages.

                :param model: name of model for given provider or alias
                :type model: str

                :param messages: messages to be processed during call
                :type messages: list[dict]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | Iterator
                """
                request_json = {"messages": messages, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_chat_completions_href()

                if stream:
                    return _streaming_create(
                        api_client=self._client, url=url, request_json=request_json
                    )

                response = self._client.httpx_client.post(
                    url=url,
                    headers=self._client._get_headers(),
                    json=request_json,
                )

                return self._handle_response(200, "chat completion creation", response)

            @overload
            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> AsyncIterator: ...

            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | AsyncIterator | httpx.Response:
                """Generate chat completions for given model and messages asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param messages: messages to be processed during call
                :type messages: list[dict]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | AsyncIterator
                """
                request_json = {"messages": messages, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_chat_completions_href()

                if stream:
                    return _streaming_acreate(
                        api_client=self._client, url=url, request_json=request_json
                    )

                response = await self._client.async_httpx_client.post(
                    url=url,
                    headers=await self._client._aget_headers(),
                    json=request_json,
                )

                return self._handle_response(200, "chat completion creation", response)

        class _Chat:
            def __init__(self, api_client: APIClient):
                self.completions = _ChatCompletions(api_client)

        self.chat = _Chat(self._client)

        # Text completions
        class _Completions(WMLResource):
            def __init__(self, api_client: APIClient):
                WMLResource.__init__(self, __name__, api_client)

            @overload
            def create(
                self,
                model: str,
                prompt: str | list[str] | list[int],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            def create(
                self,
                model: str,
                prompt: str | list[str] | list[int],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> Iterator: ...

            def create(
                self,
                model: str,
                prompt: str | list[str] | list[int],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | Iterator:
                """Generate text completions for given model and prompt.

                :param model: name of model for given provider or alias
                :type model: str

                :param prompt: prompt for processing
                :type prompt: str or list[str] or list[int]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | Iterator
                """
                request_json = {"prompt": prompt, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_text_completions_href()

                if stream:
                    return _streaming_create(
                        api_client=self._client, url=url, request_json=request_json
                    )
                else:
                    response = self._client.httpx_client.post(
                        url=url,
                        headers=self._client._get_headers(),
                        json=request_json,
                    )

                    return self._handle_response(
                        200, "text completion creation", response
                    )

            @overload
            async def acreate(
                self,
                model: str,
                prompt: str | list[str] | list[int],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            async def acreate(
                self,
                model: str,
                prompt: str | list[str] | list[int],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> AsyncIterator: ...

            async def acreate(
                self,
                model: str,
                prompt: str | list[str] | list[int],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | AsyncIterator:
                """Generate text completions for given model and prompt asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param prompt: prompt for processing
                :type prompt: str or list[str] or list[int]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | AsyncIterator
                """
                request_json = {"prompt": prompt, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_text_completions_href()

                if stream:
                    return _streaming_acreate(
                        api_client=self._client, url=url, request_json=request_json
                    )
                else:
                    response = await self._client.async_httpx_client.post(
                        url=url,
                        headers=self._client._get_headers(),
                        json=request_json,
                    )

                    return self._handle_response(
                        200, "text completion creation", response
                    )

        self.completions = _Completions(self._client)

        # Embeddings
        class _Embeddings(WMLResource):
            def __init__(self, api_client: APIClient):
                WMLResource.__init__(self, __name__, api_client)

            def create(
                self, model: str, input: str | list[str] | list[int], **kwargs: Any
            ) -> dict:
                """Generate embeddings for given model and input.

                :param model: name of model for given provider or alias
                :type model: str

                :param input: prompt for processing
                :type input: str or list[str] or list[int]

                :returns: embeddings for given model and input
                :rtype: dict
                """
                request_json = {"input": input, "model": model, **kwargs}

                response = self._client.httpx_client.post(
                    self._client._href_definitions.get_gateway_embeddings_href(),
                    headers=self._client._get_headers(),
                    json=request_json,
                )

                return self._handle_response(200, "embedding creation", response)

            async def acreate(
                self, model: str, input: str | list[str] | list[int], **kwargs: Any
            ) -> dict:
                """Generate embeddings for given model and input asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param input: prompt for processing
                :type input: str or list[str] or list[int]

                :returns: embeddings for given model and input
                :rtype: dict
                """
                request_json = {"input": input, "model": model, **kwargs}

                response = await self._client.async_httpx_client.post(
                    self._client._href_definitions.get_gateway_embeddings_href(),
                    headers=self._client._get_headers(),
                    json=request_json,
                )

                return self._handle_response(200, "embedding creation", response)

        self.embeddings = _Embeddings(self._client)
