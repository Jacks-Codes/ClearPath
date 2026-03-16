#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import fields
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Literal,
)
from warnings import warn

import httpx

from ibm_watsonx_ai._wrappers import httpx_wrapper
from ibm_watsonx_ai._wrappers.httpx_wrapper import TokenBucket, _httpx_transport_params
from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
    TextGenParameters,
)
from ibm_watsonx_ai.foundation_models.utils.utils import (
    GraniteGuardianDetectionWarning,
    HAPDetectionWarning,
    PIIDetectionWarning,
)
from ibm_watsonx_ai.utils.utils import get_from_json
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient

__all__ = ["BaseModelInference"]

_RETRY_STATUS_CODES = [429, 503, 504, 520]
LIMIT_RATE_HEADER = "x-requests-limit-rate"


class BaseModelInference(WMLResource, ABC):
    """Base interface class for the model interface."""

    DEFAULT_CONCURRENCY_LIMIT = 8

    def __init__(
        self,
        name: str,
        client: APIClient,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
        validate: bool = True,
    ):
        # to use in get_identifying_params(
        self._validate = validate

        self._transport_params = _httpx_transport_params(client)

        WMLResource.__init__(self, name, client)

        # Set initially 8 requests per second as it is default for prod instances
        # if header "x-requests-limit-rate" is different capacity will be updated
        self.rate_limiter = TokenBucket(rate=8, capacity=8)

        self.retry_status_codes = retry_status_codes
        self.max_retries = max_retries
        self.delay_time = delay_time

    @abstractmethod
    def get_details(self) -> dict:
        """Get model interface's details

        :return: details of model or deployment
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> dict:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat response.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> Generator:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat as generator.
        """
        raise NotImplementedError

    @abstractmethod
    async def achat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def achat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> AsyncGenerator:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat as a async generator.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> dict | list[dict] | Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        raise NotImplementedError

    @abstractmethod
    async def _agenerate_single(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        guardrails_granite_guardian_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> dict:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        raise NotImplementedError

    @abstractmethod
    async def agenerate_stream(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> AsyncGenerator:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_text_stream(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generator.
        """
        raise NotImplementedError

    @abstractmethod
    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        raise NotImplementedError

    def _prepare_chat_payload(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        context: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> dict:
        raise NotImplementedError

    def _prepare_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        guardrails_granite_guardian_params: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        raise NotImplementedError

    def _send_inference_payload_raw(
        self,
        payload: dict,
        generate_url: str,
    ) -> httpx.Response:
        post_params: dict[str, Any] = dict(
            url=generate_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._post(http_client=self._client.httpx_client, **post_params)

    def _send_inference_payload(
        self,
        payload: dict,
        generate_url: str,
    ) -> dict:
        response_scoring = self._send_inference_payload_raw(
            payload=payload,
            generate_url=generate_url,
        )

        return self._handle_response(
            200,
            "generate",
            response_scoring,
            _field_to_hide="generated_text",
        )

    async def _asend_inference_payload(
        self,
        payload: dict,
        generate_url: str,
    ) -> dict:
        response = await self._apost(
            self._client.async_httpx_client,
            url=generate_url,
            json=payload,
            headers=await self._client._aget_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        return self._handle_response(200, "agenerate", response)

    async def _agenerate_stream_with_url(
        self,
        payload: dict,
        generate_url: str,
    ) -> AsyncGenerator:
        if hasattr(self._client.async_httpx_client, "post_stream"):
            stream_function = self._client.async_httpx_client.post_stream
        else:
            stream_function = self._client.async_httpx_client.stream

        kw_args: dict = dict(
            method="POST",
            url=generate_url,
            json=payload,
            headers=await self._client._aget_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        async with self._astream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = resp.aiter_lines()

                async for chunk in resp_iter:
                    if chunk.rstrip() == "event: error":
                        chunk = await anext(resp_iter)
                        field_name, _, response = chunk.partition(":")
                        raise WMLClientError(
                            error_msg="Error event occurred during generating stream.",
                            reason=response,
                        )

                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and "generated_text" in chunk:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")

                        yield parsed_response

            elif resp.status_code != 200:
                await resp.aread()
                raise WMLClientError(
                    f"Request failed with: ({resp.text} {resp.status_code})"
                )

    @httpx_wrapper._with_retry_stream()
    @contextmanager
    def _stream(
        self,
        stream_function: Callable,
        **kw_args: Any,
    ) -> Generator[httpx.Response, None, None]:
        """Handles streaming with retry."""
        with stream_function(**kw_args) as resp:
            yield resp

    @httpx_wrapper._with_async_retry_stream()
    @asynccontextmanager
    async def _astream(
        self,
        stream_function: Callable,
        **kw_args: Any,
    ) -> AsyncGenerator[httpx.Response, None]:
        async with stream_function(**kw_args) as resp:
            yield resp

    def __make_request(
        self,
        payload: dict,
        generate_url: str,
    ) -> dict:
        """Rate-limited request with dynamic token adjustment and retry logic."""
        self.rate_limiter.acquire()

        inference_response = self._send_inference_payload_raw(
            payload=payload,
            generate_url=generate_url,
        )
        rate_limit = int(inference_response.headers.get(LIMIT_RATE_HEADER, 8))
        if rate_limit and rate_limit != self.rate_limiter.capacity:
            self.rate_limiter.capacity = rate_limit

        rate_limit_remaining = int(
            inference_response.headers.get(
                httpx_wrapper.REMAINING_LIMIT_HEADER, self.rate_limiter.capacity
            )
        )
        self.rate_limiter.adjust_tokens(rate_limit_remaining)

        return self._handle_response(
            200,
            "generate",
            inference_response,
            _field_to_hide="generated_text",
        )

    def _generate_with_url(
        self,
        payloads: list[dict],
        generate_url: str,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    ) -> list | dict:
        """
        Helper method which implements multi-threading for with passed generate_url.
        """
        if len(payloads) > 1:
            # If CLOUD, use __make_request which uses token bucket for throttling
            if self._client.CLOUD_PLATFORM_SPACES:
                func = self.__make_request
            else:  # If CDP, don't use Token Bucket
                func = self._send_inference_payload

            inference_fn = partial(
                func,
                generate_url=generate_url,
            )  # If CDP, don't use Token Bucket

            if (payloads_length := len(payloads)) <= concurrency_limit:
                with ThreadPoolExecutor(max_workers=payloads_length) as executor:
                    generated_responses = list(executor.map(inference_fn, payloads))
            else:
                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    generated_responses = list(executor.map(inference_fn, payloads))
            return generated_responses

        else:
            response = [
                self._send_inference_payload(
                    payload=payloads[0],
                    generate_url=generate_url,
                )
            ]
        return response

    def _generate_with_url_async(
        self,
        payloads: list[dict],
        generate_url: str,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    ) -> Generator:
        """
        Helper method which implements multi-threading for with passed generate_url.
        """
        payloads = copy.deepcopy(payloads)

        for payload in payloads:
            payload.setdefault("parameters", {})["return_options"] = {
                "input_text": True
            }

        if len(payloads) > 1:
            # If CLOUD, use __make_request which uses token bucket for throttling
            if self._client.CLOUD_PLATFORM_SPACES:
                func = self.__make_request
            else:  # If CDP, don't use Token Bucket
                func = self._send_inference_payload

            inference_fn = partial(
                func,
                generate_url=generate_url,
            )

            if (payloads_length := len(payloads)) <= concurrency_limit:
                with ThreadPoolExecutor(max_workers=payloads_length) as executor:
                    generate_futures = [
                        executor.submit(inference_fn, payload=payload)
                        for payload in payloads
                    ]
                    try:
                        for future in as_completed(generate_futures):
                            yield future.result()
                    except:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

            else:
                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    generate_futures = [
                        executor.submit(inference_fn, payload=payload)
                        for payload in payloads
                    ]
                    try:
                        for future in as_completed(generate_futures):
                            yield future.result()
                    except:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
        else:
            response = self._send_inference_payload(
                payload=payloads[0],
                generate_url=generate_url,
            )
            yield response

    def _generate_stream_with_url(
        self,
        payload: dict,
        generate_stream_url: str,
        raw_response: bool = False,
    ) -> Generator:
        if hasattr(self._client.httpx_client, "post_stream"):
            stream_function = self._client.httpx_client.post_stream
        else:
            stream_function = self._client.httpx_client.stream

        kw_args: dict = dict(
            method="POST",
            url=generate_stream_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        with self._stream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = (
                    resp.iter_lines()
                    if isinstance(resp, httpx.Response)
                    else resp.iter_lines(decode_unicode=False)
                )
                for chunk in resp_iter:
                    if chunk.rstrip() == "event: error":
                        chunk = next(resp_iter)
                        field_name, _, response = chunk.partition(":")
                        raise WMLClientError(
                            error_msg="Error event occurred during generating stream.",
                            reason=response,
                        )

                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and "generated_text" in chunk:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        if raw_response:
                            yield parsed_response
                            continue
                        yield self._return_guardrails_stats(parsed_response)[
                            "generated_text"
                        ]

            else:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    @staticmethod
    def _return_guardrails_stats(single_response: dict) -> dict:
        results = single_response["results"][0]
        hap_details = get_from_json(
            results,
            ["moderations", "hap"],
        )
        if hap_details:
            harmful_text_warning = f"Potentially harmful text detected: {hap_details}"
            if hap_details[0].get("input"):
                # overwrite with UNSUITABLE_INPUT warning from API if present in response
                harmful_text_warning = next(
                    (
                        warning.get("message")
                        for warning in get_from_json(
                            single_response, ["system", "warnings"]
                        )
                        if warning.get("id") == "UNSUITABLE_INPUT"
                    ),
                    harmful_text_warning,
                )
            warn(harmful_text_warning, category=HAPDetectionWarning)
        pii_details = get_from_json(
            results,
            ["moderations", "pii"],
        )
        if pii_details:
            identifiable_information_warning = (
                f"Personally identifiable information detected: {pii_details}"
            )
            if pii_details[0].get("input"):
                # overwrite with UNSUITABLE_INPUT warning from API if present in response
                identifiable_information_warning = next(
                    (
                        warning.get("message")
                        for warning in get_from_json(
                            single_response, ["system", "warnings"]
                        )
                        if warning.get("id") == "UNSUITABLE_INPUT"
                    ),
                    identifiable_information_warning,
                )

            warn(identifiable_information_warning, category=PIIDetectionWarning)
        granite_guardian_details = get_from_json(
            results, ["moderations", "granite_guardian"]
        )
        if granite_guardian_details:
            granite_guardian_warning = (
                f"Potentially granite guardian detected: {granite_guardian_details}"
            )
            if granite_guardian_details[0].get("input"):
                granite_guardian_warning = next(
                    (
                        warning.get("message")
                        for warning in get_from_json(
                            single_response, ["system", "warnings"]
                        )
                        if warning.get("id") == "UNSUITABLE_INPUT"
                    ),
                    granite_guardian_warning,
                )
            warn(granite_guardian_warning, category=GraniteGuardianDetectionWarning)
        return results

    @staticmethod
    def _update_moderations_params(additional_params: dict) -> dict:
        default_params = {"input": {"enabled": True}, "output": {"enabled": True}}
        if additional_params:
            for key, value in default_params.items():
                if key in additional_params:
                    if additional_params[key]:
                        if "threshold" in additional_params:
                            default_params[key]["threshold"] = additional_params[
                                "threshold"
                            ]
                    else:
                        default_params[key]["enabled"] = False
                else:
                    if "threshold" in additional_params:
                        default_params[key]["threshold"] = additional_params[
                            "threshold"
                        ]
            if "mask" in additional_params:
                default_params.update({"mask": additional_params["mask"]})
        return default_params

    @staticmethod
    def _validate_and_overwrite_params(
        params: dict[str, Any], valid_param: TextChatParameters | TextGenParameters
    ) -> dict[str, Any]:
        """Validate and fix parameters"""
        chat_valid_params = {field.name.lower() for field in fields(valid_param)}
        valid_params = {}
        invalid_params = {}

        for param, value in params.items():
            if param.lower() in chat_valid_params:
                valid_params[param] = value
            else:
                invalid_params[param] = value

        if invalid_params:
            invalid_params_warning = f"Parameters [{', '.join(invalid_params)}] is/are not recognized and will be ignored."
            warn(invalid_params_warning)

        return valid_params

    @httpx_wrapper._with_retry()
    def _post(self, http_client: Any, *args: Any, **kwargs: Any) -> httpx.Response:
        return http_client.post(*args, **kwargs)

    @httpx_wrapper._with_async_retry()
    async def _apost(
        self, async_http_client: Any, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        return await async_http_client.post(*args, **kwargs)
