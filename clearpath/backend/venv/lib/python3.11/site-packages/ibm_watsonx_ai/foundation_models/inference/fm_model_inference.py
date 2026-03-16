#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    Literal,
    cast,
)

import httpx

__all__ = ["FMModelInference"]


from ibm_watsonx_ai.foundation_models.schema import (
    BaseSchema,
    Crypto,
    TextChatParameters,
    TextGenParameters,
)
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.utils.utils import (
    _check_model_state,
)
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.wml_client_error import WMLClientError

from .base_model_inference import _RETRY_STATUS_CODES, BaseModelInference

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class FMModelInference(BaseModelInference):
    """Base abstract class for the model interface."""

    def __init__(
        self,
        *,
        model_id: str,
        api_client: APIClient,
        params: dict | TextChatParameters | TextGenParameters | None = None,
        validate: bool = True,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
    ):
        self.model_id = model_id
        if isinstance(self.model_id, Enum):
            self.model_id = self.model_id.value

        self.params = params
        FMModelInference._validate_type(
            params, "params", [dict, TextChatParameters, TextGenParameters], False, True
        )

        self._client = api_client
        self._tech_preview = False
        if validate:
            model_specs = cast(dict, self._client.foundation_models.get_model_specs())

            supported_models = [
                spec["model_id"] for spec in model_specs.get("resources", [])
            ]

            if self.model_id not in supported_models:
                model_specs = cast(
                    dict,
                    self._client.foundation_models.get_model_specs(tech_preview=True),
                )
                supported_models.clear()
                for spec in model_specs.get("resources", []):
                    supported_models.append(spec["model_id"])
                    if self.model_id == spec["model_id"]:
                        if "tech_preview" in spec:  # check if tech_preview model
                            self._tech_preview = True
                        break

                if not self._tech_preview:
                    raise WMLClientError(
                        error_msg=f"Model '{self.model_id}' is not supported for this environment. "
                        f"Supported models: {supported_models}"
                    )

            # check if model is in constricted mode
            _check_model_state(
                self._client,
                self.model_id,
                tech_preview=self._tech_preview,
                model_specs=model_specs,
            )

        BaseModelInference.__init__(
            self,
            __name__,
            self._client,
            max_retries,
            delay_time,
            retry_status_codes,
            validate=validate,
        )

    def get_details(self) -> dict:
        """Get model's details

        :return: details of model or deployment
        :rtype: dict
        """
        return self._client.foundation_models.get_model_specs(
            self.model_id, tech_preview=self._tech_preview
        )  # type: ignore[return-value]

    def chat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        text_chat_url = self._client._href_definitions.get_fm_chat_href("chat")

        return self._send_chat_payload(
            messages=messages,
            params=params,
            generate_url=text_chat_url,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            crypto=crypto,
        )

    def chat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> Generator:
        text_chat_stream_url = self._client._href_definitions.get_fm_chat_href(
            "chat_stream"
        )

        return self._generate_chat_stream_with_url(
            messages=messages,
            params=params,
            chat_stream_url=text_chat_stream_url,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    async def achat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        text_chat_url = self._client._href_definitions.get_fm_chat_href("chat")

        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        if isinstance(crypto, BaseSchema):
            crypto = crypto.to_dict()

        if crypto:
            payload["crypto"] = crypto

        response = await self._apost(
            self._client.async_httpx_client,
            url=text_chat_url,
            json=payload,
            headers=await self._client._aget_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        return self._handle_response(200, "achat", response, _field_to_hide="choices")

    async def achat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> AsyncGenerator:
        text_chat_stream_url = self._client._href_definitions.get_fm_chat_href(
            "chat_stream"
        )

        return self._agenerate_chat_stream_with_url(
            messages=messages,
            params=params,
            chat_stream_url=text_chat_stream_url,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    def generate(
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = BaseModelInference.DEFAULT_CONCURRENCY_LIMIT,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
        crypto: dict | Crypto | None = None,
    ) -> dict | list[dict] | Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        # if user change default value for `validate_prompt_variables` params raise an error
        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change its value for other scenarios."
            )
        self._validate_type(
            prompt, "prompt", [str, list], True, raise_error_for_list=True
        )
        self._validate_type(
            guardrails_hap_params, "guardrails_hap_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_pii_params, "guardrails_pii_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_granite_guardian_params,
            "guardrails_granite_guardian_params",
            dict,
            mandatory=False,
        )

        generate_text_url = self._client._href_definitions.get_fm_generation_href(
            "text"
        )
        prompt = cast(str | list, prompt)

        prompts = prompt if isinstance(prompt, list) else [prompt]

        payloads = [
            self._prepare_inference_payload(
                prompt=p,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
                guardrails_granite_guardian_params=guardrails_granite_guardian_params,
                crypto=crypto,
            )
            for p in prompts
        ]

        if async_mode:
            return self._generate_with_url_async(
                payloads=payloads,
                generate_url=generate_text_url,
                concurrency_limit=concurrency_limit,
            )
        else:
            results = self._generate_with_url(
                payloads=payloads,
                generate_url=generate_text_url,
                concurrency_limit=concurrency_limit,
            )
            return results if isinstance(prompt, list) else list(results)[0]

    async def _agenerate_single(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        guardrails_granite_guardian_params: dict | None = None,
        validate_prompt_variables: bool = True,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change its value for other scenarios."
            )

        self._validate_type(prompt, "prompt", str, True)
        self._validate_type(
            guardrails_hap_params, "guardrails_hap_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_pii_params, "guardrails_pii_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_granite_guardian_params,
            "guardrails_granite_guardian_params",
            dict,
            mandatory=False,
        )
        generate_text_url = self._client._href_definitions.get_fm_generation_href(
            "text"
        )

        payload = self._prepare_inference_payload(
            prompt=prompt,  # type: ignore[arg-type]
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
            crypto=crypto,
        )

        return await self._asend_inference_payload(
            generate_url=generate_text_url,
            payload=payload,
        )

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
        will generate a completion text as async generator response.
        """
        # if user change default value for `validate_prompt_variables` params raise an error
        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change its value for other scenarios."
            )

        self._validate_type(
            guardrails_hap_params, "guardrails_hap_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_pii_params, "guardrails_pii_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_granite_guardian_params,
            "guardrails_granite_guardian_params",
            dict,
            mandatory=False,
        )

        self._validate_type(prompt, "prompt", str, True)

        generate_stream_url = (
            self._client._href_definitions.get_fm_generation_stream_href()
        )

        payload = self._prepare_inference_payload(
            prompt=prompt,  # type: ignore[arg-type]
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

        return self._agenerate_stream_with_url(
            generate_url=generate_stream_url,
            payload=payload,
        )

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
        # if user change default value for `validate_prompt_variables` params raise an error
        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change it value for others scenarios."
            )
        self._validate_type(prompt, "prompt", str, True)
        generate_text_stream_url = (
            self._client._href_definitions.get_fm_generation_stream_href()
        )
        prompt = cast(str, prompt)

        payload = self._prepare_inference_payload(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

        return self._generate_stream_with_url(
            generate_stream_url=generate_text_stream_url,
            payload=payload,
            raw_response=raw_response,
        )

    def tokenize(
        self,
        prompt: str,
        return_tokens: bool = False,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        """
        Given a text prompt as input, and return_tokens parameter will return tokenized input text.
        """
        self._validate_type(prompt, "prompt", str, True)
        self._validate_type(crypto, "crypto", [dict, Crypto], False, True)
        generate_tokenize_url = self._client._href_definitions.get_fm_tokenize_href()

        return self._tokenize_with_url(
            prompt=prompt,
            tokenize_url=generate_tokenize_url,
            return_tokens=return_tokens,
            crypto=crypto,
        )

    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        return {
            "model_id": self.model_id,
            "params": (
                self.params.to_dict()
                if isinstance(self.params, BaseSchema)
                else self.params
            ),
            "project_id": self._client.default_project_id,
            "space_id": self._client.default_space_id,
            "validate": self._validate,
        }

    def _prepare_inference_payload(  # type: ignore[override]
        self,
        prompt: str,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        guardrails_granite_guardian_params: dict | None = None,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        payload: dict = {
            "model_id": self.model_id,
            "input": prompt,
        }

        if isinstance(crypto, BaseSchema):
            crypto = crypto.to_dict()

        if crypto:
            payload["crypto"] = crypto

        if guardrails:
            if (
                guardrails_hap_params is None
                and guardrails_granite_guardian_params is None
            ):
                guardrails_hap_params = dict(
                    input=True, output=True
                )  # HAP enabled if guardrails = True

            for guardrail_type, guardrails_params in zip(
                ("hap", "pii", "granite_guardian"),
                (
                    guardrails_hap_params,
                    guardrails_pii_params,
                    guardrails_granite_guardian_params,
                ),
            ):
                if guardrails_params is not None:
                    if "moderations" not in payload:
                        payload["moderations"] = {}
                    payload["moderations"].update(
                        {
                            guardrail_type: self._update_moderations_params(
                                guardrails_params
                            )
                        }
                    )

        if params is not None:
            parameters = params

            if isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        elif self.params is not None:
            self.params = cast(dict | TextGenParameters, self.params)
            parameters = deepcopy(self.params)

            if isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

            if isinstance(parameters, dict):
                parameters = self._validate_and_overwrite_params(
                    parameters, TextGenParameters()
                )
        else:
            parameters = None

        if parameters:
            payload["parameters"] = parameters

        if (
            "parameters" in payload
            and GenTextParamsMetaNames.DECODING_METHOD in payload["parameters"]
        ):
            if isinstance(
                payload["parameters"][GenTextParamsMetaNames.DECODING_METHOD],
                DecodingMethods,
            ):
                payload["parameters"][GenTextParamsMetaNames.DECODING_METHOD] = payload[
                    "parameters"
                ][GenTextParamsMetaNames.DECODING_METHOD].value

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        return payload

    def _prepare_chat_payload(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        context: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> dict:
        payload: dict = {
            "model_id": self.model_id,
            "messages": messages,
        }

        if params is not None:
            parameters = params

            if isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        elif self.params is not None:
            self.params = cast(dict | TextChatParameters, self.params)
            parameters = deepcopy(self.params)

            if isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

            if isinstance(parameters, dict):
                parameters = self._validate_and_overwrite_params(
                    parameters, TextChatParameters()
                )

        else:
            parameters = None

        if parameters:
            payload.update(parameters)

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if tool_choice_option:
            payload["tool_choice_option"] = tool_choice_option

        return payload

    def _send_chat_payload(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        generate_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        if isinstance(crypto, BaseSchema):
            crypto = crypto.to_dict()

        if crypto:
            payload["crypto"] = crypto

        post_params: dict[str, Any] = {
            "url": generate_url,
            "json": payload,
            "params": self._client._params(skip_for_create=True, skip_userfs=True),
            "headers": self._client._get_headers(),
        }

        response_scoring = self._post(self._client.httpx_client, **post_params)

        return self._handle_response(
            200,
            "chat",
            response_scoring,
            _field_to_hide="choices",
        )

    def _generate_chat_stream_with_url(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        chat_stream_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> Generator:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        if hasattr(self._client.httpx_client, "post_stream"):
            stream_function = self._client.httpx_client.post_stream
        else:
            stream_function = self._client.httpx_client.stream

        kw_args: dict = {
            "method": "POST",
            "url": chat_stream_url,
            "json": payload,
            "headers": self._client._get_headers(),
            "params": self._client._params(skip_for_create=True, skip_userfs=True),
        }

        with self._stream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = (
                    resp.iter_lines()
                    if isinstance(resp, httpx.Response)
                    else resp.iter_lines(decode_unicode=False)
                )
                for chunk in resp_iter:
                    if chunk.strip() == "event: error":
                        chunk = next(resp_iter)
                        field_name, _, response = chunk.partition(":")
                        raise WMLClientError(
                            error_msg="Error event occurred during chat stream.",
                            reason=response,
                        )

                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and response:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        yield parsed_response

            else:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    async def _agenerate_chat_stream_with_url(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        chat_stream_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> AsyncGenerator:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        if hasattr(self._client.async_httpx_client, "post_stream"):
            stream_function = self._client.async_httpx_client.post_stream
        else:
            stream_function = self._client.async_httpx_client.stream

        kw_args: dict = {
            "method": "POST",
            "url": chat_stream_url,
            "json": payload,
            "headers": await self._client._aget_headers(),
            "params": self._client._params(skip_for_create=True, skip_userfs=True),
        }

        async with self._astream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = resp.aiter_lines()

                async for chunk in resp_iter:
                    if chunk.strip() == "event: error":
                        chunk = await anext(resp_iter)
                        field_name, _, response = chunk.partition(":")
                        raise WMLClientError(
                            error_msg="Error event occurred during achat stream.",
                            reason=response,
                        )

                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and response:
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

    def _tokenize_with_url(
        self,
        prompt: str,
        tokenize_url: str,
        return_tokens: bool,
        crypto: dict | Crypto | None,
    ) -> dict:
        payload = self._prepare_inference_payload(prompt, crypto=crypto)
        payload.setdefault("parameters", {})["return_tokens"] = return_tokens

        post_params: dict[str, Any] = {
            "url": tokenize_url,
            "json": payload,
            "params": self._client._params(skip_for_create=True, skip_userfs=True),
            "headers": self._client._get_headers(),
        }
        if not isinstance(self._client.httpx_client, httpx.Client):
            post_params["_retry_status_codes"] = _RETRY_STATUS_CODES

        response_scoring = self._post(self._client.httpx_client, **post_params)

        if response_scoring.status_code == 404:
            raise WMLClientError("Tokenize is not supported for this release")

        return self._handle_response(200, "tokenize", response_scoring)
