#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    Literal,
    cast,
)

import httpx

from ibm_watsonx_ai.foundation_models.schema import (
    BaseSchema,
    TextChatParameters,
    TextGenParameters,
)
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.utils.utils import _check_model_state
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.utils.utils import get_from_json
from ibm_watsonx_ai.wml_client_error import (
    MissingValue,
    PromptVariablesError,
    UnsupportedOperation,
    WMLClientError,
)

from .base_model_inference import BaseModelInference

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient

__all__ = ["DeploymentModelInference"]


class DeploymentModelInference(BaseModelInference):
    """Base abstract class for the model interface."""

    def __init__(
        self,
        *,
        deployment_id: str,
        api_client: APIClient,
        params: dict | TextChatParameters | TextGenParameters | None = None,
        validate: bool = True,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
    ) -> None:
        self.deployment_id = deployment_id

        self.params = params
        DeploymentModelInference._validate_type(
            params, "params", [dict, TextChatParameters, TextGenParameters], False, True
        )

        self._client = api_client
        self._validate = validate
        if self._validate:
            self._deployment_details = self._client.deployments.get_details(
                deployment_id=self.deployment_id, _silent=True
            )
            _check_model_state(
                self._client,
                get_from_json(self._deployment_details, ["entity", "base_model_id"]),
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
        """Get deployment's details

        :return: details of model or deployment
        :rtype: dict
        """
        return self._client.deployments.get_details(
            deployment_id=self.deployment_id, _silent=True
        )

    def chat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> dict:
        self._validate_type(messages, "messages", list, True)

        infer_chat_url = self._client._href_definitions.get_fm_deployment_chat_href(
            deployment_id=self.deployment_id
        )

        return self._send_deployment_chat_payload(
            deployment_chat_url=infer_chat_url,
            messages=messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
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
        self._validate_type(messages, "messages", list, True)

        infer_chat_url = (
            self._client._href_definitions.get_fm_deployment_chat_stream_href(
                deployment_id=self.deployment_id
            )
        )

        return self._generate_deployment_chat_stream_with_url(
            deployment_chat_stream_url=infer_chat_url,
            messages=messages,
            params=params,
            context=context,
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
    ) -> dict:
        self._validate_type(messages, "messages", list, True)

        infer_chat_url = self._client._href_definitions.get_fm_deployment_chat_href(
            deployment_id=self.deployment_id
        )

        return await self._asend_deployment_chat_payload(
            deployment_chat_url=infer_chat_url,
            messages=messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    async def achat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
        context: str | None = None,
    ) -> AsyncGenerator:
        self._validate_type(messages, "messages", list, True)

        infer_chat_url = (
            self._client._href_definitions.get_fm_deployment_chat_stream_href(
                deployment_id=self.deployment_id
            )
        )

        return self._agenerate_deployment_chat_stream_with_url(
            deployment_chat_stream_url=infer_chat_url,
            messages=messages,
            params=params,
            context=context,
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
    ) -> dict | list[dict] | Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        prompt_required = self._deployment_type_validation(
            params, validate_prompt_variables
        )
        self._validate_type(
            prompt, "prompt", [str, list], prompt_required, raise_error_for_list=True
        )
        self._validate_type(guardrails_hap_params, "guardrails_hap_params", dict, False)
        self._validate_type(guardrails_pii_params, "guardrails_pii_params", dict, False)
        self._validate_type(
            guardrails_granite_guardian_params,
            "guardrails_granite_guardian_params",
            dict,
            False,
        )
        generate_text_url = (
            self._client._href_definitions.get_fm_deployment_generation_href(
                deployment_id=self.deployment_id, item="text"
            )
        )

        prompts = prompt if isinstance(prompt, list) else [prompt]

        payloads = [
            self._prepare_inference_payload(
                prompt=p,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
                guardrails_granite_guardian_params=guardrails_granite_guardian_params,
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
    ) -> dict:
        prompt_required = self._deployment_type_validation(
            params, validate_prompt_variables
        )
        self._validate_type(prompt, "prompt", str, prompt_required)
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
        generate_text_url = (
            self._client._href_definitions.get_fm_deployment_generation_href(
                deployment_id=self.deployment_id, item="text"
            )
        )

        payload = self._prepare_inference_payload(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

        return await self._asend_inference_payload(
            payload=payload,
            generate_url=generate_text_url,
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
        will generate a completion text as a async generator.
        """

        prompt_required = self._deployment_type_validation(
            params, validate_prompt_variables
        )

        self._validate_type(prompt, "prompt", str, prompt_required)
        self._validate_type(guardrails_hap_params, "guardrails_hap_params", dict, False)
        self._validate_type(guardrails_pii_params, "guardrails_pii_params", dict, False)
        self._validate_type(
            guardrails_granite_guardian_params,
            "guardrails_granite_guardian_params",
            dict,
            False,
        )

        generate_stream_url = (
            self._client._href_definitions.get_fm_deployment_generation_stream_href(
                deployment_id=self.deployment_id
            )
        )

        payload = self._prepare_inference_payload(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

        return self._agenerate_stream_with_url(
            payload=payload,
            generate_url=generate_stream_url,
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
        prompt_required = self._deployment_type_validation(
            params, validate_prompt_variables
        )
        self._validate_type(prompt, "prompt", str, prompt_required)
        self._validate_type(guardrails_hap_params, "guardrails_hap_params", dict, False)
        self._validate_type(guardrails_pii_params, "guardrails_pii_params", dict, False)
        self._validate_type(
            guardrails_granite_guardian_params,
            "guardrails_granite_guardian_params",
            dict,
            False,
        )
        generate_text_stream_url = (
            self._client._href_definitions.get_fm_deployment_generation_stream_href(
                deployment_id=self.deployment_id
            )
        )

        payload = self._prepare_inference_payload(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

        return self._generate_stream_with_url(
            payload=payload,
            generate_stream_url=generate_text_stream_url,
            raw_response=raw_response,
        )

    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        return {
            "deployment_id": self.deployment_id,
            "params": (
                self.params.to_dict()
                if isinstance(self.params, BaseSchema)
                else self.params
            ),
            "project_id": self._client.default_project_id,
            "space_id": self._client.default_space_id,
            "validate": self._validate,
        }

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
        payload: dict = {
            "input": prompt,
        }

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
        payload: dict = {"messages": messages}

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
            if self._client.ICP_PLATFORM_SPACES and self._client.CPD_version <= 5.1:
                raise WMLClientError(
                    "Text chat parameters for Deployment Model Inference are not supported for IBM Cloud Pak® for Data 5.1 release and earlier."
                )
            payload.update(parameters)

        if context:
            payload["context"] = context
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if tool_choice_option:
            payload["tool_choice_option"] = tool_choice_option

        return payload

    def _send_deployment_chat_payload(
        self,
        deployment_chat_url: str,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> dict:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        post_params: dict[str, Any] = {
            "url": deployment_chat_url,
            "json": payload,
            "params": self._client._params(skip_for_create=True, skip_userfs=True),
            "headers": self._client._get_headers(),
        }

        response_scoring = self._post(self._client.httpx_client, **post_params)

        if response_scoring.status_code == 404:
            raise UnsupportedOperation(
                Messages.get_message(message_id="chat_deployment_not_supported")
            )

        return self._handle_response(
            200,
            "chat",
            response_scoring,
            _field_to_hide="choices",
        )

    async def _asend_deployment_chat_payload(
        self,
        deployment_chat_url: str,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> dict:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        post_params: dict[str, Any] = {
            "url": deployment_chat_url,
            "json": payload,
            "params": self._client._params(skip_for_create=True, skip_userfs=True),
            "headers": await self._client._aget_headers(),
        }

        response = await self._apost(self._client.async_httpx_client, **post_params)

        if response.status_code == 404:
            raise UnsupportedOperation(
                Messages.get_message(message_id="chat_deployment_not_supported")
            )

        return self._handle_response(
            200,
            "achat",
            response,
            _field_to_hide="choices",
        )

    def _deployment_type_validation(
        self, params: dict | TextGenParameters | None, validate_prompt_variables: bool
    ) -> bool:
        prompt_required = False
        prompt_id = (
            get_from_json(self._deployment_details, ["entity", "prompt_template", "id"])
            if self._validate
            else None
        )
        if prompt_id is None and self._validate:
            prompt_required = True
        elif prompt_id is not None:
            if validate_prompt_variables:
                from ibm_watsonx_ai.foundation_models.prompts import (
                    PromptTemplateManager,
                )

                prompt_template = PromptTemplateManager(
                    api_client=self._client
                ).load_prompt(prompt_id)

                if not hasattr(
                    prompt_template, "input_variables"
                ):  # validate only if it is prompt template
                    return prompt_required

                prompt_template.input_variables = cast(
                    dict, prompt_template.input_variables
                )
                # params may be not specified but instead self.params is specified
                parameters = params if params is not None else self.params

                if isinstance(parameters, BaseSchema):
                    parameters = parameters.to_dict()

                template_inputs: dict | None = (
                    parameters.get("prompt_variables")
                    if parameters is not None
                    else None
                )
                if (
                    template_inputs is None
                    and prompt_template.input_variables is not None
                ):
                    raise MissingValue(
                        "prompt_variables",
                        reason=(
                            "Prompt template contains input variables but "
                            "`prompt_variables` parameter not provided in `params`."
                        ),
                    )
                if (
                    input_variables := set(prompt_template.input_variables.keys())
                ) != set(template_inputs.keys()):
                    raise PromptVariablesError(
                        str(input_variables - set(template_inputs.keys()))
                    )

        return prompt_required

    def _generate_deployment_chat_stream_with_url(
        self,
        deployment_chat_stream_url: str,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> Generator:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            context=context,
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
            "url": deployment_chat_stream_url,
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
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and response:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        yield parsed_response

            elif resp.status_code == 404:
                raise UnsupportedOperation(
                    Messages.get_message(message_id="chat_deployment_not_supported")
                )

            else:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    async def _agenerate_deployment_chat_stream_with_url(
        self,
        deployment_chat_stream_url: str,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto", "required"] | None = None,
    ) -> AsyncGenerator:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            context=context,
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
            "url": deployment_chat_stream_url,
            "json": payload,
            "headers": await self._client._aget_headers(),
            "params": self._client._params(skip_for_create=True, skip_userfs=True),
        }

        async with self._astream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = resp.aiter_lines()

                async for chunk in resp_iter:
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
