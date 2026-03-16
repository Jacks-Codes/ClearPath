#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Literal,
    cast,
)
from warnings import warn

from ibm_watsonx_ai._wrappers.httpx_wrapper import (
    _get_async_httpx_client,
    _get_httpx_client,
    _httpx_transport_params,
)
from ibm_watsonx_ai.foundation_models.schema import (
    Crypto,
    TextChatParameters,
    TextGenParameters,
)
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.utils.utils import is_lib_installed
from ibm_watsonx_ai.wml_client_error import (
    InvalidMultipleArguments,
    MissingExtension,
    ParamOutOfRange,
    UnsupportedOperation,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

from .base_model_inference import BaseModelInference
from .deployment_model_inference import DeploymentModelInference
from .fm_model_inference import FMModelInference

if TYPE_CHECKING:
    from langchain_ibm import WatsonxLLM

    from ibm_watsonx_ai import APIClient, Credentials


class ModelInference(WMLResource):
    """Instantiate the model interface.

    .. hint::
        To use the ModelInference class with LangChain, use the :func:`WatsonxLLM <langchain_ibm.WatsonxLLM>` wrapper.

    :param model_id: type of model to use
    :type model_id: str, optional

    :param deployment_id: ID of tuned model's deployment
    :type deployment_id: str, optional

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: Credentials or dict, optional

    :param params: parameters to use during request generation
    :type params: dict, TextGenParameters, TextChatParameters, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of the following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool | str | Path, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    :param validate: Model ID validation, defaults to True
    :type validate: bool, optional

    :param max_retries: number of retries performed when request was not successful and status code is in retry_status_codes, defaults to 10
    :type max_retries: int, optional

    :param delay_time: delay time to retry request, factor in exponential backoff formula: wx_delay_time * pow(2.0, attempt), defaults to 0.5s
    :type delay_time: float, optional

    :param retry_status_codes: list of status codes which will be considered for retry mechanism, defaults to [429, 503, 504, 520]
    :type retry_status_codes: list[int], optional

    .. note::
        * You must provide one of these parameters: [``model_id``, ``deployment_id``]
        * When the ``credentials`` parameter is passed, you must provide one of these parameters: [``project_id``, ``space_id``].
        * For any "chat" method you can also pass tools from :ref:`Toolkit<fm_toolkit>` converted with
          :func:`convert_to_watsonx_tool() <ibm_watsonx_ai.foundation_models.utils.convert_to_watsonx_tool>`.

    .. hint::
        You can copy the project_id from the Project's Manage tab (Project -> Manage -> General -> Details).

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        from ibm_watsonx_ai.foundation_models.utils.enums import (
            DecodingMethods,
        )
        from ibm_watsonx_ai import APIClient

        # Client initialization
        credentials = Credentials(
            api_key="<USER API KEY>", url="https://us-south.ml.cloud.ibm.com"
        )
        project_id = "<USER PROJECT ID>"
        api_client = APIClient(credentials=credentials, project_id=project_id)

        # To display example params enter
        GenParams().get_example_values()

        generate_params = {GenParams.MAX_NEW_TOKENS: 25}

        model_inference = ModelInference(
            api_client.foundation_models.TextModels.GRANITE_3_2_8B_INSTRUCT,
            params=generate_params,
            credentials=credentials,
            project_id=project_id,
        )

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials

        deployment_inference = ModelInference(
            deployment_id="<ID of deployed model>",
            credentials=Credentials(
                api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com"
            ),
            project_id="*****",
        )

    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        deployment_id: str | None = None,
        params: dict | TextChatParameters | TextGenParameters | None = None,
        credentials: dict | Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: bool | str | Path | None = None,
        api_client: APIClient | None = None,
        validate: bool = True,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if "persistent_connection" in kwargs:
            warn(
                "The `persistent_connection` parameter is no longer supported and any value provided for this parameter will be ignored."
            )
            kwargs.pop("persistent_connection")

        self._model_id = model_id
        if isinstance(self._model_id, Enum):
            self._model_id = self._model_id.value

        self._deployment_id = deployment_id

        if self._model_id and self._deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="Both arguments were provided.",
            )
        elif not self._model_id and not self._deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="None of the arguments were provided.",
            )

        self.params = params
        ModelInference._validate_type(
            params, "params", [dict, TextChatParameters, TextGenParameters], False, True
        )
        if isinstance(verify, str):
            verify = Path(verify)

        if credentials:
            from ibm_watsonx_ai import APIClient

            self.set_api_client(APIClient(credentials, verify=verify))
        elif api_client:
            self.set_api_client(api_client)
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if space_id:
            self._client.set.default_space(space_id)
        elif project_id:
            self._client.set.default_project(project_id)

        # 1. When `deployment_id` is provided and `validate` is set to True:
        #    space, project or api_client with default project/space is required.
        # 2. When `model_id` is provided: space, project or `api_client` is required.
        #    If `api_client` is provided, we don't check if it has default
        #    space/project set due to compatibility with lightweight clusters.
        is_project_or_space_required = bool(
            (model_id and not api_client) or (deployment_id and validate)
        )
        if is_project_or_space_required and not (
            self._client.default_project_id or self._client.default_space_id
        ):
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )

        self._inference: BaseModelInference
        if self._model_id:
            self._inference = FMModelInference(
                model_id=self._model_id,
                api_client=self._client,
                params=self.params,
                validate=validate,
                max_retries=max_retries,
                delay_time=delay_time,
                retry_status_codes=retry_status_codes,
            )
        else:
            self._deployment_id = cast(str, self._deployment_id)
            self._inference = DeploymentModelInference(
                deployment_id=self._deployment_id,
                api_client=self._client,
                params=self.params,
                validate=validate,
                max_retries=max_retries,
                delay_time=delay_time,
                retry_status_codes=retry_status_codes,
            )

        self._transport_params = _httpx_transport_params(self._client)

        WMLResource.__init__(self, __name__, self._client)

    @property
    def model_id(self) -> str | None:
        """Get the model ID from the underlying inference object.

        :return: model ID if using Foundation Model, None otherwise
        :rtype: str | None
        """
        if isinstance(self._inference, FMModelInference):
            return self._inference.model_id
        return None

    @model_id.setter
    def model_id(self, value: str) -> None:
        """Set the model ID in the underlying inference object.

        :param value: new model ID value
        :type value: str

        :raises WMLClientError: if trying to set model_id on a deployment-based inference
        """
        if isinstance(self._inference, FMModelInference):
            self._inference.model_id = value
            self._model_id = value
        else:
            error_message = (
                "Cannot set 'model_id' on a deployment-based ModelInference. "
                "This instance was initialized with 'deployment_id'."
            )
            raise WMLClientError(error_message)

    @property
    def deployment_id(self) -> str | None:
        """Get the deployment ID from the underlying inference object.

        :return: deployment ID if using Deployed Model, None otherwise
        :rtype: str | None
        """
        if isinstance(self._inference, DeploymentModelInference):
            return self._inference.deployment_id
        return None

    @deployment_id.setter
    def deployment_id(self, value: str) -> None:
        """Set the deployment ID in the underlying inference object.

        :param value: new deployment ID value
        :type value: str

        :raises WMLClientError: if trying to set deployment_id on a model-based inference
        """
        if isinstance(self._inference, DeploymentModelInference):
            self._inference.deployment_id = value
            self._deployment_id = value
        else:
            error_message = (
                "Cannot set 'deployment_id' on a model-based ModelInference. "
                "This instance was initialized with 'model_id'."
            )
            raise WMLClientError(error_message)

    def get_details(self) -> dict:
        """Get the details of a model interface

        :return: details of the model or deployment
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_inference.get_details()

        """
        return self._inference.get_details()

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
        """
        Given a list of messages comprising a conversation, the model will return a response.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto", "required"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :param crypto: configuration for tenant-level encryption
        :type crypto: dict, Crypto, optional

        :return: scoring result containing generated chat content.
        :rtype: dict

        **Examples:**

        .. tab-set::

            .. tab-item:: Simple conversation

                .. code-block:: python

                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": "Who won the world series in 2020?",
                        },
                    ]
                    generated_response = model.chat(messages=messages)

                    # Print full response
                    print(generated_response)

                    # Print only content
                    print(generated_response["choices"][0]["message"]["content"])

            .. tab-item:: Audio/Video input

                .. code-block:: python

                    import base64

                    # Path to your MP3 file
                    file_path = "sample_audio_file.mp3"

                    # Read file as binary and encode to Base64
                    with open(file_path, "rb") as mp3_file:
                        encoded_bytes = base64.b64encode(mp3_file.read())

                    encoded_string = encoded_bytes.decode("utf-8")

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please extract the text from the mp3 file",
                                },
                                {
                                    "type": "input_audio",  # `input_video`
                                    "input_audio": {
                                        "data": encoded_string,
                                        "format": "mp3",
                                    },
                                },
                            ],
                        }
                    ]
                    generated_response = model.chat(messages=messages)

                    # Print full response
                    print(generated_response)

                    # Print only content
                    print(generated_response["choices"][0]["message"]["content"])


            .. tab-item:: Control messages

                .. note::

                    Control messages enable inclusion of enhanced reasoning to chat response content.
                    They are available only for IBM Granite reasoning models 3.2 and above.
                    For more information, visit: https://www.ibm.com/granite/docs/models/granite/

                .. code-block:: python

                    messages = [
                        {"role": "control", "content": "thinking"},
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": "Who won the world series in 2020?",
                        },
                    ]
                    generated_response = model.chat(messages=messages)

                    # Print full response
                    print(generated_response)

                    # Print only content
                    print(generated_response["choices"][0]["message"]["content"])
        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        if isinstance(self._inference, FMModelInference):
            chat_fn: Callable = partial(self._inference.chat, crypto=crypto)
        else:
            if crypto:
                raise WMLClientError(
                    "The `crypto` parameter is not supported for inferring a chat prompt deployment."
                )
            chat_fn = self._inference.chat

        return chat_fn(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
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
        """
        Given a list of messages comprising a conversation, the model will return a response in stream.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto", "required"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :return: scoring result containing generated chat content.
        :rtype: generator

        **Example:**

        .. code-block:: python

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
            ]
            generated_response = model.chat_stream(messages=messages)

            for chunk in generated_response:
                if chunk["choices"]:
                    print(
                        chunk["choices"][0]["delta"].get("content", ""),
                        end="",
                        flush=True,
                    )

        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        return self._inference.chat_stream(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
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
        """
        Given a list of messages comprising a conversation with a chat model in an asynchronous manner.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto", "required"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :param crypto: configuration for tenant-level encryption
        :type crypto: dict, Crypto, optional

        :return: scoring result containing generated chat content.
        :rtype: dict

        **Example:**

        .. code-block:: python

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
            ]
            generated_response = await model.achat(messages=messages)

            # Print all response
            print(generated_response)

            # Print only content
            print(generated_response["choices"][0]["message"]["content"])

        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        if isinstance(self._inference, FMModelInference):
            achat_fn: Callable = partial(self._inference.achat, crypto=crypto)
        else:
            if crypto:
                raise WMLClientError(
                    "The `crypto` parameter is not supported for inferring a chat prompt deployment."
                )
            achat_fn = self._inference.achat

        return await achat_fn(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
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
        """
        Given a list of messages comprising a conversation, the model will return a response in stream in an asynchronous manner.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto", "required"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :return: scoring result containing generated chat content.
        :rtype: AsyncGenerator

        **Example:**

        .. code-block:: python

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
            ]
            generated_response = await model.achat_stream(messages=messages)

            async for chunk in generated_response:
                if chunk["choices"]:
                    print(
                        chunk["choices"][0]["delta"].get("content", ""),
                        end="",
                        flush=True,
                    )

        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        return await self._inference.achat_stream(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
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
        """Generates a completion text as generated_text after getting a text prompt as input and parameters for the
        selected model (model_id) or deployment (deployment_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param concurrency_limit: number of requests to be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: prompt string or list of strings. If list of strings is passed, requests will be managed in parallel with the rate of concurrency_limit, defaults to None
        :type prompt: (str | list | None), optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP)
                        is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param async_mode: If True, yields results asynchronously (using a generator). In this case, both prompt and
                           generated text will be concatenated in the final response - under `generated_text`, defaults
                           to False
        :type async_mode: bool

        :param validate_prompt_variables: If True and `ModelInference` instance has been initialized with `validate=True`, prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool, optional

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :param crypto: configuration for tenant-level encryption
        :type crypto: dict, Crypto, optional

        :return: scoring result the contains the generated content
        :rtype: dict

        **Examples:**

        .. tab-set::

            .. tab-item:: Generate

                .. code-block:: python

                    q = "What is 1 + 1?"

                    generated_response = model_inference.generate(prompt=q)

                    print(generated_response["results"][0]["generated_text"])

            .. tab-item:: Generate with Params

                .. code-block:: python

                    from ibm_watsonx_ai.foundation_models.schema import (
                        TextGenParameters,
                        TextGenDecodingMethod,
                    )

                    q = "What is 1 + 1?"

                    generate_params = TextGenParameters(
                        decoding_method=TextGenDecodingMethod.SAMPLE,
                        temperature=0.8,
                        top_p=0.3,
                    )

                    generated_response = model_inference.generate(
                        prompt=q,
                        params=generate_params,
                    )

                    print(generated_response["results"][0]["generated_text"])

            .. tab-item:: Generate with Granite Guardian

                .. code-block:: python

                    from ibm_watsonx_ai.metanames import GenTextModerationsMetaNames

                    q = "<YOUR-QUESTION-TO-DETECT>"

                    guardrails_granite_guardian_params = {
                        GenTextModerationsMetaNames.INPUT: True,
                        GenTextModerationsMetaNames.THRESHOLD: 0.01,
                    }

                    generated_response = model_inference.generate(
                        prompt=q,
                        guardrails=True,
                        guardrails_granite_guardian_params=guardrails_granite_guardian_params,
                    )

        """
        self._validate_type(params, "params", [dict, TextGenParameters], False, True)
        self._validate_type(
            concurrency_limit,
            "concurrency_limit",
            [int, float],
            False,
            raise_error_for_list=True,
        )
        self._validate_type(crypto, "crypto", [dict, Crypto], False, True)

        if isinstance(concurrency_limit, float):  # convert float (ex. 10.0) to int
            concurrency_limit = int(concurrency_limit)

        if concurrency_limit > 10 or concurrency_limit < 1:
            raise ParamOutOfRange(
                param_name="concurrency_limit", value=concurrency_limit, min=1, max=10
            )
        if async_mode:
            warning_async_mode = (
                "In this mode, the results will be returned in the order in which the server returns the responses. "
                "Please notice that it does not support non-blocking requests scheduling. "
                "To use non-blocking native async inference method you may use `ModelInference.agenerate(...)`"
            )
            warn(warning_async_mode)

        if isinstance(self._inference, FMModelInference):
            generate_fn: Callable = partial(self._inference.generate, crypto=crypto)

        else:
            if crypto:
                raise WMLClientError(
                    "The `crypto` parameter is not supported for inferring a generate prompt deployment."
                )
            generate_fn = self._inference.generate

        return generate_fn(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            async_mode=async_mode,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    async def _agenerate_single(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> dict:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        self._validate_type(params, "params", [dict, TextGenParameters], False, True)

        return await self._inference._agenerate_single(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
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
        """Generates a stream as agenerate_stream after getting a text prompt as input and
        parameters for the selected model (model_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param prompt: prompt string, defaults to None
        :type prompt: str, optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :return: scoring result that contains the generated content
        :rtype: AsyncGenerator

        .. note::
            By default, only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        **Example:**

        .. code-block:: python

            q = "Write an epigram about the sun"
            generated_response = await model_inference.agenerate_stream(prompt=q)

            async for chunk in generated_response:
                print(chunk, end="", flush=True)

        """

        self._validate_type(params, "params", [dict, TextGenParameters], False, True)

        return await self._inference.agenerate_stream(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
        )

    def generate_text(
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = BaseModelInference.DEFAULT_CONCURRENCY_LIMIT,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
        crypto: dict | Crypto | None = None,
    ) -> str | list[str | dict] | dict:
        """Generates a completion text as generated_text after getting a text prompt as input and
        parameters for the selected model (model_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param concurrency_limit: number of requests to be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: prompt string or list of strings. If list of strings is passed, requests will be managed in parallel with the rate of concurrency_limit, defaults to None
        :type prompt: (str | list | None), optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param raw_response: returns the whole response object
        :type raw_response: bool, optional

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :param crypto: configuration for tenant-level encryption
        :type crypto: dict, Crypto, optional

        :return: generated content
        :rtype: str | list | dict

        .. note::
            By default, only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        **Examples:**

        .. tab-set::

            .. tab-item:: Generate Text

                .. code-block:: python

                    q = "What is 1 + 1?"

                    generated_text = model_inference.generate_text(prompt=q)

                    print(generated_text)

            .. tab-item:: Generate Text with Params

                .. code-block:: python

                    from ibm_watsonx_ai.foundation_models.schema import (
                        TextGenParameters,
                        TextGenDecodingMethod,
                    )

                    q = "What is 1 + 1?"

                    generate_params = TextGenParameters(
                        decoding_method=TextGenDecodingMethod.SAMPLE,
                        temperature=0.8,
                        top_p=0.3,
                    )

                    generated_text = model_inference.generate_text(
                        prompt=q, params=generate_params
                    )

                    print(generated_text)

        """
        metadata = ModelInference.generate(
            self,
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
            crypto=crypto,
        )
        if raw_response:
            return metadata  # type: ignore[return-value]
        else:
            if isinstance(prompt, list):
                return [
                    self._inference._return_guardrails_stats(single_response)[
                        "generated_text"
                    ]
                    for single_response in metadata
                ]
            else:
                return self._inference._return_guardrails_stats(cast(dict, metadata))[
                    "generated_text"
                ]

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
        """Generates a streamed text as generate_text_stream after getting a text prompt as input and
        parameters for the selected model (model_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param prompt: prompt string, defaults to None
        :type prompt: str, optional

        :param raw_response: yields the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :return: scoring result that contains the generated content
        :rtype: generator

        .. note::
            By default, only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        **Examples:**

        .. tab-set::

            .. tab-item:: Generate Text Stream

                .. code-block:: python

                    q = "Write an epigram about the sun"
                    generated_response = model_inference.generate_text_stream(prompt=q)

                    for chunk in generated_response:
                        print(chunk, end="", flush=True)

            .. tab-item:: Generate Text Stream with Params

                .. code-block:: python

                    from ibm_watsonx_ai.foundation_models.schema import (
                        TextGenParameters,
                        TextGenDecodingMethod,
                    )

                    q = "Write an epigram about the sun"

                    generate_params = TextGenParameters(
                        decoding_method=TextGenDecodingMethod.SAMPLE,
                        temperature=0.8,
                        top_p=0.3,
                    )

                    generated_response = model_inference.generate_text_stream(
                        prompt=q,
                        params=generate_params,
                    )

                    for chunk in generated_response:
                        print(chunk, end="", flush=True)

        """
        self._validate_type(params, "params", [dict, TextGenParameters], False, True)

        return self._inference.generate_text_stream(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    def tokenize(
        self,
        prompt: str,
        return_tokens: bool = False,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        """
        The text tokenize operation allows you to check the conversion of provided input to tokens for a given model.
        It splits text into words or sub-words, which then are converted to IDs through a look-up table (vocabulary).
        Tokenization allows the model to have a reasonable vocabulary size.

        .. note::
            The tokenization method is available only for base models and is not supported for deployments.

        :param prompt: prompt string, defaults to None
        :type prompt: str, optional

        :param return_tokens: parameter for text tokenization, defaults to False
        :type return_tokens: bool

        :param crypto: configuration for tenant-level encryption
        :type crypto: dict, Crypto, optional

        :return: result of tokenizing the input string
        :rtype: dict

        **Example:**

        .. code-block:: python

            q = "Write an epigram about the moon"
            tokenized_response = model_inference.tokenize(
                prompt=q, return_tokens=True
            )
            print(tokenized_response["result"])

        """

        if isinstance(self._inference, FMModelInference):
            return self._inference.tokenize(
                prompt=prompt, return_tokens=return_tokens, crypto=crypto
            )
        else:
            raise UnsupportedOperation(
                Messages.get_message(message_id="fm_tokenize_no_supported_deployment")
            )

    def to_langchain(self) -> WatsonxLLM:
        """

        :return: WatsonxLLM wrapper for watsonx foundation models
        :rtype: WatsonxLLM

        **Example:**

        .. code-block:: python

            from langchain import PromptTemplate
            from langchain.chains import LLMChain
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference

            from ibm_watsonx_ai import APIClient

            # Client initialization
            credentials = Credentials(
                api_key="<USER API KEY>", url="https://us-south.ml.cloud.ibm.com"
            )
            project_id = "<USER PROJECT ID>"
            api_client = APIClient(credentials=credentials, project_id=project_id)

            flan_ul2_model = ModelInference(
                api_client.foundation_models.TextModels.FLAN_UL2,
                credentials=Credentials(
                    api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com"
                ),
                project_id="*****",
            )

            prompt_template = "What color is the {flower}?"

            llm_chain = LLMChain(
                llm=flan_ul2_model.to_langchain(),
                prompt=PromptTemplate.from_template(prompt_template),
            )
            llm_chain.invoke("sunflower")

        .. code-block:: python

            from langchain import PromptTemplate
            from langchain.chains import LLMChain
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference

            deployed_model = ModelInference(
                deployment_id="<ID of deployed model>",
                credentials=Credentials(
                    api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com"
                ),
                space_id="*****",
            )

            prompt_template = "What color is the {car}?"

            llm_chain = LLMChain(
                llm=deployed_model.to_langchain(),
                prompt=PromptTemplate.from_template(prompt_template),
            )
            llm_chain.invoke("sunflower")

        """
        if not is_lib_installed(ext := "langchain-ibm"):
            raise MissingExtension(ext, extra_info="rag")
        from langchain_ibm import WatsonxLLM

        return WatsonxLLM(watsonx_model=self)

    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        return self._inference.get_identifying_params()

    def close_persistent_connection(self) -> None:
        """
        Calling this method closes the current `httpx.Client` and recreates a new `httpx.Client` with default values:
        timeout: httpx.Timeout(timeout=30 * 60, connect=10)
        limit: httpx.Limits(max_connections=10, max_keepalive_connections=10, keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY)
        """
        self._client.httpx_client.close()
        self._client.httpx_client = _get_httpx_client(transport=self._transport_params)

    def set_api_client(self, api_client: APIClient) -> None:
        """
        Set or refresh the APIClient object associated with ModelInference object.

        :param api_client: initialized APIClient object with a set project ID or space ID.
        :type api_client: APIClient, optional

        **Example:**

        .. code-block:: python

            api_client = APIClient(credentials=..., space_id=...)
            model_inference.set_api_client(api_client=api_client)

        """

        self._client = api_client
        if hasattr(self, "_inference"):
            self._inference._client = api_client

    async def agenerate(
        self,
        prompt: str | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
        crypto: dict | Crypto | None = None,
    ) -> dict:
        """Generate a response in an asynchronous manner.

        :param prompt: prompt string, defaults to None
        :type prompt: str | None, optional

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: dict | None, optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool, optional

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict | None, optional

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool, optional

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :param crypto: configuration for tenant-level encryption
        :type crypto: dict, Crypto, optional

        :return: raw response that contains the generated content
        :rtype: dict
        """
        self._validate_type(params, "params", dict, False)
        self._validate_type(crypto, "crypto", [dict, Crypto], False, True)

        if isinstance(self._inference, FMModelInference):
            agenerate_fn: Callable = partial(
                self._inference._agenerate_single, crypto=crypto
            )

        else:
            agenerate_fn = self._inference._agenerate_single
            if crypto:
                raise WMLClientError(
                    "The `crypto` parameter is not supported for inferring a generate prompt deployment."
                )

        return await agenerate_fn(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    async def aclose_persistent_connection(self) -> None:
        """
        Calling this method closes the current `httpx.AsyncClient` and recreates a new `httpx.AsyncClient` with default values:
        timeout: httpx.Timeout(timeout=30 * 60, connect=10)
        limit: httpx.Limits(max_connections=10, max_keepalive_connections=10, keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY)
        """
        await self._client.async_httpx_client.aclose()
        self._client.async_httpx_client = _get_async_httpx_client(
            transport=self._transport_params
        )

    def _return_guardrails_stats(  # Added for backward compatibility
        self, single_response: dict
    ) -> dict:
        return self._inference._return_guardrails_stats(single_response=single_response)
