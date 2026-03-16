#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    TypeAlias,
    cast,
)

from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.metanames import TrainingConfigurationMetaNames
from ibm_watsonx_ai.utils import (
    TRAINING_RUN_DETAILS_TYPE,
    StatusLogger,
    print_text_header_h1,
    print_text_header_h2,
)
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid, get_from_json
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure, WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

logging.getLogger("lomond").setLevel(logging.CRITICAL)
ListType: TypeAlias = list

if TYPE_CHECKING:
    from httpx import Response
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient


class Training(WMLResource):
    """Train new models."""

    ConfigurationMetaNames = TrainingConfigurationMetaNames()

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    @staticmethod
    def _get_status_process_response(details: dict, training_id: str) -> dict:
        """Helper method for `(a)get_status` methods.
        Return training status retrieved from details.
        """
        if details is not None:
            return WMLResource._get_required_element_from_dict(
                details, "details", ["entity", "status"], dict
            )
        else:
            raise WMLClientError(
                "Getting trained model status failed. Unable to get model details for training_id: '{}'.".format(
                    training_id
                )
            )

    def get_status(self, training_id: str | None = None, **kwargs: Any) -> dict:
        """Get the status of a created training.

        :param training_id: ID of the training
        :type training_id: str

        :return: training_status
        :rtype: dict

        **Example:**

        .. code-block:: python

            training_status = client.training.get_status(training_id)
        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        Training._validate_type(training_id, "training_id", str, True)

        details = self.get_details(
            training_id, _internal=True, _is_fine_tuning=_is_fine_tuning
        )

        return self._get_status_process_response(details, training_id)

    async def aget_status(self, training_id: str, **kwargs: Any) -> dict:
        """Get the status of a created training asynchronously.

        :param training_id: ID of the training
        :type training_id: str

        :return: training_status
        :rtype: dict

        **Example:**

        .. code-block:: python

            training_status = await client.training.aget_status(training_id)
        """
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        Training._validate_type(training_id, "training_id", str, True)

        details = await self.aget_details(
            training_id, _internal=True, _is_fine_tuning=_is_fine_tuning
        )

        return self._get_status_process_response(details, training_id)

    @staticmethod
    def _get_details_prepare_query_params(
        training_type: str | None = None,
        state: str | None = None,
        tag_value: str | list[str] | None = None,
        training_definition_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Helper method for `(a)get_details` methods.
        Return query parameters for a request to get training details.
        """
        query_params: dict | None = {
            param_name: param_value
            for param_name, param_value in (
                ("type", training_type),
                ("state", state),
                ("tag.value", tag_value),
                ("training_definition_id", training_definition_id),
            )
            if param_value is not None
        }
        # note: If query params is an empty dict convert it back to None value
        query_params = query_params or None

        return query_params

    def _get_url(self, is_fine_tuning: bool, training_id: str | None = None) -> str:
        """Return proper URL for a request based on `is_fine_tuning`."""
        if training_id is None:
            return (
                self._client._href_definitions.get_fine_tunings_href()
                if is_fine_tuning
                else self._client._href_definitions.get_trainings_href()
            )
        else:
            return (
                self._client._href_definitions.get_fine_tuning_href(training_id)
                if is_fine_tuning
                else self._client._href_definitions.get_training_href(training_id)
            )

    def get_details(
        self,
        training_id: str | None = None,
        limit: int | None = None,
        asynchronous: Literal[True, False] = False,
        get_all: Literal[True, False] = False,
        training_type: str | None = None,
        state: str | None = None,
        tag_value: str | list[str] | None = None,
        training_definition_id: str | None = None,
        _internal: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of training(s). If training_id is not specified, the metadata of all model spaces are returned.

        :param training_id: unique ID of the training
        :type training_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param training_type: filter the fetched list of trainings based on the training type ["pipeline" or "experiment"]
        :type training_type: str, optional

        :param state: filter the fetched list of training based on their state:
            [`queued`, `running`, `completed`, `failed`]
        :type state: str, optional

        :param tag_value: filter the fetched list of training based on their tag value
        :type tag_value: str, list[str], optional

        :param training_definition_id: filter the fetched trainings that are using the given training definition
        :type training_definition_id: str, optional

        :return: metadata of training(s)
        :rtype:
          - **dict** - if training_id is not None
          - **{"resources": [dict]}** - if training_id is None

        **Examples**

        .. code-block:: python

            training_run_details = client.training.get_details(training_id)
            training_runs_details = client.training.get_details()
            training_runs_details = client.training.get_details(limit=100)
            training_runs_details = client.training.get_details(
                limit=100, get_all=True
            )
            training_runs_details = []
            for entry in client.training.get_details(
                limit=100, asynchronous=True, get_all=True
            ):
                training_runs_details.extend(entry)

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=True
        )

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_id, "training_id", str, False)

        url = self._get_url(kwargs.get("_is_fine_tuning", False))

        if training_id is None:
            query_params = self._get_details_prepare_query_params(
                training_type, state, tag_value, training_definition_id
            )

            return self._get_artifact_details(
                base_url=url,
                id=training_id,
                limit=limit,
                resource_name="trained models",
                _async=asynchronous,
                _all=get_all,
                query_params=query_params,
            )
        else:
            return self._get_artifact_details(url, training_id, limit, "trained models")

    async def aget_details(
        self,
        training_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        training_type: str | None = None,
        state: str | None = None,
        tag_value: str | list[str] | None = None,
        training_definition_id: str | None = None,
        _internal: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of training(s) asynchronously. If training_id is not specified, the metadata of all model spaces are returned.

        :param training_id: unique ID of the training
        :type training_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param training_type: filter the fetched list of trainings based on the training type ["pipeline" or "experiment"]
        :type training_type: str, optional

        :param state: filter the fetched list of training based on their state:
            [`queued`, `running`, `completed`, `failed`]
        :type state: str, optional

        :param tag_value: filter the fetched list of training based on their tag value
        :type tag_value: str, list[str], optional

        :param training_definition_id: filter the fetched trainings that are using the given training definition
        :type training_definition_id: str, optional

        :return: metadata of training(s)
        :rtype:
          - **dict** - if training_id is not None
          - **{"resources": [dict]}** - if training_id is None

        **Examples**

        .. code-block:: python

            training_run_details = await client.training.aget_details(training_id)
            training_runs_details = await client.training.aget_details()
            training_runs_details = await client.training.aget_details(limit=100)
            training_runs_details = await client.training.aget_details(
                limit=100, get_all=True
            )
            training_runs_details = []
            for entry in await client.training.aget_details(
                limit=100, asynchronous=True, get_all=True
            ):
                training_runs_details.extend(entry)

        """
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_id, "training_id", str, False)

        url = self._get_url(kwargs.get("_is_fine_tuning", False))

        if training_id is None:
            query_params = self._get_details_prepare_query_params(
                training_type, state, tag_value, training_definition_id
            )

            return await self._aget_artifact_details(  # type: ignore[call-overload]
                base_url=url,
                id=training_id,
                limit=limit,
                resource_name="trained models",
                _async=asynchronous,
                _all=get_all,
                query_params=query_params,
            )
        else:
            return await self._aget_artifact_details(
                url, training_id, limit, "trained models"
            )

    @staticmethod
    def get_href(training_details: dict) -> str:
        """Get the training href from the training details.

        :param training_details: metadata of the created training
        :type training_details: dict

        :return: training href
        :rtype: str

        **Example:**

        .. code-block:: python

            training_details = client.training.get_details(training_id)
            run_url = client.training.get_href(training_details)
        """

        Training._validate_type(training_details, "training_details", dict, True)
        if "id" in training_details.get("metadata", {}):
            training_id = WMLResource._get_required_element_from_dict(
                training_details, "training_details", ["metadata", "id"], str
            )
            return "/ml/v4/trainings/" + training_id
        else:
            Training._validate_type_of_details(
                training_details, TRAINING_RUN_DETAILS_TYPE
            )
            return WMLResource._get_required_element_from_dict(
                training_details, "training_details", ["metadata", "href"], str
            )

    @staticmethod
    def get_id(training_details: dict) -> str:
        """Get the training ID from the training details.

        :param training_details: metadata of the created training
        :type training_details: dict

        :return: unique ID of the training
        :rtype: str

        **Example:**

        .. code-block:: python

            training_details = client.training.get_details(training_id)
            training_id = client.training.get_id(training_details)

        """
        Training._validate_type(training_details, "training_details", dict, True)
        return WMLResource._get_required_element_from_dict(
            training_details, "training_details", ["metadata", "id"], str
        )

    def _run_validate(self, meta_props: dict, asynchronous: bool) -> None:
        """Helper method for `(a)run` methods.
        Validate `meta_props` and `asynchronous` passed to `(a)run` method.
        Also, check if either space or project ID is set.
        """
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Training._validate_type(meta_props, "meta_props", dict, True)
        Training._validate_type(asynchronous, "asynchronous", bool, True)

        self.ConfigurationMetaNames._validate(meta_props)

    def _get_training_configuration_meta(self, meta_props: dict) -> dict:
        """Helper method for `(a)run` methods.
        Return training configuration metadata for a request to run training.
        """
        training_configuration_metadata = {
            "training_data_references": meta_props[
                self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
            ],
            "results_reference": meta_props[
                self.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE
            ],
        }

        if self.ConfigurationMetaNames.TEST_DATA_REFERENCES in meta_props:
            training_configuration_metadata["test_data_references"] = meta_props[
                self.ConfigurationMetaNames.TEST_DATA_REFERENCES
            ]

        if self.ConfigurationMetaNames.TEST_OUTPUT_DATA in meta_props:
            training_configuration_metadata["test_output_data"] = meta_props[
                self.ConfigurationMetaNames.TEST_OUTPUT_DATA
            ]

        if self.ConfigurationMetaNames.TAGS in meta_props:
            training_configuration_metadata["tags"] = meta_props[
                self.ConfigurationMetaNames.TAGS
            ]

        if self.ConfigurationMetaNames.PROMPT_TUNING in meta_props:
            training_configuration_metadata["prompt_tuning"] = meta_props[
                self.ConfigurationMetaNames.PROMPT_TUNING
            ]

        if self.ConfigurationMetaNames.FINE_TUNING in meta_props:
            training_configuration_metadata["parameters"] = meta_props[
                self.ConfigurationMetaNames.FINE_TUNING
            ]

        if self.ConfigurationMetaNames.AUTO_UPDATE_MODEL in meta_props:
            training_configuration_metadata["auto_update_model"] = meta_props[
                self.ConfigurationMetaNames.AUTO_UPDATE_MODEL
            ]

        # TODO remove when training service starts copying such data on their own

        training_configuration_metadata["name"] = meta_props[
            self.ConfigurationMetaNames.NAME
        ]
        training_configuration_metadata["description"] = meta_props[
            self.ConfigurationMetaNames.DESCRIPTION
        ]

        if self.ConfigurationMetaNames.PIPELINE in meta_props:
            training_configuration_metadata["pipeline"] = meta_props[
                self.ConfigurationMetaNames.PIPELINE
            ]
        if self.ConfigurationMetaNames.EXPERIMENT in meta_props:
            training_configuration_metadata["experiment"] = meta_props[
                self.ConfigurationMetaNames.EXPERIMENT
            ]
        if self.ConfigurationMetaNames.MODEL_DEFINITION in meta_props:
            training_configuration_metadata["model_definition"] = meta_props[
                self.ConfigurationMetaNames.MODEL_DEFINITION
            ]
        if self.ConfigurationMetaNames.SPACE_UID in meta_props:
            training_configuration_metadata["space_id"] = meta_props[
                self.ConfigurationMetaNames.SPACE_UID
            ]
        if "type" in meta_props:
            training_configuration_metadata["type"] = meta_props["type"]

        # _check_if_either_is_set is performed on the beginning of processing function call
        if self._client.default_space_id is not None:
            training_configuration_metadata["space_id"] = self._client.default_space_id
        else:
            training_configuration_metadata["project_id"] = (
                self._client.default_project_id
            )

        return training_configuration_metadata

    def _run_prepare_query_params(self) -> dict:
        """Helper method for `(a)run` methods.
        Return query parameters for training run.
        """
        params = self._client._params()
        if "space_id" in params.keys():
            params.pop("space_id")
        if "project_id" in params.keys():
            params.pop("project_id")

        if self._client.ICP_PLATFORM_SPACES:
            if "userfs" in params.keys():
                params.pop("userfs")

        return params

    def _print_final_training_result(
        self, state: str, status: dict, trained_model_id: str, run_details: dict
    ) -> None:
        """Helper method for `(a)run` methods.
        Print final training result.
        """
        if "completed" in state:
            print(
                "\nTraining of '{}' finished successfully.".format(
                    str(trained_model_id)
                )
            )
        else:
            print(
                "\nTraining of '{}' failed with status: '{}'.".format(
                    trained_model_id, str(status)
                )
            )

        self._logger.debug("Response({}): {}".format(state, run_details))

    def run(self, meta_props: dict, asynchronous: bool = True, **kwargs: Any) -> dict:
        """Create a new Machine Learning training.

        :param meta_props: metadata of the training configuration. To see available meta names, use:

            .. code-block:: python

                client.training.ConfigurationMetaNames.show()

        :type meta_props: dict
        :param asynchronous:
            * `True` - training job is submitted and progress can be checked later
            * `False` - method will wait till job completion and print training stats
        :type asynchronous: bool, optional

        :return: metadata of the training created
        :rtype: dict

        .. note::

            You can provide one of the following values for training:
             * client.training.ConfigurationMetaNames.EXPERIMENT
             * client.training.ConfigurationMetaNames.PIPELINE
             * client.training.ConfigurationMetaNames.MODEL_DEFINITION

        **Examples**

        Example of meta_props for creating a training run in IBM Cloud Pak® for Data version 3.0.1 or above:

        .. code-block:: python

            metadata = {
                client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
                client.training.ConfigurationMetaNames.DESCRIPTION: 'Hand-written Digit Recognition Training',
                client.training.ConfigurationMetaNames.PIPELINE: {
                    "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                    "rev": "12",
                    "model_type": "string",
                    "data_bindings": [
                        {
                            "data_reference_name": "string",
                            "node_id": "string"
                        }
                    ],
                    "nodes_parameters": [
                        {
                            "node_id": "string",
                            "parameters": {}
                        }
                    ],
                    "hardware_spec": {
                        "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                        "rev": "12",
                        "name": "string",
                        "num_nodes": "2"
                    }
                },
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
                    'type': 's3',
                    'connection': {},
                    'location': {'href': 'v2/assets/asset1233456'},
                    'schema': { 'id': 't1', 'name': 'Tasks', 'fields': [ { 'name': 'duration', 'type': 'number' } ]}
                }],
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    'id' : 'string',
                    'connection': {
                        'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
                        'access_key_id': '***',
                        'secret_access_key': '***'
                    },
                    'location': {
                        'bucket': 'wml-dev-results',
                        'path' : "path"
                    }
                    'type': 's3'
                }
            }

        """
        self._run_validate(meta_props, asynchronous)

        training_configuration_metadata = self._get_training_configuration_meta(
            meta_props
        )

        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)
        url = self._get_url(_is_fine_tuning)

        params = self._run_prepare_query_params()

        response_train_post = self._client.httpx_client.post(
            url=url,
            json=training_configuration_metadata,
            params=params,
            headers=self._client._get_headers(),
        )

        run_details = self._handle_response(201, "training", response_train_post)

        trained_model_id = self.get_id(run_details)

        if asynchronous is True:
            return run_details
        else:
            print_text_header_h1("Running '{}'".format(trained_model_id))

            status = self.get_status(trained_model_id, _is_fine_tuning=_is_fine_tuning)
            state = status["state"]

            with StatusLogger(state) as status_logger:
                while state not in ["error", "completed", "canceled", "failed"]:
                    time.sleep(5)
                    status = self.get_status(
                        trained_model_id, _is_fine_tuning=_is_fine_tuning
                    )
                    state = status["state"]
                    status_logger.log_state(state)

            self._print_final_training_result(
                state, status, trained_model_id, run_details
            )

            return self.get_details(
                trained_model_id, _internal=True, _is_fine_tuning=_is_fine_tuning
            )

    async def arun(
        self, meta_props: dict, asynchronous: bool = True, **kwargs: Any
    ) -> dict:
        """Create a new Machine Learning training asynchronously.

        :param meta_props: metadata of the training configuration. To see available meta names, use:

            .. code-block:: python

                client.training.ConfigurationMetaNames.show()

        :type meta_props: dict
        :param asynchronous:
            * `True` - training job is submitted and progress can be checked later
            * `False` - method will wait till job completion and print training stats
        :type asynchronous: bool, optional

        :return: metadata of the training created
        :rtype: dict

        .. note::

            You can provide one of the following values for training:
             * client.training.ConfigurationMetaNames.EXPERIMENT
             * client.training.ConfigurationMetaNames.PIPELINE
             * client.training.ConfigurationMetaNames.MODEL_DEFINITION

        **Examples**

        Example of meta_props for creating a training run in IBM Cloud Pak® for Data version 3.0.1 or above:

        .. code-block:: python

            metadata = {
                client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
                client.training.ConfigurationMetaNames.DESCRIPTION: 'Hand-written Digit Recognition Training',
                client.training.ConfigurationMetaNames.PIPELINE: {
                    "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                    "rev": "12",
                    "model_type": "string",
                    "data_bindings": [
                        {
                            "data_reference_name": "string",
                            "node_id": "string"
                        }
                    ],
                    "nodes_parameters": [
                        {
                            "node_id": "string",
                            "parameters": {}
                        }
                    ],
                    "hardware_spec": {
                        "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                        "rev": "12",
                        "name": "string",
                        "num_nodes": "2"
                    }
                },
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
                    'type': 's3',
                    'connection': {},
                    'location': {'href': 'v2/assets/asset1233456'},
                    'schema': { 'id': 't1', 'name': 'Tasks', 'fields': [ { 'name': 'duration', 'type': 'number' } ]}
                }],
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    'id' : 'string',
                    'connection': {
                        'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
                        'access_key_id': '***',
                        'secret_access_key': '***'
                    },
                    'location': {
                        'bucket': 'wml-dev-results',
                        'path' : "path"
                    }
                    'type': 's3'
                }
            }

        """
        self._run_validate(meta_props, asynchronous)

        training_configuration_metadata = self._get_training_configuration_meta(
            meta_props
        )

        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)
        url = self._get_url(_is_fine_tuning)

        params = self._run_prepare_query_params()

        response_train_post = await self._client.async_httpx_client.post(
            url=url,
            json=training_configuration_metadata,
            params=params,
            headers=await self._client._aget_headers(),
        )

        run_details = self._handle_response(201, "training", response_train_post)

        trained_model_id = self.get_id(run_details)

        if asynchronous is True:
            return run_details
        else:
            print_text_header_h1("Running '{}'".format(trained_model_id))

            status = await self.aget_status(
                trained_model_id, _is_fine_tuning=_is_fine_tuning
            )
            state = status["state"]

            with StatusLogger(state) as status_logger:
                while state not in ["error", "completed", "canceled", "failed"]:
                    await asyncio.sleep(5)
                    status = await self.aget_status(
                        trained_model_id, _is_fine_tuning=_is_fine_tuning
                    )
                    state = status["state"]
                    status_logger.log_state(state)

            self._print_final_training_result(
                state, status, trained_model_id, run_details
            )

            return await self.aget_details(
                trained_model_id, _internal=True, _is_fine_tuning=_is_fine_tuning
            )

    def list(
        self,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> DataFrame | Iterator | ListType:
        """List stored trainings in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: pandas.DataFrame with listed experiments
        :rtype: pandas.DataFrame

        **Examples**

        .. code-block:: python

            client.training.list()
            training_runs_df = client.training.list(limit=100)
            training_runs_df = client.training.list(limit=100, get_all=True)
            training_runs_df = []
            for entry in client.training.list(
                limit=100, asynchronous=True, get_all=True
            ):
                training_runs_df.extend(entry)
        """
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        def preprocess_details(details: dict) -> DataFrame | ListType:
            resources = details["resources"]
            values = [
                (
                    m["metadata"].get("id") or m["metadata"].get("guid"),
                    m["entity"]["status"]["state"],
                    m["metadata"]["created_at"],
                )
                for m in resources
            ]

            return self._list(
                values,
                ["ID (training)", "STATE", "CREATED"],
                limit=None,
                sort_by=None,
            )

        if asynchronous:
            return (
                preprocess_details(details)
                for details in self.get_details(
                    limit=limit,
                    asynchronous=asynchronous,
                    get_all=get_all,
                    _internal=True,
                )
            )
        else:
            details = self.get_details(limit=limit, get_all=get_all, _internal=True)
            table = preprocess_details(details)
            return table

    def list_intermediate_models(
        self, training_id: str | None = None, **kwargs: Any
    ) -> None:
        """Print the intermediate_models in a table format.

        :param training_id: ID of the training
        :type training_id: str

        .. note::

            This method is not supported for IBM Cloud Pak® for Data.

        **Example:**

        .. code-block:: python

            client.training.list_intermediate_models()

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )

        # For CP4D, check if either space or project ID is set
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                "This method is not supported for IBM Cloud Pak® for Data. "
            )

        self._client._check_if_either_is_set()
        details = self.get_details(training_id, _internal=True)
        # if status is completed then only lists global_output else display message saying "state value"
        training_state = details["entity"]["status"]["state"]
        if training_state == "completed":
            if (
                "metrics" in details["entity"]["status"]
                and details["entity"]["status"].get("metrics") is not None
            ):
                metrics_list = details["entity"]["status"]["metrics"]
                new_list = []
                for ml in metrics_list:
                    if "context" in ml and "intermediate_model" in ml["context"]:
                        name = ml["context"]["intermediate_model"].get("name", "")
                        if "location" in ml["context"]["intermediate_model"]:
                            path = ml["context"]["intermediate_model"]["location"].get(
                                "model", ""
                            )
                        else:
                            path = ""
                    else:
                        name = ""
                        path = ""

                    accuracy = ml["ml_metrics"].get("training_accuracy", "")
                    F1Micro = round(ml["ml_metrics"].get("training_f1_micro", 0), 2)
                    F1Macro = round(ml["ml_metrics"].get("training_f1_macro", 0), 2)
                    F1Weighted = round(
                        ml["ml_metrics"].get("training_f1_weighted", 0), 2
                    )
                    logLoss = round(ml["ml_metrics"].get("training_neg_log_loss", 0), 2)
                    PrecisionMicro = round(
                        ml["ml_metrics"].get("training_precision_micro", 0), 2
                    )
                    PrecisionWeighted = round(
                        ml["ml_metrics"].get("training_precision_weighted", 0), 2
                    )
                    PrecisionMacro = round(
                        ml["ml_metrics"].get("training_precision_macro", 0), 2
                    )
                    RecallMacro = round(
                        ml["ml_metrics"].get("training_recall_macro", 0), 2
                    )
                    RecallMicro = round(
                        ml["ml_metrics"].get("training_recall_micro", 0), 2
                    )
                    RecallWeighted = round(
                        ml["ml_metrics"].get("training_recall_weighted", 0), 2
                    )
                    createdAt = details["metadata"]["created_at"]
                    new_list.append(
                        [
                            name,
                            path,
                            accuracy,
                            F1Micro,
                            F1Macro,
                            F1Weighted,
                            logLoss,
                            PrecisionMicro,
                            PrecisionMacro,
                            PrecisionWeighted,
                            RecallMicro,
                            RecallMacro,
                            RecallWeighted,
                            createdAt,
                        ]
                    )
                    new_list.append([])

                from tabulate import tabulate

                header = [
                    "NAME",
                    "PATH",
                    "Accuracy",
                    "F1Micro",
                    "F1Macro",
                    "F1Weighted",
                    "LogLoss",
                    "PrecisionMicro",
                    "PrecisionMacro",
                    "PrecisionWeighted",
                    "RecallMicro",
                    "RecallMacro",
                    "RecallWeighted",
                    "CreatedAt",
                ]
                table = tabulate([header] + new_list)

                print(table)
            else:
                print(
                    " There is no intermediate model metrics are available for this training id. "
                )
        else:
            self._logger.debug("state is not completed")

    def _cancel_prepare_query_params(self, hard_delete: bool) -> dict:
        """Helper method for `(a)cancel` methods.
        Return query parameters for cancelling training run.
        """
        params = self._client._params()

        if hard_delete is True:
            params.update({"hard_delete": "true"})

        return params

    def _cancel_process_response(self, response: Response) -> Literal["SUCCESS"]:
        """Helper method for `(a)cancel` methods.
        Return training status retrieved from details.
        """
        if (
            response.status_code == 400
            and response.text is not None
            and "Job already completed with state" in response.text
        ):
            print(
                "Job is not running currently. Please use 'hard_delete=True' parameter to force delete"
                " completed or canceled training runs."
            )
            return "SUCCESS"
        else:
            return cast(
                Literal["SUCCESS"],
                self._handle_response(204, "trained model deletion", response, False),
            )

    def cancel(
        self,
        training_id: str | None = None,
        hard_delete: bool = False,
        **kwargs: Any,
    ) -> Literal["SUCCESS"]:
        """Cancel a training that is currently running. This method can delete metadata
        details of a completed or canceled training run when `hard_delete` parameter is set to `True`.

        :param training_id: ID of the training
        :type training_id: str

        :param hard_delete: specify `True` or `False`:

            * `True` - to delete the completed or canceled training run
            * `False` - to cancel the currently running training run
        :type hard_delete: bool, optional

        :return: status "SUCCESS" if cancellation is successful
        :rtype: Literal["SUCCESS"]

        :raises WMLClientError: if cancellation (deletion) failed

        **Example:**

        .. code-block:: python

            client.training.cancel(training_id)
        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_id, "training_id", str, True)

        params = self._cancel_prepare_query_params(hard_delete)

        train_endpoint = self._get_url(_is_fine_tuning, training_id)

        response_delete = self._client.httpx_client.delete(
            url=train_endpoint,
            headers=self._client._get_headers(),
            params=params,
        )

        return self._cancel_process_response(response_delete)

    async def acancel(
        self, training_id: str, hard_delete: bool = False, **kwargs: Any
    ) -> Literal["SUCCESS"]:
        """Cancel a training that is currently running asynchronously. This method can delete metadata
        details of a completed or canceled training run when `hard_delete` parameter is set to `True`.

        :param training_id: ID of the training
        :type training_id: str

        :param hard_delete: specify `True` or `False`:

            * `True` - to delete the completed or canceled training run
            * `False` - to cancel the currently running training run
        :type hard_delete: bool, optional

        :return: status "SUCCESS" if cancellation is successful
        :rtype: Literal["SUCCESS"]

        :raises WMLClientError: if cancellation (deletion) failed

        **Example:**

        .. code-block:: python

            await client.training.acancel(training_id)
        """
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_id, "training_id", str, True)

        params = self._cancel_prepare_query_params(hard_delete)

        train_endpoint = self._get_url(_is_fine_tuning, training_id)

        response_delete = await self._client.async_httpx_client.delete(
            url=train_endpoint,
            headers=await self._client._aget_headers(),
            params=params,
        )

        return self._cancel_process_response(response_delete)

    def _prepare_connection_to_results_file(
        self, run_details: dict, results_file_key: str
    ) -> DataConnection:
        results_reference = run_details["entity"]["results_reference"]
        location = get_from_json(
            run_details, ["entity", "results_reference", "location"]
        )

        if "file_name" in location:
            results_reference["location"]["file_name"] = location[results_file_key]
        elif "path" in location:
            results_reference["location"]["path"] = location[results_file_key]

        results_conn = DataConnection.from_dict(results_reference)
        results_conn.set_client(self._client)

        return results_conn

    def _print_result_if_finished(self, run_details: dict, training_id: str) -> bool:
        """Helper method for `(a)monitor_logs` methods.
        Check training state and print result if training is finished. If so, return `True`, otherwise return `False`.
        """
        state = run_details["entity"]["status"]["state"]

        print_text_header_h1(
            "Log monitor started for training run: " + str(training_id)
        )

        if state in {"completed", "error", "failed", "canceled"}:
            results_conn = self._prepare_connection_to_results_file(
                run_details, "training_log"
            )
            result = results_conn.read(raw=True, binary=True)
            print(cast(bytes, result).decode("utf-8"))

            return True

        return False

    @staticmethod
    def _print_not_found_msg_if_404(exception: Exception) -> bool:
        """Helper method for `(a)monitor_logs` and `(a)monitor_metrics` methods.
        Check if '404' present in exception. If so, print message and return `True`, otherwise return `False`.
        """
        if "404" in str(exception.args[1]):
            print(
                "Could not find the training run details for the given training run id."
            )
            return True
        return False

    def monitor_logs(self, training_id: str | None = None, **kwargs: Any) -> None:
        """Print the logs of a training created.

        :param training_id: training ID
        :type training_id: str

        **Example:**

        .. code-block:: python

            client.training.monitor_logs(training_id)

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )
        Training._validate_type(training_id, "training_id", str, True)

        try:
            run_details = self.get_details(training_id, _internal=True)
        except ApiRequestFailure as ex:
            if self._print_not_found_msg_if_404(ex):
                return
            else:
                raise ex

        if not self._print_result_if_finished(run_details, training_id):
            self._monitor_connection(
                training_id,
                token=self._client.token,
                process_event=Training._process_logs,
            )

        print_text_header_h2("Log monitor done.")

    async def amonitor_logs(self, training_id: str) -> None:
        """Print the logs of a training created asynchronously.

        :param training_id: training ID
        :type training_id: str

        **Example:**

        .. code-block:: python

            await client.training.amonitor_logs(training_id)

        """
        Training._validate_type(training_id, "training_id", str, True)

        try:
            run_details = await self.aget_details(training_id, _internal=True)
        except ApiRequestFailure as ex:
            if self._print_not_found_msg_if_404(ex):
                return
            else:
                raise ex

        if not self._print_result_if_finished(run_details, training_id):
            self._monitor_connection(
                training_id,
                token=await self._client._auth_method.aget_token(),
                process_event=Training._process_logs,
            )

        print_text_header_h2("Log monitor done.")

    @staticmethod
    def _process_logs(event: Any) -> None:
        if event.name != "text":
            return

        text = json.loads(event.text)
        message = get_from_json(text, ["entity", "status", "message"])

        if not message:
            return

        if "level" in message and "text" in message:
            print(f"{message['level'].upper()}: {message['text']}")
        else:
            print(message)

    @staticmethod
    def _process_metrics(event: Any) -> None:
        if event.name != "text":
            return

        text = json.loads(event.text)
        metrics = get_from_json(text, ["entity", "status", "metrics"])

        if metrics:
            print(metrics[0])

    def _monitor_connection(
        self,
        training_id: str,
        *,
        token: str,
        process_event: Callable[[Any], None] = lambda _: None,
    ) -> None:
        from lomond import WebSocket

        ws_param = self._client._params()
        ws_param["version"] = "2025-02-27"

        url = self._credentials.url if self._credentials.url is not None else ""
        monitor_endpoint = (
            url.replace("https", "wss")
            + "/ml/v4/trainings/"
            + training_id
            + "?"
            + "&".join(f"{k}={v}" for k, v in ws_param.items())
        )

        websocket = WebSocket(monitor_endpoint)

        try:
            websocket.add_header(
                b"Authorization",
                ("Bearer " + token).encode("utf-8"),
            )
        except Exception:
            websocket.add_header(
                b"Authorization",
                ("bearer " + token).encode("utf-8"),
            )

        for event in websocket:
            process_event(event)

        websocket.close()

    def _monitor_metrics(self, training_id: str, run_details: dict) -> None:
        """Print the metrics of a created training."""

        status = run_details["entity"]["status"]["state"]

        print_text_header_h1(
            "Metric monitor started for training run: " + str(training_id)
        )

        if (
            status == "completed"
            or status == "error"
            or status == "failed"
            or status == "canceled"
        ):
            for m in self._get_metrics_from_details(run_details):
                print(m)
        else:
            self._monitor_connection(
                training_id,
                token=self._client.token,
                process_event=Training._process_metrics,
            )

        print_text_header_h2("Metric monitor done.")

    def monitor_metrics(self, training_id: str | None = None, **kwargs: Any) -> None:
        """Print the metrics of a created training.

        :param training_id: ID of the training
        :type training_id: str

        **Example:**

        .. code-block:: python

            client.training.monitor_metrics(training_id)
        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )

        Training._validate_type(training_id, "training_id", str, True)
        try:
            run_details = self.get_details(training_id, _internal=True)
        except ApiRequestFailure as ex:
            if self._print_not_found_msg_if_404(ex):
                return
            else:
                raise ex

        self._monitor_metrics(training_id, run_details)

    async def amonitor_metrics(self, training_id: str) -> None:
        """Print the metrics of a created training asynchronously.

        :param training_id: ID of the training
        :type training_id: str

        **Example:**

        .. code-block:: python

            await client.training.amonitor_metrics(training_id)
        """
        Training._validate_type(training_id, "training_id", str, True)
        try:
            run_details = await self.aget_details(training_id, _internal=True)
        except ApiRequestFailure as ex:
            if self._print_not_found_msg_if_404(ex):
                return
            else:
                raise ex

        self._monitor_metrics(training_id, run_details)

    @staticmethod
    def _get_metrics_from_details(run_details: dict) -> ListType[dict]:
        status = get_from_json(run_details, ["entity", "status"])
        if "metrics" in status:
            return status["metrics"]
        else:
            if "metrics" in run_details:
                return run_details["metrics"]
            else:
                raise WMLClientError(
                    "No metrics details are available for the given training_id"
                )

    def get_metrics(
        self, training_id: str | None = None, **kwargs: Any
    ) -> ListType[dict]:
        """Get metrics of a training run.

        :param training_id: ID of the training
        :type training_id: str

        :return: metrics of the training run
        :rtype: list of dict

        **Example:**

        .. code-block:: python

            training_status = client.training.get_metrics(training_id)

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )

        Training._validate_type(training_id, "training_id", str, True)

        return self._get_metrics_from_details(self.get_details(training_id))

    async def aget_metrics(self, training_id: str) -> ListType[dict]:
        """Get metrics of a training run asynchronously.

        :param training_id: ID of the training
        :type training_id: str

        :return: metrics of the training run
        :rtype: list of dict

        **Example:**

        .. code-block:: python

            training_status = await client.training.aget_metrics(training_id)

        """
        Training._validate_type(training_id, "training_id", str, True)

        return self._get_metrics_from_details(await self.aget_details(training_id))

    def _get_experiment_asset_id_to_delete(
        self, tags: ListType[str], training_details: dict, trainings_with_tags: dict
    ) -> str | None:
        """Helper method for `(a)delete` methods.
        Return the experiment asset id unless there are still other trainings assigned to it. If so, return `None`.
        """
        experiment_asset_id: str | None = None

        if trainings_with_tags["resources"]:
            return experiment_asset_id

        if tags[0] == "autoai" or tags[0].startswith("dsx-project"):
            experiment_asset_id = training_details["entity"]["pipeline"]["id"]
        elif tags[0] == "prompt_tuning":
            experiment_asset_id = tags[1].split(".", maxsplit=1)[1]
        else:
            self._logger.warning(
                "Unknown training type, skipping asset deletion. Training details: %s",
                training_details,
            )

        return experiment_asset_id

    def delete(self, training_id: str) -> None:
        """Delete a training run. If the experiment asset exists and contains only this training, delete the asset.

        :param training_id: ID of the training
        :type training_id: str

        **Example:**

        .. code-block:: python

            client.training.delete(training_id)
        """

        training_details = self.get_details(training_id)
        tags: list[str] = get_from_json(training_details, ["metadata", "tags"], [])

        self.cancel(training_id, hard_delete=True)

        if not tags:
            return

        # Delete the asset unless there are still other trainings assigned to it
        trainings_with_tags = self.get_details(tag_value=tags)
        if (
            experiment_asset_id := self._get_experiment_asset_id_to_delete(
                tags, training_details, trainings_with_tags
            )
        ) is None:
            return

        self._client.repository.delete(experiment_asset_id)

    async def adelete(self, training_id: str) -> None:
        """Delete a training run asynchronously. If the experiment asset exists and contains only this training, delete the asset.

        :param training_id: ID of the training
        :type training_id: str

        **Example:**

        .. code-block:: python

            await client.training.adelete(training_id)
        """

        training_details = await self.aget_details(training_id)
        tags: list[str] = get_from_json(training_details, ["metadata", "tags"], [])

        await self.acancel(training_id, hard_delete=True)

        if not tags:
            return

        # Delete the asset unless there are still other trainings assigned to it
        trainings_with_tags = await self.aget_details(tag_value=tags)
        if (
            experiment_asset_id := self._get_experiment_asset_id_to_delete(
                tags, training_details, trainings_with_tags
            )
        ) is None:
            return

        await self._client.repository.adelete(experiment_asset_id)
