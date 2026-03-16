#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ibm_watsonx_ai.foundation_models.schema import TextClassificationParameters
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.utils import get_from_json
from ibm_watsonx_ai.wml_client_error import (
    InvalidMultipleArguments,
    InvalidValue,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    import pandas

    from ibm_watsonx_ai import APIClient, Credentials


class TextClassification(WMLResource):
    """Instantiate the text classification service.

    :param credentials: credentials to the watsonx.ai instance
    :type credentials: Credentials, optional

    :param project_id: ID of the project, defaults to None
    :type project_id: str, optional

    :param space_id: ID of the space, defaults to None
    :type space_id: str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required, defaults to None
    :type api_client: APIClient, optional

    :raises InvalidMultipleArguments: raised when neither `api_client` nor `credentials` alongside `space_id` or `project_id` are provided

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models.classifications import (
            TextClassification,
        )

        text_classification = TextClassification(
            credentials=Credentials(
                api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com"
            ),
            project_id="*****",
        )

    """

    def __init__(
        self,
        credentials: Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        api_client: APIClient | None = None,
    ) -> None:
        if credentials is not None:
            from ibm_watsonx_ai import APIClient

            self._client = APIClient(credentials)
        elif api_client is not None:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if space_id is not None:
            self._client.set.default_space(space_id)
        elif project_id is not None:
            self._client.set.default_project(project_id)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )

        if self._client.ICP_PLATFORM_SPACES and self._client.CPD_version < 5.3:
            raise WMLClientError(
                Messages.get_message(">= 5.3", message_id="invalid_cpd_version")
            )

        super().__init__(__name__, self._client)

    def run_job(
        self,
        document_reference: DataConnection,
        parameters: TextClassificationParameters | dict,
        custom: dict | None = None,
    ) -> dict:
        """Start a request to classify text in the document.

        :param document_reference: reference to the document in the bucket from which text will be classified
        :type document_reference: DataConnection

        :param parameters: the parameters for the text classification
        :type parameters: TextClassificationParameters or dict

        :param custom: user defined properties for the text classification, defaults to None
        :type custom: dict, optional

        :return: text classification response
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.helpers import DataConnection, S3Location
            from ibm_watsonx_ai.foundation_models.schema import (
                TextClassificationParameters,
                ClassificationMode,
                OCRMode,
            )

            document_reference = DataConnection(
                connection_asset_id="<connection_id>",
                location=S3Location(bucket="<bucket_name>", path="path/to/file"),
            )

            response = text_classification.run_job(
                document_reference=document_reference,
                parameters=TextClassificationParameters(
                    ocr_mode=OCRMode.ENABLED,
                    classification_mode=ClassificationMode.EXACT,
                    auto_rotation_correction=True,
                    languages=["en"],
                    semantic_config=TextClassificationSemanticConfig(
                        schemas_merge_strategy=SchemasMergeStrategy.MERGE,
                        schemas=[...],
                    ),
                ),
                custom={},
            )

        """
        TextClassification._validate_type(
            document_reference, "document_reference", DataConnection, True
        )
        TextClassification._validate_type(
            parameters, "parameters", [TextClassificationParameters, dict], True
        )
        TextClassification._validate_type(custom, "custom", dict, False)
        if isinstance(parameters, TextClassificationParameters):
            parameters = parameters.to_dict()

        payload: dict[str, Any] = {
            "document_reference": document_reference.to_dict(),
            "parameters": parameters,
        }

        if custom:
            payload["custom"] = custom

        if self._client.default_space_id is not None:
            payload["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            payload["project_id"] = self._client.default_project_id

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_text_classifications_href(),
            json=payload,
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
        )

        return self._handle_response(201, "run_job", response)

    def list_jobs(self, limit: int | None = None) -> pandas.DataFrame:
        """List text classification jobs. If limit is None, all jobs will be listed.

        :param limit: limit number of fetched records, defaults to None
        :type limit: int, optional

        :return: text classification jobs information as a pandas DataFrame
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            text_classification.list_jobs()

        """
        import pandas

        columns_mapping = {
            "metadata.id": "CLASSIFICATION_JOB_ID",
            "metadata.created_at": "CREATED",
            "entity.results.status": "STATUS",
        }

        details = self.get_job_details(limit=limit)

        df_details = (
            pandas.json_normalize(details["resources"])
            .reindex(columns=list(columns_mapping.keys()))
            .rename(columns=columns_mapping)
        )

        return df_details

    def get_results(self, classification_job_id: str) -> dict:
        """Get the text classification results.

        :param classification_job_id: ID of text classification job
        :type classification_job_id: str

        :return: text classification job results
        :rtype: dict

        **Example:**

        .. code-block:: python

            results = text_classification.get_results(
                classification_job_id="<CLASSIFICATION_JOB_ID>"
            )

        """
        self._validate_type(classification_job_id, "classification_job_id", str, True)

        job_details = self.get_job_details(classification_job_id)

        return get_from_json(job_details, ["entity", "results"])

    def get_status(self, classification_job_id: str) -> str:
        """Get the text classification status.

        :param classification_job_id: ID of text classification job
        :type classification_job_id: str

        :return: text classification job status, possible values: [submitted, uploading, running, downloading, downloaded, completed, failed]
        :rtype: str

        **Example:**

        .. code-block:: python

            status = text_classification.get_status(
                classification_job_id="<CLASSIFICATION_JOB_ID>"
            )

        """
        self._validate_type(classification_job_id, "classification_job_id", str, True)

        job_details = self.get_job_details(classification_job_id)

        return get_from_json(job_details, ["entity", "results", "status"])

    def get_job_details(
        self, classification_job_id: str | None = None, limit: int | None = None
    ) -> dict:
        """Return text classification job details. If `classification_job_id` is None, return the details of all text classification jobs.

        :param classification_job_id: ID of the text classification job, defaults to None
        :type classification_job_id: str, optional

        :param limit: limit number of fetched records, defaults to None
        :type limit: int, optional

        :return: details of the text classification job
        :rtype: dict

        **Example:**

        .. code-block:: python

            text_classification.get_job_details(
                classification_job_id="<CLASSIFICATION_JOB_ID>"
            )

        """
        self._validate_type(classification_job_id, "classification_job_id", str, False)

        if classification_job_id is not None:
            response = self._client.httpx_client.get(
                url=self._client._href_definitions.get_text_classification_href(
                    classification_job_id
                ),
                params=self._client._params(skip_userfs=True),
                headers=self._client._get_headers(),
            )
        elif limit is None or 1 <= limit <= 200:
            params = self._client._params(skip_userfs=True)
            if limit is not None:
                params["limit"] = limit

            response = self._client.httpx_client.get(
                url=self._client._href_definitions.get_text_classifications_href(),
                params=params,
                headers=self._client._get_headers(),
            )
        else:
            raise InvalidValue(
                value_name="limit",
                reason=f"The given value {limit} is not in between 1 and 200",
            )

        return self._handle_response(200, "get_job_details", response)

    def delete_job(self, classification_job_id: str) -> Literal["SUCCESS"]:
        """Delete a text classification job.

        :param classification_job_id: ID of text classification job
        :type classification_job_id: str

        :return: "SUCCESS" if the deletion succeeds
        :rtype: str
        :raises WMLClientError: if deletion failed

        **Example:**

        .. code-block:: python

            text_classification.delete_job(
                classification_job_id="<CLASSIFICATION_JOB_ID>"
            )

        """
        self._validate_type(classification_job_id, "classification_job_id", str, True)

        params = self._client._params(skip_userfs=True)
        params["hard_delete"] = True

        response = self._client.httpx_client.delete(
            url=self._client._href_definitions.get_text_classification_href(
                classification_job_id
            ),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete_job", response)  # type: ignore[return-value]

    def cancel_job(self, classification_job_id: str) -> Literal["SUCCESS"]:
        """Cancel a text classification job.

        :param classification_job_id: ID of text classification job
        :type classification_job_id: str

        :return: "SUCCESS" if the cancel succeeds
        :rtype: str
        :raises WMLClientError: if cancellation failed

        **Example:**

        .. code-block:: python

            text_classification.cancel_job(
                classification_job_id="<CLASSIFICATION_JOB_ID>"
            )

        """
        self._validate_type(classification_job_id, "classification_job_id", str, True)

        response = self._client.httpx_client.delete(
            url=self._client._href_definitions.get_text_classification_href(
                classification_job_id
            ),
            params=self._client._params(skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "cancel_job", response)  # type: ignore[return-value]

    @classmethod
    def get_job_id(cls, classification_details: dict) -> str:
        """Get the unique ID of a stored classification request.

        :param classification_details: metadata of the stored classification
        :type classification_details: dict

        :return: unique ID of the stored classification request
        :rtype: str

        **Example:**

        .. code-block:: python

            classification_details = text_classification.run_job(...)
            classification_job_id = text_classification.get_job_id(
                classification_details
            )

        """
        cls._validate_type(classification_details, "classification_details", dict, True)

        return cls._get_required_element_from_dict(
            classification_details, "classification_details", ["metadata", "id"], str
        )
