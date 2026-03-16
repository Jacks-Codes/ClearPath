#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any

import ibm_watsonx_ai.messages

from .globalization_util import GlobalizationUtil


def get_message_dict(locale: str) -> dict[str, str]:
    file_name = "messages_" + locale + ".json"
    message_dict = {}
    path = Path(ibm_watsonx_ai.messages.__file__).parent
    messages = []
    # try to load the respective json file for the locale
    try:
        messages = json.loads((path / file_name).read_text())
    # load the english dictionary if the json file for the locale doesn't exist
    except Exception:
        try:
            return get_message_dict("en")
        except Exception:
            raise Exception(
                "An error occurred while trying to load the message json file for the {} locale. "
                "Make sure the json file exists and is located in the correct folder.".format(
                    locale
                )
            )

    messages_list = [{item["code"]: item["message"]} for item in messages]
    for i in range(len(messages_list)):
        message_dict.update(messages_list[i])
    return message_dict


MESSAGE_DICT = get_message_dict(GlobalizationUtil.get_language())


def replace_args_in_message(message: str, *args: Any) -> str:
    if args:
        varss = []
        for x in args:
            if x is not None and type(x) is not Exception:
                varss.append(x)
        if "{0}" in message:
            message = message.format(*varss)
        else:
            message = message % tuple(varss)
    return message


class Messages:
    @classmethod
    def get_message(cls, *args: Any, message_id: str) -> str:
        message = MESSAGE_DICT.get(message_id)

        if args and message:
            message = replace_args_in_message(message, *args)

        return str(message)
