import os
import sys

import gitlab

PROJECT_ID = int(os.getenv("CI_PROJECT_ID", 19378))
GITLAB_ENDPOINT = os.getenv('GITLAB_ENDPOINT')


def get_gitlab_handle():
    return gitlab.Gitlab(f"https://{GITLAB_ENDPOINT}", private_token=os.getenv("RO_API_TOKEN"))


def is_sucess():
    pipelines = (
        get_gitlab_handle().projects.get(PROJECT_ID).pipelines.list(ref="main", scope="finished")
    )

    most_recent = pipelines[0]

    return most_recent.attributes['status'] == 'success'


if __name__ == "__main__":
    sys.exit(not int(is_sucess()))
