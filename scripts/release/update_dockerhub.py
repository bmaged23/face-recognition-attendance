"""Synchronize the public image overview after qualified publication."""

import json
import os
from pathlib import Path
import urllib.request


def request(url, value, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "JWT " + token
    message = urllib.request.Request(
        url, json.dumps(value).encode(), headers, method="PATCH" if token else "POST"
    )
    with urllib.request.urlopen(message, timeout=45) as response:
        return json.load(response)


def main():
    login = request(
        "https://hub.docker.com/v2/users/login/",
        {"username": "bmaged23", "password": os.environ["DOCKERHUB_TOKEN"]},
    )
    root = Path(__file__).resolve().parents[2]
    result = request(
        "https://hub.docker.com/v2/repositories/bmaged23/face-recognition-attendance/",
        {
            "is_private": False,
            "description": "GPU AdaFace recognition and attendance with a persistent gallery, HTTP API and evaluated releases.",
            "full_description": (root / "deployment/dockerhub/README.md").read_text(),
        },
        login["token"],
    )
    if result.get("is_private"):
        raise RuntimeError("Docker Hub repository is not public")
    print("Public Docker Hub overview synchronized; source and API links included.")


if __name__ == "__main__":
    main()
