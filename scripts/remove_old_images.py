import datetime
import json
import subprocess


def get_untagged_images(image_name: str):
    image_versions = []
    page = 1
    while True:
        result = subprocess.run(
            " ".join(
                [
                    "gh",
                    "api",
                    "-H",
                    '"Accept: application/vnd.github+json"',
                    f'"https://api.github.com/orgs/tinkoff-ai/packages/container/etna%2F{image_name}/versions?page={page}"',
                ]
            ),
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        parsed_result = json.loads(result.stdout.decode("utf-8"))
        if len(parsed_result) == 0:
            break
        else:
            image_versions += parsed_result
            page += 1
    # filter out tagged images
    image_versions = [image for image in image_versions if len(image["metadata"]["container"]["tags"]) == 0]
    return image_versions


def get_list_to_remove(leave_last_n_images: int, image_versions: list):
    image_versions = sorted(
        image_versions, key=lambda x: datetime.datetime.strptime(x["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    )
    return image_versions[:-leave_last_n_images]


def remove_images(image_versions: list):
    for image in image_versions:
        print(f"Removing {image['url']}")
        subprocess.run(
            " ".join(
                [
                    "echo -n |",
                    "gh",
                    "api",
                    "--method",
                    "DELETE",
                    "-H",
                    '"Accept: application/vnd.github+json"',
                    f'"{image["url"]}"',
                    "--input -",
                ]
            ),
            shell=True,
            check=True,
        )


def delete_pipe(image_name: str, leave_last_n_images: int):
    image_versions = get_untagged_images(image_name)
    image_versions_to_remove = get_list_to_remove(leave_last_n_images, image_versions)
    remove_images(image_versions_to_remove)


if __name__ == "__main__":
    delete_pipe("etna-cpu", 20)
    delete_pipe("etna-cuda-10.2", 20)
    delete_pipe("etna-cuda-11.1", 20)
    delete_pipe("etna-cuda-11.6.2", 20)
