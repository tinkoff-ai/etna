# Docker images

## Available image names

- `ghcr.io/tinkoff-ai/etna/etna-cpu:<tag>`
- `ghcr.io/tinkoff-ai/etna/etna-cuda-11.6.2:<tag>`

On previous versions of the library there were also:
- `ghcr.io/tinkoff-ai/etna/etna-cuda-10.2:<tag>`

## Use cases

- ### Run jupyter notebook locally

    ```bash
        docker run -v <host_path>:<path_in_docker_container> -p <host_port>:8888 --rm <image_name> jupyter notebook --ip=0.0.0.0 --allow-root
    ```

    Then you could connect to your local jupyter notebook via `http://localhost:<host_port>`

- ### Run etna cli

    ```bash
        docker run -v <host_path>:<path_in_docker_container> --rm <image_name> etna --help
    ```

- ### VSCode containers

  - Install VSCode and [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  - Go to working directory -- `cd <work_path>`
  - Create `.devcontainer/devcontainer.json` with json file

    ```json
    {
        "image": "ghcr.io/tinkoff-ai/etna/etna-cpu:latest"
    }
    ```

  - Open current working directory in VSCode and Run:  `CTRL+SHIFT+P` + `Remote-Containers: Reopen in Container`
