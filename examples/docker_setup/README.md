## Quick Start Guide to Running Your PyTorch Docker Container

### Step 1: Create the Dockerfile

1. **Open Terminal**: Open a terminal on your Ubuntu machine.
2. **Create Dockerfile**: Enter `nano Dockerfile` to create and edit a new Dockerfile.
3. **Enter Dockerfile Content**:
    ```dockerfile
    # Use an official PyTorch image as a base
    FROM nvcr.io/nvidia/pytorch:latest

    # Set the working directory inside the container
    WORKDIR /workspace

    # Install any necessary dependencies
    RUN pip install -r requirements.txt

    # Copy the local code to the container's workspace
    COPY ./ /workspace/

    # Set the default command to execute
    CMD ["/bin/bash"]
    ```
    Replace `latest` with the specific version of PyTorch you need. Modify `requirements.txt` to include all necessary Python packages.

### Step 2: Build Your Docker Image

1. **Build Image**: In your terminal, run:
   ```bash
   docker build -t my-pytorch-app .
   ```
   This command builds the Docker image named `my-pytorch-app` using the Dockerfile in the current directory.

### Step 2: Create the Docker Run Script

1. **Open Terminal**: Open a terminal on your Ubuntu machine.
2. **Create Script File**: Enter `nano run_pytorch_docker.sh` to create and edit a new shell script.
3. **Enter Script Content**:
    ```bash
    #!/bin/bash
    # This script runs a Docker container with necessary volume mounts for the PyTorch application.

    docker run --gpus all -it --rm \
      -v /path/to/megatron:/workspace/megatron \
      -v /path/to/dataset:/workspace/dataset \
      -v /path/to/checkpoints:/workspace/checkpoints \
      my-pytorch-app \
      /bin/bash
    ```
    Replace `/path/to/megatron`, `/path/to/dataset`, and `/path/to/checkpoints` with the actual paths to your resources. This will take you the interactive window.
4. **Save and Exit**: Press `Ctrl+O`, hit `Enter` to save, then `Ctrl+X` to exit `nano`.
5. **Make Executable**: Run `chmod +x run_pytorch_docker.sh` to make your script executable.

### Step 3: Run the Docker Container

- **Execute the Script**: In your terminal, type `./run_pytorch_docker.sh` to start the Docker container. This script mounts specified directories and opens a container with GPU access enabled.


### Step 4: Debugging Inside the Container

Once your Docker container is running and you're inside its interactive shell, you can proceed as if you're in a typical development environment:

- **Full Access to Libraries**: All libraries and tools installed in the Docker image are at your disposal. You can run commands, execute scripts, and use your usual debugging tools just like on a local machine.
- **Normal Operation**: Interact with the terminal as you would in any Linux environment. Edit, execute, and debug your applications directly inside the container using the command line or any terminal-based editors like Vim or Nano.

This setup provides a seamless experience for development and debugging, ensuring that your work environment is both controlled and replicable.

### Step 5: Exit the Container

- **To Exit**: Type `exit` in the container's terminal. The container will stop, and due to the `--rm` flag, it will also be automatically removed, cleaning up your system.
