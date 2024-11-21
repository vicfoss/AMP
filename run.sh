#!/bin/bash
# Grant Docker access to the X server
xhost +local:docker

# Run the Docker container with GPU and display access
sudo docker run --rm -it \
    --gpus all \
    --env="DISPLAY=$DISPLAY" \
    --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
    --env="DATABASE_URL=http://10.0.0.36:5000/add_record" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR:rw" \
    --privileged \
    alpr-jetson

# Revoke X server access once the container stops
xhost -local:docker
