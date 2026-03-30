#!/bin/bash
CONTAINER_NAME="${1:-isaac-rl}"

docker rm -f "$CONTAINER_NAME" 2>/dev/null

docker run \
    --name "$CONTAINER_NAME" \
    --entrypoint /root/entrypoint.sh \
    -e "ACCEPT_EULA=Y" \
    -it --gpus all \
    -v /dev/shm:/dev/shm:rw \
    -v ~/iCode/RL/verl:/root/code/verl \
    -v ~/iCode/RL/RobotLearningLab:/root/RobotLearningLab \
    -v /home/billyw/iDataset/VLA/openpi/checkpoint/pi05_libero_torch:/root/data/pi05_libero_torch \
    -v /home/billyw/iDataset/VLA/openpi/libero_rl:/root/data/libero_rl \
    verl-isaac-vla:latest \
    bash
