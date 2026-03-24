#!/bin/bash
docker run \
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
