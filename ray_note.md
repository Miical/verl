<!-- 安装ping -->
apt-get update
apt-get install -y iputils-ping
<!-- 设置路由表 -->
sudo ip route add 192.168.18.0/24 via 10.10.10.1 dev eno1

source /mnt/pfs/users/weijie.ke/.bashrc 
Node:A
conda activate raytest312
cd /shared_disk/users/weijie.ke/verl
ray stop --force
rm -rf /tmp/ray
ray start --head --dashboard-host=0.0.0.0 --port=6379 --resources='{"node:A": 1, "train_rollout": 1}'
Node：B
conda activate raytest2
ray stop --force
rm -rf /tmp/ray
export CUDA_VISIBLE_DEVICES=0 
ray start --address='192.168.18.151:6379' --resources='{"node:B": 1, "sim": 1}'

<!-- 机器人测试 -->
bash recipe/vla/run_pi05_robot_test.sh



PIPER_ROOT=/shared_disk/users/weijie.ke/verl/recipe/vla/envs/test_env/robot/controller/piper
export PYTHONPATH=$PIPER_ROOT:$PYTHONPATH
cd $PIPER_ROOT

