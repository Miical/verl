source /mnt/pfs/users/angen.ye/myconda/conda/etc/profile.d/conda.sh
conda activate verl_real_rl
cd /mnt/pfs/users/angen.ye/verl
bash recipe/vla/run_pi05_libero_sac_yag.sh


apt-get install -y libegl1 libgl1 libglvnd0
apt-get install -y libegl1 libglvnd0 libgles2 mesa-utils
apt-get install -y \
  libosmesa6 \
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  libglu1-mesa
  

rm -rf /tmp/replay_pools