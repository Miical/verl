import h5py
import numpy as np
from PIL import Image
import glob
import os
import argparse
from tqdm import tqdm
import sys
from pathlib import Path
import shutil

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"[ERROR] 无法导入 LeRobot 库: {e}")
    sys.exit(1)

def get_features_dict(has_gripper=True):
    """
    定义 RL 训练需要的特征（用于 hil-serl）
    基于 record_dataset.py 中的特征定义
    """
    features = {
        # observation.state: 从 qpos 和 qvel 合并
        "observation.state": {
            "dtype": "float32",
            "shape": (28,),  # qpos(14) + qvel(14)
            "names": None,
        },
        # action: 从原始 action 数据获取
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": None,  # 可以根据需要命名
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        },
    }
    
    # 添加图像特征
    features["observation.images.cam_high"] = {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    }
    features["observation.images.cam_left_wrist"] = {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    }
    
    return features

def process_episode(hdf5_file_path, dataset: LeRobotDataset, task_name="default_task"):
    """
    处理单个 episode 的 HDF5 文件
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # 加载数据
            cam_high_ds = f['/observations/images/cam_high']
            cam_left_wrist_ds = f['/observations/images/cam_left_wrist']
            qpos = f['/observations/qpos'][:].astype(np.float32)  # shape: (T, 14)
            qvel = f['/observations/qvel'][:].astype(np.float32)  # shape: (T, 14)
            actions = f['/action'][:].astype(np.float32)  # shape: (T, 14)
            success = f['/success'][:]  # shape: (T,)
            
            num_steps = qpos.shape[0]
            if num_steps <= 0:
                print(f"警告: {hdf5_file_path} 为空 (0 步)。跳过。")
                return False
            
            # 合并 qpos 和 qvel 为 observation.state
            obs_state = np.concatenate([qpos, qvel], axis=1)  # shape: (T, 28)
            
            for i in range(num_steps):
                # 处理图像数据 (HWC -> 确保正确格式)
                img_high = cam_high_ds[i]
                img_left = cam_left_wrist_ds[i]
                
                # 确保图像是 uint8 格式
                if img_high.dtype != np.uint8:
                    img_high = img_high.astype(np.uint8)
                if img_left.dtype != np.uint8:
                    img_left = img_left.astype(np.uint8)
                
                # 构建帧数据
                frame_data = {
                    "observation.images.cam_high": Image.fromarray(img_high),
                    "observation.images.cam_left_wrist": Image.fromarray(img_left),
                    "observation.state": obs_state[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "next.reward": np.array([float(success[i])], dtype=np.float32),
                    "next.done": np.array([False], dtype=bool),  # 默认 False
                    "complementary_info.discrete_penalty": np.array([0.0], dtype=np.float32),
                    "task": task_name,
                }
                
                # 最后一个时间步标记为 done
                if i == num_steps - 1:
                    frame_data["next.done"] = np.array([True], dtype=bool)
                
                dataset.add_frame(frame_data)
        
        dataset.save_episode()
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 处理文件 {hdf5_file_path} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="将 HDF5 转换为 LeRobot RL 训练数据集")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="输入 HDF5 文件所在目录")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出数据集保存路径")
    parser.add_argument("--repo_id", type=str, required=True,
                       help="数据集仓库 ID")
    parser.add_argument("--fps", type=int, default=10,
                       help="数据集帧率")
    parser.add_argument("--task_name", type=str, default="default_task",
                       help="任务名称")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"警告：目标文件夹 {output_path} 已存在，将删除。")
        shutil.rmtree(output_path)
    
    # 创建数据集
    features = get_features_dict()
    print(f"正在创建 RL 训练数据集...")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_path,
        features=features,
        fps=args.fps,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    
    # 确保启动 image writer
    if not dataset.image_writer:
        dataset.start_image_writer(num_processes=0, num_threads=4)
    
    # 处理 HDF5 文件
    source_files = sorted(glob.glob(os.path.join(args.input_dir, "episode_*.hdf5")))
    if not source_files:
        print(f"未找到 episode_*.hdf5 文件。")
        dataset.finalize()
        return
    
    print(f"找到 {len(source_files)} 个 episode 文件")
    
    success_count = 0
    for hdf5_file in tqdm(source_files, desc="转换 Episodes"):
        if process_episode(hdf5_file, dataset, task_name=args.task_name):
            success_count += 1
    
    if success_count == 0:
        print("无有效数据。")
        dataset.finalize()
        return
    
    print(f"\n成功转换 {success_count} 个 episode。")
    dataset.finalize()
    print(f"\n✅ 数据集已保存至: {output_path}")

if __name__ == "__main__":
    main()

