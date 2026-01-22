#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import time
import numpy as np
from piper_sdk import *


def _m_to_sdk_len_units(x_m: float) -> int:
        """m -> 0.001 mm（int）"""
        return int(round(x_m * 1_000_000))


def _sdk_len_units_to_m(x_units: int) -> float:
    """0.001 mm（int）-> m"""
    return x_units / 1_000_000.0


def _rad_to_sdk_angle_units(rad: float) -> int:
    """rad -> 0.001 degree（int）"""
    return int(round(np.rad2deg(rad) * 1000.0))


def _sdk_angle_units_to_rad(units: int) -> float:
    """0.001 degree（int）-> rad"""
    return np.deg2rad(units / 1000.0)

if __name__ == "__main__":
    # 1. 读取左右臂末端位姿
    left_ee_poses = np.loadtxt("/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/motors/piper/left_ee_poses.txt", delimiter=",")  # shape: (N, 6)
    right_ee_poses = np.loadtxt("/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/motors/piper/left_ee_poses.txt", delimiter=",")  # shape: (N, 6)
    assert left_ee_poses.shape == right_ee_poses.shape
    N = left_ee_poses.shape[0]

    # 2. 实例化左右臂接口
    piper_left = C_PiperInterface("can_left")
    piper_right = C_PiperInterface("can_right")
    piper_left.ConnectPort()
    piper_right.ConnectPort()
    while not piper_left.EnablePiper():
        time.sleep(0.01)
    while not piper_right.EnablePiper():
        time.sleep(0.01)
    # while True:
    #     print("1")
    #     time.sleep(0.1)
    # # 3. 以50Hz频率依次下发末端位姿
    # freq = 50
    # dt = 1.0 / freq
    # piper_left.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    # piper_right.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    # for i in range(N):
    #     '''
    #     piper_left.EndPoseCtrl(
    #         int(lx*1000), int(ly*1000), int(lz*1000),
    #         int(lroll*1000), int(lpitch*1000), int(lyaw*1000)
    #     )
    #     piper_right.EndPoseCtrl(
    #         int(rx*1000), int(ry*1000), int(rz*1000),
    #         int(rroll*1000), int(rpitch*1000), int(ryaw*1000)
    #     )
    #     lx, ly, lz, lroll, lpitch, lyaw = left_ee_poses[i]
    #     # 右臂
    #     rx, ry, rz, rroll, rpitch, ryaw = right_ee_poses[i]
    #     '''
    #     # 左臂
    #     '''
    #     2.4035111e-02 -4.6762493e-02  2.8267273e-01 -7.2138622e-02
    #     1.5109223e+00 -1.5576510e-02  7.1470000e-02  5.1607665e-02
    #     1.6162419e-03  2.1671905e-01 -8.0605963e-04  1.4835000e+00
    #     1.9687361e-03  7.2310001e-02
    #     '''
    #     print(_m_to_sdk_len_units(2.4035111e-02), _m_to_sdk_len_units(-4.6762493e-02),  _m_to_sdk_len_units(2.8267273e-01), _rad_to_sdk_angle_units(-7.2138622e-02), _rad_to_sdk_angle_units(1.5109223e+00), _rad_to_sdk_angle_units(-1.5576510e-02))
    #     print(_m_to_sdk_len_units(5.1607665e-02), _m_to_sdk_len_units(1.6162419e-03), _m_to_sdk_len_units(2.1671905e-01), _rad_to_sdk_angle_units(-8.0605963e-04), _rad_to_sdk_angle_units(1.4835000e+00), _rad_to_sdk_angle_units(1.9687361e-03) )
    #     lx, ly, lz, lroll, lpitch, lyaw = _m_to_sdk_len_units(2.4035111e-02), _m_to_sdk_len_units(-4.6762493e-02),  _m_to_sdk_len_units(2.8267273e-01), _rad_to_sdk_angle_units(-7.2138622e-02), _rad_to_sdk_angle_units(1.5109223e+00), _rad_to_sdk_angle_units(-1.5576510e-02) #42.502, -18.948, 299.893, 0.536, 85.568, 0.273 #left_ee_poses[i]
    #     # 右臂
    #     rx, ry, rz, rroll, rpitch, ryaw = _m_to_sdk_len_units(5.1607665e-02), _m_to_sdk_len_units(1.6162419e-03), _m_to_sdk_len_units(2.1671905e-01), _rad_to_sdk_angle_units(-8.0605963e-04), _rad_to_sdk_angle_units(1.4835000e+00), _rad_to_sdk_angle_units(1.9687361e-03) #61.616, -2.148, 198.667, -3.165, 86.880, 0.344 #right_ee_poses[i]
    #     # 单位转换：mm->um，deg->mdeg
    #     piper_left.EndPoseCtrl(
    #         int(lx), int(ly), int(lz),
    #         int(lroll), int(lpitch), int(lyaw)
    #     )
    #     piper_right.EndPoseCtrl(
    #         int(rx), int(ry), int(rz),
    #         int(rroll), int(rpitch), int(ryaw)
    #     )
    #     print(i)
    #     time.sleep(dt)