from typing import (
    Optional,
)
import time
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_left")
    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    piper.GripperCtrl(0,1000,0x01, 0)
    factor = 57295.7795 #1000*180/3.1415926
    position = [0.2,0.35,-0.25,0.3,-0.2,0.5,0.08]
    joint_0 = round(position[0]*factor)
    joint_1 = round(position[1]*factor)
    joint_2 = round(position[2]*factor)
    joint_3 = round(position[3]*factor)
    joint_4 = round(position[4]*factor)
    joint_5 = round(position[5]*factor)
    joint_6 = round(position[6]*1000*1000)
    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    print(piper.GetArmStatus())
    print(position)
    time.sleep(2)


    position = [0,0,0,0,0,0,0]
    joint_0 = round(position[0]*factor)
    joint_1 = round(position[1]*factor)
    joint_2 = round(position[2]*factor)
    joint_3 = round(position[3]*factor)
    joint_4 = round(position[4]*factor)
    joint_5 = round(position[5]*factor)
    joint_6 = round(position[6]*1000*1000)
    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    print(piper.GetArmStatus())
    print(position)
    time.sleep(2)
0.252452   -0.034618    0.272318    3.07350736  0.4325624   2.90283161

    position = [
        57.0, \
        0.0, \
        260.0, \
        0, \
        85.0, \
        0, \
        0]
    factor = 1000
    X = round(position[0]*factor)
    Y = round(position[1]*factor)
    Z = round(position[2]*factor)
    RX = round(position[3]*factor)
    RY = round(position[4]*factor)
    RZ = round(position[5]*factor)
    joint_6 = round(position[6]*factor)
    print(X,Y,Z,RX,RY,RZ)
    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
    time.sleep(0.5)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)