# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the OrbbecDabaiCamera class for capturing frames from Orbbec Dabai cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Union, Any, Optional

import cv2
import numpy as np

try:
    import pyorbbecsdk as ob
except Exception as e:
    logging.info(f"Could not import Orbbec SDK: {e}")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_dabai import OrbbecDabaiCameraConfig  # 可复用配置
from .utils import frame_to_numpy_image, save_color_frame
logger = logging.getLogger(__name__)


class OrbbecDabaiCamera(Camera):
    """
    Manages interactions with Orbbec Dabai cameras for color and depth capture.

    This class is modeled after RealSenseCamera but uses the Orbbec SDK (pyorbbecsdk).
    It supports RGB and depth streams, configurable FPS, resolution, and rotation.

    Example:
        ```python
        config = OrbbecDabaiCameraConfig(serial_number_or_name="ABC123XYZ")  # 实际序列号
        camera = OrbbecDabaiCamera(config)
        camera.connect()

        color_image = camera.read()
        depth_map = camera.read_depth()

        camera.disconnect()
        ```
    """

    def __init__(self, config: OrbbecDabaiCameraConfig):
        super().__init__(config)
        self.config = config

        self.serial_number = config.serial_number_or_name  # Orbbec 也支持按 SN 选择
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.pipeline: Optional[ob.Pipeline] = None
        self.device: Optional[ob.Device] = None
        self.color_profile: Optional[ob.VideoStreamProfile] = None
        self.depth_profile: Optional[ob.VideoStreamProfile] = None

        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None
        self.frame_lock = Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.new_frame_event = Event()

        self.rotation: Optional[int] = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None and self.device is not None

    def connect(self, warmup: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        import time
        
        # 添加重试机制，防止设备被其他进程占用
        max_retries = 3
        retry_delay = 1.0  # 秒
        
        for attempt in range(max_retries):
            try:
                ctx = ob.Context()
                devices = ctx.query_devices()
                if devices.get_count() == 0:
                    raise ConnectionError("No Orbbec devices detected.")

                target_device = None
                for i in range(devices.get_count()):
                    dev = devices.get_device_by_index(i)
                    device_info = dev.get_device_info()

                    #dev_info = devices_info.get_device_info_by_index(i)
                    sn = device_info.get_serial_number()
                    #sn = dev.get_info(ob.OBDeviceInfo.OB_DEVICE_INFO_SERIAL_NUMBER)
                    if sn == self.serial_number:
                        target_device = dev
                        break

                if not target_device:
                    available_sn = [
                        devices.get_device_by_index(i).get_device_info().get_serial_number()
                        for i in range(devices.get_count())
                    ]
                    raise ValueError(f"Orbbec Dabai with SN '{self.serial_number}' not found. Available: {available_sn}")

                self.device = target_device
                self.pipeline = ob.Pipeline(self.device)
                break  # 成功创建 pipeline，退出重试循环
                
            except ob.OBError as e:
                error_msg = str(e)
                if "failed,return res-6" in error_msg or "already opened" in error_msg:
                    # 设备被占用，等待后重试
                    if attempt < max_retries - 1:
                        logger.warning(f"Camera device busy (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        # 清理可能残留的资源
                        try:
                            del ctx
                            del devices
                            if 'target_device' in locals():
                                del target_device
                        except:
                            pass
                        continue
                    else:
                        raise ConnectionError(f"Failed to open camera after {max_retries} attempts. Device may be in use by another process.") from e
                else:
                    # 其他错误，直接抛出
                    raise

        config = ob.Config()
        depth_profile = None
        color_profile = None
        # 配置流
        if self.width and self.height and self.fps:
            profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            try:
                color_profile: ob.VideoStreamProfile = profile_list.get_video_stream_profile(self.width, self.height, ob.OBFormat.RGB, self.fps)
            except ob.OBError as e:
                print(f"Error getting color profile: {e}")
                color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            if self.use_depth:
                profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                try:
                    depth_profile: ob.VideoStreamProfile = profile_list.get_video_stream_profile(self.width, self.height, ob.OBFormat.Y16, self.fps)
                except ob.OBError as e:
                    print(f"Error getting depth profile: {e}")
                    depth_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(depth_profile)
        else:
            profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            try:
                color_profile: ob.VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, ob.OBFormat.RGB, 30) #get_default_video_stream_profile()
            except ob.OBError as e:
                print(f"Error getting color profile: {e}")
                color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            if self.use_depth:
                profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                try:
                    depth_profile: ob.VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, ob.OBFormat.Y16, 30) #get_default_video_stream_profile()
                except ob.OBError as e:
                    print(f"Error getting depth profile: {e}")
                    depth_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(depth_profile)

        try:
            self.pipeline.start(config)
        except Exception as e:
            self.pipeline = None
            self.device = None
            self.depth_profile = None
            self.color_profile = None
            raise ConnectionError(f"Failed to start pipeline for {self}: {e}")

        # 获取实际流配置
        # profile = self.pipeline.get_active_profile()
        # color_stream = profile.get_stream(ob.OBStreamType.OB_STREAM_COLOR)
        # if color_stream:
        #     self.color_profile = color_stream.as_video_stream_profile()
        #     if self.fps is None:
        #         self.fps = self.color_profile.fps()
        #     if self.width is None or self.height is None:
        #         w, h = self.color_profile.width(), self.color_profile.height()
        #         if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        #             self.width, self.height = h, w
        #             self.capture_width, self.capture_height = w, h
        #         else:
        #             self.width, self.height = w, h
        #             self.capture_width, self.capture_height = w, h

        # if self.use_depth:
        #     depth_stream = profile.get_stream(ob.OBStreamType.OB_STREAM_DEPTH)
        #     if depth_stream:
        #         self.depth_profile = depth_stream.as_video_stream_profile()
        self.color_profile = color_profile
        if self.use_depth:
            self.depth_profile = depth_profile

        if warmup:
            time.sleep(1)
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read()
                except:
                    pass
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    # @staticmethod
    # def find_cameras2() -> list[dict[str, Any]]:
    #     """Detect available Orbbec cameras."""
    #     ctx = ob.Context()
    #     devices_info = ctx.query_devices()
    #     cameras = []

    #     for i in range(devices_info.get_count()):
    #         dev_info = devices_info.get_device_info_by_index(i)
    #         sn = dev_info.get_info(ob.OBDeviceInfo.OB_DEVICE_INFO_SERIAL_NUMBER)
    #         name = dev_info.get_info(ob.OBDeviceInfo.OB_DEVICE_INFO_NAME)
    #         firmware = dev_info.get_info(ob.OBDeviceInfo.OB_DEVICE_INFO_FIRMWARE_VERSION)
    #         usb_type = dev_info.get_info(ob.OBDeviceInfo.OB_DEVICE_INFO_USB_TYPE)


    #         camera_info = {
    #             "name": name,
    #             "type": "Orbbec",
    #             "id": sn,
    #             "firmware_version": firmware,
    #             "usb_type_descriptor": usb_type,
    #         }

    #         # 打开设备获取流信息
    #         try:
    #             dev = ctx.get_device_by_sn(sn)
    #             pipeline = ob.Pipeline(dev)
    #             config = ob.Config()
    #             config.enable_stream(ob.OBStreamType.OB_STREAM_COLOR)
    #             profile = pipeline.start(config, None)
    #             color_stream = profile.get_stream(ob.OBStreamType.OB_STREAM_COLOR).as_video_stream_profile()
    #             camera_info["default_stream_profile"] = {
    #                 "stream_type": "Color",
    #                 "format": "RGB8",
    #                 "width": color_stream.width(),
    #                 "height": color_stream.height(),
    #                 "fps": color_stream.fps(),
    #             }
    #             pipeline.stop()
    #         except Exception as e:
    #             logger.warning(f"Could not get stream profile for {name}: {e}")
    #             camera_info["default_stream_profile"] = {}

    #         cameras.append(camera_info)

    #     return cameras

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect available Orbbec cameras."""
        ctx = ob.Context()
        devices_list = ctx.query_devices()
        cameras = []

        for i in range(devices_list.get_count()):
            device = devices_list.get_device_by_index(i)
            device_info = device.get_device_info()

            #dev_info = devices_info.get_device_info_by_index(i)
            sn = device_info.get_serial_number()
            name = device_info.get_name()
            firmware = device_info.get_firmware_version()
            connection_type = device_info.get_connection_type()
            print("-----------------------------------------------------------")
            print(f"Device info: {device_info}")
            camera_info = {
                "name": name,
                "type": "Orbbec",
                "id": sn,
                "firmware_version": firmware,
                "usb_type_descriptor": connection_type,
            }

            # 打开设备获取流信息
            has_color_sensor = False
            pipeline = None
            try:
                # 使用已经获取的设备对象
                pipeline = ob.Pipeline(device)
                config = ob.Config()
                profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                if profile_list is not None:
                    color_profile: ob.VideoStreamProfile = profile_list.get_default_video_stream_profile()
                    config.enable_stream(color_profile)
                    has_color_sensor = True
                #config.enable_stream(ob.OBStreamType.OB_STREAM_COLOR)
                depth_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                if depth_profile_list is not None:
                    depth_profile: ob.VideoStreamProfile = depth_profile_list.get_default_video_stream_profile()
                    config.enable_stream(depth_profile)
                pipeline.start(config, None)
                #color_stream = profile.get_stream(ob.OBStreamType.OB_STREAM_COLOR).as_video_stream_profile()
                # frames = pipeline.wait_for_frames(100)
                # color_frame = frames.get_color_frame()
                # if color_frame is not None:
                #     width = color_frame.get_width()
                #     height = color_frame.get_height()
                #     fps = color_frame.get_fps()
                camera_info["default_stream_profile"] = {
                    "stream_type": "Color",
                    "format": "RGB8",
                    "width": color_profile.get_width(),
                    "height": color_profile.get_height(),
                    "fps": color_profile.get_fps(),
                }
            except Exception as e:
                logger.warning(f"Could not get stream profile for {name}: {e}")
                camera_info["default_stream_profile"] = {}
            finally:
                # 确保 pipeline 被正确关闭和释放
                if pipeline is not None:
                    try:
                        pipeline.stop()
                        del pipeline
                    except:
                        pass

            cameras.append(camera_info)

        return cameras

    def _configure_capture_settings(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")
        # 已在 connect 中完成
        pass

    def read(self, color_mode: Optional[ColorMode] = None, timeout_ms: int = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                raise RuntimeError(f"{self} read failed: timeout")
            color_frame = frames.get_color_frame()
            if color_frame is None:
                raise RuntimeError(f"{self} no color frame received.")
            #save_color_frame(color_frame, 0)
            color_image_raw = frame_to_numpy_image(color_frame)
            #data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            #w, h = color_frame.get_width(), color_frame.get_height()
            #color_image_raw = data.reshape((h, w, 3))  # RGB8
        except Exception as e:
            raise RuntimeError(f"{self} read failed: {e}")

        processed = self._postprocess_image(color_image_raw, color_mode=color_mode)
        logger.debug(f"{self} read took: {(time.perf_counter() - time.perf_counter()) * 1e3:.1f}ms")
        return processed

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")
        if not self.use_depth:
            raise RuntimeError(f"Depth stream not enabled for {self}.")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                raise RuntimeError(f"{self} read_depth failed: timeout")
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                raise RuntimeError(f"{self} no depth frame received.")
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            w, h = depth_frame.get_width(), depth_frame.get_height()
            depth_map = data.reshape((h, w))
        except Exception as e:
            raise RuntimeError(f"{self} read_depth failed: {e}")

        processed = self._postprocess_image(depth_map, depth_frame=True)
        return processed

    def _postprocess_image(
        self, image: np.ndarray, color_mode: Optional[ColorMode] = None, depth_frame: bool = False
    ) -> np.ndarray:
        target_mode = color_mode or self.color_mode
        h, w = image.shape[:2]

        # if h != self.capture_height or w != self.capture_width:
        #     raise RuntimeError(
        #         f"Frame size ({w}x{h}) != configured ({self.capture_width}x{self.capture_height})"
        #     )

        processed = image

        if not depth_frame:
            if target_mode == ColorMode.BGR:
                processed = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)

        return processed

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.read(timeout_ms=500)
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error in async read loop for {self}: {e}")

    def _start_read_thread(self):
        if self.thread and self.thread.is_alive():
            self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self):
        if self.stop_event:
            self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")
        if not self.thread or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timeout waiting for frame from {self}")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: frame is None after event set for {self}")
        return frame

    def disconnect(self):
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} already disconnected.")

        if self.thread:
            self._stop_read_thread()

        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.device = None

        logger.info(f"{self} disconnected.")

