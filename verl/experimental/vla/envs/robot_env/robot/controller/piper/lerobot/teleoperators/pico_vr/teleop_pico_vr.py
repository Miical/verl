from __future__ import annotations

import json
import logging
from time import time
from typing import Any

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_pico_vr import PicoVrTeleopConfig

class PicoVrTeleop(Teleoperator):
    """
    Dual-arm teleoperation driven by Pico VR controllers via ZeroMQ.
    
    This teleoperator subscribes to teleoperation signals (button states) via ZeroMQ
    instead of directly connecting to the VR device. The actual teleoperation actions
    are handled by a separate program, and this class only processes button events
    for intervention control and episode management.
    
    Expected ZeroMQ message format (JSON):
    {
        "timestamp": <float>,
        "buttons": {
            "A": <bool>,
            "X": <bool>,
            "Y": <bool>,
            "B": <bool>,
            "right_axis_click": <bool>,
            ...
        }
    }
    """

    config_class = PicoVrTeleopConfig
    name = "pico_vr"

    def __init__(self, config: PicoVrTeleopConfig):
        super().__init__(config)
        self.config = config
        
        if zmq is None:
            raise ImportError("pyzmq is required. Install it with: pip install pyzmq")

        logging.info(f"[PicoVrTeleop] Initializing with ZeroMQ config: host={config.zmq_host}, port={config.zmq_port}")

        self._zmq_context: zmq.Context | None = None
        self._zmq_socket: zmq.Socket | None = None

        self._connected = False
        self._intervention_latched = False
        self._button_states: dict[str, bool] | None = None
        self._reset_button_prev = False
        self._intervention_reset_prev = False
        self._print_interval = 1  # Print interval in seconds
        self._last_print_time = time()
        self._events_call_count = 0  # Counter for event calls
        self._last_events: dict[str, Any] | None = None  # Track last events for change detection

    # ------------------------------------------------------------------ #
    # Teleoperator interface
    # ------------------------------------------------------------------ #
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "left_delta_x": 0,
                "left_delta_y": 1,
                "left_delta_z": 2,
                "left_delta_rx": 3,
                "left_delta_ry": 4,
                "left_delta_rz": 5,
                "left_delta_gripper": 6,
                "right_delta_x": 7,
                "right_delta_y": 8,
                "right_delta_z": 9,
                "right_delta_rx": 10,
                "right_delta_ry": 11,
                "right_delta_rz": 12,
                "right_delta_gripper": 13,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Pico VR teleop already connected.")
        
        # Initialize ZeroMQ context and socket
        address = f"tcp://{self.config.zmq_host}:{self.config.zmq_port}"
        print(f"[PicoVrTeleop] Attempting to connect to ZeroMQ publisher at {address}...")
        logging.info(f"[PicoVrTeleop] Attempting to connect to ZeroMQ publisher at {address}")
        
        try:
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.SUB)
            
            # Connect to publisher
            self._zmq_socket.connect(address)
            # Subscribe to all messages (empty string means subscribe to all)
            self._zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            # Set receive timeout (milliseconds) - non-blocking with timeout
            self._zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            
            self._connected = True
            print(f"[PicoVrTeleop] ✓ Connected to ZeroMQ publisher at {address}")
            logging.info(f"[PicoVrTeleop] Connected to ZeroMQ publisher at {address}")
        except Exception as e:
            error_msg = f"Failed to connect to ZeroMQ publisher at {address}: {e}"
            print(f"[PicoVrTeleop] ✗ {error_msg}")
            logging.error(f"[PicoVrTeleop] {error_msg}")
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Pico VR teleop is not connected. Call `connect()` before `get_action()`."
            )
        
        # Fetch button states from ZeroMQ messages
        self._button_states = self._fetch_button_states()
        self._maybe_handle_reset_buttons()
        
        # Debug: 周期性打印干预状态
        now = time()
        if now - self._last_print_time >= self._print_interval:
            logging.debug(f"[PicoVrTeleop] intervention_latched={self._intervention_latched}")
            self._last_print_time = now
        
        # 遥操动作在另外一个单独的程序中实现，这里仅获取遥操的信号，而不发送遥操动作
        # 始终返回全 0 的动作
        return self._zero_action()

    def get_teleop_events(self) -> dict[str, Any]:
        self._events_call_count += 1
        
        if not self.is_connected:
            if self._events_call_count == 1:
                print("[PicoVrTeleop] WARNING: Not connected, returning default events")
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        if self._button_states is None:
            self._button_states = self._fetch_button_states()

        terminate = self._get_button_state(self.config.terminate_button)
        success = self._get_button_state(self.config.success_button)
        rerecord = self._get_button_state(self.config.rerecord_button)
        
        if bool(success):
            print("[PicoVrTeleop] ✓ SUCCESS button pressed!")
            logging.info("Pico VR teleop: SUCCESS button pressed.")
        
        events = {
            TeleopEvents.IS_INTERVENTION: self._intervention_latched,
            TeleopEvents.TERMINATE_EPISODE: bool(terminate),
            TeleopEvents.SUCCESS: bool(success),
            TeleopEvents.RERECORD_EPISODE: bool(rerecord),
        }

        # Only print when events change or periodically (every 50 calls = ~1 second at 50fps)
        events_changed = self._last_events != events
        should_print = events_changed or (self._events_call_count % 50 == 0)
        
        if should_print:
            print(f"[PicoVrTeleop] events={events} (call_count={self._events_call_count})")
            logging.info(f"Pico VR teleop: events={events}")
        
        # Clear cache to avoid stale values on next tick
        # Note: We clear the cache here so that button states are fetched fresh
        # on the next call to get_teleop_events or get_action
        self._button_states = None
        self._last_events = events.copy()
        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback  # Not supported

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Pico VR teleop is not connected.")
        
        # Close ZeroMQ socket and context
        if self._zmq_socket is not None:
            try:
                self._zmq_socket.close()
            except Exception as exc:
                logging.warning("Failed to close ZeroMQ socket cleanly: %s", exc)
        
        if self._zmq_context is not None:
            try:
                self._zmq_context.term()
            except Exception as exc:
                logging.warning("Failed to close ZeroMQ context cleanly: %s", exc)
        
        self._zmq_socket = None
        self._zmq_context = None
        self._connected = False
        logging.info("Disconnected from ZeroMQ publisher")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _zero_action(self) -> dict[str, float]:
        return {
            "left_delta_x": 0.0,
            "left_delta_y": 0.0,
            "left_delta_z": 0.0,
            "left_delta_rx": 0.0,
            "left_delta_ry": 0.0,
            "left_delta_rz": 0.0,
            "left_delta_gripper": 0.0,
            "right_delta_x": 0.0,
            "right_delta_y": 0.0,
            "right_delta_z": 0.0,
            "right_delta_rx": 0.0,
            "right_delta_ry": 0.0,
            "right_delta_rz": 0.0,
            "right_delta_gripper": 0.0,
        }

    def _fetch_button_states(self) -> dict[str, bool]:
        """
        Fetch button states from ZeroMQ messages.
        Expected message format (JSON):
        {
            "timestamp": <float>,
            "buttons": {
                "A": <bool>,
                "X": <bool>,
                "Y": <bool>,
                "B": <bool>,
                "right_axis_click": <bool>,
                ...
            }
        }
        """
        assert self._zmq_socket is not None
        
        # Try to receive the latest message (non-blocking due to timeout)
        # We may receive multiple messages, but we only care about the latest one
        # This ensures we always have the most recent button state
        latest_message = None
        try:
            while True:
                try:
                    message = self._zmq_socket.recv_string(zmq.NOBLOCK)
                    latest_message = message
                except zmq.Again:
                    # No more messages available (timeout reached)
                    break
        except Exception as e:
            logging.debug(f"Error receiving ZeroMQ message: {e}")
        
        # Parse the latest message if available
        states: dict[str, bool] = {}
        if latest_message is not None:
            try:
                data = json.loads(latest_message)
                # Extract button states from the message
                # Support multiple possible message formats
                if "buttons" in data:
                    # Format: {"buttons": {"A": true, "X": false, ...}}
                    buttons_data = data["buttons"]
                elif "button_states" in data:
                    # Alternative format: {"button_states": {"A": true, ...}}
                    buttons_data = data["button_states"]
                else:
                    # Direct format: {"A": true, "X": false, ...}
                    buttons_data = data
                
                # Extract all button states
                for button_name, button_state in buttons_data.items():
                    if isinstance(button_state, bool):
                        states[button_name] = button_state
                    elif isinstance(button_state, (int, float)):
                        states[button_name] = bool(button_state)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse ZeroMQ message as JSON: {e}")
            except Exception as e:
                logging.warning(f"Error parsing button states from ZeroMQ message: {e}")
        
        # Ensure all required buttons have a state (default to False if not present)
        required_buttons = {
            self.config.reset_reference_button,
            self.config.intervention_reset_button,
            self.config.terminate_button,
            self.config.success_button,
            self.config.rerecord_button,
        }
        for button_name in filter(None, required_buttons):
            if button_name not in states:
                states[button_name] = False
        
        return states

    def _maybe_handle_reset_buttons(self) -> None:
        """
        Handle button presses for reset and intervention toggle.
        Since we're using ZeroMQ, we only handle button state changes here.
        """
        # Reset button handling (if needed in the future)
        reset_pressed = self._get_button_state(self.config.reset_reference_button)
        if reset_pressed and not self._reset_button_prev:
            # Reset button pressed - could be used for future functionality
            logging.debug("Reset button pressed")
        self._reset_button_prev = reset_pressed

        # 手动切换干预状态：按下按钮时切换 intervention_latched
        intervention_toggle = self._get_button_state(self.config.intervention_reset_button)
        if intervention_toggle and not self._intervention_reset_prev:
            # 按钮刚按下时，切换干预状态
            self._intervention_latched = not self._intervention_latched
            logging.info(f"Intervention toggled: {self._intervention_latched}")
        self._intervention_reset_prev = intervention_toggle

    def _get_button_state(self, button_name: str | None) -> bool:
        """
        Get the state of a button from the cached button states.
        
        Args:
            button_name: Name of the button (e.g., "A", "X", "Y", "B", "right_axis_click")
            
        Returns:
            True if button is pressed, False otherwise
        """
        if button_name is None or self._button_states is None:
            return False
        return self._button_states.get(button_name, False)

