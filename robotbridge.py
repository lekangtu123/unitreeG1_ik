import time, math
from typing import Dict, List, Tuple,Optional
import mujoco
import threading
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateMsg


# ----------------------------------------------------------------------
# 1) RobotBridge
# ----------------------------------------------------------------------

class RobotBridge:
    """Very small wrapper to publish ``LowCmd`` messages every cycle."""

    def __init__(self, iface: str, domain: int, default_mode: int = 0, kp: float = 40.0, kd: float = 1.0):
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        except Exception:
            print("[robot] SDK-2 not present – robot output disabled")
            self.ok = False
            self._latest_state: Optional[LowStateMsg] = None
            return

        self._default_mode = default_mode

        try:
            ChannelFactoryInitialize(domain, iface)
            self._state_lock = threading.Lock()
            self._latest_state = None

            self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self._pub.Init()

            self._cmd = unitree_hg_msg_dds__LowCmd_()

            # Default gains for all motors
            for mc in self._cmd.motor_cmd:
                mc.mode = self._default_mode
                mc.kp = float(kp)
                mc.kd = float(kd)
                mc.tau = 0.0   # initialize feedforward torque

            # Enable/weight slot
            if 29 < len(self._cmd.motor_cmd):
                self._cmd.motor_cmd[29].q = 1.0

            # --- Subscriber
            self._sub = ChannelSubscriber("rt/lowstate", LowState_)
            self._sub.Init(self._on_state, 1)

            try:
                from unitree_sdk2py.utils.crc import CRC
                self._crc = CRC()
            except Exception:
                self._crc = None

            self.ok = True
        except Exception as e:
            print(f"[robot] DDS init failed – robot disabled ({e})")
            self.ok = False

    # ------------------------------------------------------------------
    # Subscriber callback
    # ------------------------------------------------------------------
    def _on_state(self, msg: LowStateMsg):
        with self._state_lock:
            self._latest_state = msg

    def get_joint_states(self) -> Optional[Dict[int, Dict[str, float]]]:
        """
        Returns latest joint states as:
        { idx : { 'q': rad, 'dq': rad/s, 'tau': Nm } }
        """
        with self._state_lock:
            if self._latest_state is None:
                return None
            states = {}
            for i, m in enumerate(self._latest_state.motor_state):
                tau = getattr(m, "tau_est",
                              getattr(m, "tauEst",
                                      getattr(m, "tau", 0.0)))
                states[i] = {
                    "q":   float(m.q),
                    "dq":  float(m.dq),
                    "tau": float(tau),
                }
            return states

    def send_qpos(self, q: Dict[int, float]) -> None:
        """Send positions for a subset of joints (29-DoF indices → radians)."""
        if not self.ok:
            return

        for idx, val in q.items():
            if idx >= 29:
                continue
            if idx < len(self._cmd.motor_cmd):
                mc = self._cmd.motor_cmd[idx]
                mc.mode = self._default_mode
                mc.q = float(val)

        if 29 < len(self._cmd.motor_cmd):
            self._cmd.motor_cmd[29].q = 1.0

        if self._crc is not None:
            if hasattr(self._crc, "Crc"):
                self._cmd.crc = self._crc.Crc(self._cmd)
            elif hasattr(self._crc, "calculate_crc"):
                self._cmd.crc = self._crc.calculate_crc(self._cmd)

        self._pub.Write(self._cmd)

    def send_qpos_tau(self, q: Dict[int, float], tau: Optional[Dict[int, float]] = None) -> None:
        """
        Send desired positions and optional feedforward torques.
        q   : Dict[index, position] (radians)
        tau : Dict[index, torque]   (Nm), optional
        """
        if not self.ok:
            return

        for idx, val in q.items():
            if idx >= 29:
                continue
            if idx < len(self._cmd.motor_cmd):
                mc = self._cmd.motor_cmd[idx]
                mc.mode = self._default_mode
                mc.q = float(val)
                # 如果有提供 tau，就更新
                if tau and idx in tau:
                    mc.tau = float(tau[idx])

        if 29 < len(self._cmd.motor_cmd):
            self._cmd.motor_cmd[29].q = 1.0

        if self._crc is not None:
            if hasattr(self._crc, "Crc"):
                self._cmd.crc = self._crc.Crc(self._cmd)
            elif hasattr(self._crc, "calculate_crc"):
                self._cmd.crc = self._crc.calculate_crc(self._cmd)

        self._pub.Write(self._cmd)


    def send_tau(self, tau: Dict[int, float]) -> None:
        """Pure torque control (ignore q)."""
        if not self.ok:
            return
        for idx, val in tau.items():
            if idx < len(self._cmd.motor_cmd):
                mc = self._cmd.motor_cmd[idx]
                mc.q = 0.0
                mc.tau = float(val)
        self._pub.Write(self._cmd)

    def send_gravity_comp(self, model, data, joint_names: list[str]) -> None:
        """
        Gravity compensation using MuJoCo dynamics.
        `joint_names` : list of joint names to compensate.
        """
        if not self.ok:
            return
        mujoco.mj_forward(model, data)

        tau_gc = {}
        for jname in joint_names:
            jid = model.joint(jname).id
            tau_gc[jid] = float(data.qfrc_bias[jid])  # MuJoCo bias includes gravity
        self.send_tau(tau_gc)

    def send_impedance(self,
                       q_des: Dict[int, float],
                       dq_des: Optional[Dict[int, float]] = None,
                       kp: Optional[Dict[int, float]] = None,
                       kd: Optional[Dict[int, float]] = None,
                       tau_ff: Optional[Dict[int, float]] = None):
        """Send impedance-mode command."""
        if not self.ok:
            return

        dq_des = dq_des or {}
        kp     = kp or {}
        kd     = kd or {}
        tau_ff = tau_ff or {}

        for idx, val in q_des.items():
            if idx >= len(self._cmd.motor_cmd):
                continue
            mc = self._cmd.motor_cmd[idx]
            mc.mode = 4                      # impedance mode
            mc.q    = float(val)
            mc.dq   = float(dq_des.get(idx, 0.0))
            mc.kp   = float(kp.get(idx, mc.kp))
            mc.kd   = float(kd.get(idx, mc.kd))
            mc.tau  = float(tau_ff.get(idx, 0.0))

        # enable slot
        if 29 < len(self._cmd.motor_cmd):
            self._cmd.motor_cmd[29].q = 1.0

        if self._crc is not None:
            if hasattr(self._crc, "Crc"):
                self._cmd.crc = self._crc.Crc(self._cmd)
            else:
                self._cmd.crc = self._crc.calculate_crc(self._cmd)

        self._pub.Write(self._cmd)

# ----------------------------------------------------------------------
# 2) Joint mapping
# ----------------------------------------------------------------------

JOINTS: List[Tuple[int, str, str]] = [
    (15, "left-shoulder-pitch", "left_shoulder_pitch"),
    (16, "left-shoulder-roll",  "left_shoulder_roll"),
    (17, "left-shoulder-yaw",   "left_shoulder_yaw"),
    (18, "left-elbow",          "left_elbow"),
    (19, "left-wrist-roll",     "left_wrist_roll"),
    (20, "left-wrist-pitch",    "left_wrist_pitch"),
    (21, "left-wrist-yaw",      "left_wrist_yaw"),
    (22, "right-shoulder-pitch", "right_shoulder_pitch"),
    (23, "right-shoulder-roll",  "right_shoulder_roll"),
    (24, "right-shoulder-yaw",   "right_shoulder_yaw"),
    (25, "right-elbow",          "right_elbow"),
    (26, "right-wrist-roll",     "right_wrist_roll"),
    (27, "right-wrist-pitch",    "right_wrist_pitch"),
    (28, "right-wrist-yaw",      "right_wrist_yaw"),
]
IDX2LABEL = {idx: lbl for idx, lbl, _ in JOINTS}

IDX2MUJOCO = {
    15: "left_shoulder_pitch_joint",
    16: "left_shoulder_roll_joint",
    17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint",
    19: "left_wrist_roll_joint",
    20: "left_wrist_pitch_joint",
    21: "left_wrist_yaw_joint",
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
    27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
    # add waist or legs if you want to compensate them as well
}










def name_to_index(name: str) -> int:
    for idx, lbl in IDX2LABEL.items():
        if lbl == name:
            return idx
    key = name.lower().replace(" ", "").replace("-", "")
    for idx, lbl in IDX2LABEL.items():
        cand = lbl.lower().replace(" ", "").replace("-", "")
        if cand == key or cand.startswith(key):
            return idx
    raise ValueError(f"Unknown joint: {name!r}")