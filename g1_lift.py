#!/usr/bin/env python3
import curses
import math
import time
import warnings
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
from ruckig import Ruckig, InputParameter, OutputParameter, Result
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from g1_ik_solver import G1IKSolver, RIGHT_JOINTS, LEFT_JOINTS
from robotbridge import RobotBridge, name_to_index

warnings.filterwarnings("ignore", message=".*Wayland: The platform does not provide the window position.*")


# =========================
# Workspace definition
# =========================
WORKSPACE = {
    "frame": "world",
    "left_arm": {
        "left_bottom_front":  [0.33,  0.24, 0.02],
        "right_bottom_front": [0.33,  0.07, 0.02],
        "left_bottom_back":   [0.16,  0.24, 0.02],
        "right_bottom_back":  [0.16,  0.07, 0.02],
        "right_top_back":     [0.07,  0.20, 0.20],
        "left_top_back":      [0.07,  0.47, 0.20],
        "right_top_front":    [0.45,  0.11, 0.20],
        "left_top_front":     [0.41,  0.30, 0.20],
    },
    "right_arm": {
        "left_bottom_front":  [0.33, -0.24, 0.02],
        "right_bottom_front": [0.33, -0.07, 0.02],
        "left_bottom_back":   [0.16, -0.24, 0.02],
        "right_bottom_back":  [0.16, -0.07, 0.02],
        "right_top_back":     [0.07, -0.20, 0.20],
        "left_top_back":      [0.07, -0.47, 0.20],
        "right_top_front":    [0.45, -0.11, 0.20],
        "left_top_front":     [0.41, -0.30, 0.20],
    },
}


def ws_points_from_dict(d: dict) -> np.ndarray:
    return np.array(list(d.values()), dtype=float)


def workspace_halfspaces(points_xyz: np.ndarray):
    """
    给定 Nx3 点，构建凸包半空间：A x <= b
    hull.equations: a*x + b*y + c*z + d = 0
    对于内部：a*x + b*y + c*z + d <= 0
    """
    hull = ConvexHull(points_xyz)
    A = hull.equations[:, :3]
    b = -hull.equations[:, 3]
    return A, b


def point_in_convex_poly(A: np.ndarray, b: np.ndarray, p: np.ndarray, margin=1e-6) -> bool:
    return np.all(A.dot(p) <= b + margin)


# precompute hulls in WORKSPACE frame
WS_RIGHT = ws_points_from_dict(WORKSPACE["right_arm"])
WS_LEFT  = ws_points_from_dict(WORKSPACE["left_arm"])
A_R, b_R = workspace_halfspaces(WS_RIGHT)
A_L, b_L = workspace_halfspaces(WS_LEFT)


def world_to_body_frame(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, p_world: np.ndarray) -> np.ndarray:
    """
    把 world 坐标下的点 p_world 转到某个 body 的局部坐标系下。
    p_body = R_bw^T * (p_world - t_bw)
    其中 R_bw: body->world
    """
    bid = model.body(body_name).id
    t = data.xpos[bid].copy()
    R_bw = data.xmat[bid].reshape(3, 3).copy()  # body->world
    return R_bw.T @ (p_world - t)


def apply_yaw_delta(quat_xyzw: np.ndarray, dyaw: float) -> np.ndarray:
    r_curr = R.from_quat(quat_xyzw)
    r_d = R.from_euler("z", dyaw, degrees=False)
    return (r_d * r_curr).as_quat()


def fmt_vec(v: np.ndarray) -> str:
    return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"


def joint_deg(model: mujoco.MjModel, qpos: np.ndarray, joint_names):
    out = []
    for n in joint_names:
        jid = model.joint(n).id
        out.append(math.degrees(float(qpos[jid])))
    return out


# =========================
# Arm controller (Ruckig)
# =========================
class ArmController:
    def __init__(self, qpos, qvel, ctrl, timestep, model, joint_names):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.model = model

        self.joint_ids = [self.model.joint(j).id for j in joint_names]
        self.dof = len(self.joint_ids)

        self.otg = Ruckig(self.dof, timestep)
        self.otg_inp = InputParameter(self.dof)
        self.otg_out = OutputParameter(self.dof)

        # limits (与你原来的保持一致)
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
        self.otg_res = Result.Finished

    def move_to(self, target_qpos_full: np.ndarray):
        target = target_qpos_full[self.joint_ids]
        self.otg_inp.current_position = self.qpos[self.joint_ids].copy()
        self.otg_inp.current_velocity = self.qvel[self.joint_ids].copy()
        self.otg_inp.target_position = target.copy()
        self.otg_res = Result.Working

    def step(self):
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.qpos[self.joint_ids] = self.otg_out.new_position
            self.qvel[self.joint_ids] = self.otg_out.new_velocity
            self.ctrl[self.joint_ids] = self.otg_out.new_position
        elif self.otg_res == Result.Finished:
            self.ctrl[self.joint_ids] = self.otg_out.new_position

    def stop_hold(self):
        """立刻停止该臂轨迹，保持当前位置（不再继续走）"""
        cur = self.qpos[self.joint_ids].copy()

        # 直接让 Ruckig 输入“目标=当前”，速度清零
        self.otg_inp.current_position = cur.copy()
        self.otg_inp.current_velocity = np.zeros(self.dof)
        self.otg_inp.target_position  = cur.copy()

        # 标记 finished，step() 就不会再 update 产生新轨迹
        self.otg_res = Result.Finished

        # 控制量也锁到当前位置
        self.ctrl[self.joint_ids] = cur
        self.qvel[self.joint_ids] = 0.0

@dataclass
class ArmTarget:
    pos: np.ndarray
    quat: np.ndarray
    pos0: np.ndarray
    quat0: np.ndarray


def draw_panel(
    stdscr,
    *,
    active: str,
    frame_name: str,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    ik: G1IKSolver,
    targets: dict,
    inws: dict,
    local: dict,
    warn_msg: str
):
    stdscr.erase()
    stdscr.addstr(0, 0, "G1 Dual-Arm Keyboard IK | TAB switch arm | ESC quit")
    stdscr.addstr(1, 0, f"Active: {active.upper():<5} | Workspace frame: {frame_name}")

    stdscr.addstr(3, 0, "Keys: W/S X  A/D Y  R/F Z  Q/E Yaw  9 Reset XYZ  0 Reset Quat")

    # Current EE from FK
    r_now, _ = ik.forward("right", qpos)
    l_now, _ = ik.forward("left",  qpos)

    stdscr.addstr(5, 0, "EE current (world)")
    stdscr.addstr(6, 0, f"  Right: {fmt_vec(r_now)}")
    stdscr.addstr(7, 0, f"  Left : {fmt_vec(l_now)}")

    stdscr.addstr(9, 0, "Target (world)  |  Target (workspace frame)  |  inWS")
    stdscr.addstr(10, 0, f"  Right: {fmt_vec(targets['right'].pos)}  |  {fmt_vec(local['right'])}  |  {inws['right']}")
    stdscr.addstr(11, 0, f"  Left : {fmt_vec(targets['left'].pos)}   |  {fmt_vec(local['left'])}  |  {inws['left']}")

    # Joint angles
    rj = joint_deg(model, qpos, RIGHT_JOINTS)
    lj = joint_deg(model, qpos, LEFT_JOINTS)

    stdscr.addstr(13, 0, "Joint angles (deg) [7 joints]")
    stdscr.addstr(14, 0, "  Right: " + "  ".join([f"{a:+7.2f}" for a in rj]))
    stdscr.addstr(15, 0, "  Left : " + "  ".join([f"{a:+7.2f}" for a in lj]))

    stdscr.addstr(17, 0, f"Last warn: {warn_msg}")

    stdscr.refresh()


# =========================
# Main: keyboard dual-arm + workspace + live panel
# =========================
def visualize_dual_arm_keyboard(
    xml_path="models/g1_description/g1_dual_arm.xml",
    sim_only=True,                 # ✅ 默认纯 MuJoCo，不初始化 DDS
    iface="enp0s31f6",
    domain=0,
    rate_hz=100,
    pos_step=0.005,
    yaw_step_deg=5.0,
):
    ik = G1IKSolver(xml_path)
    frame_name = WORKSPACE.get("frame", "torso_link")

    # --- Optional RobotBridge ---
    rb = None
    if not sim_only:
        rb = RobotBridge(iface=iface, domain=domain, default_mode=0, kp=20.0, kd=1.0)
        if not rb.ok:
            rb = None

    # --- initial qpos0 ---
    qpos0 = ik.model.key("home").qpos.copy()
    if rb is not None:
        js = None
        for _ in range(50):
            js = rb.get_joint_states()
            if js is not None:
                break
            time.sleep(0.01)
        if js is not None:
            for jname in RIGHT_JOINTS + LEFT_JOINTS:
                jid = ik.model.joint(jname).id
                idx_real = name_to_index(jname.replace("_joint", "").replace("_", "-"))
                qpos0[jid] = js[idx_real]["q"]

    ik.data.qpos[:] = qpos0
    mujoco.mj_forward(ik.model, ik.data)

    # --- shared full state ---
    full_qpos = qpos0.copy()
    full_qvel = np.zeros_like(full_qpos)
    full_ctrl = full_qpos.copy()

    right_ctrl = ArmController(full_qpos, full_qvel, full_ctrl, 1.0 / rate_hz, ik.model, RIGHT_JOINTS)
    left_ctrl  = ArmController(full_qpos, full_qvel, full_ctrl, 1.0 / rate_hz, ik.model, LEFT_JOINTS)

    # --- init targets from FK (world frame) ---
    r_pos, r_quat = ik.forward("right", full_qpos)
    l_pos, l_quat = ik.forward("left",  full_qpos)

    targets = {
        "right": ArmTarget(pos=r_pos.copy(), quat=r_quat.copy(), pos0=r_pos.copy(), quat0=r_quat.copy()),
        "left":  ArmTarget(pos=l_pos.copy(), quat=l_quat.copy(), pos0=l_pos.copy(), quat0=l_quat.copy()),
    }

    yaw_step = math.radians(yaw_step_deg)
    dt = 1.0 / rate_hz

    def ws_local(arm: str, p_world: np.ndarray) -> np.ndarray:
        if frame_name == "world":
            return p_world
        return world_to_body_frame(ik.model, ik.data, frame_name, p_world)


    def ws_ok(arm: str, p_world: np.ndarray) -> bool:
        p_loc = ws_local(arm, p_world)
        if arm == "right":
            return point_in_convex_poly(A_R, b_R, p_loc)
        else:
            return point_in_convex_poly(A_L, b_L, p_loc)

    # --- curses ---
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active = "right"
    warn_msg = "OK"
    last_warn_t = 0.0
    last_out_t = 0.0

    try:
        with mujoco.viewer.launch_passive(ik.model, ik.data) as viewer:
            while viewer.is_running():
                key = stdscr.getch()
                changed = False

                if key != -1:
                    if key == 27:  # ESC
                        break

                    if key == 9:  # TAB
                        active = "left" if active == "right" else "right"

                    tgt = targets[active]

                    if key in (ord("w"), ord("W")):
                        tgt.pos[0] += pos_step; changed = True
                    elif key in (ord("s"), ord("S")):
                        tgt.pos[0] -= pos_step; changed = True
                    elif key in (ord("a"), ord("A")):
                        tgt.pos[1] += pos_step; changed = True
                    elif key in (ord("d"), ord("D")):
                        tgt.pos[1] -= pos_step; changed = True
                    elif key in (ord("r"), ord("R")):
                        tgt.pos[2] += pos_step; changed = True
                    elif key in (ord("f"), ord("F")):
                        tgt.pos[2] -= pos_step; changed = True
                    elif key in (ord("q"), ord("Q")):
                        tgt.quat = apply_yaw_delta(tgt.quat, +yaw_step); changed = True
                    elif key in (ord("e"), ord("E")):
                        tgt.quat = apply_yaw_delta(tgt.quat, -yaw_step); changed = True
                    elif key == ord("9"):
                        tgt.pos[:] = tgt.pos0; changed = True
                    elif key == ord("0"):
                        tgt.quat[:] = tgt.quat0; changed = True

                    # Workspace gate: OUT -> reject + stop motion + restore target to current FK
                    if changed and (not ws_ok(active, tgt.pos)):
                        now = time.time()
                        if now - last_warn_t > 0.2:
                            p_loc = ws_local(active, tgt.pos)
                            warn_msg = f"OUT-OF-WS ({active}) target_{frame_name}={fmt_vec(p_loc)} -> rejected+STOP"
                            last_warn_t = now
                            last_out_t = now

                        # STOP the active arm motion immediately
                        if active == "right":
                            right_ctrl.stop_hold()
                        else:
                            left_ctrl.stop_hold()

                        # restore target to current FK to avoid drift
                        cur_pos, cur_quat = ik.forward(active, full_qpos)
                        tgt.pos[:] = cur_pos
                        tgt.quat[:] = cur_quat
                        changed = False

                # IK only if accepted
                if changed:
                    tgt = targets[active]
                    q_seed = full_qpos.copy()
                    q_goal = ik.solve(active, tgt.pos, tgt.quat, q_seed)

                    if active == "right":
                        right_ctrl.move_to(q_goal)
                    else:
                        left_ctrl.move_to(q_goal)

                # step both controllers
                right_ctrl.step()
                left_ctrl.step()

                # push to mujoco
                ik.data.qpos[:] = full_qpos
                mujoco.mj_forward(ik.model, ik.data)

                # optional: stream to real robot
                if rb is not None:
                    q_real = {}
                    for jname in RIGHT_JOINTS + LEFT_JOINTS:
                        sim_jid = ik.model.joint(jname).id
                        q_val = float(ik.data.qpos[sim_jid])
                        idx_real = name_to_index(jname.replace("_joint", "").replace("_", "-"))
                        q_real[idx_real] = q_val
                    rb.send_qpos(q_real)

                # panel info
                local = {
                    "right": ws_local("right", targets["right"].pos),
                    "left":  ws_local("left",  targets["left"].pos),
                }
                inws = {
                    "right": ws_ok("right", targets["right"].pos),
                    "left":  ws_ok("left",  targets["left"].pos),
                }

                if inws["right"] and inws["left"]:
                    if time.time() - last_out_t > 0.3:
                        warn_msg = "OK"
                
                draw_panel(
                    stdscr,
                    active=active,
                    frame_name=frame_name,
                    model=ik.model,
                    qpos=full_qpos,
                    ik=ik,
                    targets=targets,
                    inws=inws,
                    local=local,
                    warn_msg=warn_msg,
                )

                viewer.sync()
                time.sleep(dt)

    finally:
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


if __name__ == "__main__":
    visualize_dual_arm_keyboard(sim_only=True)










# import mujoco
# import mujoco.viewer
# import numpy as np
# import time
# import math
# from time import sleep
# from ruckig import Ruckig, InputParameter, OutputParameter, Result

# from g1_ik_solver import G1IKSolver, RIGHT_JOINTS, LEFT_JOINTS
# from robotbridge import RobotBridge, name_to_index

# from scipy.spatial.transform import Rotation as R

# class ArmController:
#     def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper,
#                  timestep, model, joint_names, ee_name=None):
#         self.qpos = qpos
#         self.qvel = qvel
#         self.ctrl = ctrl
#         self.qpos_gripper = qpos_gripper
#         self.ctrl_gripper = ctrl_gripper
#         self.model = model
#         self.ee_name = ee_name

#         # joints index mapping
#         self.joint_ids = [self.model.joint(j).id for j in joint_names]
#         self.dof = len(self.joint_ids)

#         # Ruckig setup
#         self.otg = Ruckig(self.dof, timestep)
#         self.otg_inp = InputParameter(self.dof)
#         self.otg_out = OutputParameter(self.dof)

#         # limits
#         self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
#         self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
#         self.otg_res = Result.Finished

#     def move_to(self, target_qpos):
#         target = target_qpos[self.joint_ids]
#         self.otg_inp.current_position = self.qpos[self.joint_ids].copy()
#         self.otg_inp.current_velocity = self.qvel[self.joint_ids].copy()
#         self.otg_inp.target_position = target.copy()
#         self.otg_res = Result.Working

#     def step(self):
#         if self.otg_res == Result.Working:
#             self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
#             self.otg_out.pass_to_input(self.otg_inp)

#             self.qpos[self.joint_ids] = self.otg_out.new_position
#             self.qvel[self.joint_ids] = self.otg_out.new_velocity
#             self.ctrl[self.joint_ids] = self.otg_out.new_position

#         elif self.otg_res == Result.Finished:
#             # Hold the last commanded position
#             #print("FINISHED!!!!!!!!!!!!!!!!!")

#             self.ctrl[self.joint_ids] = self.otg_out.new_position
#     def is_busy(self):
#         return self.otg_res == Result.Working


# # --- 雙臂同步可視化 ---
# def visualize_dual_arm():
#     ik = G1IKSolver("models/g1_description/g1_dual_arm.xml")

#     # === 初始化 RobotBridge ===

#     rb = RobotBridge(iface="enp0s31f6", domain=0, default_mode=0, kp=20.0, kd=1.0)
#     if not rb.ok:
#         print("[warn] RobotBridge disabled – running sim only")
#     # 建立左右手控制器

#     # --- Initialize qpos0 from robot ---
#     if rb.ok:
#         print("Reading current joint state from robot...")
#         js = None
#         for _ in range(50):   # try for ~0.5s
#             js = rb.get_joint_states()
#             if js is not None:
#                 break
#             time.sleep(0.01)

#         if js is None:
#             print("[warn] Failed to read joint state; using home posture")
#             qpos0 = ik.model.key("home").qpos.copy()
#         else:
#             qpos0 = ik.model.key("home").qpos.copy()
#             for jname in RIGHT_JOINTS + LEFT_JOINTS:
#                 jid = ik.model.joint(jname).id
#                 idx_real = name_to_index(jname.replace("_joint", "").replace("_", "-"))
#                 qpos0[jid] = js[idx_real]['q']

#             ik.data.qpos[:] = qpos0
#             mujoco.mj_forward(ik.model, ik.data)
#     else:
#         qpos0 = ik.model.key("home").qpos.copy()

#     right_arm_ctrl = ArmController(
#         qpos=qpos0.copy(),
#         qvel=np.zeros_like(qpos0),
#         ctrl=qpos0.copy(),
#         qpos_gripper=np.zeros(1),
#         ctrl_gripper=np.zeros(1),
#         timestep=0.01,
#         model=ik.model,
#         joint_names=RIGHT_JOINTS,
#         ee_name="ee_site_right",
#     )
#     left_arm_ctrl = ArmController(
#         qpos=right_arm_ctrl.qpos,
#         qvel=right_arm_ctrl.qvel,
#         ctrl=right_arm_ctrl.ctrl,
#         qpos_gripper=np.zeros(1),
#         ctrl_gripper=np.zeros(1),
#         timestep=0.01,
#         model=ik.model,
#         joint_names=LEFT_JOINTS,
#         ee_name="ee_site_left",
#     )



#     # Waypoints (右手、左手一組)
#     waypoints = [
#         # (right_pos, right_quat, left_pos, left_quat)
#         (np.array([0.00062608, - 0.21180004, - 0.07396974 ]), np.array([-0.12861246,0.62181813,0.05116128,0.77083303]),
#          np.array([0.00062608,  0.21180004, - 0.07396974]), np.array([-0.12861246,0.62181813,0.05116128,0.77083303])),
#         (np.array([0.23952028, -0.3038806, -0.07864896]), np.array([-0.0093351, 0.10210487, -0.05137731, 0.99340215]),
#          np.array([0.23952028, 0.3038806, -0.07864896]), np.array([-0.0093351, 0.10210487, -0.05137731, 0.99340215])),
#         (np.array([0.23952028, -0.17538806, -0.07864896]), np.array([-0.0093351 ,  0.10210487 ,-0.05137731 , 0.99340215]),
#          np.array([0.23952028, 0.17538806, -0.07864896]), np.array([-0.0093351 ,  0.10210487 ,-0.05137731 , 0.99340215])),
#     ]
#     with mujoco.viewer.launch_passive(ik.model, ik.data) as viewer:
#         for (r_pos, r_quat, l_pos, l_quat) in waypoints:

#             # 分別 IK 解算
#             qpos_goal_r = ik.solve("right", r_pos, r_quat, right_arm_ctrl.qpos)
#             qpos_goal_l = ik.solve("left",  l_pos, l_quat,  left_arm_ctrl.qpos)

#             # Ruckig 啟動
#             right_arm_ctrl.move_to(qpos_goal_r)
#             left_arm_ctrl.move_to(qpos_goal_l)

#             # 執行到兩邊都完成
#             while right_arm_ctrl.is_busy() or left_arm_ctrl.is_busy():
#                 right_arm_ctrl.step()
#                 left_arm_ctrl.step()

#                 ik.data.qpos[:] = right_arm_ctrl.qpos
#                 mujoco.mj_forward(ik.model, ik.data)

#                 # === MuJoCo → Real robot streaming ===
#                 if rb.ok:
#                     q_real = {}
#                     for jname in RIGHT_JOINTS + LEFT_JOINTS:
#                         #print(jname)
#                         sim_jid = ik.model.joint(jname).id
#                         q_val = ik.data.qpos[sim_jid]

#                         idx_real = name_to_index(jname.replace("_joint", "").replace("_", "-"))
#                         q_real[idx_real] = float(q_val)

#                     rb.send_qpos(q_real)

#                 # === 即時顯示末端座標與姿態 ===
#                 ee_r = ik.data.site_xpos[ik.model.site("ee_site_right").id]
#                 ee_l = ik.data.site_xpos[ik.model.site("ee_site_left").id]
#                 # 取得 site 的旋轉矩陣 (3x3)
#                 mat_r = ik.data.site_xmat[ik.model.site("ee_site_right").id].reshape(3, 3)
#                 mat_l = ik.data.site_xmat[ik.model.site("ee_site_left").id].reshape(3, 3)

#                 # 轉成四元數
#                 quat_r = R.from_matrix(mat_r).as_quat()  # xyzw
#                 quat_l = R.from_matrix(mat_l).as_quat()

#                 rpy_r = R.from_quat(quat_r[[1, 2, 3, 0]]).as_euler('xyz', degrees=True)  # MuJoCo xyzw → SciPy wxyz
#                 rpy_l = R.from_quat(quat_l[[1, 2, 3, 0]]).as_euler('xyz', degrees=True)

#                 if viewer.user_scn.ngeom % 10 == 0:
#                     print(f"[RightEE] pos={ee_r.round(3)} | rpy={rpy_r.round(1)}")
#                     print(f"[LeftEE] pos={ee_l.round(3)} | rpy={rpy_l.round(1)}")
#                 # 標記 EE 位置
#                 for ee_name, color in [("ee_site_right", [0, 1, 0, 1]),
#                                        ("ee_site_left",  [0, 0, 1, 1])]:
#                     site_id = ik.model.site(ee_name).id
#                     ee_pos = ik.data.site_xpos[site_id].copy()
#                     viewer.user_scn.ngeom = min(viewer.user_scn.ngeom + 1, viewer.user_scn.maxgeom)
#                     g = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
#                     mujoco.mjv_initGeom(
#                         g,
#                         type=mujoco.mjtGeom.mjGEOM_SPHERE,
#                         size=[0.005, 0, 0],
#                         pos=ee_pos,
#                         mat=np.eye(3).flatten(),
#                         rgba=color,
#                     )

#                 viewer.sync()
#                 time.sleep(0.01)

#         print("Dual-arm trajectory completed.")
#         while viewer.is_running():
#             mujoco.mj_forward(ik.model, ik.data)
#             viewer.sync()
#             sleep(0.01)


# if __name__ == "__main__":
#     visualize_dual_arm()
