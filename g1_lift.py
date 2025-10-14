import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from time import sleep
from ruckig import Ruckig, InputParameter, OutputParameter, Result

from g1_ik_solver import G1IKSolver, RIGHT_JOINTS, LEFT_JOINTS
from robotbridge import RobotBridge, name_to_index

from scipy.spatial.transform import Rotation as R

class ArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper,
                 timestep, model, joint_names, ee_name=None):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper
        self.model = model
        self.ee_name = ee_name

        # joints index mapping
        self.joint_ids = [self.model.joint(j).id for j in joint_names]
        self.dof = len(self.joint_ids)

        # Ruckig setup
        self.otg = Ruckig(self.dof, timestep)
        self.otg_inp = InputParameter(self.dof)
        self.otg_out = OutputParameter(self.dof)

        # limits
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
        self.otg_res = Result.Finished

    def move_to(self, target_qpos):
        target = target_qpos[self.joint_ids]
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
            # Hold the last commanded position
            #print("FINISHED!!!!!!!!!!!!!!!!!")

            self.ctrl[self.joint_ids] = self.otg_out.new_position
    def is_busy(self):
        return self.otg_res == Result.Working


# --- 雙臂同步可視化 ---
def visualize_dual_arm():
    ik = G1IKSolver("models/g1_description/g1_dual_arm.xml")

    # === 初始化 RobotBridge ===

    rb = RobotBridge(iface="enp0s31f6", domain=0, default_mode=0, kp=20.0, kd=1.0)
    if not rb.ok:
        print("[warn] RobotBridge disabled – running sim only")
    # 建立左右手控制器

    # --- Initialize qpos0 from robot ---
    if rb.ok:
        print("Reading current joint state from robot...")
        js = None
        for _ in range(50):   # try for ~0.5s
            js = rb.get_joint_states()
            if js is not None:
                break
            time.sleep(0.01)

        if js is None:
            print("[warn] Failed to read joint state; using home posture")
            qpos0 = ik.model.key("home").qpos.copy()
        else:
            qpos0 = ik.model.key("home").qpos.copy()
            for jname in RIGHT_JOINTS + LEFT_JOINTS:
                jid = ik.model.joint(jname).id
                idx_real = name_to_index(jname.replace("_joint", "").replace("_", "-"))
                qpos0[jid] = js[idx_real]['q']

            ik.data.qpos[:] = qpos0
            mujoco.mj_forward(ik.model, ik.data)
    else:
        qpos0 = ik.model.key("home").qpos.copy()

    right_arm_ctrl = ArmController(
        qpos=qpos0.copy(),
        qvel=np.zeros_like(qpos0),
        ctrl=qpos0.copy(),
        qpos_gripper=np.zeros(1),
        ctrl_gripper=np.zeros(1),
        timestep=0.01,
        model=ik.model,
        joint_names=RIGHT_JOINTS,
        ee_name="ee_site_right",
    )
    left_arm_ctrl = ArmController(
        qpos=right_arm_ctrl.qpos,
        qvel=right_arm_ctrl.qvel,
        ctrl=right_arm_ctrl.ctrl,
        qpos_gripper=np.zeros(1),
        ctrl_gripper=np.zeros(1),
        timestep=0.01,
        model=ik.model,
        joint_names=LEFT_JOINTS,
        ee_name="ee_site_left",
    )



    # Waypoints (右手、左手一組)
    waypoints = [
        # (right_pos, right_quat, left_pos, left_quat)
        (np.array([0.00062608, - 0.21180004, - 0.07396974 ]), np.array([-0.12861246,0.62181813,0.05116128,0.77083303]),
         np.array([0.00062608,  0.21180004, - 0.07396974]), np.array([-0.12861246,0.62181813,0.05116128,0.77083303])),
        (np.array([0.23952028, -0.3038806, -0.07864896]), np.array([-0.0093351, 0.10210487, -0.05137731, 0.99340215]),
         np.array([0.23952028, 0.3038806, -0.07864896]), np.array([-0.0093351, 0.10210487, -0.05137731, 0.99340215])),
        (np.array([0.23952028, -0.17538806, -0.07864896]), np.array([-0.0093351 ,  0.10210487 ,-0.05137731 , 0.99340215]),
         np.array([0.23952028, 0.17538806, -0.07864896]), np.array([-0.0093351 ,  0.10210487 ,-0.05137731 , 0.99340215])),
    ]
    with mujoco.viewer.launch_passive(ik.model, ik.data) as viewer:
        for (r_pos, r_quat, l_pos, l_quat) in waypoints:

            # 分別 IK 解算
            qpos_goal_r = ik.solve("right", r_pos, r_quat, right_arm_ctrl.qpos)
            qpos_goal_l = ik.solve("left",  l_pos, l_quat,  left_arm_ctrl.qpos)

            # Ruckig 啟動
            right_arm_ctrl.move_to(qpos_goal_r)
            left_arm_ctrl.move_to(qpos_goal_l)

            # 執行到兩邊都完成
            while right_arm_ctrl.is_busy() or left_arm_ctrl.is_busy():
                right_arm_ctrl.step()
                left_arm_ctrl.step()

                ik.data.qpos[:] = right_arm_ctrl.qpos
                mujoco.mj_forward(ik.model, ik.data)

                # === MuJoCo → Real robot streaming ===
                if rb.ok:
                    q_real = {}
                    for jname in RIGHT_JOINTS + LEFT_JOINTS:
                        #print(jname)
                        sim_jid = ik.model.joint(jname).id
                        q_val = ik.data.qpos[sim_jid]

                        idx_real = name_to_index(jname.replace("_joint", "").replace("_", "-"))
                        q_real[idx_real] = float(q_val)

                    rb.send_qpos(q_real)

                # === 即時顯示末端座標與姿態 ===
                ee_r = ik.data.site_xpos[ik.model.site("ee_site_right").id]
                ee_l = ik.data.site_xpos[ik.model.site("ee_site_left").id]
                # 取得 site 的旋轉矩陣 (3x3)
                mat_r = ik.data.site_xmat[ik.model.site("ee_site_right").id].reshape(3, 3)
                mat_l = ik.data.site_xmat[ik.model.site("ee_site_left").id].reshape(3, 3)

                # 轉成四元數
                quat_r = R.from_matrix(mat_r).as_quat()  # xyzw
                quat_l = R.from_matrix(mat_l).as_quat()

                rpy_r = R.from_quat(quat_r[[1, 2, 3, 0]]).as_euler('xyz', degrees=True)  # MuJoCo xyzw → SciPy wxyz
                rpy_l = R.from_quat(quat_l[[1, 2, 3, 0]]).as_euler('xyz', degrees=True)

                if viewer.user_scn.ngeom % 10 == 0:
                    print(f"[RightEE] pos={ee_r.round(3)} | rpy={rpy_r.round(1)}")
                    print(f"[LeftEE] pos={ee_l.round(3)} | rpy={rpy_l.round(1)}")
                # 標記 EE 位置
                for ee_name, color in [("ee_site_right", [0, 1, 0, 1]),
                                       ("ee_site_left",  [0, 0, 1, 1])]:
                    site_id = ik.model.site(ee_name).id
                    ee_pos = ik.data.site_xpos[site_id].copy()
                    viewer.user_scn.ngeom = min(viewer.user_scn.ngeom + 1, viewer.user_scn.maxgeom)
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
                    mujoco.mjv_initGeom(
                        g,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.005, 0, 0],
                        pos=ee_pos,
                        mat=np.eye(3).flatten(),
                        rgba=color,
                    )

                viewer.sync()
                time.sleep(0.01)

        print("Dual-arm trajectory completed.")
        while viewer.is_running():
            mujoco.mj_forward(ik.model, ik.data)
            viewer.sync()
            sleep(0.01)


if __name__ == "__main__":
    visualize_dual_arm()
