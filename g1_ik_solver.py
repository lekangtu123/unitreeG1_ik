# Refs:
# - https://github.com/bulletphysics/bullet3/blob/master/examples/ThirdPartyLibs/BussIK/Jacobian.cpp
# - https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
# - https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py

import mujoco
import numpy as np

DAMPING_COEFF = 1e-3
MAX_ANGLE_CHANGE = np.deg2rad(45.0)

LEFT_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

SITE_BY_ARM = {
    "left":  "ee_site_left",
    "right": "ee_site_right",
}


class G1IKSolver:
    """
    Damped least-squares IK with a nullspace bias to the 'home' keyframe.
    Works per-arm; other joints remain untouched.
    """

    def __init__(self, xml_path="models/g1_description/g1_dual_arm.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Preallocations for full-model Jacobian; we'll slice the arm columns
        self.jac_pos = np.zeros((3, self.model.nv))
        self.jac_rot = np.zeros((3, self.model.nv))
        self.jac6   = np.zeros((6, self.model.nv))

        # 6x6 damping (task space)
        self.damp6 = DAMPING_COEFF * np.eye(6)

        # Cache the 'home' keyframe qpos as the nullspace target
        try:
            self.qpos_home = self.model.key("home").qpos.copy()
        except Exception:
            # Fallback if keyframe missing
            self.qpos_home = np.zeros(self.model.nq, dtype=float)

    # ---------- helpers ----------
    def _joint_ids_and_qadr(self, joint_names):
        jids = [self.model.joint(name).id for name in joint_names]
        qadr = np.array([self.model.jnt_qposadr[jid] for jid in jids], dtype=int)
        vadr = np.array([self.model.jnt_dofadr[jid]  for jid in jids], dtype=int)
        return jids, qadr, vadr

    def _arm_index_block(self, vadr, dof_per_joint=1):
        """
        For hinge joints (1 dof each), arm dof indices in qvel space are exactly vadr.
        We still keep it generic in case you later add multi-dof joints.
        """
        cols = []
        for va in vadr:
            cols.extend(range(va, va + dof_per_joint))
        return np.array(cols, dtype=int)

    # ---------- main API ----------
    def solve(
        self,
        arm: str,
        target_pos_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray,
        qpos_init_full: np.ndarray,
        max_iters: int = 100,
        err_thresh: float = 1e-4,
    ) -> np.ndarray:
        """
        arm: 'left' or 'right'
        target_pos_xyz: (3,)
        target_quat_xyzw: (4,) in (x,y,z,w)
        qpos_init_full: full nq vector (all joints)
        returns: full nq solution (only arm joints updated)
        """
        assert arm in ("left", "right"), "arm must be 'left' or 'right'"

        # Select arm resources
        joint_names = LEFT_JOINTS if arm == "left" else RIGHT_JOINTS
        site_name   = SITE_BY_ARM[arm]
        site_id     = self.model.site(site_name).id

        # Indices for the arm in qpos/qvel
        _, qadr, vadr = self._joint_ids_and_qadr(joint_names)
        cols = self._arm_index_block(vadr, dof_per_joint=1)  # 7 columns

        # Nullspace target (home) sliced to the arm
        qpos_home_arm = self.qpos_home[qadr].copy()

        # Prepare state
        self.data.qpos[:] = qpos_init_full
        mujoco.mj_forward(self.model, self.data)

        # Convert target quat (xyzw) -> (wxyz) for mujoco
        target_quat_wxyz = np.array(
            [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]],
            dtype=float,
        )

        # Buffers for orientation math
        site_quat = np.empty(4)
        site_quat_inv = np.empty(4)
        err_quat = np.empty(4)
        err6 = np.zeros(6)
        err_pos = err6[:3]
        err_rot = err6[3:]

        for _ in range(max_iters):
            # Current site pose
            mujoco.mj_forward(self.model, self.data)  # ensures xpos/xmat up-to-date
            site_pos = self.data.site_xpos[site_id]          # (3,)
            site_mat = self.data.site_xmat[site_id]          # (9,) row-major


            # Position error
            err_pos[:] = target_pos_xyz - site_pos

            # Rotation error: site_mat -> site_quat (wxyz)
            mujoco.mju_mat2Quat(site_quat, site_mat)
            mujoco.mju_negQuat(site_quat_inv, site_quat)           # inverse
            mujoco.mju_mulQuat(err_quat, target_quat_wxyz, site_quat_inv)
            mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)

            # Converged?
            if np.linalg.norm(err6) < err_thresh:
                break

            # 6×nv Jacobian at site, then slice arm columns
            self.jac_pos.fill(0.0)
            self.jac_rot.fill(0.0)
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, site_id)
            self.jac6[:3, :] = self.jac_pos
            self.jac6[3:, :] = self.jac_rot
            J = self.jac6[:, cols]                               # 6×7

            # Damped least-squares with nullspace bias toward home
            # dq_arm = J^T (J J^T + λI)^-1 e  +  (I - J^T (J J^T + λI)^-1 J) (q_home - q_curr)
            JJt = J @ J.T
            A = JJt + self.damp6  # 6×6
            rhs = err6
            y = np.linalg.solve(A, rhs)                          # 6,
            dq_arm = J.T @ y                                     # 7,

            # Nullspace bias
            q_curr_arm = self.data.qpos[qadr]
            qbias = self._wrap_to_pi(qpos_home_arm - q_curr_arm)  # 7,
            # Use the same (J J^T + λI)^-1 for projector for stability
            J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
            N = np.eye(J.shape[1]) - J_pinv @ J                  # 7×7
            dq_arm += N @ qbias

            # Step size limit
            maxmag = np.max(np.abs(dq_arm))
            if maxmag > MAX_ANGLE_CHANGE:
                dq_arm *= (MAX_ANGLE_CHANGE / maxmag)

            # Apply update only to arm joints
            dq_full = np.zeros(self.model.nv)
            dq_full[cols] = dq_arm

            # Integrate position with full nv update
            mujoco.mj_integratePos(self.model, self.data.qpos, dq_full, 1.0)

        return self.data.qpos.copy()

    @staticmethod
    def _wrap_to_pi(x):
        # elementwise wrap to [-pi, pi]
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def forward(self, arm: str, qpos_full: np.ndarray):
        """
        Forward kinematics for the chosen arm.
        Args:
            arm: 'left' or 'right'
            qpos_full: full nq vector of joint positions
        Returns:
            pos: (3,) end-effector position in world frame
            quat_xyzw: (4,) end-effector orientation quaternion (x,y,z,w)
        """
        assert arm in ("left", "right")
        site_name = SITE_BY_ARM[arm]
        site_id = self.model.site(site_name).id

        # Set joint configuration and run FK
        self.data.qpos[:] = qpos_full
        mujoco.mj_forward(self.model, self.data)

        # Extract pose
        pos = self.data.site_xpos[site_id].copy()
        mat = self.data.site_xmat[site_id].copy()  # 3x3 rotation (flattened)
        quat_wxyz = np.empty(4)
        mujoco.mju_mat2Quat(quat_wxyz, mat)

        # Convert MuJoCo's (w,x,y,z) to (x,y,z,w)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos, quat_xyzw

if __name__ == "__main__":
    # Example: solve for right arm to a reachable pose/orientation
    ik = G1IKSolver(xml_path="models/g1_description/g1_dual_arm.xml")

    # Start from 'home'
    qpos0 = ik.model.key("home").qpos.copy()

    # Target pose (meters) and quat (xyzw)
    target_pos = np.array([0.00062608, - 0.21180004, - 0.07396974])        # tweak to your workspace
    target_quat_xyzw = np.array([-0.12861246, 0.62181813, 0.05116128, 0.77083303]) # identity

    q_sol = ik.solve(
        arm="right",
        target_pos_xyz=target_pos,
        target_quat_xyzw=target_quat_xyzw,
        qpos_init_full=qpos0,
        max_iters=60,
        err_thresh=1e-4,
    )
    print("Solution (deg, arm joints only shown):")
    # pretty print the right arm joints
    _, qadr_r, _ = ik._joint_ids_and_qadr(RIGHT_JOINTS)
    print(np.rad2deg(q_sol[qadr_r]))

    pos_r, quat_r = ik.forward("right", q_sol)

    print("Right hand FK:", pos_r, quat_r)

