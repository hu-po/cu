# parallel_tattoo_ik.py
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import os
import time
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# --- Configuration ---
@dataclass
class IKConfig:
    # --- Input/Output ---
    ik_targets_path: str = "outputs/tattoo_ik_poses.npy" # Source of all targets
    urdf_path: str = "~/dev/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf" # Source URDF
    output_dir: str = "ik_batches_output" # Directory for batch USDs

    # --- Simulation & Solver ---
    seed: int = 42
    num_iters: int = 200 # Number of IK iterations per batch
    batch_size: int = 64 # Max environments per batch
    device: str = None # Auto-select unless specified
    headless: bool = False
    fps: int = 30 # Rendering FPS
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0) # Offset from ee_gripper_link
    # IK Solver Parameters (from sampling_gn morph)
    ik_sigma: float = 1e-2
    ik_num_samples: int = 10
    ik_damping: float = 0.1

    # --- Arm Initialization ---
    arm_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_rot_offset: list[tuple[tuple[float, float, float], float]] = field(default_factory=lambda: [
        ((1.0, 0.0, 0.0), -math.pi * 0.5),
    ])
    qpos_home: list[float] = field(default_factory=lambda: [
        0, np.pi/12, np.pi/12, 0, 0, 0, 0, 0
    ])
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: [
        (-3.054, 3.054), (0.0, 3.14), (0.0, 2.356), (-1.57, 1.57),
        (-1.57, 1.57), (-3.14, 3.14), (0.0, 0.044), (0.0, 0.044)
    ])
    joint_attach_ke: float = 1600.0
    joint_attach_kd: float = 20.0

    # --- Visualization ---
    gizmo_radius: float = 0.005
    gizmo_length: float = 0.02
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0)
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0)
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5)
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5)
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0)

# --- Warp Math & Kernels (Unchanged from previous version) ---
# [Keep the @wp.func and @wp.kernel definitions here:
# quat_mul, quat_conjugate, quat_orientation_error,
# compute_ee_error_kernel, clip_joints_kernel,
# calculate_gizmo_transforms_kernel]
@wp.func
def quat_mul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    # ... (implementation from previous script) ...
        return wp.quat(
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
        q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
    )

@wp.func
def quat_conjugate(q: wp.quat) -> wp.quat:
    # ... (implementation from previous script) ...
    return wp.quat(-q[0], -q[1], -q[2], q[3])

@wp.func
def quat_orientation_error(target: wp.quat, current: wp.quat) -> wp.vec3:
    # ... (implementation from previous script) ...
    q_err = quat_mul(target, quat_conjugate(current))
    qw = wp.where(q_err[3] < 0.0, -q_err[3], q_err[3])
    qx = wp.where(q_err[3] < 0.0, -q_err[0], q_err[0])
    qy = wp.where(q_err[3] < 0.0, -q_err[1], q_err[1])
    qz = wp.where(q_err[3] < 0.0, -q_err[2], q_err[2])
    return wp.vec3(qx, qy, qz) * 2.0

@wp.kernel
def compute_ee_error_kernel(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    target_pos: wp.array(dtype=wp.vec3),
    target_ori: wp.array(dtype=wp.quat),
    error_out: wp.array(dtype=wp.float32)
):
    # ... (implementation from previous script) ...
    tid = wp.tid() # Environment index
    t_flat = body_q[tid * num_links + ee_link_index]
    t_pos = wp.vec3(t_flat[0], t_flat[1], t_flat[2])
    t_ori = wp.quat(t_flat[3], t_flat[4], t_flat[5], t_flat[6])
    current_pos = wp.transform_point(wp.transform(t_pos, t_ori), ee_link_offset)
    current_ori = t_ori
    pos_err = target_pos[tid] - current_pos
    ori_err = quat_orientation_error(target_ori[tid], current_ori)
    base = tid * 6
    error_out[base + 0] = pos_err.x
    error_out[base + 1] = pos_err.y
    error_out[base + 2] = pos_err.z
    error_out[base + 3] = ori_err.x
    error_out[base + 4] = ori_err.y
    error_out[base + 5] = ori_err.z

@wp.kernel
def clip_joints_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_limits_min: wp.array(dtype=wp.float32),
    joint_limits_max: wp.array(dtype=wp.float32),
    num_envs: int,
    dof: int
):
    # ... (implementation from previous script) ...
    tid = wp.tid()
    joint_idx = tid % dof
    current_val = joint_q[tid]
    min_val = joint_limits_min[joint_idx]
    max_val = joint_limits_max[joint_idx]
    joint_q[tid] = wp.clamp(current_val, min_val, max_val)

@wp.kernel
def calculate_gizmo_transforms_kernel(
    body_q: wp.array(dtype=wp.transform),
    targets_pos: wp.array(dtype=wp.vec3),
    targets_ori: wp.array(dtype=wp.quat),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    num_envs: int,
    rot_x_axis_q: wp.quat,
    rot_y_axis_q: wp.quat,
    rot_z_axis_q: wp.quat,
    cone_half_height: float,
    out_gizmo_pos: wp.array(dtype=wp.vec3),
    out_gizmo_rot: wp.array(dtype=wp.quat)
):
    # ... (implementation from previous script) ...
    tid = wp.tid()
    # Target Gizmos
    target_pos = targets_pos[tid]
    target_ori = targets_ori[tid]
    target_rot_x = quat_mul(target_ori, rot_x_axis_q)
    target_rot_y = quat_mul(target_ori, rot_y_axis_q)
    target_rot_z = quat_mul(target_ori, rot_z_axis_q)
    offset_vec = wp.vec3(0.0, cone_half_height, 0.0)
    offset_x = wp.quat_rotate(target_rot_x, offset_vec)
    offset_y = wp.quat_rotate(target_rot_y, offset_vec)
    offset_z = wp.quat_rotate(target_rot_z, offset_vec)
    base_idx = tid * 6
    out_gizmo_pos[base_idx + 0] = target_pos - offset_x
    out_gizmo_rot[base_idx + 0] = target_rot_x
    out_gizmo_pos[base_idx + 1] = target_pos - offset_y
    out_gizmo_rot[base_idx + 1] = target_rot_y
    out_gizmo_pos[base_idx + 2] = target_pos - offset_z
    out_gizmo_rot[base_idx + 2] = target_rot_z
    # End-Effector Gizmos
    t_flat = body_q[tid * num_links + ee_link_index]
    ee_link_pos = wp.vec3(t_flat[0], t_flat[1], t_flat[2])
    ee_link_ori = wp.quat(t_flat[3], t_flat[4], t_flat[5], t_flat[6])
    ee_tip_pos = wp.transform_point(wp.transform(ee_link_pos, ee_link_ori), ee_link_offset)
    ee_rot_x = quat_mul(ee_link_ori, rot_x_axis_q)
    ee_rot_y = quat_mul(ee_link_ori, rot_y_axis_q)
    ee_rot_z = quat_mul(ee_link_ori, rot_z_axis_q)
    ee_offset_x = wp.quat_rotate(ee_rot_x, offset_vec)
    ee_offset_y = wp.quat_rotate(ee_rot_y, offset_vec)
    ee_offset_z = wp.quat_rotate(ee_rot_z, offset_vec)
    out_gizmo_pos[base_idx + 3] = ee_tip_pos - ee_offset_x
    out_gizmo_rot[base_idx + 3] = ee_rot_x
    out_gizmo_pos[base_idx + 4] = ee_tip_pos - ee_offset_y
    out_gizmo_rot[base_idx + 4] = ee_rot_y
    out_gizmo_pos[base_idx + 5] = ee_tip_pos - ee_offset_z
    out_gizmo_rot[base_idx + 5] = ee_rot_z

# --- Solver Class (Modified for Batching) ---
class ParallelTattooIKSolver:
    # Takes config, target data/size for *this batch*, and output path for *this batch*
    def __init__(self, config: IKConfig, target_poses_batch_np: np.ndarray, batch_num_envs: int, batch_usd_path: str):
        self.config = config
        self.profiler = {}
        self.render_time = 0.0
        self.frame_dt = 1.0 / config.fps if config.fps > 0 else 0.0
        self.num_envs = batch_num_envs # Use batch size here

        # --- Use provided Targets for this batch ---
        if target_poses_batch_np.ndim != 2 or target_poses_batch_np.shape[1] != 7:
             raise ValueError(f"Batch targets have wrong shape: {target_poses_batch_np.shape}")
        if len(target_poses_batch_np) != batch_num_envs:
             raise ValueError(f"Target array length {len(target_poses_batch_np)} != batch_num_envs {batch_num_envs}")

        target_pos_np = target_poses_batch_np[:, :3].copy()
        target_ori_np = target_poses_batch_np[:, 3:].copy()
        self.targets_pos_wp = wp.array(target_pos_np, dtype=wp.vec3, device=config.device)
        self.targets_ori_wp = wp.array(target_ori_np, dtype=wp.quat, device=config.device)
        log.info(f"Solver instance created for batch with {self.num_envs} environments.")

        # --- Build Model for this batch size ---
        with wp.ScopedTimer("model_build", print=False, active=True, dict=self.profiler):
            self.rng = np.random.default_rng(self.config.seed) # Seed potentially reused, maybe advance seed per batch?

            # Parse URDF once (can be cached or done outside if needed)
            urdf_path = os.path.expanduser(self.config.urdf_path)
            if not os.path.exists(urdf_path): raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            articulation_builder = wp.sim.ModelBuilder()
            wp.sim.parse_urdf(urdf_path, articulation_builder, floating=False)

            # Store arm properties
            self.num_links = len(articulation_builder.body_name)
            self.dof = len(articulation_builder.joint_q)
            self.joint_limits = np.array(self.config.joint_limits, dtype=np.float32)
            self.joint_limits_min_wp = wp.array(self.joint_limits[:, 0], dtype=wp.float32, device=config.device)
            self.joint_limits_max_wp = wp.array(self.joint_limits[:, 1], dtype=wp.float32, device=config.device)
            # Only log DoF once outside if running multiple batches
            # log.info(f"Parsed URDF with {self.num_links} links and {self.dof} DoF.")

            # Find EE link index
            self.ee_link_offset = wp.vec3(self.config.ee_link_offset)
            self.ee_link_index = -1
            ee_link_name = "ee_gripper_link"
            for i, name in enumerate(articulation_builder.body_name):
                if name == ee_link_name: self.ee_link_index = i; break
            if self.ee_link_index == -1:
                ee_joint_name = "ee_gripper"
                found_joint_idx = -1
                for j_idx, j_name in enumerate(articulation_builder.joint_name):
                     if j_name == ee_joint_name: found_joint_idx = j_idx; break
                if found_joint_idx != -1:
                     self.ee_link_index = articulation_builder.joint_child[found_joint_idx]
                else: raise ValueError(f"Could not find link '{ee_link_name}' or joint '{ee_joint_name}' in URDF")
            # log.info(f"Using End-Effector Link Index: {self.ee_link_index}")

            # Calculate initial arm orientation
            _initial_arm_orientation_wp = wp.quat_identity()
            for axis, angle in self.config.arm_rot_offset:
                rot_quat_wp = wp.quat_from_axis_angle(wp.vec3(axis), angle)
                _initial_arm_orientation_wp = quat_mul(rot_quat_wp, _initial_arm_orientation_wp)
            initial_arm_transform = wp.transform(wp.vec3(self.config.arm_base_pos), _initial_arm_orientation_wp)

            # Build environments for the current batch
            builder = wp.sim.ModelBuilder()
            for _ in range(self.num_envs): # Use current batch size
                builder.add_builder(articulation_builder, xform=initial_arm_transform)

            # Finalize model
            self.model = builder.finalize(device=config.device)
            self.model.ground = False
            self.model.joint_attach_ke = self.config.joint_attach_ke
            self.model.joint_attach_kd = self.config.joint_attach_kd
            self.model.joint_q.requires_grad = False

            # Set initial joint angles (all arms start at home in this batch)
            initial_joint_q = []
            qpos_home_adjusted = (self.config.qpos_home + [0.0]*self.dof)[:self.dof]
            for _ in range(self.num_envs): # Use current batch size
                initial_joint_q.extend(qpos_home_adjusted)
            self.model.joint_q.assign(wp.array(initial_joint_q, dtype=wp.float32, device=config.device))

            # --- State & Solver Prep ---
            self.state = self.model.state()
            self.ee_error_wp = wp.zeros(self.num_envs * 6, dtype=wp.float32, device=config.device)

        # --- Renderer Setup ---
        self.renderer = None
        if not config.headless:
            # Use the specific path passed for this batch
            log.info(f"Initializing renderer for batch, outputting to {batch_usd_path}")
            try:
                self.renderer = wp.sim.render.SimRenderer(self.model, batch_usd_path, scaling=1.0)
                self.gizmo_pos_wp = wp.zeros(self.num_envs * 6, dtype=wp.vec3, device=config.device)
                self.gizmo_rot_wp = wp.zeros(self.num_envs * 6, dtype=wp.quat, device=config.device)
                self.rot_x_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), math.pi / 2.)
                self.rot_y_axis_q_wp = wp.quat_identity()
                self.rot_z_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(1., 0., 0.), -math.pi / 2.)
            except Exception as e:
                 log.error(f"Failed to initialize renderer for batch: {e}. Running headless.", exc_info=True)
                 # Don't globally set config.headless=True, just skip rendering for this batch if needed
                 self.renderer = None # Ensure renderer is None if setup fails

    # --- Methods: compute_ee_error, _ik_step, step, render_gizmos, render ---
    # [These methods remain largely the same as in the previous version,
    #  they will operate on self.num_envs which is now the batch size]
    def compute_ee_error(self, joint_q_wp: wp.array) -> np.ndarray:
        # ... (implementation from previous script) ...
        with wp.ScopedTimer("eval_fk", print=False, active=True, dict=self.profiler):
            # Use the provided joint angles, not necessarily self.model.joint_q
            wp.sim.eval_fk(self.model, joint_q_wp, None, None, self.state)

        with wp.ScopedTimer("error_kernel", print=False, active=True, dict=self.profiler):
            wp.launch(
                compute_ee_error_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state.body_q, # body_q is updated by eval_fk
                    self.num_links,
                    self.ee_link_index,
                    self.ee_link_offset,
                    self.targets_pos_wp, # Use loaded targets
                    self.targets_ori_wp, # Use loaded targets
                ],
                outputs=[self.ee_error_wp], # ee_error is updated in-place
                device=self.config.device
            )
        return self.ee_error_wp.numpy() # Return numpy array

    def _ik_step(self):
        # ... (implementation from previous script - sampling_gn logic) ...
        # Config parameters for the solver
        sigma = self.config.ik_sigma
        num_samples = self.config.ik_num_samples
        damping = self.config.ik_damping
        dof = self.dof
        num_envs = self.num_envs # This is now the batch size

        # Get current joint angles (GPU -> CPU)
        q_current_np = self.model.joint_q.numpy().copy() # shape: (num_envs * dof,)
        q_updated_np = q_current_np.copy() # Array to store updates

        # Compute baseline error (using current joints on GPU)
        q_current_wp = wp.array(q_current_np, dtype=wp.float32, device=self.config.device)
        error_base_all_np = self.compute_ee_error(q_current_wp) # (num_envs * 6,)

        # Temporary array for perturbed joint angles on GPU
        q_temp_wp = wp.empty_like(self.model.joint_q)

        # Process each environment in the current batch
        for e in range(num_envs): # Loop up to batch size
            base_idx_q = e * dof
            base_idx_err = e * 6
            q_e = q_current_np[base_idx_q : base_idx_q + dof].copy() # (dof,)
            error_base = error_base_all_np[base_idx_err : base_idx_err + 6].copy() # (6,)

            # Generate random perturbations
            D = self.rng.normal(0.0, sigma, size=(num_samples, dof)).astype(np.float32)
            Delta = np.zeros((num_samples, 6), dtype=np.float32)

            # Sampling Loop (Inefficient version from reference morph)
            with wp.ScopedTimer("sampling_loop", print=False, active=True, dict=self.profiler):
                 error_sample_all_np = np.zeros((num_samples, num_envs * 6), dtype=np.float32)
                 for i in range(num_samples):
                      q_sample = q_e + D[i]
                      q_temp_np = q_current_np.copy() # Get the full current state for the batch
                      q_temp_np[base_idx_q : base_idx_q + dof] = q_sample # Modify only env 'e'
                      q_temp_wp.assign(q_temp_np) # Assign the modified batch state to GPU
                      error_sample_all_np[i, :] = self.compute_ee_error(q_temp_wp) # Compute error for the whole batch again

                 # Extract the error difference for environment 'e'
                 for i in range(num_samples):
                     error_sample_e = error_sample_all_np[i, base_idx_err : base_idx_err + 6]
                     Delta[i] = error_sample_e - error_base

            # Estimate Jacobian
            with wp.ScopedTimer("jacobian_estimation", print=False, active=True, dict=self.profiler):
                try:
                    J_transpose_est, _, _, _ = np.linalg.lstsq(D, Delta, rcond=None)
                    J_est = J_transpose_est.T
                except np.linalg.LinAlgError: continue # Skip env e

            # Compute Gauss-Newton update
            with wp.ScopedTimer("gn_update", print=False, active=True, dict=self.profiler):
                try:
                    JJt = J_est @ J_est.T
                    A = JJt + damping * np.eye(6, dtype=np.float32)
                    y = np.linalg.solve(A, error_base)
                    delta_q = -J_est.T @ y
                except np.linalg.LinAlgError: continue # Skip env e

            # Update joint configuration for this environment
            q_updated_np[base_idx_q : base_idx_q + dof] += delta_q

        # Assign the updated joint angles back to the GPU model
        self.model.joint_q.assign(q_updated_np)

        # Clip joint angles
        with wp.ScopedTimer("clip_joints", print=False, active=True, dict=self.profiler):
             wp.launch(
                 kernel=clip_joints_kernel,
                 dim=self.num_envs * self.dof, # Still uses batch size num_envs
                 inputs=[ self.model.joint_q, self.joint_limits_min_wp, self.joint_limits_max_wp, self.num_envs, self.dof ],
                 device=self.config.device
             )

    def step(self):
        # ... (implementation from previous script) ...
        with wp.ScopedTimer("ik_step", print=False, active=True, dict=self.profiler):
            self._ik_step()
        self.render_time += self.frame_dt

    def render_gizmos(self):
        # ... (implementation from previous script) ...
         if self.renderer is None or self.gizmo_pos_wp is None:
             return
         # Ensure FK is up-to-date
         wp.sim.eval_fk(self.model, self.model.joint_q, None, None, self.state)

         radius = self.config.gizmo_radius
         half_height = self.config.gizmo_length / 2.0

         with wp.ScopedTimer("gizmo_kernel", print=False, active=True, dict=self.profiler):
             wp.launch(
                 kernel=calculate_gizmo_transforms_kernel,
                 dim=self.num_envs,
                 inputs=[
                     self.state.body_q,
                     self.targets_pos_wp, # Use loaded targets
                     self.targets_ori_wp, # Use loaded targets
                     self.num_links,
                     self.ee_link_index,
                     self.ee_link_offset,
                     self.num_envs,
                     self.rot_x_axis_q_wp,
                     self.rot_y_axis_q_wp,
                     self.rot_z_axis_q_wp,
                     half_height
                 ],
                 outputs=[self.gizmo_pos_wp, self.gizmo_rot_wp],
                 device=self.config.device
             )

         gizmo_pos_np = self.gizmo_pos_wp.numpy()
         gizmo_rot_np = self.gizmo_rot_wp.numpy()

         for e in range(self.num_envs): # Loops up to batch size
             base_idx = e * 6
             # Target Gizmos
             self.renderer.render_cone(name=f"target_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 0]), rot=tuple(gizmo_rot_np[base_idx + 0]), radius=radius, half_height=half_height, color=self.config.gizmo_color_x_target)
             self.renderer.render_cone(name=f"target_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 1]), rot=tuple(gizmo_rot_np[base_idx + 1]), radius=radius, half_height=half_height, color=self.config.gizmo_color_y_target)
             self.renderer.render_cone(name=f"target_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 2]), rot=tuple(gizmo_rot_np[base_idx + 2]), radius=radius, half_height=half_height, color=self.config.gizmo_color_z_target)
             # End-Effector Gizmos
             self.renderer.render_cone(name=f"ee_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 3]), rot=tuple(gizmo_rot_np[base_idx + 3]), radius=radius, half_height=half_height, color=self.config.gizmo_color_x_ee)
             self.renderer.render_cone(name=f"ee_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 4]), rot=tuple(gizmo_rot_np[base_idx + 4]), radius=radius, half_height=half_height, color=self.config.gizmo_color_y_ee)
             self.renderer.render_cone(name=f"ee_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 5]), rot=tuple(gizmo_rot_np[base_idx + 5]), radius=radius, half_height=half_height, color=self.config.gizmo_color_z_ee)

    def render(self):
        # ... (implementation from previous script) ...
        if self.renderer is None: return
        with wp.ScopedTimer("render", print=False, active=True, dict=self.profiler):
             self.renderer.begin_frame(self.render_time)
             self.renderer.render(self.state)
             self.render_gizmos()
             self.renderer.end_frame()

    def get_final_joint_q(self) -> np.ndarray:
        """Returns the final joint configurations for the current batch."""
        return self.model.joint_q.numpy()


# --- Main Execution (Modified for Batching) ---
if __name__ == "__main__":
    import argparse
    import math # Ensure math is imported for ceil

    parser = argparse.ArgumentParser(description="Parallel IK Solver for Tattoo Targets using Batching")
    parser.add_argument("--targets", type=str, default=IKConfig.ik_targets_path, help="Path to the .npy file containing N x 7 IK targets (pos, quat).")
    parser.add_argument("--urdf", type=str, default=IKConfig.urdf_path, help="Path to the robot URDF file.")
    parser.add_argument("--output_dir", type=str, default=IKConfig.output_dir, help="Directory to save the output batch USD files.")
    parser.add_argument("--iters", type=int, default=IKConfig.num_iters, help="Number of IK iterations per batch.")
    parser.add_argument("--batch_size", type=int, default=IKConfig.batch_size, help="Number of environments per batch.")
    parser.add_argument("--device", type=str, default=IKConfig.device, help="Compute device (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--headless", action="store_true", help="Run without visualization.")
    parser.add_argument("--fps", type=int, default=IKConfig.fps, help="Rendering FPS for USD.")

    args = parser.parse_args()

    config = IKConfig(
        ik_targets_path=args.targets,
        urdf_path=args.urdf,
        output_dir=args.output_dir,
        num_iters=args.iters,
        batch_size=args.batch_size,
        device=args.device,
        headless=args.headless,
        fps=args.fps,
    )

    # --- Setup ---
    # Ensure output directory exists
    output_dir = os.path.expanduser(config.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Created output directory: {output_dir}")

    wp.init()
    sim_device = config.device if config.device else wp.get_preferred_device()
    log.info(f"Using device: {sim_device}")

    # --- Load ALL targets ---
    all_targets_np = None
    try:
        target_path = os.path.expanduser(config.ik_targets_path)
        log.info(f"Loading all targets from: {target_path}")
        all_targets_np = np.load(target_path).astype(np.float32)
        if all_targets_np.ndim != 2 or all_targets_np.shape[1] != 7:
             raise ValueError(f"Expected N x 7 array, got {all_targets_np.shape}")
        total_targets = len(all_targets_np)
        log.info(f"Successfully loaded {total_targets} targets.")
    except Exception as e:
        log.error(f"Failed to load targets: {e}", exc_info=True)
        exit(1)

    # --- Batch Processing ---
    num_batches = math.ceil(total_targets / config.batch_size)
    log.info(f"Processing {total_targets} targets in {num_batches} batches of size {config.batch_size}")

    all_final_q = [] # To store final joint angles from all batches
    solver = None # Keep solver variable outside loop scope for finally block

    overall_start_time = time.monotonic()

    try:
        with wp.ScopedDevice(sim_device):
            for batch_idx in range(num_batches):
                batch_start_time = time.monotonic()
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, total_targets)
                targets_batch_np = all_targets_np[start_idx:end_idx]
                current_batch_size = len(targets_batch_np)
                current_usd_path = os.path.join(output_dir, f"ik_batch_{batch_idx:03d}.usd")

                log.info(f"--- Starting Batch {batch_idx+1}/{num_batches} ({current_batch_size} targets) ---")
                log.info(f"Outputting USD to: {current_usd_path}")

                # Instantiate solver for this batch
                solver = ParallelTattooIKSolver(config, targets_batch_np, current_batch_size, current_usd_path)

                # Run IK iterations for this batch
                for i in range(config.num_iters):
                    solver.step()
                    if not config.headless:
                        solver.render()
                    # Optional: Log progress within batch if needed
                    # if i % 50 == 0: log.debug(f" Batch {batch_idx+1}, Iter {i}")

                # Save USD for this batch
                if solver.renderer:
                    log.info(f"Saving USD for batch {batch_idx+1}...")
                    solver.renderer.save()

                # Store final joint configurations for this batch
                all_final_q.append(solver.get_final_joint_q())

                batch_end_time = time.monotonic()
                log.info(f"--- Finished Batch {batch_idx+1}/{num_batches} (Duration: {batch_end_time - batch_start_time:.2f}s) ---")

                # Optional: Clear cache or delete solver to free memory between batches if needed
                # del solver
                # wp.context.empty_cache() # Might help free CUDA memory
                # solver = None

    except Exception as e:
        log.error(f"An error occurred during batch processing: {e}", exc_info=True)

    finally:
        overall_end_time = time.monotonic()
        log.info(f"\nTotal processing time: {overall_end_time - overall_start_time:.2f} seconds")

        # --- Consolidate and Save Final Results ---
        if all_final_q:
             try:
                 final_q_all_targets = np.concatenate(all_final_q, axis=0)
                 # Ensure the concatenated shape matches total targets * dof
                 expected_shape = (total_targets * (solver.dof if solver else 0))
                 if final_q_all_targets.size == expected_shape:
                      final_q_reshaped = final_q_all_targets.reshape(total_targets, solver.dof)
                      save_path = os.path.join(output_dir, "final_joint_configs.npy")
                      np.save(save_path, final_q_reshaped)
                      log.info(f"Saved final joint configurations for all {total_targets} targets to: {save_path}")
                 else:
                      log.warning(f"Could not concatenate final joint angles correctly. Expected size {expected_shape}, got {final_q_all_targets.size}")
             except Exception as e:
                  log.error(f"Failed to save consolidated final joint configurations: {e}", exc_info=True)


        # --- Profiling Output (Optional: Could aggregate across batches) ---
        # Note: Profiling stats from the last batch might not be representative.
        if solver and solver.profiler:
            log.info("\n--- Performance Profile (Last Batch) ---")
            # [Same profiling print logic as before, using the last 'solver' instance]
            profiling_data = solver.profiler
            if "model_build" in profiling_data: log.info(f"  Model Build Time: {profiling_data['model_build'][0]:.2f} ms")
            if "ik_step" in profiling_data:
                times = np.array(profiling_data['ik_step']); avg_t, std_t, min_t, max_t = times.mean(), times.std(), times.min(), times.max()
                log.info(f"  IK Step Time: Avg: {avg_t:.3f} ms, Std: {std_t:.3f}, Min: {min_t:.3f}, Max: {max_t:.3f}")
            log.info("  Avg Time per Internal Operation (within IK step):")
            internal_ops = ["eval_fk", "error_kernel", "sampling_loop", "jacobian_estimation", "gn_update", "clip_joints"]
            for key in internal_ops:
                if key in profiling_data and profiling_data[key]: log.info(f"    {key}: {np.mean(profiling_data[key]):.4f} ms")
            if "render" in profiling_data: log.info(f"  Render Time (Avg): {np.mean(profiling_data['render']):.2f} ms")
            if "gizmo_kernel" in profiling_data: log.info(f"    Gizmo Kernel (Avg): {np.mean(profiling_data['gizmo_kernel']):.3f} ms")

        log.info("Script finished.")