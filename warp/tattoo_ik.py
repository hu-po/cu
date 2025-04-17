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
    # Default path assumes the npy file is in the same directory
    ik_targets_path: str = "outputs/tattoo_ik_poses.npy"
    urdf_path: str = "~/dev/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
    usd_output_path: str = "scenes/parallel_tattoo_ik_output.usd"

    # --- Simulation & Solver ---
    seed: int = 42
    num_iters: int = 200 # Number of IK iterations to run
    device: str = None # Auto-select unless specified
    headless: bool = False
    fps: int = 30 # Rendering FPS
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0) # Offset from ee_gripper_link
    # IK Solver Parameters (from sampling_gn morph)
    ik_sigma: float = 1e-2      # Std deviation for random samples
    ik_num_samples: int = 10    # Samples per env per step for Jacobian estimation
    ik_damping: float = 0.1     # Damping factor (lambda) for Gauss-Newton

    # --- Arm Initialization ---
    # ALL arms will start here
    arm_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_rot_offset: list[tuple[tuple[float, float, float], float]] = field(default_factory=lambda: [
        ((1.0, 0.0, 0.0), -math.pi * 0.5), # Match previous setup if needed
    ])
    # Start all arms exactly at home
    qpos_home: list[float] = field(default_factory=lambda: [
        0, np.pi/12, np.pi/12, 0, 0, 0, 0, 0
    ])
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: [
        (-3.054, 3.054), (0.0, 3.14), (0.0, 2.356), (-1.57, 1.57),
        (-1.57, 1.57), (-3.14, 3.14), (0.0, 0.044), (0.0, 0.044)
    ])
    # Physical properties (can be adjusted)
    joint_attach_ke: float = 1600.0
    joint_attach_kd: float = 20.0

    # --- Visualization ---
    gizmo_radius: float = 0.005
    gizmo_length: float = 0.02 # Smaller gizmos might be clearer
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0)
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0)
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5)
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5)
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0)

# --- Warp Math & Kernels (Copied from reference IK script) ---
@wp.func
def quat_mul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    return wp.quat(
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
        q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
    )

@wp.func
def quat_conjugate(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])

@wp.func
def quat_orientation_error(target: wp.quat, current: wp.quat) -> wp.vec3:
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

# --- Main Solver Class ---
class ParallelTattooIKSolver:
    def __init__(self, config: IKConfig):
        self.config = config
        self.profiler = {}
        self.render_time = 0.0
        self.frame_dt = 1.0 / config.fps if config.fps > 0 else 0.0

        # --- Load Targets ---
        self.target_poses_np = self._load_targets()
        self.num_envs = self.target_poses_np.shape[0]
        log.info(f"Loaded {self.num_envs} IK targets.")

        # Create target arrays on device
        target_pos_np = self.target_poses_np[:, :3].copy() # Nx3
        target_ori_np = self.target_poses_np[:, 3:].copy() # Nx4 (qx,qy,qz,qw)
        self.targets_pos_wp = wp.array(target_pos_np, dtype=wp.vec3, device=config.device)
        self.targets_ori_wp = wp.array(target_ori_np, dtype=wp.quat, device=config.device)

        # --- Build Model ---
        with wp.ScopedTimer("model_build", print=False, active=True, dict=self.profiler):
            # Seed random number generator
            self.rng = np.random.default_rng(self.config.seed)

            # Parse URDF once
            urdf_path = os.path.expanduser(self.config.urdf_path)
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            log.info(f"Parsing URDF: {urdf_path}")
            articulation_builder = wp.sim.ModelBuilder()
            wp.sim.parse_urdf(urdf_path, articulation_builder, floating=False)

            # Store arm properties
            self.num_links = len(articulation_builder.body_name) # Use body_name length
            self.dof = len(articulation_builder.joint_q)
            self.joint_limits = np.array(self.config.joint_limits, dtype=np.float32)
            self.joint_limits_min_wp = wp.array(self.joint_limits[:, 0], dtype=wp.float32, device=config.device)
            self.joint_limits_max_wp = wp.array(self.joint_limits[:, 1], dtype=wp.float32, device=config.device)
            log.info(f"Parsed URDF with {self.num_links} links and {self.dof} DoF.")

            # Find EE link index (copied logic)
            self.ee_link_offset = wp.vec3(self.config.ee_link_offset)
            self.ee_link_index = -1
            ee_link_name = "ee_gripper_link" # Make sure this matches your URDF
            for i, name in enumerate(articulation_builder.body_name):
                if name == ee_link_name:
                    self.ee_link_index = i
                    break
            if self.ee_link_index == -1:
                 # Try fallback using joint name convention
                 ee_joint_name = "ee_gripper"
                 found_joint_idx = -1
                 for j_idx, j_name in enumerate(articulation_builder.joint_name):
                     if j_name == ee_joint_name:
                         found_joint_idx = j_idx
                         break
                 if found_joint_idx != -1:
                     self.ee_link_index = articulation_builder.joint_child[found_joint_idx]
                     log.info(f"Found '{ee_joint_name}' joint, using child link index: {self.ee_link_index} (Name: {articulation_builder.body_name[self.ee_link_index]})")
                 else:
                     raise ValueError(f"Could not find link '{ee_link_name}' or joint '{ee_joint_name}' in URDF")
            log.info(f"Using End-Effector Link: {articulation_builder.body_name[self.ee_link_index]} (Index: {self.ee_link_index})")


            # Calculate the single initial arm orientation
            _initial_arm_orientation_wp = wp.quat_identity()
            for axis, angle in self.config.arm_rot_offset:
                rot_quat_wp = wp.quat_from_axis_angle(wp.vec3(axis), angle)
                _initial_arm_orientation_wp = quat_mul(rot_quat_wp, _initial_arm_orientation_wp)
            self.initial_arm_orientation = _initial_arm_orientation_wp
            self.arm_base_pos_wp = wp.vec3(self.config.arm_base_pos)
            initial_arm_transform = wp.transform(self.arm_base_pos_wp, self.initial_arm_orientation)
            log.info(f"Initializing all arms at Pos: {list(self.arm_base_pos_wp)}, Rot: {list(self.initial_arm_orientation)}")

            # Build all environments at the same starting pose
            builder = wp.sim.ModelBuilder()
            log.info(f"Adding {self.num_envs} arm environments...")
            for _ in range(self.num_envs):
                builder.add_builder(articulation_builder, xform=initial_arm_transform)

            # Finalize the model
            self.model = builder.finalize(device=config.device)
            self.model.ground = False # Usually good for IK tasks
            self.model.joint_attach_ke = self.config.joint_attach_ke
            self.model.joint_attach_kd = self.config.joint_attach_kd
            # Jacobian estimation doesn't require grads on joint angles
            self.model.joint_q.requires_grad = False

            # Set initial joint angles (all arms start at home)
            initial_joint_q = []
            if len(self.config.qpos_home) != self.dof:
                 log.warning(f"Config qpos_home length ({len(self.config.qpos_home)}) != Arm DoF ({self.dof}). Check config.")
                 # Adjust qpos_home or URDF if necessary
                 qpos_home_adjusted = (self.config.qpos_home + [0.0]*self.dof)[:self.dof]
            else:
                 qpos_home_adjusted = self.config.qpos_home

            for _ in range(self.num_envs):
                initial_joint_q.extend(qpos_home_adjusted)
            self.model.joint_q.assign(wp.array(initial_joint_q, dtype=wp.float32, device=config.device))
            log.info(f"Set initial joint configuration for {self.num_envs} environments.")

            # --- Simulation State & Solver Prep ---
            # No integrator needed for pure IK
            self.state = self.model.state() # Get state AFTER setting initial joint_q
            # Allocate space for errors
            self.ee_error_wp = wp.zeros(self.num_envs * 6, dtype=wp.float32, device=config.device)

        # --- Renderer Setup ---
        self.renderer = None
        if not config.headless:
            log.info(f"Initializing renderer, outputting to {config.usd_output_path}")
            try:
                self.renderer = wp.sim.render.SimRenderer(self.model, config.usd_output_path, scaling=1.0) # Adjust scaling if needed
                # Init gizmo arrays
                self.gizmo_pos_wp = wp.zeros(self.num_envs * 6, dtype=wp.vec3, device=config.device)
                self.gizmo_rot_wp = wp.zeros(self.num_envs * 6, dtype=wp.quat, device=config.device)
                # Precompute gizmo rotations (cone points along +Y)
                self.rot_x_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), math.pi / 2.)
                self.rot_y_axis_q_wp = wp.quat_identity()
                self.rot_z_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(1., 0., 0.), -math.pi / 2.)
            except Exception as e:
                 log.error(f"Failed to initialize renderer: {e}. Running headless.", exc_info=True)
                 self.config.headless = True

    def _load_targets(self):
        """Loads 6D target poses (pos, quat) from .npy file."""
        target_path = os.path.expanduser(self.config.ik_targets_path)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"IK targets file not found: {target_path}")
        try:
            targets = np.load(target_path)
            if targets.ndim != 2 or targets.shape[1] != 7:
                raise ValueError(f"Expected N x 7 array (pos, quat), but got shape {targets.shape}")
            # Ensure float32 for consistency
            return targets.astype(np.float32)
        except Exception as e:
            log.error(f"Failed to load or validate IK targets from {target_path}: {e}", exc_info=True)
            raise

    def compute_ee_error(self, joint_q_wp: wp.array) -> np.ndarray:
        """
        Computes the 6D end-effector error for all envs given joint angles.
        Input: joint_q_wp - Warp array of joint angles (num_envs * dof)
        Output: Numpy array of flattened errors (num_envs * 6)
        """
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
        """Performs one IK step using sampling-based Jacobian + Gauss-Newton."""
        # Config parameters for the solver
        sigma = self.config.ik_sigma
        num_samples = self.config.ik_num_samples
        damping = self.config.ik_damping
        dof = self.dof
        num_envs = self.num_envs

        # Get current joint angles (GPU -> CPU)
        q_current_np = self.model.joint_q.numpy().copy() # shape: (num_envs * dof,)
        q_updated_np = q_current_np.copy() # Array to store updates

        # Compute baseline error (using current joints on GPU)
        # Create a temporary Warp array for the *current* q for error calculation
        q_current_wp = wp.array(q_current_np, dtype=wp.float32, device=self.config.device)
        error_base_all_np = self.compute_ee_error(q_current_wp) # (num_envs * 6,)

        # Temporary array for perturbed joint angles on GPU
        q_temp_wp = wp.empty_like(self.model.joint_q)

        # Process each environment (Jacobian estimation on CPU)
        for e in range(num_envs):
            base_idx_q = e * dof
            base_idx_err = e * 6
            q_e = q_current_np[base_idx_q : base_idx_q + dof].copy() # (dof,)
            error_base = error_base_all_np[base_idx_err : base_idx_err + 6].copy() # (6,)

            # Generate random perturbations
            D = self.rng.normal(0.0, sigma, size=(num_samples, dof)).astype(np.float32)
            Delta = np.zeros((num_samples, 6), dtype=np.float32)

            # --- Sampling Loop ---
            with wp.ScopedTimer("sampling_loop", print=False, active=True, dict=self.profiler):
                 q_batch_np = np.tile(q_current_np, (num_samples, 1)) # (num_samples * num_envs * dof)
                 # Apply perturbations only to the current environment 'e' within the batch
                 for i in range(num_samples):
                     q_sample = q_e + D[i]
                     row_start = i * num_envs * dof
                     q_batch_np[row_start + base_idx_q : row_start + base_idx_q + dof] = q_sample

                 # TODO: This approach is inefficient as it recomputes error for all envs for each sample.
                 # A more efficient approach would require batching FK/error computation differently or
                 # a kernel that handles perturbations directly. Sticking to reference morph logic for now.

                 # Compute errors for all samples (inefficiently)
                 error_sample_all_np = np.zeros((num_samples, num_envs * 6), dtype=np.float32)
                 for i in range(num_samples):
                      q_sample = q_e + D[i]
                      # Create a temporary copy of current joint angles, replace env e
                      q_temp_np = q_current_np.copy()
                      q_temp_np[base_idx_q : base_idx_q + dof] = q_sample
                      # Assign to temporary GPU array and compute error
                      q_temp_wp.assign(q_temp_np)
                      error_sample_all_np[i, :] = self.compute_ee_error(q_temp_wp)

                 # Extract the error difference for environment 'e'
                 for i in range(num_samples):
                     error_sample_e = error_sample_all_np[i, base_idx_err : base_idx_err + 6]
                     Delta[i] = error_sample_e - error_base
            # --- End Sampling Loop ---


            # Estimate Jacobian J (6 x dof) via least squares: D @ J^T = Delta
            with wp.ScopedTimer("jacobian_estimation", print=False, active=True, dict=self.profiler):
                try:
                    J_transpose_est, _, _, _ = np.linalg.lstsq(D, Delta, rcond=None)
                    J_est = J_transpose_est.T
                except np.linalg.LinAlgError:
                     log.warning(f"LinAlgError during Jacobian estimation for env {e}. Skipping update.")
                     continue # Skip update for this environment

            # Compute damped Gauss-Newton update: (J J^T + lambda*I) y = error; dq = - J^T y
            with wp.ScopedTimer("gn_update", print=False, active=True, dict=self.profiler):
                try:
                    JJt = J_est @ J_est.T
                    A = JJt + damping * np.eye(6, dtype=np.float32)
                    y = np.linalg.solve(A, error_base)
                    delta_q = -J_est.T @ y
                except np.linalg.LinAlgError:
                     log.warning(f"LinAlgError during Gauss-Newton update for env {e}. Skipping update.")
                     continue # Skip update for this environment

            # Update joint configuration for this environment in the numpy array
            q_updated_np[base_idx_q : base_idx_q + dof] += delta_q

        # Assign the fully updated joint angles back to the GPU model
        self.model.joint_q.assign(q_updated_np)

        # Clip joint angles
        with wp.ScopedTimer("clip_joints", print=False, active=True, dict=self.profiler):
             wp.launch(
                 kernel=clip_joints_kernel,
                 dim=self.num_envs * self.dof,
                 inputs=[
                     self.model.joint_q, # Operate directly on model's array
                     self.joint_limits_min_wp,
                     self.joint_limits_max_wp,
                     self.num_envs,
                     self.dof
                 ],
                 device=self.config.device
             )

    def step(self):
        """Advances the IK solver by one iteration."""
        with wp.ScopedTimer("ik_step", print=False, active=True, dict=self.profiler):
            self._ik_step()
        self.render_time += self.frame_dt

    def render_gizmos(self):
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

         for e in range(self.num_envs):
             base_idx = e * 6
             # Target Gizmos (Semi-transparent Red/Green/Blue)
             self.renderer.render_cone(name=f"target_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 0]), rot=tuple(gizmo_rot_np[base_idx + 0]), radius=radius, half_height=half_height, color=self.config.gizmo_color_x_target)
             self.renderer.render_cone(name=f"target_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 1]), rot=tuple(gizmo_rot_np[base_idx + 1]), radius=radius, half_height=half_height, color=self.config.gizmo_color_y_target)
             self.renderer.render_cone(name=f"target_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 2]), rot=tuple(gizmo_rot_np[base_idx + 2]), radius=radius, half_height=half_height, color=self.config.gizmo_color_z_target)
             # End-Effector Gizmos (Solid Red/Green/Blue)
             self.renderer.render_cone(name=f"ee_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 3]), rot=tuple(gizmo_rot_np[base_idx + 3]), radius=radius, half_height=half_height, color=self.config.gizmo_color_x_ee)
             self.renderer.render_cone(name=f"ee_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 4]), rot=tuple(gizmo_rot_np[base_idx + 4]), radius=radius, half_height=half_height, color=self.config.gizmo_color_y_ee)
             self.renderer.render_cone(name=f"ee_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 5]), rot=tuple(gizmo_rot_np[base_idx + 5]), radius=radius, half_height=half_height, color=self.config.gizmo_color_z_ee)


    def render(self):
        if self.renderer is None:
            return
        with wp.ScopedTimer("render", print=False, active=True, dict=self.profiler):
             self.renderer.begin_frame(self.render_time)
             # State should be reasonably up-to-date from last FK call in compute_ee_error
             # or call FK again if _ik_step modified state significantly without FK
             # wp.sim.eval_fk(self.model, self.model.joint_q, None, None, self.state)
             self.renderer.render(self.state)
             self.render_gizmos()
             self.renderer.end_frame()


# --- Main Execution ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel IK Solver for Tattoo Targets")
    parser.add_argument("--targets", type=str, default=IKConfig.ik_targets_path, help="Path to the .npy file containing N x 7 IK targets (pos, quat).")
    parser.add_argument("--urdf", type=str, default=IKConfig.urdf_path, help="Path to the robot URDF file.")
    parser.add_argument("--output_usd", type=str, default=IKConfig.usd_output_path, help="Path to save the output USD recording.")
    parser.add_argument("--iters", type=int, default=IKConfig.num_iters, help="Number of IK iterations.")
    parser.add_argument("--device", type=str, default=IKConfig.device, help="Compute device (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--headless", action="store_true", help="Run without visualization.")
    parser.add_argument("--fps", type=int, default=IKConfig.fps, help="Rendering FPS for USD.")
    # Add args for IK params if needed: --sigma, --samples, --damping

    args = parser.parse_args()

    config = IKConfig(
        ik_targets_path=args.targets,
        urdf_path=args.urdf,
        usd_output_path=args.output_usd,
        num_iters=args.iters,
        device=args.device,
        headless=args.headless,
        fps=args.fps,
        # Can override other config defaults here if needed
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.expanduser(config.usd_output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wp.init()
    sim_device = config.device if config.device else wp.get_preferred_device()
    log.info(f"Using device: {sim_device}")

    solver = None
    try:
        with wp.ScopedDevice(sim_device):
            solver = ParallelTattooIKSolver(config)

            log.info(f"--- Starting Parallel IK ({solver.num_envs} environments) ---")
            start_time = time.monotonic()

            for i in range(config.num_iters):
                solver.step()
                if not config.headless:
                    solver.render()

                if i % 50 == 0 or i == config.num_iters - 1: # Log progress
                     log.info(f"Iteration {i+1}/{config.num_iters}")

            end_time = time.monotonic()
            log.info(f"--- IK Solving Finished ({end_time - start_time:.2f} seconds) ---")

            if solver.renderer:
                log.info("Saving USD recording...")
                solver.renderer.save()
                log.info(f"USD saved to {config.usd_output_path}")

    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)

    finally:
        # --- Profiling Output ---
        if solver and solver.profiler:
            log.info("\n--- Performance Profile ---")
            profiling_data = solver.profiler

            if "model_build" in profiling_data:
                log.info(f"  Model Build Time: {profiling_data['model_build'][0]:.2f} ms")

            if "ik_step" in profiling_data:
                 times = np.array(profiling_data['ik_step'])
                 avg_t, std_t, min_t, max_t = times.mean(), times.std(), times.min(), times.max()
                 log.info(f"  IK Step Time: Avg: {avg_t:.3f} ms, Std: {std_t:.3f}, Min: {min_t:.3f}, Max: {max_t:.3f}")

            log.info("  Avg Time per Internal Operation (within IK step):")
            internal_ops = ["eval_fk", "error_kernel", "sampling_loop", "jacobian_estimation", "gn_update", "clip_joints"]
            for key in internal_ops:
                if key in profiling_data and profiling_data[key]:
                    times = np.array(profiling_data[key])
                    avg_time = times.mean()
                    log.info(f"    {key}: {avg_time:.4f} ms")

            if "render" in profiling_data:
                 times = np.array(profiling_data['render'])
                 log.info(f"  Render Time (Avg): {times.mean():.2f} ms")
                 if "gizmo_kernel" in profiling_data:
                     times_giz = np.array(profiling_data['gizmo_kernel'])
                     log.info(f"    Gizmo Kernel (Avg): {times_giz.mean():.3f} ms")

    log.info("Script finished.")