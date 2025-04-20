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
# Use torch for efficient GPU<->CPU slicing if available, otherwise fallback
try:
    import torch
    USE_TORCH_FOR_DEBUG_SLICING = True
except ImportError:
    USE_TORCH_FOR_DEBUG_SLICING = False
    log.warning("torch not found. Falling back to numpy for debug slicing (potentially slower).")

logging.basicConfig(
    level=logging.INFO, # Set to INFO for production, DEBUG for detailed tracing
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Adjust level here (INFO or DEBUG)

# --- Configuration ---
@dataclass
class IKConfig:
    # --- Input/Output ---
    ik_targets_path: str = "outputs/tattoo_ik_poses.npy" # Source of all targets (N x 7: px,py,pz, qx,qy,qz,qw)
    urdf_path: str = "~/dev/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf" # Source URDF
    output_dir: str = "outputs/ik_batches" # Directory for batch USDs and final results
    # --- Simulation & Solver ---
    seed: int = 42
    num_iters: int = 100 # Number of IK iterations per batch (adjust as needed)
    batch_size: int = 128 # Max environments per batch (adjust based on GPU memory)
    device: str = None # Auto-select unless specified (e.g., 'cuda:0')
    headless: bool = False # Set to True to disable rendering
    fps: int = 30 # Rendering FPS (only relevant if not headless)
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0) # Offset from ee_gripper_link frame
    # --- IK Solver Parameters (Sampling-based Gauss-Newton) ---
    ik_sigma: float = 1e-2 # Standard deviation for random joint perturbations
    ik_num_samples: int = 10 # Number of samples per env to estimate Jacobian
    ik_damping: float = 0.1 # Damping factor (lambda) for Gauss-Newton update
    # --- Arm Initialization ---
    arm_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0) # Base position of the *first* arm (others offset)
    arm_rot_offset: list[tuple[tuple[float, float, float], float]] = field(default_factory=lambda: [
        ((1.0, 0.0, 0.0), -math.pi * 0.5), # Rotate base to point forward initially
    ])
    qpos_home: list[float] = field(default_factory=lambda: [
        0, np.pi/12, np.pi/12, 0, 0, 0, 0, 0 # Waist, Shoulder, Elbow, Wr1, Wr2, Wr3, Finger1, Finger2
    ])
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: [
        (-3.054, 3.054), (0.0, 3.14), (0.0, 2.356), (-1.57, 1.57),
        (-1.57, 1.57), (-3.14, 3.14), (0.0, 0.044), (0.0, 0.044)
    ])
    joint_attach_ke: float = 1600.0 # Stiffness for joint targets (helps stability)
    joint_attach_kd: float = 20.0 # Damping for joint targets
    # --- Visualization (Only relevant if not headless) ---
    gizmo_radius: float = 0.005
    gizmo_length: float = 0.02
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0)
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0)
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5)
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5)
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0)

# --- Warp Math & Kernels ---
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
    """Computes 3D orientation error vector from target and current quaternions."""
    q_err = quat_mul(target, quat_conjugate(current))
    # Ensure scalar part is non-negative for consistency
    qw = wp.where(q_err[3] < 0.0, -q_err[3], q_err[3])
    qx = wp.where(q_err[3] < 0.0, -q_err[0], q_err[0])
    qy = wp.where(q_err[3] < 0.0, -q_err[1], q_err[1])
    qz = wp.where(q_err[3] < 0.0, -q_err[2], q_err[2])
    # Return axis * angle (scaled by 2), which is approx 2 * axis * sin(angle/2)
    return wp.vec3(qx, qy, qz) * 2.0

@wp.kernel
def compute_ee_error_kernel(
    body_q: wp.array(dtype=wp.transform), # Result of FK for *current* joint angles
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    target_pos: wp.array(dtype=wp.vec3), # Target positions for this batch
    target_ori: wp.array(dtype=wp.quat), # Target orientations for this batch
    error_out: wp.array(dtype=wp.float32) # Output: Flattened error (num_envs * 6)
):
    tid = wp.tid() # Environment index (0 to num_envs-1)

    # Get EE link transform [px,py,pz, qx,qy,qz,qw]
    # Note: body_q is flattened: [env0_link0, env0_link1, ..., env1_link0, ...]
    T_flat = body_q[tid * num_links + ee_link_index]
    T_pos = wp.vec3(T_flat[0], T_flat[1], T_flat[2])
    T_ori = wp.quat(T_flat[3], T_flat[4], T_flat[5], T_flat[6])
    T = wp.transform(T_pos, T_ori)

    # Calculate current EE tip position and orientation based on FK result
    current_pos = wp.transform_point(T, ee_link_offset)
    current_ori = wp.transform_get_rotation(T) # which is just T_ori

    # Calculate errors
    pos_err = target_pos[tid] - current_pos
    ori_err = quat_orientation_error(target_ori[tid], current_ori)

    # Write to output array (flattened)
    base = tid * 6
    error_out[base + 0] = pos_err.x
    error_out[base + 1] = pos_err.y
    error_out[base + 2] = pos_err.z
    error_out[base + 3] = ori_err.x
    error_out[base + 4] = ori_err.y
    error_out[base + 5] = ori_err.z

@wp.kernel
def clip_joints_kernel(
    joint_q: wp.array(dtype=wp.float32), # Input/Output: flattened joint angles (num_envs * dof)
    joint_limits_min: wp.array(dtype=wp.float32), # Min limits per DOF type (dof,)
    joint_limits_max: wp.array(dtype=wp.float32), # Max limits per DOF type (dof,)
    num_envs: int,
    dof: int
):
    tid = wp.tid() # Global index across all joints in all envs (0 to num_envs*dof - 1)

    # Calculate which DOF this thread corresponds to within an arm
    joint_idx_in_arm = tid % dof

    # Clamp the value using the limits for this specific DOF
    current_val = joint_q[tid]
    min_val = joint_limits_min[joint_idx_in_arm]
    max_val = joint_limits_max[joint_idx_in_arm]
    joint_q[tid] = wp.clamp(current_val, min_val, max_val)

@wp.kernel
def calculate_gizmo_transforms_kernel(
    body_q: wp.array(dtype=wp.transform), # Current body transforms from FK
    targets_pos: wp.array(dtype=wp.vec3), # Target positions for the batch
    targets_ori: wp.array(dtype=wp.quat), # Target orientations for the batch
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    num_envs: int,
    rot_x_axis_q: wp.quat, # Precomputed rotation for X axis gizmo
    rot_y_axis_q: wp.quat, # Precomputed rotation for Y axis gizmo
    rot_z_axis_q: wp.quat, # Precomputed rotation for Z axis gizmo
    cone_half_height: float,
    # Outputs (flat array, order: TgtX,TgtY,TgtZ, EEX,EEY,EEZ per env)
    out_gizmo_pos: wp.array(dtype=wp.vec3),
    out_gizmo_rot: wp.array(dtype=wp.quat)
):
    tid = wp.tid() # Environment index

    # --- Target Gizmos ---
    target_pos = targets_pos[tid]
    target_ori = targets_ori[tid]

    # Calculate orientations for the X, Y, Z target gizmos
    target_rot_x = quat_mul(target_ori, rot_x_axis_q)
    target_rot_y = quat_mul(target_ori, rot_y_axis_q)
    target_rot_z = quat_mul(target_ori, rot_z_axis_q)

    # Calculate the offset vector (along the cone's height axis, assumed Y)
    offset_vec = wp.vec3(0.0, cone_half_height, 0.0)

    # Rotate the offset vector by the gizmo orientations
    offset_x = wp.quat_rotate(target_rot_x, offset_vec)
    offset_y = wp.quat_rotate(target_rot_y, offset_vec)
    offset_z = wp.quat_rotate(target_rot_z, offset_vec)

    # Calculate final positions (center - rotated offset) and store
    base_idx = tid * 6 # 6 gizmos per environment (3 target, 3 ee)
    out_gizmo_pos[base_idx + 0] = target_pos - offset_x
    out_gizmo_rot[base_idx + 0] = target_rot_x
    out_gizmo_pos[base_idx + 1] = target_pos - offset_y
    out_gizmo_rot[base_idx + 1] = target_rot_y
    out_gizmo_pos[base_idx + 2] = target_pos - offset_z
    out_gizmo_rot[base_idx + 2] = target_rot_z

    # --- End-Effector Gizmos ---
    # Get EE link transform from the current body state
    T_flat = body_q[tid * num_links + ee_link_index]
    ee_link_pos = wp.vec3(T_flat[0], T_flat[1], T_flat[2])
    ee_link_ori = wp.quat(T_flat[3], T_flat[4], T_flat[5], T_flat[6])
    T_ee = wp.transform(ee_link_pos, ee_link_ori)

    # Calculate the actual end-effector tip position
    ee_tip_pos = wp.transform_point(T_ee, ee_link_offset)

    # Calculate orientations for the X, Y, Z EE gizmos (relative to EE frame)
    ee_rot_x = quat_mul(ee_link_ori, rot_x_axis_q)
    ee_rot_y = quat_mul(ee_link_ori, rot_y_axis_q)
    ee_rot_z = quat_mul(ee_link_ori, rot_z_axis_q)

    # Rotate the offset vector by the EE gizmo orientations
    ee_offset_x = wp.quat_rotate(ee_rot_x, offset_vec)
    ee_offset_y = wp.quat_rotate(ee_rot_y, offset_vec)
    ee_offset_z = wp.quat_rotate(ee_rot_z, offset_vec)

    # Calculate final positions and store
    out_gizmo_pos[base_idx + 3] = ee_tip_pos - ee_offset_x
    out_gizmo_rot[base_idx + 3] = ee_rot_x
    out_gizmo_pos[base_idx + 4] = ee_tip_pos - ee_offset_y
    out_gizmo_rot[base_idx + 4] = ee_rot_y
    out_gizmo_pos[base_idx + 5] = ee_tip_pos - ee_offset_z
    out_gizmo_rot[base_idx + 5] = ee_rot_z


@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),      # Input: current joint angles (num_envs * dof)
    delta_q: wp.array(dtype=wp.float32),      # Input: computed joint updates (num_envs * dof)
    out_joint_q: wp.array(dtype=wp.float32)   # Output: updated joint angles (num_envs * dof)
):
    tid = wp.tid() # Global index across all joints in all envs
    out_joint_q[tid] = joint_q[tid] + delta_q[tid]


# --- Solver Class (Handles one batch) ---
class ParallelTattooIKSolver:
    def __init__(self, config: IKConfig, target_poses_batch_np: np.ndarray, batch_num_envs: int, batch_usd_path: str):
        """
        Initializes the solver for a specific batch of targets.
        Args:
            config: The main IK configuration.
            target_poses_batch_np: Numpy array (batch_num_envs x 7) of targets for this batch.
            batch_num_envs: The number of environments (targets) in this specific batch.
            batch_usd_path: Path to save the USD file for this batch (if rendering).
        """
        self.config = config
        self.profiler = {} # For timing internal operations
        self.render_time = 0.0
        self.frame_dt = 1.0 / config.fps if config.fps > 0 and not config.headless else 0.0
        self.num_envs = batch_num_envs # Number of environments for *this* batch

        log.info(f"Initializing solver for batch with {self.num_envs} environments.")
        log.debug(f"Using device: {config.device}")

        # --- Process Targets for this Batch ---
        if target_poses_batch_np.ndim != 2 or target_poses_batch_np.shape[1] != 7:
            raise ValueError(f"Batch targets have incorrect shape: {target_poses_batch_np.shape}. Expected ({batch_num_envs}, 7).")
        if len(target_poses_batch_np) != batch_num_envs:
            raise ValueError(f"Target array length {len(target_poses_batch_np)} != batch_num_envs {batch_num_envs}")

        # Separate position (xyz) and orientation (quat xyzw) and send to GPU
        target_pos_np = target_poses_batch_np[:, :3].copy().astype(np.float32)
        target_ori_np = target_poses_batch_np[:, 3:].copy().astype(np.float32) # xyzw order from input typical
        self.targets_pos_wp = wp.array(target_pos_np, dtype=wp.vec3, device=config.device)
        self.targets_ori_wp = wp.array(target_ori_np, dtype=wp.quat, device=config.device)
        log.debug(f"Targets loaded to Warp arrays: pos shape {self.targets_pos_wp.shape}, ori shape {self.targets_ori_wp.shape}")

        # --- Build Model ---
        with wp.ScopedTimer("model_build", print=False, active=True, dict=self.profiler):
            self.rng = np.random.default_rng(self.config.seed) # Seed for sampling

            # Parse URDF once
            urdf_path_expanded = os.path.expanduser(self.config.urdf_path)
            if not os.path.exists(urdf_path_expanded):
                raise FileNotFoundError(f"URDF file not found: {urdf_path_expanded}")
            log.debug(f"Parsing URDF: {urdf_path_expanded}")
            articulation_builder = wp.sim.ModelBuilder()
            wp.sim.parse_urdf(urdf_path_expanded, articulation_builder, floating=False)

            # Store arm properties
            self.num_links = len(articulation_builder.body_name)
            self.dof = len(articulation_builder.joint_q) # Number of degrees of freedom
            self.joint_limits_np = np.array(self.config.joint_limits, dtype=np.float32)
            if self.joint_limits_np.shape != (self.dof, 2):
                 log.warning(f"Joint limits shape {self.joint_limits_np.shape} mismatch with DoF ({self.dof}). Using provided limits.")
                 # Attempt to resize/pad if necessary, or raise error
                 if self.joint_limits_np.shape[0] < self.dof:
                      pad_width = self.dof - self.joint_limits_np.shape[0]
                      # Pad with very large limits, assuming missing are unconstrained (adjust if needed)
                      padded_limits = np.pad(self.joint_limits_np, ((0, pad_width), (0,0)), constant_values=((-1e6, 1e6)))
                      self.joint_limits_np = padded_limits
                      log.warning(f"Padded joint limits to shape {self.joint_limits_np.shape}")
                 elif self.joint_limits_np.shape[0] > self.dof:
                      self.joint_limits_np = self.joint_limits_np[:self.dof, :]
                      log.warning(f"Truncated joint limits to shape {self.joint_limits_np.shape}")

            # Create Warp arrays for joint limits on GPU
            self.joint_limits_min_wp = wp.array(self.joint_limits_np[:, 0], dtype=wp.float32, device=config.device)
            self.joint_limits_max_wp = wp.array(self.joint_limits_np[:, 1], dtype=wp.float32, device=config.device)

            log.info(f"URDF parsed: {self.num_links} links, {self.dof} DoF.")

            # Find EE link index (robust check)
            self.ee_link_offset = wp.vec3(self.config.ee_link_offset)
            self.ee_link_index = -1
            ee_link_name = "ee_gripper_link" # Common convention
            ee_joint_name = "ee_gripper" # Fallback joint name convention
            for i, name in enumerate(articulation_builder.body_name):
                if name == ee_link_name:
                    self.ee_link_index = i
                    log.debug(f"Found EE link by name '{ee_link_name}' at index {self.ee_link_index}")
                    break
            if self.ee_link_index == -1:
                log.debug(f"EE link '{ee_link_name}' not found by name, checking joint children...")
                found_joint_idx = -1
                for j_idx, j_name in enumerate(articulation_builder.joint_name):
                     if j_name == ee_joint_name:
                          found_joint_idx = j_idx
                          break
                if found_joint_idx != -1:
                     # joint_child array maps joint index to the index of its child link/body
                     self.ee_link_index = articulation_builder.joint_child[found_joint_idx]
                     child_link_name = articulation_builder.body_name[self.ee_link_index]
                     log.debug(f"Found joint '{ee_joint_name}', using its child link index: {self.ee_link_index} (Name: {child_link_name})")
                else:
                     raise ValueError(f"Could not find EE link ('{ee_link_name}') or EE joint ('{ee_joint_name}') in URDF")
            log.info(f"Using End-Effector Link Index: {self.ee_link_index} with offset {tuple(self.config.ee_link_offset)}")

            # Calculate initial arm orientation based on config offsets
            _initial_arm_orientation_wp = wp.quat_identity()
            for axis, angle in self.config.arm_rot_offset:
                rot_quat_wp = wp.quat_from_axis_angle(wp.vec3(axis), angle)
                _initial_arm_orientation_wp = quat_mul(rot_quat_wp, _initial_arm_orientation_wp)
            initial_arm_transform = wp.transform(wp.vec3(self.config.arm_base_pos), _initial_arm_orientation_wp)
            log.debug(f"Base arm transform: pos={self.config.arm_base_pos}, ori={list(_initial_arm_orientation_wp)}")

            # Build multi-environment model for the current batch size
            builder = wp.sim.ModelBuilder()
            for _ in range(self.num_envs): # Create one arm instance per target in this batch
                # Add arm instance to the main builder. For simplicity here, all arms start at the same base pose.
                # If spacing is needed for viz, it should be handled during rendering/USD setup,
                # as the IK solver operates independently per environment.
                builder.add_builder(articulation_builder, xform=initial_arm_transform)

            # Finalize the model
            self.model = builder.finalize(device=config.device)
            self.model.ground = False # Ensure no ground plane interaction
            self.model.joint_attach_ke = self.config.joint_attach_ke
            self.model.joint_attach_kd = self.config.joint_attach_kd
            # We don't need gradients for joints in sampling-based approach
            self.model.joint_q.requires_grad = False
            self.model.body_q.requires_grad = False # FK output doesn't need grad here

            # Set initial joint angles (all arms start at home pose for this batch)
            qpos_home_np = np.array(self.config.qpos_home, dtype=np.float32)
            # Ensure home pose matches DOF, truncate or pad if necessary
            if len(qpos_home_np) != self.dof:
                 log.warning(f"qpos_home length ({len(qpos_home_np)}) != DoF ({self.dof}). Adjusting.")
                 qpos_home_adjusted = np.zeros(self.dof, dtype=np.float32)
                 copy_len = min(len(qpos_home_np), self.dof)
                 qpos_home_adjusted[:copy_len] = qpos_home_np[:copy_len]
                 qpos_home_np = qpos_home_adjusted

            # Tile the home pose for all environments in the batch
            initial_joint_q_np = np.tile(qpos_home_np, self.num_envs)
            log.debug(f"Setting initial joint q (shape {initial_joint_q_np.shape}) for {self.num_envs} envs.")
            self.model.joint_q.assign(wp.array(initial_joint_q_np, dtype=wp.float32, device=config.device))

            # --- State & Solver Prep ---
            # Get initial simulation state (includes body transforms, etc.)
            self.state = self.model.state(requires_grad=False) # Get state AFTER setting initial joint_q
            # Allocate GPU arrays for error and joint updates
            self.ee_error_wp = wp.zeros(self.num_envs * 6, dtype=wp.float32, device=config.device)
            self.delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=config.device)
            # Allocate GPU array for temporary perturbed joint states (used in _ik_step)
            self.q_temp_wp = wp.empty_like(self.model.joint_q)

        # --- Renderer Setup (Optional) ---
        self.renderer = None
        if not config.headless:
            log.info(f"Initializing renderer for batch, outputting to {batch_usd_path}")
            try:
                self.renderer = wp.sim.render.SimRenderer(self.model, batch_usd_path, scaling=1.0)
                log.debug("SimRenderer created successfully.")
                # Initialize gizmo arrays on GPU for rendering
                self.gizmo_pos_wp = wp.zeros(self.num_envs * 6, dtype=wp.vec3, device=config.device)
                self.gizmo_rot_wp = wp.zeros(self.num_envs * 6, dtype=wp.quat, device=config.device)
                # Precompute gizmo rotations (relative to frame axes)
                self.rot_x_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0) # Rotate around Z for X gizmo
                self.rot_y_axis_q_wp = wp.quat_identity()                                            # No rotation needed for Y gizmo
                self.rot_z_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)# Rotate around X for Z gizmo
                log.debug("Renderer and gizmo arrays initialized.")
            except Exception as e:
                log.error(f"Failed to initialize renderer for batch: {e}. Running headless for this batch.", exc_info=True)
                self.renderer = None # Ensure renderer is None if setup fails


    def compute_ee_error(self, joint_q_wp: wp.array, body_q_wp: wp.array) -> np.ndarray:
        """
        Computes the 6D end-effector error for all envs using a kernel.
        Args:
            joint_q_wp: Warp array of joint angles (num_envs * dof) - NOT DIRECTLY USED, but indicates state.
            body_q_wp: Warp array of body transforms (num_envs * num_links) - RESULTING FROM FK(joint_q_wp).
        Returns:
            Numpy array of flattened errors (num_envs * 6).
        """
        if log.isEnabledFor(logging.DEBUG):
             log.debug("--- compute_ee_error ---")
             log.debug(f"  Input shapes: body_q={body_q_wp.shape}, targets_pos={self.targets_pos_wp.shape}, targets_ori={self.targets_ori_wp.shape}")
             log.debug(f"  Params: num_links={self.num_links}, ee_idx={self.ee_link_index}, ee_offset={tuple(self.ee_link_offset)}")

        with wp.ScopedTimer("error_kernel", print=False, active=True, dict=self.profiler):
            try:
                wp.launch(
                    compute_ee_error_kernel,
                    dim=self.num_envs,
                    inputs=[
                        body_q_wp, # Pass the current body transforms
                        self.num_links,
                        self.ee_link_index,
                        self.ee_link_offset,
                        self.targets_pos_wp, # Batch targets (pos)
                        self.targets_ori_wp, # Batch targets (ori)
                    ],
                    outputs=[self.ee_error_wp], # Output array
                    device=self.config.device
                )
                wp.synchronize(device=self.config.device) # Ensure kernel finishes before numpy conversion

                if log.isEnabledFor(logging.DEBUG) and self.num_envs > 0:
                    # Sample only first environment's error for debugging
                    if USE_TORCH_FOR_DEBUG_SLICING:
                         first_error_t = wp.to_torch(self.ee_error_wp[0:6])
                         log.debug(f"  Sample error (env 0): pos={first_error_t[0:3].cpu().numpy()}, ori={first_error_t[3:6].cpu().numpy()}")
                    else:
                         first_error_np = self.ee_error_wp.numpy()[0:6]
                         log.debug(f"  Sample error (env 0): pos={first_error_np[0:3]}, ori={first_error_np[3:6]}")

            except Exception as e:
                log.error(f"Error during compute_ee_error_kernel launch: {e}", exc_info=True)
                raise

        # Return error as numpy array (required by the sampling logic)
        return self.ee_error_wp.numpy()


    def _ik_step(self):
        """
        Performs one IK step using Sampling-based Jacobian Estimation with Gauss-Newton update.
        This logic is adapted from the provided Morph._step method.
        """
        log.debug("=== Starting IK Step ===")

        # --- Get IK Parameters ---
        sigma = self.config.ik_sigma
        num_samples = self.config.ik_num_samples
        damping = self.config.ik_damping
        dof = self.dof
        num_envs = self.num_envs

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"  Params: sigma={sigma}, num_samples={num_samples}, damping={damping}")
            log.debug(f"  System: num_envs={num_envs}, dof={dof}")

        # --- 1. Calculate Baseline Error ---
        log.debug("  Computing baseline FK and error...")
        # Perform Forward Kinematics using current joint angles (self.model.joint_q)
        # This updates self.state.body_q in-place on the GPU.
        with wp.ScopedTimer("eval_fk_base", print=False, active=True, dict=self.profiler):
            wp.sim.eval_fk(self.model, self.model.joint_q, None, None, self.state)
            wp.synchronize(device=self.config.device) # Ensure FK completes
            if log.isEnabledFor(logging.DEBUG) and num_envs > 0:
                 ee_link_transform = self.state.body_q[self.ee_link_index] # Transform of EE link in first env
                 log.debug(f"  Sample baseline EE link transform (env 0): {list(ee_link_transform)}")


        # Compute the 6D error based on the FK result (self.state.body_q)
        # This function calls the kernel and returns the error as a NumPy array
        error_base_all_np = self.compute_ee_error(self.model.joint_q, self.state.body_q) # Shape (num_envs * 6)
        log.debug(f"  Computed baseline error (shape: {error_base_all_np.shape})")

        # --- 2. Sampling Loop (CPU-side logic driving GPU computations) ---
        # Get current joint angles from GPU to CPU for manipulation in the loop
        q_current_all_np = self.model.joint_q.numpy().copy() # Shape (num_envs * dof)
        # Array to store computed joint updates (delta_q) on CPU temporarily
        delta_q_all_np = np.zeros_like(q_current_all_np)

        log.debug(f"  Starting sampling loop for {num_envs} environments...")
        # Iterate through each environment in the batch
        for e in range(num_envs):
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"  --- Processing Environment {e+1}/{num_envs} ---")
            # --- Extract data for environment 'e' ---
            base_idx_q = e * dof         # Start index for joints of env 'e'
            base_idx_err = e * 6       # Start index for error of env 'e'
            q_e_np = q_current_all_np[base_idx_q : base_idx_q + dof].copy() # Current joints for env 'e'
            error_base_e_np = error_base_all_np[base_idx_err : base_idx_err + 6].copy() # Baseline error for env 'e'

            if log.isEnabledFor(logging.DEBUG):
                 log.debug(f"    Current q: {q_e_np[:min(dof,4)]}... (first 4)") # Log first few joints
                 pos_err_norm = np.linalg.norm(error_base_e_np[0:3])
                 ori_err_norm = np.linalg.norm(error_base_e_np[3:6])
                 log.debug(f"    Base error: pos_mag={pos_err_norm:.4f}, ori_mag={ori_err_norm:.4f}")


            # --- Generate Perturbations & Prepare Sample Storage ---
            # D: Joint space perturbations (num_samples x dof)
            D_np = self.rng.normal(0.0, sigma, size=(num_samples, dof)).astype(np.float32)
            # Delta: Corresponding change in EE error (num_samples x 6)
            Delta_np = np.zeros((num_samples, 6), dtype=np.float32)

            if log.isEnabledFor(logging.DEBUG) and num_samples > 0:
                log.debug(f"    Generated {num_samples} perturbations (sample 0, first 4 joints): {D_np[0, :min(dof,4)]}...")

            # --- Inner Sampling Loop (for Jacobian estimation) ---
            with wp.ScopedTimer("sampling_inner_loop", print=False, active=True, dict=self.profiler):
                # Temporary numpy array to hold joint states for *all* envs, modified one sample at a time
                q_temp_all_np = q_current_all_np.copy()

                for i in range(num_samples):
                    # Apply perturbation D_np[i] to joints of env 'e' (q_e_np)
                    q_sample_e_np = q_e_np + D_np[i]

                    # Update the temporary *full* joint array with the perturbed joints for env 'e'
                    q_temp_all_np[base_idx_q : base_idx_q + dof] = q_sample_e_np

                    # --- Perform FK for the perturbed state ---
                    # Assign the temporary (modified) joint state to the GPU model's q_temp_wp array
                    self.q_temp_wp.assign(q_temp_all_np) # CPU -> GPU transfer

                    # Run FK using the temporary joint state (self.q_temp_wp)
                    # This updates self.state.body_q in-place on GPU based on q_temp_wp
                    with wp.ScopedTimer("eval_fk_sample", print=False, active=True, dict=self.profiler):
                        wp.sim.eval_fk(self.model, self.q_temp_wp, None, None, self.state)
                        # No sync needed here usually, error computation below will wait

                    # --- Compute Error for the perturbed state ---
                    # Compute error using the body poses resulting from the *perturbed* q
                    # Pass self.q_temp_wp (for consistency) and the updated self.state.body_q
                    error_sample_all_np = self.compute_ee_error(self.q_temp_wp, self.state.body_q) # Returns numpy

                    # Extract the error for environment 'e' from the result
                    error_sample_e_np = error_sample_all_np[base_idx_err : base_idx_err + 6]

                    # Calculate the *change* in error due to the perturbation
                    Delta_np[i] = error_sample_e_np - error_base_e_np

                    # Log first sample details if debugging
                    if i == 0 and log.isEnabledFor(logging.DEBUG):
                         log.debug(f"      Sample {i+1} - delta_q (first 4): {D_np[i,:min(dof,4)]}")
                         log.debug(f"      Sample {i+1} - resulted error: pos_mag={np.linalg.norm(error_sample_e_np[0:3]):.4f}, ori_mag={np.linalg.norm(error_sample_e_np[3:6]):.4f}")
                         log.debug(f"      Sample {i+1} - error change (Delta): {Delta_np[i]}")

            # --- 3. Estimate Jacobian using Least Squares ---
            with wp.ScopedTimer("jacobian_estimation", print=False, active=True, dict=self.profiler):
                # We want to find J_est (6 x dof) such that: Delta ≈ D @ J_est^T
                # Solve: D * J_transpose ≈ Delta for J_transpose
                try:
                    # J_transpose_est will be (dof x 6)
                    J_transpose_est, residuals, rank, s = np.linalg.lstsq(D_np, Delta_np, rcond=None)
                    J_est = J_transpose_est.T # Transpose to get J (6 x dof)
                    if log.isEnabledFor(logging.DEBUG):
                         log.debug(f"    Jacobian estimated: shape={J_est.shape}, rank={rank}") #, svals={s[:min(len(s),6)]}")
                except np.linalg.LinAlgError as lin_err:
                    log.error(f"    Linalg Error during Jacobian estimation for env {e}: {lin_err}. Skipping update.")
                    J_est = None # Skip update calculation for this env

            # --- 4. Compute Gauss-Newton Update ---
            if J_est is not None: # Proceed only if Jacobian estimation was successful
                with wp.ScopedTimer("gn_update", print=False, active=True, dict=self.profiler):
                    # Solve: (J * J^T + damping * I) * y = error_base
                    try:
                        JJt = J_est @ J_est.T # (6 x 6)
                        A = JJt + damping * np.eye(6, dtype=np.float32) # Damped matrix (6 x 6)
                        y = np.linalg.solve(A, error_base_e_np) # Solve for y (6,)

                        # Compute joint update: delta_q = -J^T * y
                        delta_q_e_np = -J_est.T @ y # (dof,)

                        # Store the computed update for this environment
                        delta_q_all_np[base_idx_q : base_idx_q + dof] = delta_q_e_np

                        if log.isEnabledFor(logging.DEBUG):
                            update_mag = np.linalg.norm(delta_q_e_np)
                            log.debug(f"    Computed delta_q (mag: {update_mag:.4e}): {delta_q_e_np[:min(dof,4)]}...")
                            log.debug(f"    Solving Ax=b: A shape={A.shape}, b shape={error_base_e_np.shape}, y shape={y.shape}")

                    except np.linalg.LinAlgError as lin_err:
                        log.error(f"    Linalg Error during GN update solve for env {e}: {lin_err}. Using zero update.")
                        # Keep delta_q_all_np as zeros for this env
            else:
                 # Jacobian estimation failed, delta_q remains zero for this env
                 log.warning(f"    Skipping GN update for env {e} due to Jacobian estimation failure.")


        # --- 5. Apply Updates and Clip Joints (GPU Operations) ---
        log.debug("  Applying all computed delta_q updates on GPU...")
        # Assign the computed delta_q (all envs) from CPU numpy array to GPU Warp array
        self.delta_q_wp.assign(delta_q_all_np)

        # Apply the updates: model.joint_q = model.joint_q + delta_q_wp
        with wp.ScopedTimer("add_delta_q", print=False, active=True, dict=self.profiler):
            wp.launch(
                add_delta_q_kernel,
                dim=num_envs * dof, # One thread per joint instance
                inputs=[self.model.joint_q, self.delta_q_wp],
                outputs=[self.model.joint_q], # Update joint_q in-place
                device=self.config.device
            )

        # Clip joint angles to their limits using the kernel
        with wp.ScopedTimer("clip_joints", print=False, active=True, dict=self.profiler):
            wp.launch(
                kernel=clip_joints_kernel,
                dim=num_envs * dof, # One thread per joint instance
                inputs=[
                    self.model.joint_q, # Input/Output array
                    self.joint_limits_min_wp,
                    self.joint_limits_max_wp,
                    num_envs,
                    dof
                ],
                device=self.config.device
            )
            wp.synchronize(device=self.config.device) # Ensure clipping finishes

        if log.isEnabledFor(logging.DEBUG) and num_envs > 0:
             if USE_TORCH_FOR_DEBUG_SLICING:
                  clipped_q_t = wp.to_torch(self.model.joint_q[0:dof])
                  log.debug(f"  Sample clipped joints (env 0): {clipped_q_t.cpu().numpy()}")
             else:
                  clipped_q_np = self.model.joint_q.numpy()[0:dof]
                  log.debug(f"  Sample clipped joints (env 0): {clipped_q_np}")

        log.debug("=== Finished IK Step ===\n")


    def step(self):
        """Performs one full IK step for the batch."""
        with wp.ScopedTimer("ik_step_total", print=False, active=True, dict=self.profiler):
            self._ik_step() # Call the core IK logic

        # Increment render time if renderer is active
        if self.renderer is not None:
            self.render_time += self.frame_dt

    def get_final_joint_q(self) -> np.ndarray:
        """Returns the final joint configurations for the current batch as a NumPy array."""
        log.debug(f"Retrieving final joint q (shape: {self.model.joint_q.shape}) from GPU.")
        # Ensure all GPU operations are complete before reading back
        wp.synchronize(device=self.config.device)
        return self.model.joint_q.numpy()

    def render_gizmos(self):
        """Renders visualization gizmos for targets and end-effector poses."""
        if self.renderer is None or self.gizmo_pos_wp is None:
            # log.debug("Skipping gizmo rendering - renderer or gizmo arrays not initialized")
            return

        log.debug("Rendering gizmos...")
        # Ensure FK is up-to-date based on the *latest* joint angles for visualization
        with wp.ScopedTimer("eval_fk_render", print=False, active=True, dict=self.profiler):
             wp.sim.eval_fk(self.model, self.model.joint_q, None, None, self.state)
             # No sync needed usually, kernel launch below will wait

        radius = self.config.gizmo_radius
        half_height = self.config.gizmo_length / 2.0

        # Calculate all gizmo transforms on GPU using the kernel
        with wp.ScopedTimer("gizmo_kernel", print=False, active=True, dict=self.profiler):
            wp.launch(
                kernel=calculate_gizmo_transforms_kernel,
                dim=self.num_envs, # One thread per environment
                inputs=[
                    self.state.body_q, # Current body transforms from FK
                    self.targets_pos_wp,
                    self.targets_ori_wp,
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
            # No sync needed usually, numpy conversion below will wait

        # Transfer calculated gizmo transforms from GPU to CPU for rendering calls
        gizmo_pos_np = self.gizmo_pos_wp.numpy() # Shape (num_envs * 6, 3)
        gizmo_rot_np = self.gizmo_rot_wp.numpy() # Shape (num_envs * 6, 4)
        log.debug("Gizmo transforms calculated and transferred to CPU.")

        # Render cones for each gizmo in each environment
        for e in range(self.num_envs):
            base_idx = e * 6
            try:
                # Target Gizmos (Red, Green, Blue for X, Y, Z) - Lighter colors
                self.renderer.render_cone(name=f"target_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 0]),
                                       rot=tuple(gizmo_rot_np[base_idx + 0]), radius=radius,
                                       half_height=half_height, color=self.config.gizmo_color_x_target)
                self.renderer.render_cone(name=f"target_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 1]),
                                       rot=tuple(gizmo_rot_np[base_idx + 1]), radius=radius,
                                       half_height=half_height, color=self.config.gizmo_color_y_target)
                self.renderer.render_cone(name=f"target_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 2]),
                                       rot=tuple(gizmo_rot_np[base_idx + 2]), radius=radius,
                                       half_height=half_height, color=self.config.gizmo_color_z_target)

                # End-Effector Gizmos (Red, Green, Blue for X, Y, Z) - Darker colors
                self.renderer.render_cone(name=f"ee_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 3]),
                                       rot=tuple(gizmo_rot_np[base_idx + 3]), radius=radius,
                                       half_height=half_height, color=self.config.gizmo_color_x_ee)
                self.renderer.render_cone(name=f"ee_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 4]),
                                       rot=tuple(gizmo_rot_np[base_idx + 4]), radius=radius,
                                       half_height=half_height, color=self.config.gizmo_color_y_ee)
                self.renderer.render_cone(name=f"ee_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 5]),
                                       rot=tuple(gizmo_rot_np[base_idx + 5]), radius=radius,
                                       half_height=half_height, color=self.config.gizmo_color_z_ee)
            except Exception as render_err:
                # Log error but continue rendering other environments
                log.error(f"Failed to render gizmos for environment {e}: {str(render_err)}", exc_info=False)
                continue
        log.debug("Gizmo rendering commands issued.")


    def render(self):
        """Renders the current state of all environments in the batch."""
        if self.renderer is None:
            # log.debug("Skipping render - renderer not initialized")
            return

        with wp.ScopedTimer("render_frame", print=False, active=True, dict=self.profiler):
            log.debug(f"Rendering frame at time {self.render_time:.2f}")
            self.renderer.begin_frame(self.render_time)

            # Render the articulated arms based on the current state (self.state)
            # FK should have been called recently (either in IK step or render_gizmos)
            log.debug("Rendering model state...")
            self.renderer.render(self.state)

            # Render the target and EE gizmos
            log.debug("Rendering gizmos...")
            self.render_gizmos()

            log.debug("Ending frame...")
            self.renderer.end_frame()


# --- Main Execution (Batch Processing) ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel IK Solver for Predefined Tattoo Targets using Batching")
    # --- File Paths ---
    parser.add_argument("--targets", type=str, default=IKConfig.ik_targets_path,
                        help="Path to the .npy file containing N x 7 IK targets (px,py,pz, qx,qy,qz,qw).")
    parser.add_argument("--urdf", type=str, default=IKConfig.urdf_path,
                        help="Path to the robot URDF file.")
    parser.add_argument("--output_dir", type=str, default=IKConfig.output_dir,
                        help="Directory to save the output batch USD files and final joint configs.")
    # --- Solver Params ---
    parser.add_argument("--iters", type=int, default=IKConfig.num_iters,
                        help="Number of IK iterations per batch.")
    parser.add_argument("--batch_size", type=int, default=IKConfig.batch_size,
                        help="Maximum number of environments (targets) per batch.")
    parser.add_argument("--sigma", type=float, default=IKConfig.ik_sigma,
                        help="Std deviation for IK sampling perturbations.")
    parser.add_argument("--num_samples", type=int, default=IKConfig.ik_num_samples,
                        help="Number of samples per env for Jacobian estimation.")
    parser.add_argument("--damping", type=float, default=IKConfig.ik_damping,
                        help="Damping factor for Gauss-Newton IK update.")
    # --- Execution Params ---
    parser.add_argument("--device", type=str, default=IKConfig.device,
                        help="Compute device (e.g., 'cuda:0' or 'cpu'). Auto-selects if None.")
    parser.add_argument("--headless", action="store_true", default=IKConfig.headless,
                        help="Run without visualization and USD saving.")
    parser.add_argument("--fps", type=int, default=IKConfig.fps,
                        help="Rendering FPS for USD (if not headless).")
    parser.add_argument("--seed", type=int, default=IKConfig.seed,
                        help="Random seed for IK sampling.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging.")


    args = parser.parse_args()

    # Update logging level if debug flag is set
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.info("DEBUG logging enabled.")


    # Create configuration object from defaults and args
    config = IKConfig(
        ik_targets_path=args.targets,
        urdf_path=args.urdf,
        output_dir=args.output_dir,
        num_iters=args.iters,
        batch_size=args.batch_size,
        ik_sigma=args.sigma,
        ik_num_samples=args.num_samples,
        ik_damping=args.damping,
        device=args.device,
        headless=args.headless,
        fps=args.fps,
        seed=args.seed,
        # Other parameters like joint limits, offsets use defaults unless added as args
    )

    # --- Setup ---
    output_dir_expanded = os.path.expanduser(config.output_dir)
    if not os.path.exists(output_dir_expanded):
        os.makedirs(output_dir_expanded, exist_ok=True)
        log.info(f"Created output directory: {output_dir_expanded}")

    wp.init()
    sim_device = config.device if config.device else wp.get_preferred_device()
    config.device = str(sim_device) # Store the resolved device name in config
    log.info(f"Using Warp device: {config.device}")
    log.info(f"Run Configuration: Headless={config.headless}, Batch Size={config.batch_size}, Iters={config.num_iters}")
    log.debug(f"Full Config: {config}")


    # --- Load ALL targets from file ---
    all_targets_np = None
    try:
        target_path_expanded = os.path.expanduser(config.ik_targets_path)
        log.info(f"Loading all targets from: {target_path_expanded}")
        all_targets_np = np.load(target_path_expanded).astype(np.float32)

        if all_targets_np.ndim != 2 or all_targets_np.shape[1] != 7:
             raise ValueError(f"Expected N x 7 target array (pos, quat xyzw), got shape {all_targets_np.shape}")

        total_targets = len(all_targets_np)
        log.info(f"Successfully loaded {total_targets} targets.")
        log.debug(f"Target array shape: {all_targets_np.shape}, dtype: {all_targets_np.dtype}")
        # Simple validation of quaternion norms (optional)
        norms = np.linalg.norm(all_targets_np[:, 3:], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
             log.warning(f"Some target quaternions are not normalized. Max deviation: {np.max(np.abs(norms - 1.0)):.4e}")


    except FileNotFoundError:
        log.error(f"Target file not found: {config.ik_targets_path}")
        exit(1)
    except Exception as e:
        log.error(f"Failed to load or validate targets: {e}", exc_info=True)
        exit(1)


    # --- Batch Processing ---
    num_batches = math.ceil(total_targets / config.batch_size)
    log.info(f"Processing {total_targets} targets in {num_batches} batches of up to size {config.batch_size}")

    all_final_q_list = [] # List to store final joint angles (numpy arrays) from all batches
    solver = None         # Keep solver variable outside loop scope for finally block/profiling access
    overall_start_time = time.monotonic()
    processed_targets_count = 0

    try:
        # Ensure operations happen on the selected device
        with wp.ScopedDevice(config.device):
            for batch_idx in range(num_batches):
                batch_start_time = time.monotonic()

                # Determine target indices for this batch
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, total_targets)
                targets_batch_np = all_targets_np[start_idx:end_idx]
                current_batch_size = len(targets_batch_np) # Actual size of this batch

                if current_batch_size == 0:
                    log.warning(f"Skipping empty batch {batch_idx+1}")
                    continue

                # Define path for the USD file for this batch (even if headless, for consistency)
                current_usd_path = os.path.join(output_dir_expanded, f"ik_batch_{batch_idx:04d}.usd")

                log.info(f"--- Starting Batch {batch_idx+1}/{num_batches} ({current_batch_size} targets) ---")
                if not config.headless:
                    log.info(f"Outputting USD to: {current_usd_path}")
                log.debug(f"Batch target indices: {start_idx} to {end_idx-1}")

                # --- Instantiate Solver for this Batch ---
                try:
                   solver = ParallelTattooIKSolver(config, targets_batch_np, current_batch_size, current_usd_path)
                except Exception as init_err:
                   log.error(f"Failed to initialize solver for batch {batch_idx+1}: {init_err}", exc_info=True)
                   log.error("Skipping this batch.")
                   continue # Skip to the next batch

                # --- Run IK Iterations for this Batch ---
                log.debug(f"Starting {config.num_iters} IK iterations...")
                iter_start_time = time.monotonic()
                for i in range(config.num_iters):
                    solver.step() # Perform one IK step

                    # Render periodically or at the end if desired (can be slow)
                    if not config.headless:
                         if i == config.num_iters - 1: # Render last frame
                              log.debug(f"Rendering final frame of batch {batch_idx+1}")
                              solver.render()
                         # Add periodic rendering if needed:
                         # elif i % 50 == 0:
                         #    log.debug(f"Rendering intermediate frame at iter {i}")
                         #    solver.render()


                    # Optional: Log progress within a batch
                    if (i + 1) % 50 == 0: # Log every 50 iterations
                         iter_elapsed = time.monotonic() - iter_start_time
                         log.debug(f"Batch {batch_idx+1}, Iteration {i+1}/{config.num_iters} completed ({iter_elapsed:.2f}s so far)")


                # --- Save USD for this Batch (if renderer exists) ---
                if solver.renderer:
                    log.info(f"Saving USD for batch {batch_idx+1}...")
                    try:
                        with wp.ScopedTimer("usd_save", print=False, active=True, dict=solver.profiler):
                            solver.renderer.save()
                        log.debug(f"Successfully saved USD: {current_usd_path}")
                    except Exception as e:
                        log.error(f"Failed to save USD file for batch {batch_idx+1}: {e}", exc_info=True)


                # --- Store Final Joint Configurations for this Batch ---
                batch_final_q_np = solver.get_final_joint_q() # Get result from GPU -> CPU
                log.debug(f"Retrieved final joint config shape for batch: {batch_final_q_np.shape}") # Should be (current_batch_size * dof,)

                # Reshape to (current_batch_size, dof) before appending
                if batch_final_q_np.size == current_batch_size * solver.dof:
                     batch_final_q_reshaped = batch_final_q_np.reshape(current_batch_size, solver.dof)
                     all_final_q_list.append(batch_final_q_reshaped)
                     processed_targets_count += current_batch_size
                     log.debug(f"Stored final q for batch, reshaped to {batch_final_q_reshaped.shape}")
                else:
                     log.error(f"Mismatch in final joint config size for batch {batch_idx+1}. "
                               f"Expected {current_batch_size * solver.dof}, got {batch_final_q_np.size}. Skipping results.")


                batch_end_time = time.monotonic()
                batch_duration = batch_end_time - batch_start_time
                log.info(f"--- Finished Batch {batch_idx+1}/{num_batches} (Duration: {batch_duration:.2f}s) ---")
                if config.num_iters > 0:
                     log.info(f"  Avg time per iteration: {batch_duration/config.num_iters:.4f}s")
                log.info(f"  Total targets processed so far: {processed_targets_count}")

    except Exception as e:
        # Catch errors during the main batch processing loop
        log.error(f"An error occurred during batch processing: {e}", exc_info=True)
        log.error("Processing stopped.")

    finally:
        overall_end_time = time.monotonic()
        total_duration = overall_end_time - overall_start_time
        log.info(f"\n=== Processing Summary ===")
        log.info(f"Total processing time: {total_duration:.2f} seconds")
        if num_batches > 0:
             log.info(f"Average time per batch: {total_duration/num_batches:.2f} seconds")
        log.info(f"Total targets processed successfully: {processed_targets_count}/{total_targets}")


        # --- Consolidate and Save Final Results ---
        if all_final_q_list and processed_targets_count == total_targets:
             log.info("Consolidating final joint configurations...")
             try:
                 # Concatenate the results from all batches along the first axis (target axis)
                 final_q_all_targets_np = np.concatenate(all_final_q_list, axis=0)

                 # Final check on the shape
                 if final_q_all_targets_np.shape == (total_targets, solver.dof): # Use solver.dof from last successful batch
                      save_path = os.path.join(output_dir_expanded, "final_joint_configs.npy")
                      np.save(save_path, final_q_all_targets_np)
                      log.info(f"Saved final joint configurations for all {total_targets} targets to: {save_path}")
                      log.info(f"Final array shape: {final_q_all_targets_np.shape}")
                 else:
                      # This case should ideally not happen if checks within the loop work
                      log.error("Concatenated final joint configurations have unexpected shape: "
                                f"{final_q_all_targets_np.shape}. Expected ({total_targets}, {solver.dof}). Saving skipped.")

             except Exception as e:
                  log.error(f"Failed to concatenate or save consolidated final joint configurations: {e}", exc_info=True)
        elif not all_final_q_list:
             log.warning("No batch results were collected. Final joint configurations not saved.")
        else: # Some results collected, but not all targets processed
             log.warning(f"Processed {processed_targets_count} targets, but expected {total_targets}. "
                         f"Final joint configurations for all targets were not saved due to errors in some batches.")


        # --- Profiling Output (from the last processed batch) ---
        if solver and solver.profiler:
            log.info("\n--- Performance Profile (Last Batch) ---")
            # Sort profiler items by average time descending for clarity
            profiler_items = sorted(solver.profiler.items(), key=lambda item: np.mean(item[1]) if item[1] else 0, reverse=True)

            for key, times in profiler_items:
                if times: # Check if list is not empty
                    times_np = np.array(times)
                    avg_t = np.mean(times_np)
                    std_t = np.std(times_np)
                    min_t = np.min(times_np)
                    max_t = np.max(times_np)
                    total_t_ms = np.sum(times_np)
                    count = len(times_np)
                    log.info(f"  {key:<25}: Avg: {avg_t:8.3f} ms | Std: {std_t:7.3f} | Min: {min_t:7.3f} | Max: {max_t:7.3f} | Total: {total_t_ms:8.1f} ms ({count} calls)")
                else:
                    log.info(f"  {key:<25}: No calls recorded")


        log.info("Script finished.")