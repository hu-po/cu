from dataclasses import dataclass, field
import math
import os
import logging
import time

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# --- End Logging Setup ---


# FIX 2: Kernel signature updated to use wp.int32
@wp.kernel
def forward_kinematics(
    body_q: wp.array(dtype=wp.transform),
    num_links: wp.int32,
    ee_link_index: wp.int32,
    ee_link_tf_offset: wp.transform,
    ee_pose: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    # Indexing with wp.int32 should work implicitly
    index = tid * num_links + ee_link_index
    ee_pose[tid] = wp.transform_multiply(body_q[index], ee_link_tf_offset)

# FIX 1: Kernel signature updated to use wp.float32
@wp.kernel
def compute_ik_loss(
    ee_pose: wp.array(dtype=wp.transform),
    target_pos: wp.array(dtype=wp.vec3),
    target_quat: wp.array(dtype=wp.quat),
    kp_pos: wp.float32,
    kp_rot: wp.float32,
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    current_pos = wp.transform_get_translation(ee_pose[tid])
    pos_err = target_pos[tid] - current_pos
    loss_pos = wp.dot(pos_err, pos_err)
    current_quat = wp.transform_get_rotation(ee_pose[tid])
    target_q = target_quat[tid]
    q_inv = wp.quat_inverse(current_quat)
    delta_q = wp.mul(target_q, q_inv)
    # Ensure positive scalar part for stable angle extraction
    if delta_q[3] < 0.0:
        delta_q = wp.quat(-delta_q[0], -delta_q[1], -delta_q[2], -delta_q[3])
    # Angle-axis representation (axis * angle) - using 2.0 * vec_part
    rot_err = 2.0 * wp.vec3(delta_q[0], delta_q[1], delta_q[2])
    loss_rot = wp.dot(rot_err, rot_err)
    loss[tid] = wp.float32(kp_pos * loss_pos + kp_rot * loss_rot)

@dataclass
class SimConfig:
    device: str = None # device to run the simulation on
    seed: int = 42 # random seed
    headless: bool = False # turns off rendering
    num_envs: int = 16 # number of parallel environments
    num_rollouts: int = 2 # number of rollouts to perform
    train_iters: int = 32 # number of training iterations per rollout
    learning_rate: float = 0.1 # learning rate for IK updates
    start_time: float = 0.0 # start time for the simulation
    fps: int = 60 # frames per second
    urdf_path: str = "~/dev/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf" # path to the urdf file
    usd_output_path: str = "~/dev/cu/warp/ik_output_6d_gemini.usd" # path to the usd file to save the model
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0) # offset from the ee_gripper_link to the end effector
    kp_pos: float = 100.0 # gain for position error
    kp_rot: float = 1.0 # gain for rotation error
    gizmo_radius: float = 0.005 # radius of the gizmo (used for arrow base radius)
    gizmo_length: float = 0.05 # total length of the gizmo arrow
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0) # color of the x gizmo for the ee
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0) # color of the y gizmo for the ee
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0) # color of the z gizmo for the ee
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5) # color of the x gizmo for the target
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5) # color of the y gizmo for the target
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0) # color of the z gizmo for the target
    arm_spacing_xz: float = 1.0 # spacing between arms in the x-z plane
    arm_height_offset: float = 0.0 # height of the arm off the floor
    target_pos_offset: tuple[float, float, float] = (0.0, 0.1, 0.3) # position offset of the target relative to arm base
    target_spawn_box_size: float = 0.1 # size of the box to spawn the target in
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: [
        (-3.054, 3.054),    # base
        (0.0, 3.14),        # shoulder
        (0.0, 2.356),       # elbow
        (-1.57, 1.57),      # wrist 1
        (-1.57, 1.57),      # wrist 2
        (-3.14, 3.14),      # wrist 3
        (0.0, 0.044),       # right finger
        (0.0, 0.044),       # left finger
    ]) # joint limits for arm
    arm_rot_offset: list[tuple[tuple[float, float, float], float]] = field(default_factory=lambda: [
        ((1.0, 0.0, 0.0), -math.pi * 0.5), # quarter turn about x-axis
        ((0.0, 0.0, 1.0), -math.pi * 0.5), # quarter turn about z-axis
    ]) # list of axis angle rotations for initial arm orientation offset
    qpos_home: list[float] = field(default_factory=lambda: [0, np.pi/12, np.pi/12, 0, 0, 0, 0, 0]) # home position for the arm
    q_angle_shuffle: list[float] = field(default_factory=lambda: [np.pi/2, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0.01, 0.01]) # amount of random noise to add to the arm joint angles
    joint_q_requires_grad: bool = True # whether to require grad for the joint q
    body_q_requires_grad: bool = True # whether to require grad for the body q
    joint_attach_ke: float = 1600.0 # stiffness for the joint attach
    joint_attach_kd: float = 20.0 # damping for the joint attach

class Sim:
    def __init__(self, config: SimConfig):
        log.info(f"Initializing Sim with config: {config}")
        self.config = config
        self.device = wp.get_device(config.device) # Ensure device object
        wp.set_device(self.device) # Set default device for subsequent allocations

        self.rng = np.random.default_rng(config.seed)
        self.num_envs = config.num_envs
        self.render_time = config.start_time
        self.frame_dt = 1.0 / config.fps
        articulation_builder = wp.sim.ModelBuilder()
        # Ensure URDF path exists
        urdf_full_path = os.path.expanduser(config.urdf_path)
        if not os.path.exists(urdf_full_path):
             raise FileNotFoundError(f"URDF file not found: {urdf_full_path}")
        log.info(f"Parsing URDF: {urdf_full_path}")

        wp.sim.parse_urdf(
            urdf_full_path,
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False,
            # arm_num_rigid_contacts=1, # Consider adding if needed
            # tendon_stiffness = 0.0, # Set if you have tendons
            # tendon_damping = 0.0, # Set if you have tendons
        )
        builder = wp.sim.ModelBuilder()
        self.num_links = len(articulation_builder.joint_type)
        self.dof = len(articulation_builder.joint_q)
        self.joint_limits = config.joint_limits # TODO: parse from URDF
        log.info(f"Parsed URDF with {self.num_links} links and {self.dof} DoF")
        if self.dof != len(config.joint_limits):
             log.warning(f"Number of DoF ({self.dof}) does not match length of provided joint_limits ({len(config.joint_limits)})")
        if self.dof != len(config.qpos_home):
             log.warning(f"Number of DoF ({self.dof}) does not match length of provided qpos_home ({len(config.qpos_home)})")
        if self.dof != len(config.q_angle_shuffle):
             log.warning(f"Number of DoF ({self.dof}) does not match length of provided q_angle_shuffle ({len(config.q_angle_shuffle)})")

        # Find the ee_gripper_link index by looking at joint connections
        _ee_link_vec3_offset = wp.vec3(config.ee_link_offset)
        self.ee_link_tf_offset = wp.transform(_ee_link_vec3_offset, wp.quat_identity())
        self.ee_link_index = -1
        log.info(f"Available link names: {articulation_builder.body_name}")
        log.info(f"Available joint names: {articulation_builder.joint_name}")
        log.info(f"Joint parents: {articulation_builder.joint_parent}")
        log.info(f"Joint children: {articulation_builder.joint_child}")

        # Attempt to find link index by name first
        ee_link_name_to_find = "ee_gripper_link" # Adjust if your link name is different
        try:
            self.ee_link_index = articulation_builder.body_name.index(ee_link_name_to_find)
            log.info(f"Found EE link '{ee_link_name_to_find}' at index {self.ee_link_index} by name.")
        except ValueError:
            log.warning(f"Could not find link named '{ee_link_name_to_find}'. Trying to find via 'ee_gripper' joint.")
            # Fallback: Find via joint child connection
            for i, joint_name in enumerate(articulation_builder.joint_name):
                if joint_name == "ee_gripper":  # The fixed joint connecting link_6 to ee_gripper_link
                    self.ee_link_index = articulation_builder.joint_child[i]
                    log.info(f"Found EE link index {self.ee_link_index} via joint '{joint_name}'.")
                    break

        if self.ee_link_index == -1:
            raise ValueError(f"Could not find end-effector link index using name '{ee_link_name_to_find}' or joint 'ee_gripper'. Please check your URDF.")

        # Initial arm orientation is composed of axis angle rotation sequence
        _initial_arm_orientation = wp.quat_identity() # Start with identity
        for i in range(len(config.arm_rot_offset)):
            axis, angle = config.arm_rot_offset[i]
            # Note: Warp quaternion multiplication is right-to-left (new_rot = rot * old_rot)
            _initial_arm_orientation = wp.mul(wp.quat_from_axis_angle(wp.vec3(axis), angle), _initial_arm_orientation)
        self.initial_arm_orientation = _initial_arm_orientation

        # Targets are 6D poses visualized with cone gizmos
        initial_target_pos_np = np.empty((self.num_envs, 3), dtype=np.float32)
        initial_target_quat_np = np.empty((self.num_envs, 4), dtype=np.float32)

        # Parallel arms are spawned in a grid on the floor (x-z plane)
        self.arm_spacing_xz = config.arm_spacing_xz
        self.arm_height_offset = config.arm_height_offset
        self.num_rows = int(math.sqrt(self.num_envs))
        log.info(f"Spawning {self.num_envs} arms in a grid of {self.num_rows}x{self.num_rows}")
        if self.num_rows * self.num_rows != self.num_envs:
             log.warning(f"num_envs ({self.num_envs}) is not a perfect square, grid layout might be uneven.")

        num_joints_in_arm = self.dof # Should match DoF derived from URDF

        for e in range(self.num_envs):
            row = e // self.num_rows
            col = e % self.num_rows
            x = col * self.arm_spacing_xz
            z = row * self.arm_spacing_xz
            base_pos = wp.vec3(x, self.arm_height_offset, z)
            base_tf = wp.transform(base_pos, self.initial_arm_orientation)

            builder.add_builder(
                articulation_builder,
                xform=base_tf,
            )

            # Calculate initial target pose relative to arm base
            # Rotate the offset by the base orientation and add to base position
            target_offset_local = wp.vec3(config.target_pos_offset)
            target_offset_world = wp.quat_rotate(self.initial_arm_orientation, target_offset_local)
            target_pos = base_pos + target_offset_world
            target_rot = self.initial_arm_orientation # Start with same orientation as arm base

            initial_target_pos_np[e] = target_pos
            initial_target_quat_np[e] = target_rot

            # Apply initial joint positions (qpos_home + noise)
            # Ensure indices match the structure built by add_builder
            current_dof_offset = e * num_joints_in_arm
            for i in range(num_joints_in_arm):
                # Check if limits and shuffle arrays have enough elements
                if i < len(config.qpos_home) and i < len(config.q_angle_shuffle) and i < len(config.joint_limits):
                    value = config.qpos_home[i] + self.rng.uniform(-config.q_angle_shuffle[i], config.q_angle_shuffle[i])
                    # Clip value using joint limits
                    clipped_value = np.clip(value, config.joint_limits[i][0], config.joint_limits[i][1])
                    builder.joint_q[current_dof_offset + i] = clipped_value
                else:
                     log.warning(f"Index {i} out of bounds for qpos_home/q_angle_shuffle/joint_limits when setting initial joint angles for env {e}.")
                     # Assign a default value if needed, e.g., 0.0
                     builder.joint_q[current_dof_offset + i] = 0.0


        # Finalize model
        self.model = builder.finalize(device=self.device) # Specify device
        self.model.ground = True # Changed to True for typical arm setup
        self.model.joint_q.requires_grad = config.joint_q_requires_grad
        # self.model.body_q requires grad is implicitly handled by state below
        self.model.joint_attach_ke = config.joint_attach_ke
        self.model.joint_attach_kd = config.joint_attach_kd

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # Renderer setup
        if not config.headless:
            usd_full_path = os.path.expanduser(config.usd_output_path)
            log.info(f"Initializing renderer, outputting to: {usd_full_path}")
            self.renderer = wp.sim.render.SimRenderer(self.model, usd_full_path, up_axis="y") # Explicitly set up_axis if needed
        else:
            self.renderer = None
            log.info("Running in headless mode, renderer disabled.")

        # Simulation state (allocate on the correct device)
        # Note: state() implicitly uses the model's device
        self.state = self.model.state(requires_grad=True)
        # Ensure body_q requires grad if needed for FK differentiation
        self.state.body_q.requires_grad = config.body_q_requires_grad

        # Allocate other arrays on the correct device
        self.ee_pose = wp.zeros(self.num_envs, dtype=wp.transform, requires_grad=True, device=self.device)
        self.initial_target_pos = wp.array(initial_target_pos_np, dtype=wp.vec3, device=self.device)
        self.initial_target_quat = wp.array(initial_target_quat_np, dtype=wp.quat, device=self.device)
        self.target_pos = wp.array(initial_target_pos_np, dtype=wp.vec3, requires_grad=False, device=self.device)
        self.target_quat = wp.array(initial_target_quat_np, dtype=wp.quat, requires_grad=False, device=self.device)
        self.ik_loss = wp.zeros(self.num_envs, dtype=wp.float32, requires_grad=False, device=self.device) # Ensure correct device

        self.profiler = {}
        self.tape = None
        self.pos_error_norm = 0.0
        self.rot_error_norm = 0.0 # Will be calculated if needed

    def compute_ee_pose(self):
        """ Performs forward kinematics to compute the end-effector pose. """
        # FIX 3: Ensure FK is evaluated to update state.body_q based on model.joint_q
        # Pass model.joint_qd=None if velocities are not used/updated elsewhere
        wp.sim.eval_fk(self.model, self.model.joint_q, self.state.joint_qd, None, self.state)

        wp.launch(
            kernel=forward_kinematics,
            dim=self.num_envs,
            inputs=[
                self.state.body_q,
                self.num_links,          # Removed wp.int32() cast
                self.ee_link_index,      # Removed wp.int32() cast
                self.ee_link_tf_offset
            ],
            outputs=[self.ee_pose],
            device=self.device,
        )
        # No return needed if self.ee_pose is used directly
        # return self.ee_pose

    def compute_ik_update(self):
        """ Computes the IK update using gradients of a loss function. """
        if self.tape is None:
             self.tape = wp.Tape()

        with self.tape:
            # Forward pass
            self.compute_ee_pose()
            wp.launch(
                kernel=compute_ik_loss,
                dim=self.num_envs,
                inputs=[
                    self.ee_pose,
                    self.target_pos,
                    self.target_quat,
                    self.config.kp_pos,  # Removed wp.float32() cast
                    self.config.kp_rot,  # Removed wp.float32() cast
                ],
                outputs=[self.ik_loss],
                device=self.device
            )

        # Create adjoint gradient (gradient of loss w.r.t. itself is 1.0)
        # Ensure it's on the correct device
        loss_adjoint = wp.ones(shape=self.ik_loss.shape, dtype=wp.float32, device=self.device)

        # Backward pass
        self.tape.backward(grads={self.ik_loss: loss_adjoint})

        # Retrieve gradients for joint angles
        delta_q = self.tape.gradients.get(self.model.joint_q) # Use .get for safer access

        # Handle case where gradients might not be computed
        if delta_q is None:
            log.warning("Gradients for model.joint_q are None. Returning zero delta_q.")
            delta_q_shape = self.model.joint_q.shape
            delta_q_dtype = self.model.joint_q.dtype # Match original joint q type
            delta_q = wp.zeros(shape=delta_q_shape, dtype=delta_q_dtype, device=self.device)

        self.tape.zero() # Clear gradients for the next iteration
        # self.tape = None # Optional: Destroy tape if memory is critical and re-create next time
        return delta_q

    def step(self):
        """Performs one step of IK optimization."""
        with wp.ScopedTimer("ik_update", print=False, active=True, dict=self.profiler):
            # Compute gradients (negative gradient for descent)
            delta_q = self.compute_ik_update()

        # Gradient descent step: q = q - lr * grad(loss)
        # Need to ensure delta_q has the same shape as model.joint_q (flatten might not be needed if already flat)
        current_q = self.model.joint_q.numpy()
        grad_q = delta_q.numpy() # This is dLoss/dq

        # Check shapes before applying update
        if current_q.shape != grad_q.shape:
             log.error(f"Shape mismatch: current_q {current_q.shape}, grad_q {grad_q.shape}")
             # Attempt reshape if grad_q is flattened unexpectedly
             if len(grad_q.flatten()) == len(current_q.flatten()):
                  log.warning("Reshaping grad_q to match current_q shape.")
                  grad_q = grad_q.reshape(current_q.shape)
             else:
                  log.error("Cannot reshape grad_q. Skipping update.")
                  return # Skip update if shapes cannot be reconciled

        new_q_np = current_q - self.config.learning_rate * grad_q

        # Clip joint angles to limits AFTER the update
        clipped_new_q_np = np.zeros_like(new_q_np)
        num_joints_total = self.num_envs * self.dof
        for i in range(num_joints_total):
             env_index = i // self.dof
             joint_index = i % self.dof
             if joint_index < len(self.joint_limits):
                  min_lim, max_lim = self.joint_limits[joint_index]
                  clipped_new_q_np[i] = np.clip(new_q_np[i], min_lim, max_lim)
             else:
                  clipped_new_q_np[i] = new_q_np[i] # No clipping if limits not defined

        # Assign back to Warp array, ensuring requires_grad is preserved
        self.model.joint_q.assign(clipped_new_q_np)
        # Re-enable grad if it was lost during numpy conversion/assignment
        if not self.model.joint_q.requires_grad:
             self.model.joint_q.requires_grad = self.config.joint_q_requires_grad


    def calculate_error_metrics(self):
        """Calculates position and rotation error metrics."""
        # Ensure latest ee_pose is computed and available as numpy
        # compute_ee_pose() # Call if not guaranteed to be up-to-date
        ee_poses_np = self.ee_pose.numpy() # Get latest computed pose
        target_pos_np = self.target_pos.numpy()
        target_quat_np = self.target_quat.numpy()

        # Position Error
        ee_pos_np = ee_poses_np['p'] # Access position component using string key 'p'
        pos_errors = target_pos_np - ee_pos_np
        self.pos_error_norm = np.linalg.norm(pos_errors, axis=1).mean()

        # Rotation Error (Angle between orientations)
        ee_quat_np = ee_poses_np['q'] # Access rotation component using string key 'q'
        rot_errors = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            # Calculate relative rotation: delta_q = target * current_inverse
            q_target = target_quat_np[i]
            q_ee = ee_quat_np[i]
            # Normalize quaternions for safety
            q_target /= np.linalg.norm(q_target)
            q_ee /= np.linalg.norm(q_ee)
            # Inverse of ee quat: (x,y,z,w) -> (-x,-y,-z,w)
            q_ee_inv = q_ee * np.array([-1, -1, -1, 1])
            # Quaternion multiplication: delta_q = q_target * q_ee_inv
            # (v1, s1) * (v2, s2) = (s1*v2 + s2*v1 + v1 x v2, s1*s2 - v1 . v2)
            tx, ty, tz, tw = q_target
            ix, iy, iz, iw = q_ee_inv
            delta_w = tw*iw - tx*ix - ty*iy - tz*iz
            # Clamp dot product to avoid domain errors in acos
            delta_w_clamped = np.clip(delta_w, -1.0, 1.0)
            # Angle is 2 * acos(|w|) or 2 * acos(w) if w is guaranteed positive
            # Ensure positive scalar part representation before calculating angle
            angle = 2.0 * np.arccos(abs(delta_w_clamped)) # Use abs() for shortest angle
            rot_errors[i] = angle

        self.rot_error_norm = np.mean(rot_errors) # Mean angle error in radians


    def render_gizmos(self):
        """Renders gizmos for EE poses and targets."""
        if self.renderer is None:
            return

        radius = self.config.gizmo_radius
        length = self.config.gizmo_length # Use total length for cone height

        # Get poses as numpy (compute if necessary, but step->render ensures they exist)
        ee_poses_np = self.ee_pose.numpy()
        target_pos_np = self.target_pos.numpy()
        target_quat_np = self.target_quat.numpy()

        for i in range(self.num_envs):
            # Extract individual pose components
            target_pos_tuple = tuple(target_pos_np[i])
            target_rot_wp = wp.quat(target_quat_np[i]) # Use wp.quat for multiplication
            ee_pos_tuple = tuple(ee_poses_np[i]['p']) # Access position field
            ee_rot_wp = wp.quat(ee_poses_np[i]['q'])   # Access rotation field

            # --- Target Gizmo ---
            self.renderer.render_transform(f"target_tf_{i}", wp.transform(target_pos_tuple, target_rot_wp), length, radius)

            # --- EE Gizmo ---
            self.renderer.render_transform(f"ee_tf_{i}", wp.transform(ee_pos_tuple, ee_rot_wp), length, radius)


    def render(self):
        """Renders the simulation state."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False, active=True, dict=self.profiler):
             self.renderer.begin_frame(self.render_time)
             self.renderer.render(self.state) # Render the robot articulation
             self.render_gizmos() # Render EE and target transforms
             self.renderer.end_frame()

        self.render_time += self.frame_dt

def run_sim(config: SimConfig):
    """Initializes and runs the simulation."""
    wp.init()
    log.info(f"Warp initialized. Device default: {wp.get_device().alias}")
    log.info(f"Running on device: {wp.get_device(config.device).alias}") # Confirm correct device context
    log.info("Starting simulation...")

    # Use ScopedDevice to ensure all operations run on the target device
    with wp.ScopedDevice(config.device):
        sim = Sim(config)

        for i in range(config.num_rollouts):
            log.info(f"--- Starting Rollout {i} ---")
            # Select new random target points for all envs
            current_target_pos_np = sim.initial_target_pos.numpy().copy()
            current_target_quat_np = sim.initial_target_quat.numpy().copy()

            # Add random translation noise
            translation_noise = sim.rng.uniform(
                -config.target_spawn_box_size / 2.0,
                config.target_spawn_box_size / 2.0,
                size=(sim.num_envs, 3),
            ).astype(np.float32)
            current_target_pos_np += translation_noise

            # Add random rotation noise (small angles)
            max_angle_noise = np.pi / 8.0 # Max random rotation
            angle_noise = sim.rng.uniform(-max_angle_noise, max_angle_noise, size=sim.num_envs)
            axis_noise = sim.rng.normal(size=(sim.num_envs, 3))
            # Normalize axes, handle potential zero vectors
            norm = np.linalg.norm(axis_noise, axis=1, keepdims=True)
            # Replace zero norms with a default axis (e.g., Z-axis) to avoid division by zero
            zero_norm_mask = (norm == 0).flatten()
            axis_noise[zero_norm_mask] = [0.0, 0.0, 1.0]
            norm[zero_norm_mask] = 1.0 # Avoid division by zero later
            axis_noise /= norm

            new_target_quat_np = np.empty_like(current_target_quat_np)
            for e in range(sim.num_envs):
                 # Convert axis-angle noise to quaternion
                 # Warp requires float for angle
                 noise_quat_wp = wp.quat_from_axis_angle(wp.vec3(axis_noise[e]), float(angle_noise[e]))
                 current_quat_wp = wp.quat(current_target_quat_np[e])
                 # Apply noise: new_rot = noise_rot * current_rot
                 new_quat_wp = wp.mul(noise_quat_wp, current_quat_wp)
                 new_target_quat_np[e] = new_quat_wp

            # Assign new targets to simulation state (Warp arrays)
            sim.target_pos.assign(current_target_pos_np)
            sim.target_quat.assign(new_target_quat_np)
            log.info(f"Rollout {i}: New targets generated.")

            # Training loop for the current rollout
            for j in range(config.train_iters):
                 sim.step() # Perform IK update
                 sim.calculate_error_metrics() # Calculate errors based on new pose
                 if not config.headless:
                      sim.render() # Render the current state

                 # Log progress periodically
                 if (j + 1) % 8 == 0 or j == config.train_iters - 1: # Log every 8 iterations or last iter
                     log.info(f"Rollout {i}, Iter: {j+1}/{config.train_iters}, Pos Error: {sim.pos_error_norm:.4f}, Rot Error (rad): {sim.rot_error_norm:.4f}")

        # Save USD after all rollouts if renderer exists
        if sim.renderer is not None:
            log.info("Saving final USD...")
            sim.renderer.save()
            log.info(f"USD saved to {os.path.expanduser(config.usd_output_path)}")

        # Print profiling info
        if "ik_update" in sim.profiler and len(sim.profiler["ik_update"]) > 0:
            avg_time = np.mean(sim.profiler["ik_update"]) # Use mean
            total_steps = config.num_rollouts * config.train_iters
            steps_per_env = total_steps
            total_env_steps = sim.num_envs * steps_per_env
            # Steps/s calculated as (num_envs * num_iters_per_env) / total_time_seconds
            # Avg time is in ms, so convert to s
            avg_steps_second = float(sim.num_envs) / (avg_time / 1000.0) if avg_time > 0 else 0
            log.info(f"Avg IK update time: {avg_time:.3f} ms")
            log.info(f"Approximate env steps/second: {avg_steps_second:.2f}")
        else:
             log.info("No 'ik_update' profiling data recorded.")

    log.info("Simulation complete!")
    total_steps_performed = config.num_rollouts * config.train_iters
    log.info(f"Performed {total_steps_performed} optimization steps across {config.num_rollouts} rollouts.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Warp 6D Inverse Kinematics Example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu', 'cuda:0', etc.). Defaults to Warp's default.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments (should be a perfect square ideally).")
    parser.add_argument("--headless", action='store_true', help="Run without rendering.")
    parser.add_argument("--num_rollouts", type=int, default=4, help="Number of times to generate new random targets.")
    parser.add_argument("--train_iters", type=int, default=64, help="Number of IK optimization iterations per rollout.")

    args = parser.parse_args() # Use parse_args() to get all arguments

    # Create config object from arguments
    config = SimConfig(
        device=args.device,
        seed=args.seed,
        headless=args.headless,
        num_envs=args.num_envs,
        num_rollouts=args.num_rollouts,
        train_iters=args.train_iters,
    )

    run_sim(config)