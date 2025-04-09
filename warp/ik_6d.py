from dataclasses import dataclass, field
import math
import os
import logging
import time

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@wp.kernel
def forward_kinematics(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_tf_offset: wp.transform,
    ee_pose: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    ee_pose[tid] = wp.transform_multiply(body_q[tid * num_links + ee_link_index], ee_link_tf_offset)

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
    if delta_q[3] < 0.0:
        delta_q = wp.quat(-delta_q[0], -delta_q[1], -delta_q[2], -delta_q[3])
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
    usd_output_path: str = "~/dev/cu/warp/ik_output_6d.usd" # path to the usd file to save the model
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
        self.device = config.device
        self.rng = np.random.default_rng(config.seed)
        self.num_envs = config.num_envs
        self.render_time = config.start_time
        self.frame_dt = 1.0 / config.fps
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.expanduser(config.urdf_path),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False,
        )
        builder = wp.sim.ModelBuilder()
        self.num_links = len(articulation_builder.joint_type)
        self.dof = len(articulation_builder.joint_q)
        self.joint_limits = config.joint_limits # TODO: parse from URDF
        log.info(f"Parsed URDF with {self.num_links} links and {self.dof} dof")
        # Find the ee_gripper_link index by looking at joint connections
        _ee_link_vec3_offset = wp.vec3(config.ee_link_offset)
        self.ee_link_tf_offset = wp.transform(_ee_link_vec3_offset, wp.quat_identity())
        self.ee_link_index = -1
        for i, joint in enumerate(articulation_builder.joint_name):
            if joint == "ee_gripper":  # The fixed joint connecting link_6 to ee_gripper_link
                self.ee_link_index = articulation_builder.joint_child[i]
                break
        if self.ee_link_index == -1:
            raise ValueError("Could not find ee_gripper joint in URDF")
        # initial arm orientation is composed of axis angle rotation sequence
        _initial_arm_orientation = None
        for i in range(len(config.arm_rot_offset)):
            axis, angle = config.arm_rot_offset[i]
            if _initial_arm_orientation is None:
                _initial_arm_orientation = wp.quat_from_axis_angle(wp.vec3(axis), angle)
            else:
                _initial_arm_orientation *= wp.quat_from_axis_angle(wp.vec3(axis), angle)
        self.initial_arm_orientation = _initial_arm_orientation
        # targets are 6D poses visualized with cone gizmos
        initial_target_pos_np = np.empty((self.num_envs, 3), dtype=np.float32)
        initial_target_quat_np = np.empty((self.num_envs, 4), dtype=np.float32)
        # parallel arms are spawned in a grid on the floor (x-z plane)
        self.arm_spacing_xz = config.arm_spacing_xz
        self.arm_height_offset = config.arm_height_offset
        self.num_rows = int(math.sqrt(self.num_envs))
        log.info(f"Spawning {self.num_envs} arms in a grid of {self.num_rows}x{self.num_rows}")
        for e in range(self.num_envs):
            x = (e % self.num_rows) * self.arm_spacing_xz
            z = (e // self.num_rows) * self.arm_spacing_xz
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(wp.vec3(x, self.arm_height_offset, z), self.initial_arm_orientation),
            )
            # Calculate initial target pose relative to arm base
            target_pos = wp.vec3(x + config.target_pos_offset[0],
                                self.arm_height_offset + config.target_pos_offset[1],
                                z + config.target_pos_offset[2])
            target_rot = self.initial_arm_orientation # Start with same orientation as arm base
            initial_target_pos_np[e] = target_pos
            initial_target_quat_np[e] = target_rot
            num_joints_in_arm = len(config.qpos_home)
            for i in range(num_joints_in_arm):
                value = config.qpos_home[i] + self.rng.uniform(-config.q_angle_shuffle[i], config.q_angle_shuffle[i])
                builder.joint_q[-num_joints_in_arm + i] = np.clip(value, config.joint_limits[i][0], config.joint_limits[i][1])

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True # Changed to True for typical arm setup
        self.model.joint_q.requires_grad = config.joint_q_requires_grad
        self.model.body_q.requires_grad = config.body_q_requires_grad
        self.model.joint_attach_ke = config.joint_attach_ke
        self.model.joint_attach_kd = config.joint_attach_kd
        self.integrator = wp.sim.SemiImplicitIntegrator()
        if not config.headless:
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.expanduser(config.usd_output_path))
        else:
            self.renderer = None
        # simulation state
        self.ee_pose = wp.zeros(self.num_envs, dtype=wp.transform, requires_grad=True, device=self.device)
        self.initial_target_pos = wp.array(initial_target_pos_np, dtype=wp.vec3, device=self.device)
        self.initial_target_quat = wp.array(initial_target_quat_np, dtype=wp.quat, device=self.device)
        self.target_pos = wp.array(initial_target_pos_np, dtype=wp.vec3, requires_grad=False, device=self.device)
        self.target_quat = wp.array(initial_target_quat_np, dtype=wp.quat, requires_grad=False, device=self.device)
        self.state = self.model.state(requires_grad=True)
        self.ik_loss = wp.zeros(self.num_envs, dtype=wp.float32, requires_grad=False, device=self.device)
        self.profiler = {}
        self.tape = None
        self.pos_error_norm = 0.0
        self.rot_error_norm = 0.0

    def compute_ee_pose(self):
        """ Performs forward kinematics to compute the end-effector pose. """
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
        wp.launch(
            forward_kinematics,
            dim=self.num_envs,
            inputs=[self.state.body_q, self.num_links, self.ee_link_index, self.ee_link_tf_offset],
            outputs=[self.ee_pose],
            device=self.device,
        )
        return self.ee_pose

    def compute_ik_update(self):
        """ Computes the IK update using gradients of a loss function. """
        self.tape = wp.Tape()
        with self.tape:
            # Forward pass
            self.compute_ee_pose()
            wp.launch(
                compute_ik_loss,
                dim=self.num_envs,
                inputs=[
                    self.ee_pose,
                    self.target_pos,
                    self.target_quat,
                    wp.float32(self.config.kp_pos),
                    wp.float32(self.config.kp_rot),
                ],
                outputs=[self.ik_loss],
                device=self.device
            )

        # Create adjoint gradient with explicit wp.float32 type
        loss_adjoint = wp.ones(shape=self.ik_loss.shape, dtype=wp.float32, device=self.device)

        # Backward pass
        self.tape.backward(grads={self.ik_loss: loss_adjoint})

        # Retrieve gradients
        if self.model.joint_q in self.tape.gradients:
            delta_q = self.tape.gradients[self.model.joint_q]
            # Ensure delta_q is not None (can happen if computations don't depend on joint_q)
            if delta_q is None:
                log.warning("Gradients for model.joint_q are None. Returning zero delta_q.")
                delta_q_shape = self.model.joint_q.shape
                delta_q_dtype = self.model.joint_q.dtype # Match original joint q type
                delta_q = wp.zeros(shape=delta_q_shape, dtype=delta_q_dtype, device=self.device)

        else:
            log.warning("No gradients entry found for model.joint_q. Returning zero delta_q.")
            delta_q_shape = self.model.joint_q.shape
            delta_q_dtype = self.model.joint_q.dtype # Match original joint q type
            delta_q = wp.zeros(shape=delta_q_shape, dtype=delta_q_dtype, device=self.device)

        self.tape.zero()
        return delta_q

    def step(self):
        with wp.ScopedTimer("ik_update", print=False, active=True, dict=self.profiler):
            delta_q = self.compute_ik_update()

        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() - self.config.learning_rate * delta_q.numpy().flatten(), # Gradient descent step
            dtype=wp.float32,
            requires_grad=True,
            device=self.device,
        )

    def render_gizmos(self):
        if self.renderer is None:
            return

        radius = self.config.gizmo_radius
        half_height = self.config.gizmo_length / 2.0

        # Base rotations to align cone (default Y up) with axes
        rot_x_axis = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 2.0) # Rotate cone from +Y to +X
        rot_y_axis = wp.quat_identity()                                               # Cone already points +Y
        rot_z_axis = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)  # Rotate cone from +Y to +Z

        # Get poses as numpy for iteration (ensure they are computed if needed)
        ee_poses_np = self.ee_pose.numpy()
        target_pos_np = self.target_pos.numpy()
        target_quat_np = self.target_quat.numpy()

        # Calculate error norms for logging (outside render loop ideally, but ok here for now)
        ee_pos_np = ee_poses_np['p'] # Access position component
        ee_quat_np = ee_poses_np['q'] # Access rotation component
        pos_errors = target_pos_np - ee_pos_np
        self.pos_error_norm = np.linalg.norm(pos_errors, axis=1).mean()
        # Note: Computing rotation error norm accurately here requires care with quaternion math in numpy or converting back to wp.quat

        for i in range(self.num_envs):
            # Extract individual pose components
            target_pos_tuple = tuple(target_pos_np[i]) # render_cone needs tuple position
            target_rot_wp = wp.quat(target_quat_np[i]) # Use wp.quat for multiplication
            ee_pos_tuple = tuple(ee_poses_np[i]['p']) # render_cone needs tuple position
            ee_rot_wp = wp.quat(ee_poses_np[i]['q'])      # Use wp.quat for multiplication

            # --- Target Gizmo ---
            # Apply the target's rotation to the base axis rotations
            final_rot_tx = wp.mul(target_rot_wp, rot_x_axis)
            final_rot_ty = wp.mul(target_rot_wp, rot_y_axis)
            final_rot_tz = wp.mul(target_rot_wp, rot_z_axis)
            self.renderer.render_cone(f"target_x_{i}", target_pos_tuple, final_rot_tx, radius, half_height, color=self.config.gizmo_color_x_target)
            self.renderer.render_cone(f"target_y_{i}", target_pos_tuple, final_rot_ty, radius, half_height, color=self.config.gizmo_color_y_target)
            self.renderer.render_cone(f"target_z_{i}", target_pos_tuple, final_rot_tz, radius, half_height, color=self.config.gizmo_color_z_target)

            # --- EE Gizmo ---
            final_rot_ex = wp.mul(ee_rot_wp, rot_x_axis)
            final_rot_ey = wp.mul(ee_rot_wp, rot_y_axis)
            final_rot_ez = wp.mul(ee_rot_wp, rot_z_axis)
            self.renderer.render_cone(f"ee_pos_x_{i}", ee_pos_tuple, final_rot_ex, radius, half_height, color=self.config.gizmo_color_x_ee)
            self.renderer.render_cone(f"ee_pos_y_{i}", ee_pos_tuple, final_rot_ey, radius, half_height, color=self.config.gizmo_color_y_ee)
            self.renderer.render_cone(f"ee_pos_z_{i}", ee_pos_tuple, final_rot_ez, radius, half_height, color=self.config.gizmo_color_z_ee)

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.render_time)
        self.renderer.render(self.state)
        self.render_gizmos()
        self.renderer.end_frame()
        self.render_time += self.frame_dt

def run_sim(config: SimConfig):
    wp.init()
    log.info(f"gpu enabled: {wp.get_device().is_cuda}")
    log.info("starting simulation")
    with wp.ScopedDevice(config.device):
        sim = Sim(config)

        for i in range(config.num_rollouts):
            # select new random target points for all envs
            current_target_pos_np = sim.initial_target_pos.numpy().copy()
            current_target_quat_np = sim.initial_target_quat.numpy().copy()

            # Add random translation
            translation_noise = sim.rng.uniform(
                -config.target_spawn_box_size/2,
                config.target_spawn_box_size/2,
                size=(sim.num_envs, 3),
            )
            current_target_pos_np += translation_noise
            
            # Add random rotation (small angle)
            angle_noise = sim.rng.uniform(-np.pi/8, np.pi/8, size=sim.num_envs)
            axis_noise = sim.rng.normal(size=(sim.num_envs, 3))
            axis_noise /= np.linalg.norm(axis_noise, axis=1, keepdims=True) # Normalize axes

            for e in range(sim.num_envs):
                 # Convert axis-angle noise to quaternion and apply
                 noise_quat = wp.quat_from_axis_angle(wp.vec3(axis_noise[e]), float(angle_noise[e]))
                 current_quat = wp.quat(current_target_quat_np[e])
                 # Multiply quaternions: new_rot = noise_rot * current_rot
                 new_quat = wp.mul(noise_quat, current_quat)
                 current_target_quat_np[e] = np.array([new_quat[0], new_quat[1], new_quat[2], new_quat[3]])

            sim.target_pos.assign(current_target_pos_np)
            sim.target_quat.assign(current_target_quat_np)

            for j in range(config.train_iters):
                 sim.step()
                 sim.render() # render updates error norms
                 log.info(f"Rollout {i}, Iter: {j}, Pos Error: {sim.pos_error_norm:.4f}, Rot Error: {sim.rot_error_norm:.4f}")
        if not config.headless and sim.renderer is not None:
            sim.renderer.save()
        if "ik_update" in sim.profiler and len(sim.profiler["ik_update"]) > 0:
            avg_time = np.array(sim.profiler["ik_update"]).mean()
            avg_steps_second = 1000.0 * float(sim.num_envs) / avg_time if avg_time > 0 else 0
            log.info(f"step time: {avg_time:.3f} ms, {avg_steps_second:.2f} steps/s")
    log.info(f"simulation complete!")
    log.info(f"performed {config.num_rollouts * config.train_iters} steps")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode, suppressing the opening of any graphical windows.")
    parser.add_argument("--num_rollouts", type=int, default=2, help="Number of rollouts to perform.")
    parser.add_argument("--train_iters", type=int, default=32, help="Number of training iterations per rollout.")
    args = parser.parse_known_args()[0]
    config = SimConfig(
        device=args.device,
        seed=args.seed,
        headless=args.headless,
        num_envs=args.num_envs,
        num_rollouts=args.num_rollouts,
        train_iters=args.train_iters,
    )
    run_sim(config)