from dataclasses import dataclass, field
import math
import os
import logging
import time

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class SimConfig:
    device: str = None # device to run the simulation on
    seed: int = 42 # random seed
    headless: bool = False # turns off rendering
    num_envs: int = 16 # number of parallel environments
    num_rollouts: int = 2 # number of rollouts to perform
    train_iters: int = 32 # number of training iterations per rollout
    start_time: float = 0.0 # start time for the simulation
    fps: int = 60 # frames per second
    step_size: float = 1.0 # step size in q space for updates
    urdf_path: str = "~/dev/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf" # path to the urdf file
    usd_output_path: str = "~/dev/cu/warp/ik_output_3d.usd" # path to the usd file to save the model
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0) # offset from the ee_gripper_link to the end effector
    gizmo_radius: float = 0.005 # radius of the gizmo (used for arrow base radius)
    gizmo_length: float = 0.05 # total length of the gizmo arrow
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0) # color of the x gizmo for the ee
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0) # color of the y gizmo for the ee
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0) # color of the z gizmo for the ee
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5) # color of the x gizmo for the target
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5) # color of the y gizmo for the target
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0) # color of the z gizmo for the target
    arm_spacing_xz: float = 1.0 # spacing between arms in the x-z plane
    arm_height: float = 0.0 # height of the arm off the floor
    target_z_offset: float = 0.3 # offset of the target in the z direction
    target_y_offset: float = 0.1 # offset of the target in the y direction
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

@wp.kernel
def forward_kinematics(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    ee_pos: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ee_pos[tid] = wp.transform_point(body_q[tid * num_links + ee_link_index], ee_link_offset)

class Sim:
    def __init__(self, config: SimConfig):
        log.debug(f"config: {config}")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.num_envs = config.num_envs
        self.render_time = config.start_time
        self.fps = config.fps
        self.frame_dt = 1.0 / self.fps
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.expanduser(config.urdf_path),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False,
        )
        builder = wp.sim.ModelBuilder()
        self.step_size = config.step_size
        self.num_links = len(articulation_builder.joint_type)
        self.dof = len(articulation_builder.joint_q)
        self.joint_limits = config.joint_limits # TODO: parse from URDF
        log.info(f"Parsed URDF with {self.num_links} links and {self.dof} dof")
        # Find the ee_gripper_link index by looking at joint connections
        self.ee_link_offset = wp.vec3(config.ee_link_offset)
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
        # targets are a 3D pose visualized with cone gizmos
        self.target_origin = []
        self.target_z_offset = config.target_z_offset
        self.target_y_offset = config.target_y_offset
        # parallel arms are spawned in a grid on the floor (x-z plane)
        self.arm_spacing_xz = config.arm_spacing_xz
        self.arm_height = config.arm_height
        self.num_rows = int(math.sqrt(self.num_envs))
        log.info(f"Spawning {self.num_envs} arms in a grid of {self.num_rows}x{self.num_rows}")
        for e in range(self.num_envs):
            x = (e % self.num_rows) * self.arm_spacing_xz
            z = (e // self.num_rows) * self.arm_spacing_xz
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(wp.vec3(x, self.arm_height, z), self.initial_arm_orientation),
            )
            self.target_origin.append((x, self.target_y_offset, z + self.target_z_offset))
            num_joints_in_arm = len(config.qpos_home)
            for i in range(num_joints_in_arm):
                value = config.qpos_home[i] + self.rng.uniform(-config.q_angle_shuffle[i], config.q_angle_shuffle[i])
                builder.joint_q[-num_joints_in_arm + i] = np.clip(value, config.joint_limits[i][0], config.joint_limits[i][1])
        self.target_origin = np.array(self.target_origin)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = False
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
        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True)
        self.state = self.model.state(requires_grad=True)
        self.targets = self.target_origin.copy()
        self.profiler = {}

    def compute_ee_position(self):
        """ Performs forward kinematics to compute the end-effector position. """
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
        wp.launch(
            forward_kinematics,
            dim=self.num_envs,
            inputs=[self.state.body_q, self.num_links, self.ee_link_index, self.ee_link_offset],
            outputs=[self.ee_pos],
        )
        return self.ee_pos

    def compute_jacobian(self):
        """ Computes the Jacobian of the end-effector position. """
        # our function has 3 outputs (EE position), so we need a 3xN jacobian per environment
        jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
        tape = wp.Tape()
        with tape:
            self.compute_ee_position()
        for o in range(3):
            # select which row of the Jacobian we want to compute
            select_index = np.zeros(3)
            select_index[o] = 1.0
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3)
            tape.backward(grads={self.ee_pos: e})
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, o, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
            tape.zero()
        return jacobians

    def compute_fd_jacobian(self, eps=1e-4):
        """ Computes the finite difference Jacobian of the end-effector position. """
        jacobians = np.zeros((self.num_envs, 3, self.dof), dtype=np.float32)
        q0 = self.model.joint_q.numpy().copy()
        for e in range(self.num_envs):
            for i in range(self.dof):
                q = q0.copy()
                q[e * self.dof + i] += eps
                self.model.joint_q.assign(q)
                self.compute_ee_position()
                f_plus = self.ee_pos.numpy()[e].copy()
                q[e * self.dof + i] -= 2 * eps
                self.model.joint_q.assign(q)
                self.compute_ee_position()
                f_minus = self.ee_pos.numpy()[e].copy()
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)
        self.model.joint_q.assign(q0)
        return jacobians

    def step(self):
        with wp.ScopedTimer("jacobian", print=False, active=True, dict=self.profiler):
            jacobians = self.compute_jacobian()
        self.ee_pos_np = self.compute_ee_position().numpy()
        error = self.targets - self.ee_pos_np
        self.error = error.reshape(self.num_envs, 3, 1)
        # compute Jacobian transpose update
        delta_q = np.matmul(jacobians.transpose(0, 2, 1), self.error)
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + self.step_size * delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=True,
        )

    def render_gizmos(self):
        if self.renderer is None:
            return

        radius = self.config.gizmo_radius
        half_height = self.config.gizmo_length / 2.0
        # Assuming default cone UP axis is Y (index 1 in render_cone)

        # Rotation to point cone along +X (e.g., rotate around Z by -90 deg)
        rot_x = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 2.0)
        # Rotation to point cone along +Y (no rotation needed if default axis is Y)
        rot_y = wp.quat_identity()
        # Rotation to point cone along +Z (e.g., rotate around X by +90 deg)
        rot_z = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)

        # Ensure positions are NumPy for iteration
        # self.targets should already be numpy from __init__/run_sim
        # self.ee_pos_np is already numpy from step()
        targets_np = self.targets
        ee_pos_np = self.ee_pos_np

        for i in range(self.num_envs):
            # Convert individual positions to tuples for render_cone
            target_pos_tuple = tuple(targets_np[i])
            ee_pos_tuple = tuple(ee_pos_np[i])

            # --- Target Gizmo (using unique names per environment) ---
            self.renderer.render_cone(f"target_x_{i}", target_pos_tuple, rot_x, radius, half_height, color=self.config.gizmo_color_x_target)
            self.renderer.render_cone(f"target_y_{i}", target_pos_tuple, rot_y, radius, half_height, color=self.config.gizmo_color_y_target)
            self.renderer.render_cone(f"target_z_{i}", target_pos_tuple, rot_z, radius, half_height, color=self.config.gizmo_color_z_target)

            # --- EE Gizmo (using unique names per environment) ---
            self.renderer.render_cone(f"ee_pos_x_{i}", ee_pos_tuple, rot_x, radius, half_height, color=self.config.gizmo_color_x_ee)
            self.renderer.render_cone(f"ee_pos_y_{i}", ee_pos_tuple, rot_y, radius, half_height, color=self.config.gizmo_color_y_ee)
            self.renderer.render_cone(f"ee_pos_z_{i}", ee_pos_tuple, rot_z, radius, half_height, color=self.config.gizmo_color_z_ee)

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
        log.debug("autodiff:")
        log.debug(sim.compute_jacobian())
        log.debug("finite diff:")
        log.debug(sim.compute_fd_jacobian())
        for i in range(config.num_rollouts):
            # select new random target points for all envs
            sim.targets = sim.target_origin.copy()
            sim.targets[:, :] += sim.rng.uniform(
                -config.target_spawn_box_size/2,
                config.target_spawn_box_size/2,
                size=(sim.num_envs, 3),
            )
            for j in range(config.train_iters):
                sim.step()
                sim.render()
                log.debug(f"rollout {i}, iter: {j}, error: {sim.error.mean()}")
        if not config.headless and sim.renderer is not None:
            sim.renderer.save()
        avg_time = np.array(sim.profiler["jacobian"]).mean()
        avg_steps_second = 1000.0 * float(sim.num_envs) / avg_time
    log.info(f"simulation complete!")
    log.info(f"performed {config.num_rollouts * config.train_iters} steps")
    log.info(f"step time: {avg_time:.3f} ms, {avg_steps_second:.2f} steps/s")

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