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
    device: str = None                   # Device to run the simulation.
    seed: int = 42                        # Random seed.
    headless: bool = False                # Turns off rendering.
    num_envs: int = 16                    # Number of parallel environments.
    num_rollouts: int = 2                 # Number of rollouts to perform.
    train_iters: int = 32                 # Training iterations per rollout.
    start_time: float = 0.0               # Start time.
    fps: int = 60                         # Frames per second.
    step_size: float = 1.0                # Step size in joint space.
    urdf_path: str = "~/dev/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
    usd_output_path: str = "~/dev/cu/warp/ik_output_6d_gpt.usd"
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gizmo_radius: float = 0.005
    gizmo_length: float = 0.05
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0)
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0)
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5)
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5)
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0)
    arm_spacing_xz: float = 1.0
    arm_height: float = 0.0
    target_z_offset: float = 0.3          # In local coordinates, forward offset (here we use negative X for "forward").
    target_y_offset: float = 0.1          # Vertical offset.
    target_spawn_box_size: float = 0.1
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: [
        (-3.054, 3.054),
        (0.0, 3.14),
        (0.0, 2.356),
        (-1.57, 1.57),
        (-1.57, 1.57),
        (-3.14, 3.14),
        (0.0, 0.044),
        (0.0, 0.044),
    ])
    arm_rot_offset: list[tuple[tuple[float, float, float], float]] = field(default_factory=lambda: [
        ((1.0, 0.0, 0.0), -math.pi * 0.5),
        ((0.0, 0.0, 1.0), -math.pi * 0.5),
    ])
    qpos_home: list[float] = field(default_factory=lambda: [0, np.pi/12, np.pi/12, 0, 0, 0, 0, 0])
    q_angle_shuffle: list[float] = field(default_factory=lambda: [np.pi/2, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0.01, 0.01])
    joint_q_requires_grad: bool = True
    body_q_requires_grad: bool = True
    joint_attach_ke: float = 1600.0
    joint_attach_kd: float = 20.0


# -------------------------------------------------------------------
# Helper: convert a quaternion (as numpy array [x,y,z,w]) to a 3x3 rotation matrix.
def quat_to_rot_matrix_np(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2),     2 * (x * y - z * w),         2 * (x * z + y * w)],
        [2 * (x * y + z * w),           1 - 2 * (x ** 2 + z ** 2),   2 * (y * z - x * w)],
        [2 * (x * z - y * w),           2 * (y * z + x * w),         1 - 2 * (x ** 2 + y ** 2)]
    ], dtype=np.float32)


# Helper: apply a transformation to a point.
def apply_transform_np(translation, quat, point):
    R = quat_to_rot_matrix_np(quat)
    return translation + R.dot(point)


# -------------------------------------------------------------------
# Device helper functions for quaternion operations.
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

# -------------------------------------------------------------------
# Device function to compute an orientation error from two quaternions.
@wp.func
def quat_orientation_error(target: wp.quat, current: wp.quat) -> wp.vec3:
    q_err = quat_mul(target, quat_conjugate(current))
    qx = wp.select(q_err[3] < 0.0, -q_err[0], q_err[0])
    qy = wp.select(q_err[3] < 0.0, -q_err[1], q_err[1])
    qz = wp.select(q_err[3] < 0.0, -q_err[2], q_err[2])
    return wp.vec3(qx, qy, qz) * 2.0

# -------------------------------------------------------------------
# Host-side helper: extract quaternions from transforms.
def get_body_quaternions(state_body_q, num_links, ee_link_index):
    num_envs = state_body_q.shape[0] // num_links
    quats = np.empty((num_envs, 4), dtype=np.float32)
    for e in range(num_envs):
        t = state_body_q[e * num_links + ee_link_index]
        quats[e, :] = t[3:7]
    return quats

# -------------------------------------------------------------------
# Device kernel to compute a 6D error (position and orientation).
@wp.kernel
def compute_ee_error_kernel(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    target_pos: wp.array(dtype=wp.vec3),
    target_ori: wp.array(dtype=wp.quat),
    current_ori: wp.array(dtype=wp.quat),  # Precomputed current orientation
    error_out: wp.array(dtype=wp.float32)  # Flattened array (num_envs*6)
):
    tid = wp.tid()
    t = body_q[tid * num_links + ee_link_index]
    pos = wp.transform_point(t, ee_link_offset)
    ori = current_ori[tid]
    pos_err = target_pos[tid] - pos
    ori_err = quat_orientation_error(target_ori[tid], ori)
    base = tid * 6
    error_out[base + 0] = pos_err.x
    error_out[base + 1] = pos_err.y
    error_out[base + 2] = pos_err.z
    error_out[base + 3] = ori_err.x
    error_out[base + 4] = ori_err.y
    error_out[base + 5] = ori_err.z

# -------------------------------------------------------------------
# Original forward kinematics kernel (position only).
@wp.kernel
def forward_kinematics(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    ee_pos: wp.array(dtype=wp.vec3)
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

        # Parse URDF and build model.
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.expanduser(config.urdf_path),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False
        )
        builder = wp.sim.ModelBuilder()
        self.step_size = config.step_size
        self.num_links = len(articulation_builder.joint_type)
        self.dof = len(articulation_builder.joint_q)
        self.joint_limits = config.joint_limits

        log.info(f"Parsed URDF with {self.num_links} links and {self.dof} dof")

        # Locate ee_gripper link.
        self.ee_link_offset = wp.vec3(config.ee_link_offset)
        self.ee_link_index = -1
        for i, joint in enumerate(articulation_builder.joint_name):
            if joint == "ee_gripper":
                self.ee_link_index = articulation_builder.joint_child[i]
                break
        if self.ee_link_index == -1:
            raise ValueError("Could not find ee_gripper joint in URDF")

        # Compute initial arm orientation.
        _initial_arm_orientation = None
        for axis, angle in config.arm_rot_offset:
            if _initial_arm_orientation is None:
                _initial_arm_orientation = wp.quat_from_axis_angle(wp.vec3(axis), angle)
            else:
                _initial_arm_orientation *= wp.quat_from_axis_angle(wp.vec3(axis), angle)
        self.initial_arm_orientation = _initial_arm_orientation

        # --- Revised target origin computation ---
        # Instead of using a raw grid, we compute each arm's target relative to its base transform.
        self.target_origin = []
        self.arm_spacing_xz = config.arm_spacing_xz
        self.arm_height = config.arm_height
        self.num_rows = int(math.sqrt(self.num_envs))
        log.info(f"Spawning {self.num_envs} arms in a grid of {self.num_rows}x{self.num_rows}")
        for e in range(self.num_envs):
            x = (e % self.num_rows) * self.arm_spacing_xz
            z = (e // self.num_rows) * self.arm_spacing_xz
            base_transform = wp.transform(wp.vec3(x, self.arm_height, z), self.initial_arm_orientation)
            # Convert base transform to numpy (for target computation).
            base_translation = np.array([x, self.arm_height, z], dtype=np.float32)
            base_quat = np.array([self.initial_arm_orientation.x, self.initial_arm_orientation.y,
                                   self.initial_arm_orientation.z, self.initial_arm_orientation.w], dtype=np.float32)
            # Define the desired target offset in the arm's local frame.
            # Here we assume the arm is meant to reach "forward" along negative X.
            target_offset_local = np.array([-config.target_z_offset, config.target_y_offset, 0.0], dtype=np.float32)
            target_world = apply_transform_np(base_translation, base_quat, target_offset_local)
            self.target_origin.append(target_world)
            builder.add_builder(articulation_builder, xform=wp.transform(wp.vec3(x, self.arm_height, z), self.initial_arm_orientation))
            num_joints_in_arm = len(config.qpos_home)
            for i in range(num_joints_in_arm):
                value = config.qpos_home[i] + self.rng.uniform(-config.q_angle_shuffle[i], config.q_angle_shuffle[i])
                builder.joint_q[-num_joints_in_arm + i] = np.clip(value, config.joint_limits[i][0], config.joint_limits[i][1])
        self.target_origin = np.array(self.target_origin)
        # Target orientation: default identity.
        self.target_ori = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (self.num_envs, 1))

        # Finalize model.
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

        # Simulation state.
        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True)
        self.ee_error = wp.zeros(self.num_envs * 6, dtype=wp.float32, requires_grad=True)
        self.state = self.model.state(requires_grad=True)
        self.targets = self.target_origin.copy()
        self.profiler = {}

    def compute_ee_error(self):
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
        host_quats = get_body_quaternions(self.state.body_q.numpy(), self.num_links, self.ee_link_index)
        device_quats = wp.array(host_quats, dtype=wp.quat)
        wp.launch(
            compute_ee_error_kernel,
            dim=self.num_envs,
            inputs=[
                self.state.body_q,
                self.num_links,
                self.ee_link_index,
                self.ee_link_offset,
                wp.array(self.targets, dtype=wp.vec3),
                wp.array(self.target_ori, dtype=wp.quat),
                device_quats,
            ],
            outputs=[self.ee_error],
        )
        return self.ee_error

    def compute_geometric_jacobian(self):
        jacobians = np.empty((self.num_envs, 6, self.dof), dtype=np.float32)
        tape = wp.Tape()
        with tape:
            self.compute_ee_error()
        for o in range(6):
            select_index = np.zeros(6, dtype=np.float32)
            select_index[o] = 1.0
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.float32)
            tape.backward(grads={self.ee_error: e})
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, o, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
            tape.zero()
        return jacobians

    def compute_fd_jacobian(self, eps=1e-4):
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)
        q0 = self.model.joint_q.numpy().copy()
        for e in range(self.num_envs):
            for i in range(self.dof):
                q = q0.copy()
                q[e * self.dof + i] += eps
                self.model.joint_q.assign(q)
                self.compute_ee_error()
                f_plus = self.ee_error.numpy()[e * 6:(e + 1) * 6].copy()
                q[e * self.dof + i] -= 2 * eps
                self.model.joint_q.assign(q)
                self.compute_ee_error()
                f_minus = self.ee_error.numpy()[e * 6:(e + 1) * 6].copy()
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)
        self.model.joint_q.assign(q0)
        return jacobians

    def step(self):
        with wp.ScopedTimer("jacobian", print=False, active=True, dict=self.profiler):
            jacobians = self.compute_geometric_jacobian()
        ee_error_flat = self.compute_ee_error().numpy()
        error = ee_error_flat.reshape(self.num_envs, 6, 1)
        delta_q = np.matmul(jacobians.transpose(0, 2, 1), error)
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

        # Create base quaternions for each axis
        rot_x = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 2.0)
        rot_y = wp.quat_identity()
        rot_z = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)

        # Convert to numpy quaternions (x,y,z,w format)
        rot_x_np = np.array([rot_x.x, rot_x.y, rot_x.z, rot_x.w], dtype=np.float32)
        rot_y_np = np.array([rot_y.x, rot_y.y, rot_y.z, rot_y.w], dtype=np.float32)
        rot_z_np = np.array([rot_z.x, rot_z.y, rot_z.z, rot_z.w], dtype=np.float32)

        # Render target gizmos with orientation.
        for e in range(self.num_envs):
            target_pos_tuple = tuple(self.targets[e])
            target_ori_np = self.target_ori[e]
            # Convert rotations to tuples for render_cone
            target_rot_x = tuple(target_ori_np * rot_x_np)
            target_rot_y = tuple(target_ori_np * rot_y_np)
            target_rot_z = tuple(target_ori_np * rot_z_np)
            self.renderer.render_cone(f"target_x_{e}", target_pos_tuple, target_rot_x, radius, half_height, color=self.config.gizmo_color_x_target)
            self.renderer.render_cone(f"target_y_{e}", target_pos_tuple, target_rot_y, radius, half_height, color=self.config.gizmo_color_y_target)
            self.renderer.render_cone(f"target_z_{e}", target_pos_tuple, target_rot_z, radius, half_height, color=self.config.gizmo_color_z_target)

            # Render end-effector gizmos.
            ee_pos_np = self.compute_ee_error().numpy().reshape(self.num_envs, 6)[:, 0:3]
            ee_pos_tuple = tuple(ee_pos_np[e])
            ee_rot_x = tuple( * rot_x_np)
            ee_rot_y = tuple( * rot_y_np)
            ee_rot_z = tuple( * rot_z_np)
            self.renderer.render_cone(f"ee_pos_x_{e}", ee_pos_tuple, ee_rot_x, radius, half_height, color=self.config.gizmo_color_x_ee)
            self.renderer.render_cone(f"ee_pos_y_{e}", ee_pos_tuple, ee_rot_y, radius, half_height, color=self.config.gizmo_color_y_ee)
            self.renderer.render_cone(f"ee_pos_z_{e}", ee_pos_tuple, ee_rot_z, radius, half_height, color=self.config.gizmo_color_z_ee)

    def render(self):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.render_time)
        self.renderer.render(self.state)
        # self.render_gizmos()
        self.renderer.end_frame()
        self.render_time += self.frame_dt


def run_sim(config: SimConfig):
    wp.init()
    log.info(f"gpu enabled: {wp.get_device().is_cuda}")
    log.info("starting simulation")
    with wp.ScopedDevice(config.device):
        sim = Sim(config)
        log.debug("autodiff geometric jacobian:")
        log.debug(sim.compute_geometric_jacobian())
        log.debug("finite diff geometric jacobian:")
        log.debug(sim.compute_fd_jacobian())
        for i in range(config.num_rollouts):
            sim.targets = sim.target_origin.copy()
            sim.targets[:, :] += sim.rng.uniform(
                -config.target_spawn_box_size / 2,
                config.target_spawn_box_size / 2,
                size=(sim.num_envs, 3)
            )
            for j in range(config.train_iters):
                sim.step()
                sim.render()
                log.debug(f"rollout {i}, iter: {j}, error: {sim.compute_ee_error().numpy().mean()}")
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
    parser.add_argument("--device", type=str, default=None, help="Override default Warp device.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode.")
    parser.add_argument("--num_rollouts", type=int, default=2, help="Number of rollouts to perform.")
    parser.add_argument("--train_iters", type=int, default=32, help="Training iterations per rollout.")
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

def quat_from_axis_angle_np(axis, angle):
    """Convert axis-angle to quaternion using numpy.
    
    Args:
        axis: numpy array [x, y, z] representing the rotation axis
        angle: rotation angle in radians
    Returns:
        quaternion as numpy array [x, y, z, w]
    """
    axis = np.array(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)  # normalize the axis
    s = np.sin(angle / 2.0)
    c = np.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)