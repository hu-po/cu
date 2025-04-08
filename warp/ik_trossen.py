###########################################################################
# Example Jacobian
#
# Demonstrates how to compute the Jacobian of a multi-valued function.
# Here, we use the simulation of a cartpole to differentiate
# through the kinematics function. We instantiate multiple copies of the
# cartpole and compute the Jacobian of the state of each cartpole in parallel
# in order to perform inverse kinematics via Jacobian transpose.
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

@wp.kernel
def compute_endeffector_position(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    ee_pos: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ee_pos[tid] = wp.transform_point(body_q[tid * num_links + ee_link_index], ee_link_offset)


class Example:
    def __init__(self, stage_path="example_jacobian_ik.usd", num_envs=10, headless=False, from_home=False):
        rng = np.random.default_rng(42)

        self.num_envs = num_envs

        fps = 60
        self.frame_dt = 1.0 / fps

        self.render_time = 0.0

        # step size to use for the IK updates
        self.step_size = 1.0

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.expanduser("~/dev/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False,
        )

        builder = wp.sim.ModelBuilder()

        self.num_links = len(articulation_builder.joint_type)
        # Find the ee_gripper_link index by looking at joint connections
        self.ee_link_index = -1
        for i, joint in enumerate(articulation_builder.joint_name):
            if joint == "ee_gripper":  # The fixed joint connecting link_6 to ee_gripper_link
                self.ee_link_index = articulation_builder.joint_child[i]
                break
        if self.ee_link_index == -1:
            raise ValueError("Could not find ee_gripper joint in URDF")
            
        self.ee_link_offset = wp.vec3(0.0, 0.0, 0.0)  # Offset from link_6 to ee_gripper_link
        self.dof = len(articulation_builder.joint_q)

        # targets will be visualized as spheres
        self.target_origin = []
        self.target_radius = 0.03

        # generate arm origins in a grid with 1m spacing
        self.arm_spacing_xz = 1.0 # floor plane is x-z plane
        self.target_z_offset = 0.3 # 0.3m in front of the arm
        self.target_y_offset = 0.1 # 10cm above floor
        self.num_rows = int(math.sqrt(self.num_envs))
        for i in range(self.num_envs):
            x = (i % self.num_rows) * self.arm_spacing_xz
            z = (i // self.num_rows) * self.arm_spacing_xz
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(
                    wp.vec3(x, 0.0, z),
                    wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5) * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi * 0.5)
                ),
            )
            self.target_origin.append((x, self.target_y_offset, z + self.target_z_offset))
            self.joint_limits = {
                'arm': [
                    (-3.054, 3.054),    # base
                    (0.0, 3.14),        # shoulder
                    (0.0, 2.356),       # elbow
                    (-1.57, 1.57),      # wrist 1
                    (-1.57, 1.57),      # wrist 2
                    (-3.14, 3.14)       # wrist 3
                ],
                'gripper': [
                    (0.0, 0.044),       # right carriage
                    (0.0, 0.044)        # left carriage
                ]
            }

            if from_home:
                # Home position with small random variations
                self.home_position = [0, np.pi/12, np.pi/12, 0, 0, 0, 0]
                self.degree_offsets = [np.pi/4, np.pi/8, np.pi/8, np.pi/8, np.pi/8, np.pi/8, 0.01, 0.01]
                # arm
                for i in range(6):
                    min_limit, max_limit = self.joint_limits['arm'][i]
                    value = self.home_position[i] + rng.uniform(-self.degree_offsets[i], self.degree_offsets[i])
                    builder.joint_q[-8 + i] = np.clip(value, min_limit, max_limit)
                # gripper
                for i in range(2):
                    min_limit, max_limit = self.joint_limits['gripper'][i]
                    value = self.home_position[-2 + i] + rng.uniform(-self.degree_offsets[-2 + i], self.degree_offsets[-2 + i])
                    builder.joint_q[-2 + i] = np.clip(value, min_limit, max_limit)

            else:
                # Full random initialization within joint limits
                # arm
                for i in range(6):
                    min_limit, max_limit = self.joint_limits['arm'][i]
                    builder.joint_q[-8 + i] = rng.uniform(min_limit, max_limit)
                # gripper
                for i in range(2):
                    min_limit, max_limit = self.joint_limits['gripper'][i]
                    builder.joint_q[-2 + i] = rng.uniform(min_limit, max_limit)
            
        self.target_origin = np.array(self.target_origin)

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.model.joint_q.requires_grad = True
        self.model.body_q.requires_grad = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if not headless and stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True)

        self.state = self.model.state(requires_grad=True)

        self.targets = self.target_origin.copy()

        self.profiler = {}

    def compute_ee_position(self):
        # computes the end-effector position from the current joint angles
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
        wp.launch(
            compute_endeffector_position,
            dim=self.num_envs,
            inputs=[self.state.body_q, self.num_links, self.ee_link_index, self.ee_link_offset],
            outputs=[self.ee_pos],
        )
        return self.ee_pos

    def compute_jacobian(self):
        # our function has 3 outputs (EE position), so we need a 3xN jacobian per environment
        jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
        tape = wp.Tape()
        with tape:
            self.compute_ee_position()
        for output_index in range(3):
            # select which row of the Jacobian we want to compute
            select_index = np.zeros(3)
            select_index[output_index] = 1.0
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3)
            tape.backward(grads={self.ee_pos: e})
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, output_index, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
            tape.zero()
        return jacobians

    def compute_fd_jacobian(self, eps=1e-4):
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
        return jacobians

    def step(self):
        with wp.ScopedTimer("jacobian", print=False, active=True, dict=self.profiler):
            # compute jacobian
            jacobians = self.compute_jacobian()

        # compute error
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

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.render_time)
        self.renderer.render(self.state)
        self.renderer.render_points("targets", self.targets, radius=self.target_radius)
        self.renderer.render_points("ee_pos", self.ee_pos_np, radius=self.target_radius)
        self.renderer.end_frame()
        self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="ik_trossen.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--train_iters", type=int, default=36, help="Total number of training iterations.")
    parser.add_argument("--num_envs", type=int, default=64, help="Total number of simulated environments.")
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=3,
        help="Total number of rollouts. In each rollout, a new set of target points is resampled for all environments.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument(
        "--from_home",
        action="store_true",
        help="Initialize the arm to the home position instead of a random position.",
    )

    args = parser.parse_known_args()[0]

    rng = np.random.default_rng(42)
    wp.init()

    print(f"USING GPU: {wp.get_device().is_cuda}")
    with wp.ScopedDevice(args.device):
        example = Example(
            headless=args.headless,
            num_envs=args.num_envs,
            stage_path=args.stage_path,
            from_home=args.from_home,
        )

        print("autodiff:")
        print(example.compute_jacobian())
        print("finite diff:")
        print(example.compute_fd_jacobian())

        for _ in range(args.num_rollouts):
            # select new random target points for all envs
            example.targets = example.target_origin.copy()
            # targets move in a 10cm square around the target origin
            target_spawn_box_size = 0.1
            example.targets[:, :] += rng.uniform(-target_spawn_box_size/2, target_spawn_box_size/2, size=(example.num_envs, 3))

            for iter in range(args.train_iters):
                example.step()
                example.render()
                print("iter:", iter, "error:", example.error.mean())

        if example.renderer is not None:
            example.renderer.save()

        avg_time = np.array(example.profiler["jacobian"]).mean()
        avg_steps_second = 1000.0 * float(example.num_envs) / avg_time

        print(f"envs: {example.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")