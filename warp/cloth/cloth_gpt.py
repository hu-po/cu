#!/usr/bin/env python3
"""
Refactored Cloth Simulation Example

This script demonstrates a simulation of an FEM cloth model colliding
against a static rigid body mesh. The code structure follows a similar
style as the refactored IK example: using a dataclass for configuration,
a dedicated simulation class, logging, and a run_sim function.
"""

from dataclasses import dataclass, field
import math
import os
import logging
import numpy as np
from enum import Enum
from pxr import Usd, UsdGeom

import warp as wp
import warp.sim
import warp.sim.render
import warp.examples

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"

    def __str__(self):
        return self.value


@dataclass
class ClothSimConfig:
    # Device and output settings
    device: str = None
    stage_path: str = "cloth_sim_gpt.usd"

    # Integrator and simulation grid parameters
    integrator: IntegratorType = IntegratorType.EULER
    sim_width: int = 64    # cloth resolution in x
    sim_height: int = 32   # cloth resolution in y
    cell_x: float = 0.1
    cell_y: float = 0.1
    mass: float = 0.1
    fix_left: bool = True

    # Cloth material parameters (for Euler)
    tri_ke: float = 1.0e3
    tri_ka: float = 1.0e3
    tri_kd: float = 1.0e1

    # Parameters for XPBD / VBD
    edge_ke: float = 1.0e2
    spring_ke: float = 1.0e3
    spring_kd: float = 0.0

    # Simulation timing settings
    sim_substeps: int = 32
    fps: int = 60

    # Collision mesh (e.g., bunny) parameters
    collision_mesh_path: str = "bunny.usd"
    collision_mesh_scale: float = 2.0
    collision_mesh_pos: tuple = (1.0, 0.0, 1.0)
    collision_mesh_rot_deg: float = 90.0  # rotation about Y axis (in degrees)
    collision_ke: float = 1.0e2
    collision_kd: float = 1.0e2
    collision_kf: float = 1.0e1

    # Initial simulation time
    sim_time: float = 0.0


class ClothSim:
    def __init__(self, config: ClothSimConfig):
        log.info(f"Initializing cloth simulation with config: {config}")
        self.config = config
        self.rng = np.random.default_rng()
        self.sim_time = config.sim_time

        # Compute time step sizes
        self.frame_dt = 1.0 / config.fps
        self.sim_dt = self.frame_dt / config.sim_substeps

        # Build the simulation model
        builder = wp.sim.ModelBuilder()

        # Define the initial cloth pose, orientation and velocity.
        cloth_pos = wp.vec3(0.0, 4.0, 0.0)
        cloth_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
        cloth_vel = wp.vec3(0.0, 0.0, 0.0)

        if config.integrator == IntegratorType.EULER:
            builder.add_cloth_grid(
                pos=cloth_pos,
                rot=cloth_rot,
                vel=cloth_vel,
                dim_x=config.sim_width,
                dim_y=config.sim_height,
                cell_x=config.cell_x,
                cell_y=config.cell_y,
                mass=config.mass,
                fix_left=config.fix_left,
                tri_ke=config.tri_ke,
                tri_ka=config.tri_ka,
                tri_kd=config.tri_kd,
            )
        elif config.integrator == IntegratorType.XPBD:
            builder.add_cloth_grid(
                pos=cloth_pos,
                rot=cloth_rot,
                vel=cloth_vel,
                dim_x=config.sim_width,
                dim_y=config.sim_height,
                cell_x=config.cell_x,
                cell_y=config.cell_y,
                mass=config.mass,
                fix_left=config.fix_left,
                edge_ke=config.edge_ke,
                add_springs=True,
                spring_ke=config.spring_ke,
                spring_kd=config.spring_kd,
            )
        else:  # VBD
            # Adjust stiffness values for VBD as needed
            builder.add_cloth_grid(
                pos=cloth_pos,
                rot=cloth_rot,
                vel=cloth_vel,
                dim_x=config.sim_width,
                dim_y=config.sim_height,
                cell_x=config.cell_x,
                cell_y=config.cell_y,
                mass=config.mass,
                fix_left=config.fix_left,
                tri_ke=config.tri_ke * 10,  # example scaling factor
                tri_ka=config.tri_ka * 10,
                tri_kd=config.tri_kd * 1e-5,
                edge_ke=config.edge_ke,
            )

        # Load the collision mesh (e.g., bunny mesh)
        mesh_usd_path = os.path.join(warp.examples.get_asset_directory(), config.collision_mesh_path)
        usd_stage = Usd.Stage.Open(mesh_usd_path)
        prim = usd_stage.GetPrimAtPath("/root/bunny")
        usd_geom = UsdGeom.Mesh(prim)
        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        mesh = wp.sim.Mesh(mesh_points, mesh_indices)

        collision_rot_rad = math.radians(config.collision_mesh_rot_deg)
        collision_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), collision_rot_rad)
        collision_pos = wp.vec3(*config.collision_mesh_pos)

        builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            pos=collision_pos,
            rot=collision_rot,
            scale=wp.vec3(config.collision_mesh_scale,
                          config.collision_mesh_scale,
                          config.collision_mesh_scale),
            ke=config.collision_ke,
            kd=config.collision_kd,
            kf=config.collision_kf,
        )

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

        # Choose the appropriate integrator
        if config.integrator == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif config.integrator == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(iterations=1)
        else:  # VBD
            self.integrator = wp.sim.VBDIntegrator(self.model, iterations=1)

        # Initialize simulation state variables
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Create renderer if a stage_path is provided.
        if config.stage_path:
            # Using scaling similar to original cloth example (e.g., 40.0)
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.expanduser(config.stage_path), scaling=40.0)
        else:
            self.renderer = None

        self.profiler = {}

        # If CUDA is enabled, pre-capture the simulation graph.
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Advance the simulation for one frame across substeps."""
        wp.sim.collide(self.model, self.state_0)
        for _ in range(self.config.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            # Swap simulation states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Perform one simulation step and increment simulation time."""
        with wp.ScopedTimer("step", dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


def run_cloth_sim(config: ClothSimConfig, num_frames: int):
    wp.init()
    log.info(f"GPU enabled: {wp.get_device().is_cuda}")
    log.info("Starting cloth simulation")
    with wp.ScopedDevice(config.device):
        sim = ClothSim(config)
        for frame in range(num_frames):
            sim.step()
            sim.render()
            log.debug(f"Frame {frame}, sim time: {sim.sim_time:.3f}")

        # Log average simulation step time if available
        frame_times = np.array(sim.profiler.get("step", [0]))
        if frame_times.size > 0:
            avg_time = frame_times.mean()
            avg_steps_per_sec = 1000.0 * (1.0 / avg_time) if avg_time > 0 else float("inf")
            log.info(f"Average frame sim time: {avg_time:.2f} ms, {avg_steps_per_sec:.2f} steps/s")
        else:
            log.info("No timing data collected.")

        if sim.renderer is not None:
            sim.renderer.save()
    log.info("Cloth simulation complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--stage_path", type=lambda s: None if s == "None" else str(s),
                        default="cloth_sim.usd", help="Path to the output USD file.")
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--integrator", type=IntegratorType, choices=list(IntegratorType),
                        default=IntegratorType.EULER, help="Type of integrator to use.")
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")
    args = parser.parse_args()

    config = ClothSimConfig(
        device=args.device,
        stage_path=args.stage_path,
        integrator=args.integrator,
        sim_width=args.width,
        sim_height=args.height,
    )
    run_cloth_sim(config, num_frames=args.num_frames)
