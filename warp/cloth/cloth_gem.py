# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Sim Cloth (Refactored)
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
# Refactored to match a specific coding style using dataclasses and logging.
#
###########################################################################

from dataclasses import dataclass, field
import math
import os
import logging
import time
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom # Import for loading USD mesh

import warp as wp
import warp.examples # Required for get_asset_directory
import warp.sim
import warp.sim.render

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
class SimConfig:
    # General Setup
    device: str = None # device to run the simulation on
    seed: int = 42 # random seed (not used heavily in this example, but good practice)
    headless: bool = False # turns off rendering
    usd_output_path: str = "~/dev/cu/warp/cloth_output_gem.usd" # path to the usd file to save the model
    num_frames: int = 300 # total number of frames to simulate
    start_time: float = 0.0 # start time for the simulation

    # Simulation Timing
    fps: int = 60 # frames per second
    sim_substeps: int = 32 # physics substeps per frame

    # Integrator Choice
    integrator: IntegratorType = IntegratorType.EULER # type of integrator to use

    # Cloth Grid Parameters
    cloth_width: int = 64 # number of vertices in cloth width
    cloth_height: int = 32 # number of vertices in cloth height
    cloth_pos: tuple[float, float, float] = (0.0, 4.0, 0.0) # initial position of the cloth center
    cloth_rot_axis: tuple[float, float, float] = (1.0, 0.0, 0.0) # axis for initial cloth rotation
    cloth_rot_angle_deg: float = 90.0 # angle for initial cloth rotation (degrees)
    cloth_vel: tuple[float, float, float] = (0.0, 0.0, 0.0) # initial velocity of the cloth
    cloth_cell_size: float = 0.1 # distance between vertices
    cloth_mass_per_particle: float = 0.1 # mass of each cloth particle
    cloth_fix_left_edge: bool = True # fix the vertices on the left edge

    # Cloth Material Properties (tuned per integrator in __init__)
    # Euler / VBD specific
    tri_ke: float = 1.0e3 # triangle stiffness (stretch/shear)
    tri_ka: float = 1.0e3 # triangle stiffness (area)
    tri_kd: float = 1.0e1 # triangle damping
    # XPBD specific
    edge_ke: float = 1.0e2 # edge stiffness
    add_springs: bool = True # Add diagonal springs for shear constraints
    spring_ke: float = 1.0e3 # spring stiffness
    spring_kd: float = 0.0   # spring damping (often 0 for XPBD)
    # VBD specific (uses tri_*, edge_ke, different values often needed)
    vbd_tri_ke: float = 1e4
    vbd_tri_ka: float = 1e4
    vbd_tri_kd: float = 1e-5
    vbd_edge_ke: float = 100

    # Integrator Parameters
    xpbd_iterations: int = 1
    vbd_iterations: int = 1

    # Collider Mesh Parameters (Bunny)
    collider_usd_path: str = "bunny.usd" # Relative path within warp asset directory
    collider_pos: tuple[float, float, float] = (1.0, 0.0, 1.0)
    collider_rot_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    collider_rot_angle_deg: float = 90.0
    collider_scale: tuple[float, float, float] = (2.0, 2.0, 2.0)
    collider_ke: float = 1.0e2 # contact stiffness
    collider_kd: float = 1.0e2 # contact damping
    collider_kf: float = 1.0e1 # contact friction

    # Model Global Properties
    model_ground: bool = True
    model_soft_contact_ke: float = 1.0e4
    model_soft_contact_kd: float = 1.0e2

    # Rendering
    renderer_scaling: float = 40.0 # Scene scaling for renderer camera

    # Performance
    use_cuda_graph: bool = True # Attempt to use CUDA graph capture for performance

class Sim:
    def __init__(self, config: SimConfig):
        log.debug(f"Initializing Sim with config: {config}")
        self.config = config
        self.rng = np.random.default_rng(config.seed) # Initialize RNG
        self.sim_time = config.start_time
        self.frame_dt = 1.0 / config.fps
        self.sim_dt = self.frame_dt / config.sim_substeps
        self.profiler = {}

        builder = wp.sim.ModelBuilder()

        cloth_pos_wp = wp.vec3(config.cloth_pos)
        cloth_rot_wp = wp.quat_from_axis_angle(wp.vec3(config.cloth_rot_axis), math.radians(config.cloth_rot_angle_deg))
        cloth_vel_wp = wp.vec3(config.cloth_vel)

        log.info(f"Building cloth grid with integrator: {config.integrator}")
        if config.integrator == IntegratorType.EULER:
            builder.add_cloth_grid(
                pos=cloth_pos_wp,
                rot=cloth_rot_wp,
                vel=cloth_vel_wp,
                dim_x=config.cloth_width,
                dim_y=config.cloth_height,
                cell_x=config.cloth_cell_size,
                cell_y=config.cloth_cell_size,
                mass=config.cloth_mass_per_particle,
                fix_left=config.cloth_fix_left_edge,
                tri_ke=config.tri_ke,
                tri_ka=config.tri_ka,
                tri_kd=config.tri_kd,
            )
        elif config.integrator == IntegratorType.XPBD:
            builder.add_cloth_grid(
                pos=cloth_pos_wp,
                rot=cloth_rot_wp,
                vel=cloth_vel_wp,
                dim_x=config.cloth_width,
                dim_y=config.cloth_height,
                cell_x=config.cloth_cell_size,
                cell_y=config.cloth_cell_size,
                mass=config.cloth_mass_per_particle,
                fix_left=config.cloth_fix_left_edge,
                edge_ke=config.edge_ke,
                add_springs=config.add_springs,
                spring_ke=config.spring_ke,
                spring_kd=config.spring_kd,
            )
        elif config.integrator == IntegratorType.VBD:
            builder.add_cloth_grid(
                pos=cloth_pos_wp,
                rot=cloth_rot_wp,
                vel=cloth_vel_wp,
                dim_x=config.cloth_width,
                dim_y=config.cloth_height,
                cell_x=config.cloth_cell_size,
                cell_y=config.cloth_cell_size,
                mass=config.cloth_mass_per_particle,
                fix_left=config.cloth_fix_left_edge,
                # Using VBD specific parameters from config
                tri_ke=config.vbd_tri_ke,
                tri_ka=config.vbd_tri_ka,
                tri_kd=config.vbd_tri_kd,
                edge_ke=config.vbd_edge_ke,
            )
        else:
             raise ValueError(f"Unsupported integrator type: {config.integrator}")

        # --- Add Collider Mesh (Bunny) ---
        collider_usd_full_path = os.path.join(warp.examples.get_asset_directory(), config.collider_usd_path)
        log.info(f"Loading collider mesh from: {collider_usd_full_path}")
        try:
            usd_stage = Usd.Stage.Open(collider_usd_full_path)
            # Simple assumption about prim path, adjust if needed for other USDs
            usd_prim = next((p for p in usd_stage.Traverse() if UsdGeom.Mesh(p)), None)
            if not usd_prim:
                 raise RuntimeError(f"Could not find a UsdGeom.Mesh prim in {collider_usd_full_path}")
            usd_geom = UsdGeom.Mesh(usd_prim)
            mesh_points = np.array(usd_geom.GetPointsAttr().Get())
            mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
            log.info(f"Collider mesh loaded: {len(mesh_points)} vertices, {len(mesh_indices)//3} faces")
        except Exception as e:
             log.error(f"Failed to load collider mesh: {e}")
             raise

        mesh = wp.sim.Mesh(mesh_points, mesh_indices)

        collider_pos_wp = wp.vec3(config.collider_pos)
        collider_rot_wp = wp.quat_from_axis_angle(wp.vec3(config.collider_rot_axis), math.radians(config.collider_rot_angle_deg))
        collider_scale_wp = wp.vec3(config.collider_scale)

        builder.add_shape_mesh(
            body=-1, # Static body
            mesh=mesh,
            pos=collider_pos_wp,
            rot=collider_rot_wp,
            scale=collider_scale_wp,
            ke=config.collider_ke,
            kd=config.collider_kd,
            kf=config.collider_kf,
        )

        # VBD needs coloring for parallel processing
        if config.integrator == IntegratorType.VBD:
            log.info("Applying graph coloring for VBD integrator.")
            builder.color() # Needed for VBD integrator

        # --- Finalize Model ---
        log.info("Finalizing simulation model.")
        self.model = builder.finalize()
        self.model.ground = config.model_ground
        self.model.soft_contact_ke = config.model_soft_contact_ke
        self.model.soft_contact_kd = config.model_soft_contact_kd
        log.info(f"Model finalized with {self.model.particle_q.shape[0]} particles.")

        # --- Initialize Integrator ---
        if config.integrator == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif config.integrator == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(iterations=config.xpbd_iterations)
        elif config.integrator == IntegratorType.VBD:
            self.integrator = wp.sim.VBDIntegrator(self.model, iterations=config.vbd_iterations)
        else:
            # This case should have been caught earlier, but belt-and-suspenders
            raise ValueError(f"Unsupported integrator type: {config.integrator}")
        log.info(f"Using integrator: {self.integrator.__class__.__name__}")

        # --- Initialize State ---
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        log.info("Simulation states initialized.")

        # --- Initialize Renderer ---
        if not config.headless:
            renderer_output_path = os.path.expanduser(config.usd_output_path)
            log.info(f"Initializing renderer, output to: {renderer_output_path}")
            self.renderer = wp.sim.render.SimRenderer(self.model, renderer_output_path, scaling=config.renderer_scaling)
        else:
            log.info("Headless mode enabled, renderer not initialized.")
            self.renderer = None

        # --- CUDA Graph Capture ---
        self.graph = None
        self.can_use_cuda_graph = config.use_cuda_graph and wp.get_device().is_cuda
        if self.can_use_cuda_graph:
            log.info("Attempting CUDA graph capture...")
            try:
                wp.capture_begin()
                self.simulate() # Run one simulation step to capture
                self.graph = wp.capture_end()
                log.info("CUDA graph captured successfully.")
            except Exception as e:
                log.warning(f"CUDA graph capture failed: {e}. Will run without graph.")
                self.can_use_cuda_graph = False # Disable if capture fails
        else:
             log.info("CUDA graph capture skipped (not requested or not on CUDA device).")

    def simulate(self):
        """Runs the core physics simulation for one frame (multiple substeps)."""
        # Initial collision check for the frame
        wp.sim.collide(self.model, self.state_0)

        for _ in range(self.config.sim_substeps):
            self.state_0.clear_forces()

            # Integrate physics
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # Swap states for next substep
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        """Advances the simulation by one frame dt."""
        with wp.ScopedTimer("step", active=True, dict=self.profiler):
            if self.can_use_cuda_graph and self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Renders the current simulation state if renderer is available."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True, dict=self.profiler):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

def run_sim(config: SimConfig):
    """Initializes and runs the cloth simulation based on the config."""
    wp.init()
    log.info(f"Warp initialized. Device: {wp.get_device().name}, CUDA enabled: {wp.get_device().is_cuda}")
    log.info("Starting cloth simulation")

    sim_start_time = time.perf_counter()

    with wp.ScopedDevice(config.device): # Use specified device or default
        try:
            sim = Sim(config)

            log.info(f"Starting simulation loop for {config.num_frames} frames...")
            for i in range(config.num_frames):
                sim.step()
                sim.render()
                if (i + 1) % 10 == 0: # Log progress periodically
                    log.info(f"Simulated frame {i+1}/{config.num_frames}")

            sim_end_time = time.perf_counter()
            log.info("Simulation loop finished.")

            if not config.headless and sim.renderer is not None:
                log.info(f"Saving USD output to: {os.path.expanduser(config.usd_output_path)}")
                sim.renderer.save()

            # Print profiling information
            if "step" in sim.profiler and len(sim.profiler["step"]) > 0:
                step_times = sim.profiler["step"]
                avg_step_time_ms = sum(step_times) / len(step_times)
                log.info(f"\n--- Profiling ---")
                log.info(f"Total simulation time: {sim_end_time - sim_start_time:.2f} s")
                log.info(f"Average frame simulation time (step): {avg_step_time_ms:.3f} ms")
                if "render" in sim.profiler and len(sim.profiler["render"]) > 0:
                    render_times = sim.profiler["render"]
                    avg_render_time_ms = sum(render_times) / len(render_times)
                    log.info(f"Average frame render time: {avg_render_time_ms:.3f} ms")
            else:
                log.warning("No step timing information recorded in profiler.")

        except Exception as e:
            log.exception(f"An error occurred during simulation setup or execution: {e}") # Log traceback
            # Optionally re-raise or handle cleanup

    log.info("Cloth simulation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Warp Cloth Simulation Example (Refactored)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments corresponding to SimConfig fields
    parser.add_argument("--seed", type=int, default=SimConfig.seed, help="Random seed.")
    parser.add_argument("--device", type=str, default=SimConfig.device, help="Compute device ID (e.g., 'cuda:0' or 'cpu'). Defaults to Warp's default.")
    parser.add_argument("--num_frames", type=int, default=SimConfig.num_frames, help="Number of frames to simulate.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode (no rendering window or USD output).")
    parser.add_argument("--usd_output_path", type=str, default=SimConfig.usd_output_path, help="Path to save the output USD file (if not headless).")
    parser.add_argument("--integrator", type=IntegratorType, choices=list(IntegratorType), default=SimConfig.integrator, help="Physics integrator type.")
    parser.add_argument("--width", type=int, default=SimConfig.cloth_width, help="Cloth resolution in width.")
    parser.add_argument("--height", type=int, default=SimConfig.cloth_height, help="Cloth resolution in height.")
    parser.add_argument("--substeps", type=int, default=SimConfig.sim_substeps, help="Number of physics substeps per frame.")
    # Add more arguments if you want finer control from CLI, e.g., cloth stiffness, collider position, etc.

    args = parser.parse_known_args()[0]

    # Create SimConfig instance from arguments
    config = SimConfig(
        seed=args.seed,
        device=args.device,
        num_frames=args.num_frames,
        headless=args.headless,
        usd_output_path=args.usd_output_path,
        integrator=args.integrator,
        cloth_width=args.width,
        cloth_height=args.height,
        sim_substeps=args.substeps,
        # Keep other SimConfig defaults unless overridden by more CLI args
    )

    run_sim(config)