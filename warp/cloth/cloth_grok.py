from dataclasses import dataclass, field
import math
import os
from enum import Enum
import logging

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Integrator types
class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"

    def __str__(self):
        return self.value

@dataclass
class SimConfig:
    device: str = None  # Device to run the simulation on
    seed: int = 42  # Random seed
    headless: bool = False  # Turns off rendering
    num_frames: int = 300  # Total number of frames to simulate
    fps: int = 60  # Frames per second
    sim_substeps: int = 32  # Number of simulation substeps per frame
    integrator_type: IntegratorType = IntegratorType.EULER  # Type of integrator
    cloth_width: int = 64  # Cloth resolution in x
    cloth_height: int = 32  # Cloth resolution in y
    cloth_cell_size: float = 0.1  # Size of each cloth cell
    cloth_mass: float = 0.1  # Mass per cloth particle
    cloth_pos: tuple[float, float, float] = (0.0, 4.0, 0.0)  # Initial position of cloth
    cloth_rot_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # Axis for cloth rotation
    cloth_rot_angle: float = math.pi * 0.5  # Angle for cloth rotation (radians)
    cloth_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Initial velocity of cloth
    fix_left: bool = True  # Fix the left edge of the cloth
    usd_output_path: str = "~/dev/cu/warp/cloth_output_grok.usd"  # Path to save USD file
    bunny_usd_path: str = "bunny.usd"  # Path to bunny USD file
    bunny_pos: tuple[float, float, float] = (1.0, 0.0, 1.0)  # Position of bunny mesh
    bunny_rot_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)  # Axis for bunny rotation
    bunny_rot_angle: float = math.pi * 0.5  # Angle for bunny rotation (radians)
    bunny_scale: tuple[float, float, float] = (2.0, 2.0, 2.0)  # Scale of bunny mesh
    # Integrator-specific parameters
    euler_tri_ke: float = 1.0e3  # Triangle stiffness for Euler
    euler_tri_ka: float = 1.0e3  # Triangle area stiffness for Euler
    euler_tri_kd: float = 1.0e1  # Triangle damping for Euler
    xpbd_edge_ke: float = 1.0e2  # Edge stiffness for XPBD
    xpbd_spring_ke: float = 1.0e3  # Spring stiffness for XPBD
    xpbd_spring_kd: float = 0.0  # Spring damping for XPBD
    vbd_tri_ke: float = 1.0e4  # Triangle stiffness for VBD
    vbd_tri_ka: float = 1.0e4  # Triangle area stiffness for VBD
    vbd_tri_kd: float = 1.0e-5  # Triangle damping for VBD
    vbd_edge_ke: float = 100.0  # Edge stiffness for VBD
    # Contact parameters
    soft_contact_ke: float = 1.0e4  # Soft contact stiffness
    soft_contact_kd: float = 1.0e2  # Soft contact damping
    mesh_ke: float = 1.0e2  # Mesh contact stiffness
    mesh_kd: float = 1.0e2  # Mesh contact damping
    mesh_kf: float = 1.0e1  # Mesh friction coefficient

class Sim:
    def __init__(self, config: SimConfig):
        log.debug(f"config: {config}")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.frame_dt = 1.0 / config.fps
        self.sim_dt = self.frame_dt / config.sim_substeps
        self.sim_time = 0.0
        self.profiler = {}

        # Build the simulation model
        builder = wp.sim.ModelBuilder()
        self._build_cloth(builder)
        self._build_collider(builder)

        # Finalize model
        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = config.soft_contact_ke
        self.model.soft_contact_kd = config.soft_contact_kd

        # Set up integrator
        self._setup_integrator()

        # Simulation states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Renderer setup
        if not config.headless:
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.expanduser(config.usd_output_path), scaling=40.0)
        else:
            self.renderer = None

        # CUDA graph setup
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _build_cloth(self, builder: wp.sim.ModelBuilder):
        """Builds the cloth grid based on integrator type."""
        cloth_args = {
            "pos": wp.vec3(*self.config.cloth_pos),
            "rot": wp.quat_from_axis_angle(wp.vec3(*self.config.cloth_rot_axis), self.config.cloth_rot_angle),
            "vel": wp.vec3(*self.config.cloth_vel),
            "dim_x": self.config.cloth_width,
            "dim_y": self.config.cloth_height,
            "cell_x": self.config.cloth_cell_size,
            "cell_y": self.config.cloth_cell_size,
            "mass": self.config.cloth_mass,
            "fix_left": self.config.fix_left,
        }

        if self.config.integrator_type == IntegratorType.EULER:
            cloth_args.update({
                "tri_ke": self.config.euler_tri_ke,
                "tri_ka": self.config.euler_tri_ka,
                "tri_kd": self.config.euler_tri_kd,
            })
        elif self.config.integrator_type == IntegratorType.XPBD:
            cloth_args.update({
                "edge_ke": self.config.xpbd_edge_ke,
                "add_springs": True,
                "spring_ke": self.config.xpbd_spring_ke,
                "spring_kd": self.config.xpbd_spring_kd,
            })
        else:  # VBD
            cloth_args.update({
                "tri_ke": self.config.vbd_tri_ke,
                "tri_ka": self.config.vbd_tri_ka,
                "tri_kd": self.config.vbd_tri_kd,
                "edge_ke": self.config.vbd_edge_ke,
            })

        builder.add_cloth_grid(**cloth_args)
        if self.config.integrator_type == IntegratorType.VBD:
            builder.color()

        log.info(f"Built cloth grid: {self.config.cloth_width}x{self.config.cloth_height}")

    def _build_collider(self, builder: wp.sim.ModelBuilder):
        """Builds the static bunny mesh collider."""
        mesh_usd_path = os.path.join(warp.examples.get_asset_directory(), self.config.bunny_usd_path)
        usd_stage = Usd.Stage.Open(mesh_usd_path)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        mesh = wp.sim.Mesh(mesh_points, mesh_indices)

        builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            pos=wp.vec3(*self.config.bunny_pos),
            rot=wp.quat_from_axis_angle(wp.vec3(*self.config.bunny_rot_axis), self.config.bunny_rot_angle),
            scale=wp.vec3(*self.config.bunny_scale),
            ke=self.config.mesh_ke,
            kd=self.config.mesh_kd,
            kf=self.config.mesh_kf,
        )
        log.info("Added bunny mesh collider")

    def _setup_integrator(self):
        """Sets up the appropriate integrator based on type."""
        if self.config.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif self.config.integrator_type == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(iterations=1)
        else:  # VBD
            self.integrator = wp.sim.VBDIntegrator(self.model, iterations=1)
        log.info(f"Using integrator: {self.config.integrator_type}")

    def simulate(self):
        """Performs one frame of simulation."""
        wp.sim.collide(self.model, self.state_0)
        for _ in range(self.config.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advances the simulation by one frame."""
        with wp.ScopedTimer("step", print=False, active=True, dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Renders the current simulation state."""
        if self.renderer is None:
            return
        with wp.ScopedTimer("render", print=False, active=True, dict=self.profiler):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

def run_sim(config: SimConfig):
    wp.init()
    log.info(f"GPU enabled: {wp.get_device().is_cuda}")
    log.info("Starting cloth simulation")
    
    with wp.ScopedDevice(config.device):
        sim = Sim(config)
        for i in range(config.num_frames):
            sim.step()
            sim.render()
            if i % (config.fps) == 0:  # Log every second
                log.debug(f"Frame {i}/{config.num_frames}")
        
        if not config.headless and sim.renderer is not None:
            sim.renderer.save()
        
        avg_step_time = np.array(sim.profiler["step"]).mean()
        steps_per_second = 1000.0 / avg_step_time
        log.info(f"Simulation complete! Performed {config.num_frames} frames")
        log.info(f"Average step time: {avg_step_time:.2f} ms, {steps_per_second:.2f} steps/s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode.")
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--integrator", type=IntegratorType, choices=list(IntegratorType), default=IntegratorType.EULER, help="Type of integrator.")
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")
    
    args = parser.parse_known_args()[0]
    
    config = SimConfig(
        device=args.device,
        seed=args.seed,
        headless=args.headless,
        num_frames=args.num_frames,
        integrator_type=args.integrator,
        cloth_width=args.width,
        cloth_height=args.height,
    )
    
    run_sim(config)