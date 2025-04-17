# tattoo_targets.py
from dataclasses import dataclass, field
import math
import os
from enum import Enum
import logging
from typing import Optional # Added

import numpy as np
from pxr import Usd, UsdGeom
from PIL import Image # Added for image loading

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

# --- Warp Math Functions (copied from your IK script for orientation) ---
@wp.func
def quat_mul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    return wp.quat(
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
        q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
    )

@wp.func
def quat_from_matrix(m: wp.mat33) -> wp.quat:
    """Convert matrix to quaternion"""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = wp.sqrt(tr + 1.0) * 2.0  # s=4*qw
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
        qw = 0.25 * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = wp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0  # s=4*qx
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
        qw = (m[2, 1] - m[1, 2]) / s
    elif m[1, 1] > m[2, 2]:
        s = wp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0  # s=4*qy
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
        qw = (m[0, 2] - m[2, 0]) / s
    else:
        s = wp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0  # s=4*qz
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
        qw = (m[1, 0] - m[0, 1]) / s
    return wp.quat(qx, qy, qz, qw)
# --- End Warp Math Functions ---

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
    num_frames: int = 60 # Increased frames for settling
    fps: int = 60  # Frames per second
    sim_substeps: int = 32  # Number of simulation substeps per frame
    integrator_type: IntegratorType = IntegratorType.XPBD # XPBD often good for cloth contact
    cloth_width: int = 128 # Increased resolution for potentially better mapping
    cloth_height: int = 128
    cloth_cell_size: float = 0.005 # Smaller cell size if resolution increased
    cloth_particle_radius: float = 0.005 # Match cell size
    cloth_mass: float = 0.01  # Mass per cloth particle (adjust as needed)
    cloth_pos: tuple[float, float, float] = (0.0, 2.8, 0.0)  # Initial position of cloth
    cloth_rot_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # Axis for cloth rotation
    cloth_rot_angle: float = math.pi * 0.5  # Angle for cloth rotation (radians)
    cloth_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Initial velocity of cloth
    fix_left: bool = False  # Fix the left edge of the cloth
    usd_output_path: str = "scenes/cloth_tattoo_output.usd"  # Path to save USD file
    mesh_target_usd_path: str = "assets/arm.usda"  # Path to mesh_target USD file
    mesh_target_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Position of mesh_target mesh
    mesh_target_rot_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)  # Axis for mesh_target rotation
    mesh_target_rot_angle: float = math.pi / 4  # Angle for mesh_target rotation (radians)
    mesh_target_scale: tuple[float, float, float] = (12.0, 12.0, 12.0)  # Scale of mesh_target mesh
    tattoo_image_path: str = "assets/zorya-224x224.png" # Path to your tattoo PNG
    tattoo_ik_poses_path: str = "outputs/tattoo_ik_poses.npy" # Path to save IK poses
    tattoo_gizmo_scale: float = 0.005 # Scale for the visualization gizmos
    # Integrator-specific parameters (Defaults adjusted slightly)
    euler_tri_ke: float = 1.0e3
    euler_tri_ka: float = 1.0e3
    euler_tri_kd: float = 1.0e1
    xpbd_edge_ke: float = 5.0e2 # Slightly increased stiffness
    xpbd_spring_ke: float = 1.0e3
    xpbd_spring_kd: float = 1.0 # Add a bit of spring damping
    vbd_tri_ke: float = 1.0e4
    vbd_tri_ka: float = 1.0e4
    vbd_tri_kd: float = 1.0e-5
    vbd_edge_ke: float = 100.0
    # Contact parameters
    soft_contact_ke: float = 1.0e4
    soft_contact_kd: float = 1.0e2
    mesh_ke: float = 5.0e2 # Increased contact stiffness
    mesh_kd: float = 1.0e2
    mesh_kf: float = 1.0e3

# --- New Kernels ---
@wp.kernel
def calculate_vertex_orientations_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    target_vertex_indices: wp.array(dtype=int),
    cloth_width_vertices: int, # width + 1
    cloth_height_vertices: int,# height + 1
    # Outputs
    target_orientations: wp.array(dtype=wp.quat), # Output quaternions for each target vertex
):
    tid = wp.tid() # Thread ID corresponds to index in target_vertex_indices
    vertex_idx = target_vertex_indices[tid]

    # Get current vertex position
    p = particle_q[vertex_idx]

    # Calculate grid row and column for boundary checks
    row = vertex_idx // cloth_width_vertices
    col = vertex_idx % cloth_width_vertices

    # --- Find valid neighbors ---
    idx_right = vertex_idx + 1
    idx_left = vertex_idx - 1
    idx_up = vertex_idx + cloth_width_vertices
    idx_down = vertex_idx - cloth_width_vertices

    # Get neighbor positions, handling boundaries (use current pos if neighbor is out of bounds)
    p_right = wp.where(col < cloth_width_vertices - 1, particle_q[idx_right], p) # Use wp.where
    p_left = wp.where(col > 0, particle_q[idx_left], p)                          # Use wp.where
    p_up = wp.where(row < cloth_height_vertices - 1, particle_q[idx_up], p)     # Use wp.where
    p_down = wp.where(row > 0, particle_q[idx_down], p)                        # Use wp.where

    # --- Calculate tangent vectors ---
    tangent_u = p_right - p_left # Vector roughly along cloth width
    tangent_v = p_up - p_down   # Vector roughly along cloth height

    if col == 0: tangent_u = p_right - p
    elif col == cloth_width_vertices - 1: tangent_u = p - p_left

    if row == 0: tangent_v = p_up - p
    elif row == cloth_height_vertices - 1: tangent_v = p - p_down

    min_len_sq = 1.0e-9
    if wp.length_sq(tangent_u) < min_len_sq: tangent_u = wp.vec3(1.0, 0.0, 0.0)
    if wp.length_sq(tangent_v) < min_len_sq: tangent_v = wp.vec3(0.0, 1.0, 0.0)

    # --- Calculate normal (Z axis) ---
    normal = wp.cross(tangent_u, tangent_v)
    if wp.length_sq(normal) < min_len_sq:
        up_vec = wp.vec3(0.0, 0.0, 1.0)
        if wp.abs(wp.dot(wp.normalize(tangent_u), up_vec)) > 0.99:
            up_vec = wp.vec3(0.0, 1.0, 0.0)
        normal = wp.cross(tangent_u, up_vec)
        if wp.length_sq(normal) < min_len_sq:
             normal = wp.vec3(0.0, 0.0, 1.0)

    normal = wp.normalize(normal)

    # --- Calculate orthonormal basis (X, Y axes) ---
    axis_x = wp.normalize(tangent_u)
    axis_y = wp.normalize(wp.cross(normal, axis_x))
    axis_x = wp.cross(axis_y, normal) # Ensure orthogonal

    # --- Construct rotation matrix and convert to quaternion ---
    rot_mat = wp.matrix_from_cols(axis_x, axis_y, normal) # Use matrix_from_cols
    orientation_quat = quat_from_matrix(rot_mat)

    target_orientations[tid] = orientation_quat

@wp.kernel
def calculate_tattoo_gizmo_transforms_kernel(
    target_positions: wp.array(dtype=wp.vec3),
    target_orientations: wp.array(dtype=wp.quat),
    num_targets: int,
    gizmo_scale: float,
    # Precomputed axis rotations relative to local frame (cone points along Y)
    rot_x_axis_q: wp.quat,
    rot_y_axis_q: wp.quat,
    rot_z_axis_q: wp.quat,
    # Outputs (flat array, order: TgtX,TgtY,TgtZ per target)
    out_gizmo_pos: wp.array(dtype=wp.vec3),
    out_gizmo_rot: wp.array(dtype=wp.quat)
):
    tid = wp.tid() # Target index

    target_pos = target_positions[tid]
    target_ori = target_orientations[tid]

    # Cone points along +Y axis in its local frame
    cone_height = gizmo_scale
    cone_half_height = cone_height / 2.0
    offset_vec = wp.vec3(0.0, cone_half_height, 0.0) # Offset from center to base

    # Calculate world rotation for each axis gizmo
    world_rot_x = quat_mul(target_ori, rot_x_axis_q)
    world_rot_y = quat_mul(target_ori, rot_y_axis_q)
    world_rot_z = quat_mul(target_ori, rot_z_axis_q)

    # Calculate world offset for each axis gizmo base
    offset_x = wp.quat_rotate(world_rot_x, offset_vec)
    offset_y = wp.quat_rotate(world_rot_y, offset_vec)
    offset_z = wp.quat_rotate(world_rot_z, offset_vec)

    # Store transforms (Position is target_pos - world_offset)
    base_idx = tid * 3 # 3 gizmos per target (X, Y, Z)
    out_gizmo_pos[base_idx + 0] = target_pos - offset_x
    out_gizmo_rot[base_idx + 0] = world_rot_x
    out_gizmo_pos[base_idx + 1] = target_pos - offset_y
    out_gizmo_rot[base_idx + 1] = world_rot_y
    out_gizmo_pos[base_idx + 2] = target_pos - offset_z
    out_gizmo_rot[base_idx + 2] = world_rot_z
# --- End New Kernels ---

class Sim:
    def __init__(self, config: SimConfig):
        log.debug(f"config: {config}")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.frame_dt = 1.0 / config.fps
        self.sim_dt = self.frame_dt / config.sim_substeps
        self.sim_time = 0.0
        self.profiler = {}

        # --- Tattoo Target Setup ---
        self.target_vertex_indices_wp: Optional[wp.array] = None
        self.target_orientations_wp: Optional[wp.array] = None
        self.target_positions_wp: Optional[wp.array] = None # Final positions of targets
        self.gizmo_pos_wp: Optional[wp.array] = None
        self.gizmo_rot_wp: Optional[wp.array] = None
        self.num_tattoo_targets = 0
        self._load_and_map_tattoo()
        # --- End Tattoo Target Setup ---

        # Build the simulation model
        with wp.ScopedTimer("model_init", print=False, active=True, dict=self.profiler):
            builder = wp.sim.ModelBuilder()
            self._build_cloth(builder)
            self._build_collider(builder)

            # Finalize model
            self.model = builder.finalize(device=config.device) # Specify device
            self.model.ground = False
            self.model.soft_contact_ke = config.soft_contact_ke
            self.model.soft_contact_kd = config.soft_contact_kd

            # Set up integrator
            self._setup_integrator()

            # Simulation states
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()

        # Renderer setup
        self.renderer = None
        if not config.headless:
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                os.path.expanduser(config.usd_output_path),
                scaling=40.0,
                # Add stage up axis if needed, e.g., Y-up
                # stage_up_axis="y",
            )
            # Precompute gizmo rotations (cone points along +Y)
            self.rot_x_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0) # Rotate cone from +Y to +X
            self.rot_y_axis_q_wp = wp.quat_identity() # Cone already points along +Y
            self.rot_z_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0) # Rotate cone from +Y to +Z
        else:
            self.renderer = None

        # CUDA graph setup
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedTimer("cuda_graph_init", print=False, active=True, dict=self.profiler):
                # Capture state needs to be initialized before capture
                self.state_0.clear_forces()
                wp.sim.collide(self.model, self.state_0) # Initial collision needed for graph
                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt) # One step

                # Now capture the simulation step
                with wp.ScopedCapture() as capture:
                    # We capture the core simulate() logic
                    # Note: Collision might need to be outside the graph if scene changes,
                    # but for static collider it's fine inside.
                    wp.sim.collide(self.model, self.state_0)
                    for _ in range(self.config.sim_substeps):
                          self.state_0.clear_forces()
                          self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                          # Swap states within the capture if integrator needs it (like SemiImplicit)
                          # If integrator updates state_1 *from* state_0, swap *after* loop
                          self.state_0, self.state_1 = self.state_1, self.state_0 # Swap state_0 and state_1
                self.graph = capture.graph
                log.info("CUDA graph captured.")


    def _load_and_map_tattoo(self):
        """Loads the tattoo image and maps black pixels to cloth vertices."""
        image_path = os.path.expanduser(self.config.tattoo_image_path)
        if not os.path.exists(image_path):
            log.warning(f"Tattoo image not found at {image_path}. Skipping target generation.")
            return

        try:
            img = Image.open(image_path).convert('L') # Convert to grayscale
            img_np = np.array(img)
            img_height, img_width = img_np.shape

            target_pixels = np.argwhere(img_np == 0) # Find black pixels (value 0)
            self.num_tattoo_targets = len(target_pixels)
            if self.num_tattoo_targets == 0:
                log.warning("No black pixels found in the tattoo image.")
                return

            log.info(f"Found {self.num_tattoo_targets} target pixels in {image_path}")

            target_vertex_indices = []
            cloth_verts_x = self.config.cloth_width + 1
            cloth_verts_y = self.config.cloth_height + 1

            for py, px in target_pixels:
                # Map pixel coord [0, img_dim-1] to vertex coord [0, cloth_dim]
                # Direct mapping from image coordinates to cloth grid
                norm_x = px / (img_width - 1)
                norm_y = py / (img_height - 1) # Direct Y mapping

                # Find closest vertex index
                vx = round(norm_x * self.config.cloth_width)
                vy = round(norm_y * self.config.cloth_height)

                # Clamp to valid vertex indices
                vx = max(0, min(self.config.cloth_width, vx))
                vy = max(0, min(self.config.cloth_height, vy))

                vertex_idx = vy * cloth_verts_x + vx
                target_vertex_indices.append(vertex_idx)

            # Remove duplicate vertex indices if multiple pixels map to the same vertex
            unique_indices = sorted(list(set(target_vertex_indices)))
            self.num_tattoo_targets = len(unique_indices) # Update count
            log.info(f"Mapped to {self.num_tattoo_targets} unique cloth vertices.")

            # Create Warp array for target indices
            self.target_vertex_indices_wp = wp.array(unique_indices, dtype=int, device=self.config.device)

            # Allocate arrays for final poses and gizmos (filled after simulation)
            self.target_positions_wp = wp.zeros(self.num_tattoo_targets, dtype=wp.vec3, device=self.config.device)
            self.target_orientations_wp = wp.zeros(self.num_tattoo_targets, dtype=wp.quat, device=self.config.device)
            self.gizmo_pos_wp = wp.zeros(self.num_tattoo_targets * 3, dtype=wp.vec3, device=self.config.device)
            self.gizmo_rot_wp = wp.zeros(self.num_tattoo_targets * 3, dtype=wp.quat, device=self.config.device)


        except Exception as e:
            log.error(f"Error loading or mapping tattoo image: {e}", exc_info=True)
            self.num_tattoo_targets = 0
            self.target_vertex_indices_wp = None


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
            "particle_radius": self.config.cloth_particle_radius, # <-- ADDED BACK HERE
        }

        # Integrator specific params...
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
        # builder.particle_radius = self.config.cloth_particle_radius # <-- REMOVE THIS LINE

        log.info(f"Built cloth grid: {self.config.cloth_width}x{self.config.cloth_height}")
        log.info(f"Total vertices: {(self.config.cloth_width+1)*(self.config.cloth_height+1)}")

    def _build_collider(self, builder: wp.sim.ModelBuilder):
        """Builds the static target mesh collider."""
        mesh_usd_path = os.path.expanduser(self.config.mesh_target_usd_path)
        if not os.path.exists(mesh_usd_path):
            log.error(f"Collider mesh not found at: {mesh_usd_path}")
            raise FileNotFoundError(f"Collider mesh not found: {mesh_usd_path}")

        # REMOVED the incorrect attempt using path=...

        # Use manual loading as the primary method
        log.info(f"Loading collider mesh from: {mesh_usd_path} (manual USD method)")
        try:
            usd_stage = Usd.Stage.Open(mesh_usd_path)
            # Try a common prim path structure
            prim_path = "/object/geom/mesh" # Adjust if your prim path is different
            usd_geom = UsdGeom.Mesh.Get(usd_stage, prim_path)
            if not usd_geom:
                prim_path = "/Root/mesh_0" # Try another common path
                usd_geom = UsdGeom.Mesh.Get(usd_stage, prim_path)
            if not usd_geom:
                # Add more potential paths if necessary or search the stage
                log.error(f"Could not find UsdGeom.Mesh at common paths in {mesh_usd_path}")
                # Example search:
                # for prim in usd_stage.Traverse():
                #     if prim.IsA(UsdGeom.Mesh):
                #         usd_geom = UsdGeom.Mesh(prim)
                #         prim_path = prim.GetPath()
                #         log.info(f"Found mesh prim via traversal: {prim_path}")
                #         break
                # if not usd_geom: # Check again after traversal
                raise ValueError(f"Could not find mesh prim in {mesh_usd_path}")

            log.info(f"Reading mesh data from prim: {prim_path}")
            mesh_points = np.array(usd_geom.GetPointsAttr().Get())
            mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

            # Check if face vertex counts exist and are triangles (optional but good practice)
            counts_attr = usd_geom.GetFaceVertexCountsAttr()
            if counts_attr.HasValue():
                mesh_counts = np.array(counts_attr.Get())
                if not np.all(mesh_counts == 3):
                    log.warning(f"Mesh {prim_path} faces are not all triangles. Warp collision/rendering might expect triangles.")
                    # You might need to triangulate the mesh here if it's not already triangles.
            else:
                log.warning(f"No face vertex counts found for {prim_path}. Assuming triangles.")


            mesh = wp.sim.Mesh(mesh_points, mesh_indices)

            builder.add_shape_mesh(
                body=-1, # Static body
                mesh=mesh, # Pass the created mesh object
                pos=wp.vec3(*self.config.mesh_target_pos),
                rot=wp.quat_from_axis_angle(wp.vec3(*self.config.mesh_target_rot_axis), self.config.mesh_target_rot_angle),
                scale=wp.vec3(*self.config.mesh_target_scale),
                ke=self.config.mesh_ke,
                kd=self.config.mesh_kd,
                kf=self.config.mesh_kf,
            )
            log.info("Added target mesh collider via manual USD loading.")
        except Exception as e_fallback:
            log.error(f"Manual USD mesh loading failed: {e_fallback}", exc_info=True)
            raise


    def _setup_integrator(self):
        """Sets up the appropriate integrator based on type."""
        if self.config.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif self.config.integrator_type == IntegratorType.XPBD:
             # iterations > 1 often needed for stiffness with XPBD
            self.integrator = wp.sim.XPBDIntegrator(iterations=5)
        else:  # VBD
            self.integrator = wp.sim.VBDIntegrator(self.model, iterations=5)
        log.info(f"Using integrator: {self.config.integrator_type}")

    def simulate_step_kinematic(self):
         """Performs one frame of simulation - Kinematic update version."""
         # This version is simpler and might be sufficient if forces aren't critical
         # It focuses on constraint satisfaction (like XPBD/VBD)
         with wp.ScopedTimer("collision_detection", print=False, active=True, dict=self.profiler):
             wp.sim.collide(self.model, self.state_0) # Detect collisions based on current state

         with wp.ScopedTimer("integration", print=False, active=True, dict=self.profiler):
             # For XPBD/VBD, the simulate call handles substeps internally based on iterations
             # For Euler, you'd need the explicit loop
             if self.config.integrator_type == IntegratorType.EULER:
                 for _ in range(self.config.sim_substeps):
                     self.state_0.clear_forces() # Euler needs force clearing
                     # Apply gravity or other forces here if needed for Euler
                     self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                     self.state_0, self.state_1 = self.state_1, self.state_0
             else: # XPBD / VBD
                 # These integrators often don't need explicit force clearing per substep (handled internally)
                 # They might apply gravity internally too. Check Warp docs for specifics.
                 # The 'iterations' parameter in their constructor acts like substeps for constraint projection.
                 self.integrator.simulate(self.model, self.state_0, self.state_1, self.frame_dt) # Use frame_dt
                 self.state_0, self.state_1 = self.state_1, self.state_0 # Swap state

    def step(self):
        """Advances the simulation by one frame."""
        with wp.ScopedTimer("step", print=False, active=True, dict=self.profiler):
            if self.use_cuda_graph and self.graph:
                wp.capture_launch(self.graph)
                # After graph launch, state_0 holds the *new* state because of the swap *inside* the graph
            else:
                 self.simulate_step_kinematic() # Call the simulation logic
        self.sim_time += self.frame_dt


    def calculate_and_save_final_poses(self):
        """Calculates orientations for target vertices and saves 6D poses."""
        if self.target_vertex_indices_wp is None or self.num_tattoo_targets == 0:
            log.info("No target vertices defined, skipping pose calculation.")
            return

        log.info("Calculating final orientations for tattoo target vertices...")
        # Ensure state_0 has the final particle positions
        final_particle_q = self.state_0.particle_q

        # --- Calculate Orientations ---
        with wp.ScopedTimer("calc_orientations", print=False, active=True, dict=self.profiler):
            wp.launch(
                kernel=calculate_vertex_orientations_kernel,
                dim=self.num_tattoo_targets,
                inputs=[
                    final_particle_q,
                    self.target_vertex_indices_wp,
                    self.config.cloth_width + 1,
                    self.config.cloth_height + 1,
                ],
                outputs=[self.target_orientations_wp],
                device=self.config.device,
            )

        # --- Extract Positions ---
        # Gather the positions corresponding to the target indices using indexing
        with wp.ScopedTimer("gather_positions", print=False, active=True, dict=self.profiler):
             # Select elements using indexing and assign to the pre-allocated array
             selected_positions = final_particle_q[self.target_vertex_indices_wp]
             self.target_positions_wp.assign(selected_positions) # <-- CORRECTED LINE


        # --- Combine and Save ---
        # Copy data to CPU
        final_pos_np = self.target_positions_wp.numpy()
        final_ori_np = self.target_orientations_wp.numpy()

        # Combine into a single N x 7 array [px, py, pz, qx, qy, qz, qw]
        # Ensure shapes match before hstack
        if final_pos_np.shape[0] != final_ori_np.shape[0]:
            log.error(f"Position ({final_pos_np.shape}) and orientation ({final_ori_np.shape}) array lengths mismatch!")
            return # Avoid crashing hstack

        log.debug(f"Final pos shape: {final_pos_np.shape}, Final ori shape: {final_ori_np.shape}")
        final_poses_6d = np.hstack((final_pos_np, final_ori_np))

        output_path = os.path.expanduser(self.config.tattoo_ik_poses_path)
        try:
            np.save(output_path, final_poses_6d)
            log.info(f"Saved {self.num_tattoo_targets} 6D IK poses to: {output_path}")
        except Exception as e:
            log.error(f"Failed to save IK poses to {output_path}: {e}", exc_info=True)


    def render_tattoo_gizmos(self):
         """Renders XYZ gizmos at the tattoo target locations."""
         if (self.renderer is None or
             self.target_positions_wp is None or
             self.target_orientations_wp is None or
             self.num_tattoo_targets == 0):
             return

         # Calculate gizmo transforms based on final target poses
         with wp.ScopedTimer("calc_gizmos", print=False, active=True, dict=self.profiler):
              wp.launch(
                  kernel=calculate_tattoo_gizmo_transforms_kernel,
                  dim=self.num_tattoo_targets,
                  inputs=[
                      self.target_positions_wp, # Use the gathered target positions
                      self.target_orientations_wp, # Use the calculated target orientations
                      self.num_tattoo_targets,
                      self.config.tattoo_gizmo_scale,
                      self.rot_x_axis_q_wp,
                      self.rot_y_axis_q_wp,
                      self.rot_z_axis_q_wp,
                  ],
                  outputs=[self.gizmo_pos_wp, self.gizmo_rot_wp],
                  device=self.config.device
              )

         # Copy gizmo transforms to CPU for rendering
         # Note: Rendering individual primitives can be slow for many targets.
         # Consider batched rendering or instancing if performance is critical.
         gizmo_pos_np = self.gizmo_pos_wp.numpy()
         gizmo_rot_np = self.gizmo_rot_wp.numpy()

         radius = self.config.tattoo_gizmo_scale * 0.1 # Make radius smaller than length
         half_height = self.config.tattoo_gizmo_scale / 2.0
         # Define colors (similar to IK script)
         color_x = (1.0, 0.2, 0.2)
         color_y = (0.2, 1.0, 0.2)
         color_z = (0.2, 0.2, 1.0)

         for i in range(self.num_tattoo_targets):
              base_idx = i * 3
              # Render X (Red)
              self.renderer.render_cone(name=f"tgt_x_{i}", pos=tuple(gizmo_pos_np[base_idx + 0]), rot=tuple(gizmo_rot_np[base_idx + 0]), radius=radius, half_height=half_height, color=color_x)
              # Render Y (Green)
              self.renderer.render_cone(name=f"tgt_y_{i}", pos=tuple(gizmo_pos_np[base_idx + 1]), rot=tuple(gizmo_rot_np[base_idx + 1]), radius=radius, half_height=half_height, color=color_y)
              # Render Z (Blue) - Normal direction
              self.renderer.render_cone(name=f"tgt_z_{i}", pos=tuple(gizmo_pos_np[base_idx + 2]), rot=tuple(gizmo_rot_np[base_idx + 2]), radius=radius, half_height=half_height, color=color_z)


    def render_final_state_with_gizmos(self):
         """Renders only the final frame including tattoo gizmos."""
         if self.renderer is None:
             return

         log.info("Rendering final state with gizmos...")
         with wp.ScopedTimer("render_final", print=False, active=True, dict=self.profiler):
             # Ensure the final poses are calculated
             if self.target_positions_wp is None:
                  log.warning("Final poses not calculated, cannot render gizmos.")
                  return

             self.renderer.begin_frame(self.sim_time) # Use the final sim time
             self.renderer.render(self.state_0) # Render the final simulation state
             self.render_tattoo_gizmos() # Render the gizmos on top
             self.renderer.end_frame()
             # Save the renderer's output (which now includes this final frame)
             self.renderer.save()
             log.info(f"Final state rendering saved to {self.config.usd_output_path}")


def run_sim(config: SimConfig):
    # Ensure output directory exists if paths are relative
    output_dir = os.path.dirname(os.path.expanduser(config.usd_output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_dir_npy = os.path.dirname(os.path.expanduser(config.tattoo_ik_poses_path))
    if output_dir_npy and not os.path.exists(output_dir_npy):
        os.makedirs(output_dir_npy, exist_ok=True)

    wp.init()
    log.info(f"Warp using device: {wp.get_device().name}")
    log.info("Starting cloth simulation for tattoo IK pose generation")

    # Select device scope
    sim_device = config.device if config.device else wp.get_preferred_device()
    log.info(f"Running simulation on device: {sim_device}")

    with wp.ScopedDevice(sim_device):
        total_profiler = {}
        sim = None # Initialize sim variable
        try:
            with wp.ScopedTimer("total_setup_time", print=False, active=True, dict=total_profiler):
                sim = Sim(config)

            if sim.num_tattoo_targets == 0 and not config.headless:
                 log.warning("No tattoo targets loaded, gizmos will not be generated.")

            log.info("--- Starting Simulation Loop ---")
            with wp.ScopedTimer("total_sim_loop_time", print=False, active=True, dict=total_profiler):
                 for i in range(config.num_frames):
                     sim.step()
                     # Render full sequence ONLY if headless is False AND we are NOT rendering just the final state
                     # if not config.headless: # Render every frame? Can be slow.
                     #      sim.render()
                     if i % config.fps == 0:  # Log progress every second
                         log.info(f"Sim Frame {i}/{config.num_frames} (Time: {sim.sim_time:.2f}s)")

            log.info("--- Simulation Loop Finished ---")

            # Calculate final poses AFTER simulation loop
            if sim: # Check if sim was successfully initialized
                 sim.calculate_and_save_final_poses()

                 # Render the final frame WITH gizmos if not headless
                 if not config.headless:
                     sim.render_final_state_with_gizmos()
                 elif sim.renderer: # Still need to save if renderer exists but headless
                     sim.renderer.save() # Saves whatever frames were rendered (maybe none)

        except Exception as e:
            log.error(f"Simulation failed: {e}", exc_info=True)
            # Optionally re-raise the exception if you want the script to exit with an error
            # raise e
        finally:
             # Clean up resources if needed
             # sim.renderer.clear() ? Check Warp docs for renderer cleanup
             pass


        # --- Profiling Output ---
        log.info("\n--- Performance Profile ---")
        profiling_data = {**total_profiler, **(sim.profiler if sim else {})}

        # One-time costs
        for key in ["total_setup_time", "model_init", "cuda_graph_init"]:
            if key in profiling_data and profiling_data[key]:
                time_ms = profiling_data[key][0] # Usually only one measurement
                log.info(f"  {key.replace('_',' ').title()}: {time_ms:.2f} ms")

        # Loop costs (Simulation)
        if "total_sim_loop_time" in profiling_data and profiling_data["total_sim_loop_time"]:
            total_sim_ms = profiling_data["total_sim_loop_time"][0]
            log.info(f"  Total Sim Loop Time: {total_sim_ms / 1000.0:.2f} s ({config.num_frames} frames)")
            avg_frame_ms = total_sim_ms / config.num_frames
            log.info(f"  Average Frame Time (Sim Loop): {avg_frame_ms:.2f} ms ({1000.0/avg_frame_ms:.2f} FPS)")

        # Per-step/operation averages (inside sim loop)
        if sim:
            log.info("  Average Time per Operation (within sim step):")
            for key in ["step", "collision_detection", "integration"]:
                if key in sim.profiler and sim.profiler[key]:
                    times = np.array(sim.profiler[key])
                    avg_time = times.mean()
                    std_time = times.std()
                    # steps_per_second = 1000.0 / avg_time
                    log.info(f"    {key.title()}: {avg_time:.3f} +/- {std_time:.3f} ms")

        # Post-simulation costs
        log.info("  Post-Simulation Operations:")
        for key in ["calc_orientations", "gather_positions", "calc_gizmos", "render_final"]:
            if key in profiling_data and profiling_data[key]:
                 time_ms = np.mean(profiling_data[key]) # May be called once or multiple times
                 log.info(f"    {key.replace('_',' ').title()}: {time_ms:.2f} ms")

        log.info(f"Performed {config.num_frames} simulation frames.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cloth Simulation for Tattoo IK Target Generation")
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device (e.g., 'cuda:0').")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode (no rendering).")
    parser.add_argument("--seed", type=int, default=SimConfig.seed, help="Random seed.")
    parser.add_argument("--num_frames", type=int, default=SimConfig.num_frames, help="Total number of simulation frames.")
    parser.add_argument("--integrator", type=IntegratorType, choices=list(IntegratorType), default=SimConfig.integrator_type, help="Type of integrator.")
    parser.add_argument("--width", type=int, default=SimConfig.cloth_width, help="Cloth resolution in width.")
    parser.add_argument("--height", type=int, default=SimConfig.cloth_height, help="Cloth resolution in height.")
    parser.add_argument("--image", type=str, default=SimConfig.tattoo_image_path, help="Path to the black and white tattoo PNG image.")
    parser.add_argument("--output_poses", type=str, default=SimConfig.tattoo_ik_poses_path, help="Path to save the generated 6D IK poses (.npy).")
    parser.add_argument("--output_usd", type=str, default=SimConfig.usd_output_path, help="Path to save the final state USD render.")
    parser.add_argument("--collider", type=str, default=SimConfig.mesh_target_usd_path, help="Path to the collider mesh USD file.")

    args = parser.parse_args() # Use parse_args()

    config = SimConfig(
        device=args.device,
        headless=args.headless,
        seed=args.seed,
        num_frames=args.num_frames,
        integrator_type=args.integrator,
        cloth_width=args.width,
        cloth_height=args.height,
        tattoo_image_path=args.image,
        tattoo_ik_poses_path=args.output_poses,
        usd_output_path=args.output_usd,
        mesh_target_usd_path=args.collider,
        # Add other args here if you want to expose more SimConfig options
    )

    # Ensure required libraries are installed
    try:
        import PIL
    except ImportError:
        print("Error: Pillow library not found. Please install it using:")
        print("pip install Pillow")
        exit(1)

    run_sim(config)