from holosoma.config_types.terrain import MeshType, TerrainManagerCfg, TerrainTermCfg

terrain_locomotion_plane = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.PLANE,
        horizontal_scale=1.0,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        max_slope=0.3,
        platform_size=2.0,
        step_width_range=[0.30, 0.40],
        amplitude_range=[0.01, 0.05],
        slope_treshold=0.75,
    )
)

terrain_locomotion_mix = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        terrain_config={
            "flat": 0.2,
            "rough": 0.6,
            "low_obstacles": 0.2,
            "smooth_slope": 0.0,
            "rough_slope": 0.0,
        },
        max_slope=0.3,
        slope_treshold=0.75,
    )
)


terrain_locomotion_isaaclab_rough = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        # Match IsaacLab ROUGH_TERRAINS_CFG global settings.
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=20,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=5,
        num_cols=5,
        max_slope=0.4,
        platform_size=3.0,
        step_width_range=[0.30, 0.30],
        amplitude_range=[0.02, 0.10],
        slope_treshold=0.75,
        # Match IsaacLab rough.py terrain names and proportions.
        terrain_config={
            "pyramid_stairs": 0.1,
            "pyramid_stairs_inv": 0.1,
            "boxes": 0.2,
            "random_rough": 0.2,
            "hf_pyramid_slope": 0.1,
            "hf_pyramid_slope_inv": 0.1,
            "flat": 0.2,
        },
    )
)

terrain_locomotion_flat_stairs = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=20,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=5,
        num_cols=5,
        max_slope=0.4,
        platform_size=3.0,
        step_width_range=[0.30, 0.30],
        amplitude_range=[0.02, 0.10],
        slope_treshold=0.75,
        terrain_config={
            "flat": 0.4,
            "pyramid_stairs": 0.3,
            "pyramid_stairs_inv": 0.3,
        },
    )
)

terrain_load_obj = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        mesh_type=MeshType.LOAD_OBJ,
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        obj_file_path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_parkour.obj",
    )
)

DEFAULTS = {
    "terrain_locomotion_plane": terrain_locomotion_plane,
    "terrain_locomotion_mix": terrain_locomotion_mix,
    "terrain_locomotion_isaaclab_rough": terrain_locomotion_isaaclab_rough,
    "terrain_locomotion_flat_stairs": terrain_locomotion_flat_stairs,
    "terrain_load_obj": terrain_load_obj,
}
