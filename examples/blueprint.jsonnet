{
    make_env: {
        "function": "mae_envs.envs.blueprint_construction:make_env",
        args: {
            # Agent Actions
            grab_box: true,
            grab_out_of_vision: false,
            grab_selective: false,
            grab_exclusive: false,

            lock_box: true,
            lock_type: "all_lock_team_specific",
            lock_out_of_vision: false,
            

            # Scenario
            n_substeps: 15,
            horizon: 150,

            # Objects
            n_boxes: 8,
            box_only_z_rot: true,
            boxid_obs: false,
            boxsize_obs: true,

            # Observations
            n_lidar_per_agent: 30,
            additional_obs: {
                hider: [[0]],
                prep_obs: [[0]],
                ramp_obs: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                mask_ar_obs: [[0]],
            },

            # Blueprint
            n_sites: [1, 4],
            site_placement: 'uniform_away_from_walls',
            reward_infos: [
                {
                    type: 'construction_dense',
                    alpha: -1.5,
                    use_corners: true,
                    reward_scale: 0.05,
                },
                {
                    type: 'construction_completed',
                    site_activation_radius: 0.1,
                    use_corners: true,
                    reward_scale: 3,
                },
            ],
        },
    },
}