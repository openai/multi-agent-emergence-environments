{
    make_env: {
        "function": "mae_envs.envs.shelter_construction:make_env",
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
            horizon: 240,

            # Objects
            n_boxes: 8,
            n_elongated_boxes: 3,
            box_only_z_rot: true,
            boxid_obs: false,

            # Observations
            n_lidar_per_agent: 30,
            additional_obs: {
                hider: [[0]],
                prep_obs: [[0]],
                ramp_obs: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                mask_ar_obs: [[0]],
            },

            # Shelter
            shelter_reward_scale: 0.001,
            objective_diameter: [1.5, 2],
            objective_placement: 'uniform_away_from_walls',
        },
    },
}