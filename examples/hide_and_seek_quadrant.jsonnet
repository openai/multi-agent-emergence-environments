{
    make_env: {
        "function": "mae_envs.envs.hide_and_seek:make_env",
        args: {
            # Agents
            n_hiders: 2,
            n_seekers: 2,
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
            horizon: 80,
            scenario: 'quadrant',
            prep_fraction: 0.4,
            rew_type: "joint_zero_sum",
            restrict_rect: [0.1, 0.1, 5.9, 5.9],
            p_door_dropout: 0.5,
            quadrant_game_hider_uniform_placement: true,

            # Objects
            n_boxes: 2,
            box_only_z_rot: true,
            boxid_obs: false,

            n_ramps: 1,
            lock_ramp: false,
            penalize_objects_out: true,

            # Food
            n_food: 0,

            # Observations
            n_lidar_per_agent: 30,
            prep_obs: true,
        },
    },
}
