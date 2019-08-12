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
            horizon: 240,
            scenario: "randomwalls",
            n_rooms: 4,
            random_room_number: true,
            prob_outside_walls: 0.5,
            prep_fraction: 0.4,
            rew_type: "joint_zero_sum",
            restrict_rect: [-6.0, -6.0, 12.0, 12.0],

            hiders_together_radius: 0.5,
            seekers_together_radius: 0.5,

            # Objects
            n_boxes: [3, 9],
            n_elongated_boxes: [3, 9],
            box_only_z_rot: true,
            boxid_obs: false,

            n_ramps: 2,

            # Food
            n_food: 0,
            max_food_health: 40,
            food_radius: 0.5,
            food_box_centered: true,
            food_together_radius: 0.25,
            food_respawn_time: 5,
            food_rew_type: "joint_mean",

            # Observations
            n_lidar_per_agent: 30,
            prep_obs: true,
        },
    },
}
