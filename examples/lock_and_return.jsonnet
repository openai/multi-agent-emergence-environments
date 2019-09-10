{
    make_env: {
        "function": "mae_envs.envs.box_locking:make_env",
        args: {
            # Agents
            n_agents: 1,
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
            horizon: 120,
            scenario: "randomwalls",
            n_rooms: 6,
            random_room_number: false,

            # Objects
            box_only_z_rot: true,
            boxid_obs: false,
            boxsize_obs: true,
            pad_ramp_size: true,

            # Observations
            n_lidar_per_agent: 30,
            additional_obs: {
                hider: [[1]],
                prep_obs: [[0]],
                ramp_obs: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                ramp_obj_lock: [[0]],
                ramp_you_lock: [[[0]]],
                ramp_team_lock: [[[0]]],
                mask_ar_obs: [[0]],
            },

            # Lock Box Task
            n_boxes: 1,
            task_type: 'all-return',
            lock_reward: 5.0,
            unlock_penalty: 5.0,
            shaped_reward_scale: 0.5,
        },
    },
}