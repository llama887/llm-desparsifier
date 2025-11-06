def dense_reward(env_params, ts_prev, action, ts_next, ctx):
    # Use helper function to compute potential based on positions.
    # If previous positions are not provided in the context, fall back to current ones.
    yellow_default = jnp.array([-100, -100], dtype=jnp.int32)
    green_default = jnp.array([-100, -100], dtype=jnp.int32)
    # For previous timestep, try to use context keys with suffix _prev if available; otherwise, use current values.
    y_pos_prev = ctx.get("yellow_square_pos_prev", ctx.get("yellow_square_pos", yellow_default))
    g_pos_prev = ctx.get("green_ball_pos_prev", ctx.get("green_ball_pos", green_default))
    # For current timestep, use available keys.
    y_pos = ctx.get("yellow_square_pos", yellow_default)
    g_pos = ctx.get("green_ball_pos", green_default)
    # Define the desired offset between yellow square and green ball.
    target_offset = jnp.array([0, 1], dtype=y_pos.dtype)
    # Compute difference vectors relative to the target.
    diff_prev = (g_pos_prev - y_pos_prev) - target_offset
    diff = (g_pos - y_pos) - target_offset
    # Compute Euclidean distances as potentials. Using sqrt(sum(square)).
    # Closeness to the goal yields a smaller distance, hence a higher potential difference.
    # We define potential as negative distance.
    # Add a small epsilon to avoid sqrt(0) issues, though not strictly necessary.
    eps = 1e-6
    potential_prev = -jnp.sqrt(jnp.sum(diff_prev * diff_prev) + eps)
    potential = -jnp.sqrt(jnp.sum(diff * diff) + eps)
    # Step reward: potential change plus a small step penalty.
    step_penalty = 0.01
    step_reward = (potential_prev - potential) - step_penalty
    # Determine if the goal is achieved in the current state:
    # That is, green ball is exactly one cell to the right of yellow square.
    goal_achieved = jnp.all((g_pos - y_pos) == target_offset)
    # If terminal (ts_next.last() > 0), then set reward to final bonus consistent with sparse reward.
    # We give a bonus of 1.0 if the goal is achieved, else 0.0.
    final_reward = jnp.where(goal_achieved, 1.0, 0.0)
    # ts_next.last() is an array value; use jnp.where for conditional selection.
    reward = jnp.where(ts_next.last() > 0, final_reward, step_reward)
    return jnp.asarray(reward, dtype=jnp.float32)