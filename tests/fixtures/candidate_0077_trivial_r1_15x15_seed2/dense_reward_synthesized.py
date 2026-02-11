```python
def dense_reward(env_params, ts_prev, action, ts_next, ctx: dict) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Calculates a dense reward for moving next to the red key.

    The reward is structured to guide the agent through the following stages:
    1.  **Exploration:** A one-time bonus is given for seeing the red key for the first time.
    2.  **Navigation:** A potential-based reward encourages the agent to decrease its
        Manhattan distance to the red key.
    3.  **Achievement:** A large, one-time bonus is awarded for becoming adjacent
        (Manhattan distance of 1) to the red key.
    4.  **Maintenance:** A small, continuous bonus is given for remaining adjacent to the key.
    5.  **Efficiency:** A small penalty is applied at each step to encourage completing
        the task quickly.

    Returns:
        A tuple containing:
        - total_reward (jnp.float32): The scalar sum of all reward components.
        - reward_components (dict): A dictionary mapping descriptive string names
          to their scalar jnp.float32 reward values.
    """
    # --- Get agent and object positions from context ---
    agent_pos = ctx.get("agent_pos", jnp.array([-1, -1], dtype=jnp.int32))
    agent_pos_prev = ctx.get("agent_pos_prev", jnp.array([-1, -1], dtype=jnp.int32))

    # Safely access nested dictionary for object positions
    object_positions = ctx.get("object_positions", {})
    red_key_pos = object_positions.get("red_key", jnp.array([-1, -1], dtype=jnp.int32))

    # --- Define a validity flag for position-based rewards ---
    # This prevents calculating rewards with invalid default positions (e.g., at episode start)
    valid_positions = jnp.all(agent_pos > -1) & jnp.all(agent_pos_prev > -1) & jnp.all(red_key_pos > -1)

    # --- 1. Potential-based reward for getting closer to the red key ---
    dist_to_key_now = jnp.sum(jnp.abs(agent_pos - red_key_pos))
    dist_to_key_prev = jnp.sum(jnp.abs(agent_pos_prev - red_key_pos))

    # Potential is the negative distance. Reward is the change in potential.
    potential_now = -dist_to_key_now.astype(jnp.float32)
    potential_prev = -dist_to_key_prev.astype(jnp.float32)
    
    # Scale the reward to balance its contribution
    distance_reward = (potential_now - potential_prev) * 0.5
    distance_reward = jnp.where(valid_positions, distance_reward, 0.0)

    # --- 2. Achievement and Maintenance bonuses for adjacency ---
    is_adjacent_now = (dist_to_key_now == 1)
    was_adjacent_prev = (dist_to_key_prev == 1)

    # Large one-time bonus for achieving adjacency
    achieved_adjacency_bonus = jnp.where(is_adjacent_now & ~was_adjacent_prev & valid_positions, 10.0, 0.0)

    # Small maintenance bonus for staying adjacent
    maintained_adjacency_bonus = jnp.where(is_adjacent_now & valid_positions, 0.1, 0.0)

    # --- 3. Exploration bonus for seeing the key ---
    # Safely access nested dictionaries for visible object positions
    visible_positions = ctx.get("visible_object_positions", {})
    visible_positions_prev = ctx.get("visible_object_positions_prev", {})
    red_key_visible_pos = visible_positions.get("red_key", jnp.array([-1, -1], dtype=jnp.int32))
    red_key_visible_pos_prev = visible_positions_prev.get("red_key", jnp.array([-1, -1], dtype=jnp.int32))

    is_visible_now = jnp.all(red_key_visible_pos > -1)
    was_visible_prev = jnp.all(red_key_visible_pos_prev > -1)

    # One-time bonus for spotting the key
    spotted_key_bonus = jnp.where(is_visible_now & ~was_visible_prev, 5.0, 0.0)

    # --- 4. Small time penalty to encourage efficiency ---
    time_penalty = -0.01

    # --- Assemble reward components and calculate total reward ---
    reward_components = {
        "distance_to_key": distance_reward.astype(jnp.float32),
        "achieved_adjacency": achieved_adjacency_bonus.astype(jnp.float32),
        "maintained_adjacency": maintained_adjacency_bonus.astype(jnp.float32),
        "spotted_key": spotted_key_bonus.astype(jnp.float32),
        "time_penalty": jnp.array(time_penalty, dtype=jnp.float32),
    }

    total_reward = (
        distance_reward
        + achieved_adjacency_bonus
        + maintained_adjacency_bonus
        + spotted_key_bonus
        + time_penalty
    )

    return total_reward.astype(jnp.float32), reward_components
```