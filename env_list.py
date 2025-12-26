import xminigrid

for env_id in xminigrid.registered_environments():
    if env_id.startswith("XLand-MiniGrid-"):
        print(env_id)
