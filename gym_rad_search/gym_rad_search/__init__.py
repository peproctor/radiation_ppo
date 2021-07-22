from gym.envs.registration import register

register(
    id='RadSearch-v0',
    entry_point='gym_rad_search.envs:RadSearch',
)