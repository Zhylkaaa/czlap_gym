from gym.envs.registration import register
register(
    id='CzlapCzlap-v0',
    entry_point='czlap_the_robot.envs:CzlapCzlapEnv'
)