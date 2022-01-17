from acme import specs


def printAttrs(obj):
    print(vars(obj))
    # Dla Assault-v0 env to
    #{'env': <gym.envs.atari.atari_env.AtariEnv object at 0x7fdc4d858c10>, 'action_space': Discrete(7),
    # 'observation_space': Box(210, 160, 3), 'reward_range': (-inf, inf), 'metadata': {'render.modes': ['human', 'rgb_array']},
    # '_max_episode_seconds': None, '_max_episode_steps': 10000, '_elapsed_steps': 0, '_episode_started_at': None}
    #
    # Z tego wynika, że nie można używać tutaj wrapera od Atari, bo nie ma obiektu 'ale' czyli liczby żyć. Jest jedynie wynik.
    # specs.make_environment_spec(env) to pobiera.
    #


def getEnvSpec(env):
    return specs.make_environment_spec(env)


def printSpec(env):
    print(getEnvSpec(env))
    #https://stackoverflow.com/questions/67656740/exception-rom-is-missing-for-ms-pacman-see-https-github-com-openai-atari-py


def printEnv(envSpec):
    print(f'actions:\n{envSpec.actions}\n')
    print(f'observations:\n{envSpec.observations}\n')
    print(f'rewards:\n{envSpec.rewards}\n')
    print(f'discounts:\n{envSpec.discounts}\n')
