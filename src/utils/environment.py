import gym
import tree
import dm_env
import numpy as np
from acme import specs
from acme.wrappers import base
from acme import types
from acme import wrappers


class MyObservationTransformWrapper(base.EnvironmentWrapper):
    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        return timestep._replace(
            observation = self._convert_value(timestep.observation)
        )

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.reset())

    def observation_spec(self):
        return self._convert_spec(self._environment.observation_spec())

    def _convert_spec(self, nested_spec: types.NestedSpec) -> types.NestedSpec:
        """Dotyczy to tylko specyfikacji. Do zmiany inputu wykorzytuje się _convert_value """
        def _convert_single_spec(spec: specs.Array):
            """Convert a single spec."""
            if spec.name == 'observation' and np.issubdtype(spec.dtype, np.integer):
                dtype = np.float32
                return spec.replace(
                    dtype = np.float32,
                    minimum = np.full(shape = spec.shape, fill_value = -1.0, dtype = dtype),
                    maximum = np.full(shape = spec.shape, fill_value = 1.0, dtype = dtype)
                )
            return spec

        return tree.map_structure(_convert_single_spec, nested_spec)

    def _convert_value(self, nested_value: types.Nest) -> types.Nest:
        """Normalizacja inputu do zakresu [-1.0; 1.0]."""
        def _convert_single_value(value):
            if value is not None and np.issubdtype(value.dtype, np.uint8):
                value = ((np.array(value, copy = False, dtype = np.float32) / 255.0) - 1.0)
            return value

        return tree.map_structure(_convert_single_value, nested_value)


def createEnv(envName, algType):
    if algType not in ['dqn', 'impala']:
        raise RuntimeError('wrong algorithm type (required: dqn/impala)')

    env = gym.make(envName)
    env = wrappers.gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    if algType == 'impala':
        env = wrappers.ObservationActionRewardWrapper(env)
        env = MyObservationTransformWrapper(env)
    timestep = env.reset()
    return env


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
