import acme
from acme.agents.jax import impala
import gym
from acme.wrappers import gym_wrapper
from acme import wrappers
from acme import specs
from acme.agents.jax.impala.networks import IMPALANetworks
import tensorflow as tf
import dm_env
from acme import types
from acme.wrappers import base
import tree
import numpy as np

Images = tf.Tensor
QValues = tf.Tensor

class MyImpalaAtariNetwork(IMPALANetworks):
    def __init__(self, env_spec):
        self.network = impala.make_atari_networks(env_spec)

    def __call__(self, inputs: Images) -> QValues:
        inputs = tf.cast(x = inputs, dtype = tf.float32)
        return self.network(inputs)

class MyObservationTransformWrapper(base.EnvironmentWrapper):
    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        return timestep._replace(
            observation=self._convert_value(timestep.observation))

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
                return spec.replace(dtype=np.float32, 
                    minimum=np.full(shape=spec.shape, fill_value=-1.0, dtype=dtype), 
                    maximum=np.full(shape=spec.shape, fill_value=1.0, dtype=dtype))
            return spec

        return tree.map_structure(_convert_single_spec, nested_spec)


    def _convert_value(self, nested_value: types.Nest) -> types.Nest:
        """Normalizacja inputu do zakresu [-1.0; 1.0]."""

        def _convert_single_value(value):
            if np.issubdtype(value.dtype, np.uint8):
                value = ((np.array(value, copy=False, dtype=np.float32) / 255.0) - 1.0)
            return value
                
        return tree.map_structure(_convert_single_value, nested_value)



def createImpala(envName):
    env = gym.make(envName)
    env = gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.ObservationActionRewardWrapper(env)
    env = MyObservationTransformWrapper(env)
    timestep = env.reset()

    env_spec = specs.make_environment_spec(env)
    
    config = impala.IMPALAConfig(
        batch_size=16,
        sequence_period=10,
        seed=111,
    )

    networks = impala.make_atari_networks(env_spec)
    #networks = MyImpalaAtariNetwork(env_spec)
    #print(env_spec)
    #exit(0)
    agent = impala.IMPALAFromConfig(
        environment_spec=env_spec,
        forward_fn=networks.forward_fn,
        unroll_init_fn=networks.unroll_init_fn,
        unroll_fn=networks.unroll_fn,
        initial_state_init_fn=networks.initial_state_init_fn,
        initial_state_fn=networks.initial_state_fn,
        config=config,
    )
    loop = acme.EnvironmentLoop(env, agent)
    loop.run(2)

createImpala('Assault-v0')