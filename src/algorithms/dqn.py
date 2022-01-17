import gym
from acme import wrappers
import acme.agents.tf as tf
from acme.tf import networks


Images = tf.Tensor
QValues = tf.Tensor

class MyDQNAtariNetwork(networks.DQNAtariNetwork):
    def __init__(self, num_actions: int):
        networks.DQNAtariNetwork.__init__(self, num_actions)

    def __call__(self, inputs: Images) -> QValues:
        inputs = tf.cast(x = inputs, dtype = tf.float32)
        return self._network(inputs)


def createEnvForDQN(envName):
    env = gym.make(envName)
    env = wrappers.gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    timestep = env.reset()
    return env
