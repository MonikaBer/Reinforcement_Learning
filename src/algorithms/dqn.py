import gym
import tensorflow
from acme import wrappers
from acme.tf import networks

Images = tensorflow.Tensor
QValues = tensorflow.Tensor

class MyDQNAtariNetwork(networks.DQNAtariNetwork):
    def __init__(self, num_actions: int):
        networks.DQNAtariNetwork.__init__(self, num_actions)

    def __call__(self, inputs: Images) -> QValues:
        inputs = tensorflow.cast(x = inputs, dtype = tensorflow.float32)
        return self._network(inputs)


def createEnvForDQN(envName):
    env = gym.make(envName)
    env = wrappers.gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    timestep = env.reset()
    return env
