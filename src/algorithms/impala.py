import tensorflow
from acme.agents.jax.impala import networks

Images = tensorflow.Tensor
QValues = tensorflow.Tensor


class MyImpalaAtariNetwork(networks.IMPALANetworks):
    def __init__(self, env_spec):
        self.network = networks.make_atari_networks(env_spec)

    def __call__(self, inputs: Images) -> QValues:
        inputs = tensorflow.cast(x = inputs, dtype = tensorflow.float32)
        return self.network(inputs)
