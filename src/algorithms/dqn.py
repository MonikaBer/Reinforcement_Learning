import tensorflow
from acme.tf import networks

Images = tensorflow.Tensor
QValues = tensorflow.Tensor


class MyDQNAtariNetwork(networks.DQNAtariNetwork):
    def __init__(self, num_actions: int):
        networks.DQNAtariNetwork.__init__(self, num_actions)

    def __call__(self, inputs: Images) -> QValues:
        inputs = tensorflow.cast(x = inputs, dtype = tensorflow.float32)
        return self._network(inputs)
