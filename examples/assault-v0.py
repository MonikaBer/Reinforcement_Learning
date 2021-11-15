
# https://gym.openai.com/envs/Assault-v0/

from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb
import os

import tensorflow as tf
from tensorflow.keras import applications
import tf_agents

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import fixed_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


import gym
from gym import envs

from tensorflow.python.keras.layers import VersionAwareLayers

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# conv
in_channels = 3
img_height = 210
img_width = 160

def disableGPU():
    tf.config.set_visible_devices([], 'GPU')

def embed_mp4(filename):
    '''Embeds an mp4 file in the notebook.'''
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    return embed_mp4(filename)
    

def specs(env):
    print('Env:')
    print(env)

    print('Observation Spec:')
    print(env.time_step_spec().observation)

    print('Reward Spec:')
    print(env.time_step_spec().reward)

    print('Action Spec:')
    print(env.action_spec())

def showRender(image):
    image = PIL.Image.fromarray(image)
    plt.imshow(image)
    plt.show()

def simpleModel(
    input_tensor=tf.keras.Input(shape=(210, 160, 3), batch_size=batch_size, dtype=tf.float32, sparse=False, tensor=None, ragged=False, type_spec=None),
    pooling='max',
    classes=7,
    classifier_activation='softmax'
):
    print("Creating simple model")
    x = []
    img_input = input_tensor
    pooling = None
    if pooling == 'avg':
        pooling = getAvgPool
    elif pooling == 'max':
        pooling = getMaxPool

    # trzeba skonwertować obraz z uint8 na float32
    """
    # Block 1
    x += [tf.keras.layers.Rescaling(1./255, input_shape=(batch_size, img_height, img_width, 3)),
        tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(batch_size, img_height, img_width, in_channels)),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
    ]
    """

    # Classification block
    x += [tf.keras.layers.Rescaling(1./255, input_shape=(batch_size, img_height, img_width, 3)),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(32, activation='relu', name='fc1'),
        tf.keras.layers.Dense(32, activation='relu', name='fc2'),
        tf.keras.layers.Dense(classes, activation=classifier_activation,
            name='predictions')
    ]

    if pooling == 'avg':
        x += [layers.GlobalAveragePooling2D()]
    elif pooling == 'max':
        x += [layers.GlobalMaxPooling2D()]

    return x


def VGG19(
        input_tensor=tf.keras.Input(shape=(210, 160, 3), batch_size=batch_size, dtype=tf.float32, sparse=False, tensor=None, ragged=False, type_spec=None),
        pooling='max',
        classes=7,
        classifier_activation='softmax'):
    """
        Instantiates the VGG19 architecture.

        Reference:
        - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
            https://arxiv.org/abs/1409.1556) (ICLR 2015)

        For image classification use cases, see
        [this page for detailed examples](
            https://keras.io/api/applications/#usage-examples-for-image-classification-models).

        For transfer learning use cases, make sure to read the
        [guide to transfer learning & fine-tuning](
            https://keras.io/guides/transfer_learning/).

        The default input size for this model is 224x224.

        Note: each Keras Application expects a specific kind of input preprocessing.
        For VGG19, call `tf.keras.applications.vgg19.preprocess_input` on your
        inputs before passing them to the model.
        `vgg19.preprocess_input` will convert the input images from RGB to BGR,
        then will zero-center each color channel with respect to the ImageNet dataset,
        without scaling.

        Args:
            include_top: whether to include the 3 fully-connected
            layers at the top of the network.
            weights: one of `None` (random initialization),
                'imagenet' (pre-training on ImageNet),
                or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
            classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
            When loading pretrained weights, `classifier_activation` can only
            be `None` or `"softmax"`.

        Returns:
            A `keras.Model` instance.
    """
    def getMaxPool():
        return tf.keras.layers.MaxPooling2D((2, 2))
    def getAvgPool():
        return tf.keras.layers.AveragePooling2D((2, 2))

    print("Creating VGG19")
    x = []
    img_input = input_tensor
    pooling = None
    if pooling == 'avg':
        pooling = getAvgPool
    elif pooling == 'max':
        pooling = getMaxPool

    # trzeba skonwertować obraz z uint8 na float32

    # Block 1
    x += [tf.keras.layers.Rescaling(1./255, input_shape=(batch_size, img_height, img_width, 3)),
        tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(batch_size, img_height, img_width, in_channels)),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
    ]
    
    # Block 2
    x += [tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
        tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
    ]

    # Block 3
    x += [tf.keras.layers.Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
        tf.keras.layers.Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
        tf.keras.layers.Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
        tf.keras.layers.Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv4'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
    ]

    # Block 4
    x += [tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
        tf.keras.layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
        tf.keras.layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
        tf.keras.layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv4'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
    ]

    # Block 5
    x += [tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
        tf.keras.layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
        tf.keras.layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
        tf.keras.layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block5_conv4'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
    ]
    
    # Classification block
    x += [tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(4096, activation='relu', name='fc1'),
        tf.keras.layers.Dense(4096, activation='relu', name='fc2'),
        tf.keras.layers.Dense(classes, activation=classifier_activation,
            name='predictions')
    ]

    if pooling == 'avg':
        x += [layers.GlobalAveragePooling2D()]
    elif pooling == 'max':
        x += [layers.GlobalMaxPooling2D()]

    return x



def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

def tensorBoardCallBack(model, logDir):
    #tensorboard --logdir logs
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def createSimpleModelCustom():
    x = simpleModel()

    # Create model.
    model = tf_agents.networks.sequential.Sequential(x)
    return model

def createVGG19Custom():
    # python3.8/site-packages/tensorflow/python/keras/applications/vgg19.py
    # trzeba skopiować, bo keras layers to nie to samo co tf-agent. Trzeba użyć

    x = VGG19()

    # Create model.
    model = tf_agents.networks.sequential.Sequential(x)
    return model

def createVGG19():
    input_image = tf.keras.Input(shape=(210, 160, 3), batch_size=batch_size, dtype=tf.float32, sparse=False, tensor=None, ragged=False, type_spec=None)
    vgg19 = tf.keras.applications.VGG19(
        include_top=True, weights=None, 
        input_tensor=input_image, 
        pooling='max', classes=7, classifier_activation='softmax'
    )
    vgg19 = tf.keras.applications.VGG19(
        include_top=True, weights=None, classifier_activation='softmax'
    )

def createNet(env):
    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    return q_net

def createAgent(env_name, q_net, learning_rate):
    train_py_env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()
    print("agent_initialized")
    return agent, train_py_env

def compute_avg_return(environment, policy, num_episodes=10):
    '''
        To może długo trwać, przechodzi przez wszystkie epizody.
    '''
    total_return = 0.0
    print("Compute average return. Max", num_episodes)
    for episode in range(num_episodes):
        print("Episode:", episode)
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def testAction(env, action, number):
    fixed_pol = fixed_policy.FixedPolicy(action, env.time_step_spec(), env.action_spec())
    tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))

    create_policy_eval_video(agent.policy, "trained-agent")
    create_policy_eval_video(random_policy, "random-agent")

def setSever(agent, replay_buffer_max_length, table_name):
    print("Setting server")
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
        
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    print("Server set")
    return reverb.Server([table])

def setReply(agent, reverb_server, batch_size, table_name):
    print("Setting reply")
    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)

    dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

    iterator = iter(dataset)
    print("Reply set")
    return iterator, rb_observer

def setDriver(agent, env, rb_observer, env_name, init_num_episodes=10):
    print("Setting driver")
    eval_py_env = suite_gym.load(env_name)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_episodes=init_num_episodes)
    returns = [0]

    print("Creating driver")
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)
    print("Driver created")

    return collect_driver, returns

def trainLoop(agent, collect_driver, train_py_env, iterator, num_iterations):
    # Reset the environment.
    time_step = train_py_env.reset()

    print("Start train loop")
    for niter in range(num_iterations):
        print("Iteration:", niter)
        # Collect a few steps and save to the replay buffer.
        print("Collect driver steps", time_step)
        time_step, _ = collect_driver.run(time_step)
        print("End collecting driver steps")

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        print("Start train")
        train_loss = agent.train(experience).loss
        print("End train")

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    print("End train loop")

def testLoad():
    env_name = 'Assault-v0'
    table_name = 'uniform_table'
    env = suite_gym.load(env_name)

    env.reset()
    #testAction(env, np.array(1, dtype=np.int32), 1000)
    #specs(env)
    #q_net = createNet(env=env)
    q_net = createSimpleModelCustom()
    agent, train_py_env = createAgent(env_name=env_name, q_net=q_net, learning_rate=learning_rate)
    server = setSever(agent=agent, replay_buffer_max_length=replay_buffer_max_length, table_name=table_name)
    replyIterator, rb_observer = setReply(agent=agent, reverb_server=server, batch_size=batch_size, table_name=table_name)
    collect_driver, returns = setDriver(agent=agent, env=env, rb_observer=rb_observer, env_name=env_name, init_num_episodes=1)
    trainLoop(agent=agent, collect_driver=collect_driver, train_py_env=train_py_env, iterator=replyIterator, num_iterations=num_iterations)

    img = env.render()
    showRender(img)

disableGPU()
testLoad()