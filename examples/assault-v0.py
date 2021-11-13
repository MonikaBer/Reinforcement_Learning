
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


from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


import gym
from gym import envs


display = pyvirtualdisplay.Display(visible=0, size=(1600, 900)).start()

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

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

def testLoad():
    env_name = 'Assault-v0'
    env = suite_gym.load(env_name)
    #env = gym.make(env_name)

    env.reset()
    specs(env)
    img = env.render()
    showRender(img)

def test():
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
    env_name = 'Assault-v0'
    env = suite_gym.load(env_name)

    env.reset()
    image = PIL.Image.fromarray(env.render())
    plt.imshow(image)
    plt.show()

testLoad()