
import copy
import pyvirtualdisplay
import imageio 
import base64
import IPython

from acme import environment_loop
from acme.tf import networks
from acme.adders import reverb as adders
from acme.agents.tf import actors as actors
from acme.datasets import reverb as datasets
from acme.wrappers import gym_wrapper
from acme import specs
from acme import wrappers
from acme.agents import tf
from acme.agents import agent
from acme.tf import utils as tf2_utils
from acme.utils import loggers

from acme.wrappers.gym_wrapper import GymAtariAdapter
import gym

import gym 
import dm_env
from dm_control import suite
#from dm_env import specs

import matplotlib.pyplot as plt
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf

from acme.agents.tf import dqn

def run():
    environment_name = 'Assault-v0' 
    env = createEnv(environment_name)
    sp = getEnvSpec(env)
    print(sp.actions[0].num_values)
    #printEnv(getEnvSpec(env))

def createDisplay(x = 160, y = 210):
	return pyvirtualdisplay.Display(visible=0, size=(x, y)).start()

def createEnv(envName):
    genv = gym.make(envName)
    env = GymAtariAdapter(genv)
    return wrappers.SinglePrecisionWrapper(env) 

def getEnvSpec(env):
    return specs.make_environment_spec(env)

def printEnv(envSpec):
    print('actions:\n', envSpec.actions, '\n')
    print('observations:\n', envSpec.observations, '\n')
    print('rewards:\n', envSpec.rewards, '\n')
    print('discounts:\n', envSpec.discounts, '\n')

def createAgent(envSpec, explorationSigma = 0.3):
    base = snt.Sequential(networks.AtariTorso.DQNAtariNetwork(envSpec.actions[0].num_values))
    #net = snt.Sequential(base + [networks.ClippedGaussian(explorationSigma),
    #                    networks.ClipToSpec(envSpec.actions)])
    return actors.dqn.DQN(envSpec, base)

def saveVideo(frames, filename='temp.mp4'):
    if(isinstance(frames, list)):
        frames = np.array(frames)
    """Save and display video."""
    # Write video
    with imageio.get_writer(filename, fps=60) as video:
        for frame in frames:
            video.append_data(frame)
    # Read video and display the video
    video = open(filename, 'rb').read()
    b64_video = base64.b64encode(video)
    video_tag = ('<video  width="160" height="210" controls alt="test" '
                'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    return IPython.display.HTML(video_tag)

def collectFrames(environment, actor, steps = 500):
    frames = []
    timestep = environment.reset()

    for _ in range(steps):
        frames.append(render(environment))
        action = actor.select_action(timestep.observation)
        timestep = environment.step(action)
    return frames

def createServer():
    replay_buffer = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        max_size=1000000,
        remover=reverb.selectors.Fifo(),
        sampler=reverb.selectors.Uniform(),
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature = adders.NStepTransitionAdder.signature(environment_spec))
    
    server = reverb.Server([replay_buffer], port=None)
    replay_server_address = 'localhost:%d' % replay_server.port
    
    return server, replay_server_address

def createExperienceBuffer(serverAddress):
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(serverAddress),
        n_step=5,
        discount=0.99)
    return adder

def collectExperience(environment, expBuffer, actor, episodes=2):
    for episode in range(episodes):
        timestep = environment.reset()
        expBuffer.add_first(timestep)

        while not timestep.last():
            action = actor.select_action(timestep.observation)
            timestep = environment.step(action)
            expBuffer.add(action=action, next_timestep=timestep)


def createReporter(serverAddress):
    dataset = iter(datasets.make_dataset(
        server_address=serverAddress,
        batch_size=256,
        transition_adder=True,
        prefetch_size=4))
    return dataset

def rest():
    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            observation_network=tf2_utils.batch_concat,
            action_network=tf.identity,
            critic_network=networks.LayerNormMLP(
                layer_sizes=(400, 300),
                activate_final=True)),
        # Value-head gives a 51-atomed delta distribution over state-action values.
        networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51)])

    # Create the target networks
    target_policy_network = copy.deepcopy(policy_network)
    target_critic_network = copy.deepcopy(critic_network)

    # We must create the variables in the networks before passing them to learner.
    tf2_utils.create_variables(network=policy_network,
                            input_spec=[environment_spec.observations])
    tf2_utils.create_variables(network=critic_network,
                            input_spec=[environment_spec.observations,
                                        environment_spec.actions])
    tf2_utils.create_variables(network=target_policy_network,
                            input_spec=[environment_spec.observations])
    tf2_utils.create_variables(network=target_critic_network,
                            input_spec=[environment_spec.observations,
                                        environment_spec.actions])

    learner = d4pg.D4PGLearner(policy_network=policy_network,
                            critic_network=critic_network,
                            target_policy_network=target_policy_network,
                            target_critic_network=target_critic_network,
                            dataset_iterator=dataset,
                            discount=0.99,
                            target_update_period=100,
                            policy_optimizer=snt.optimizers.Adam(1e-4),
                            critic_optimizer=snt.optimizers.Adam(1e-4),
                            # Log learner updates to console every 10 seconds.
                            logger=loggers.TerminalLogger(time_delta=10.),
                            checkpoint=False)

    num_training_episodes =  10 # @param {type: "integer"}
    min_actor_steps_before_learning = 1000  # @param {type: "integer"}
    num_actor_steps_per_iteration =   100 # @param {type: "integer"}
    num_learner_steps_per_iteration = 1  # @param {type: "integer"}

    learner_steps_taken = 0
    actor_steps_taken = 0
    for episode in range(num_training_episodes):
    
        timestep = environment.reset()
        actor.observe_first(timestep)
        episode_return = 0

        while not timestep.last():
            # Get an action from the agent and step in the environment.
            action = actor.select_action(timestep.observation)
            next_timestep = environment.step(action)

            # Record the transition.
            actor.observe(action=action, next_timestep=next_timestep)

            # Book-keeping.
            episode_return += next_timestep.reward
            actor_steps_taken += 1
            timestep = next_timestep

            # See if we have some learning to do.
            if (actor_steps_taken >= min_actor_steps_before_learning and
                actor_steps_taken % num_actor_steps_per_iteration == 0):
            # Learn.
                for learner_step in range(num_learner_steps_per_iteration):
                    learner.step()
            learner_steps_taken += num_learner_steps_per_iteration

    # Log quantities.
    print('Episode: %d | Return: %f | Learner steps: %d | Actor steps: %d'%(
        episode, episode_return, learner_steps_taken, actor_steps_taken))


    d4pg_agent = agent.Agent(actor=actor,
                            learner=learner,
                            min_observations=1000,
                            observations_per_step=8.)

    # This may be necessary if any of the episodes were cancelled above.
    adder.reset()

    # We also want to make sure the logger doesn't write to disk because that can
    # cause issues in colab on occasion.
    logger = loggers.TerminalLogger(time_delta=10.)

    loop = environment_loop.EnvironmentLoop(environment, d4pg_agent, logger=logger)
    loop.run(num_episodes=50)

    # Run the actor in the environment for desired number of steps.
    frames = []
    num_steps = 500
    timestep = environment.reset()

    for _ in range(num_steps):
        frames.append(render(environment))
        action = d4pg_agent.select_action(timestep.observation)
        timestep = environment.step(action)

    # Save video of the behaviour.
    display_video(np.array(frames))

    del replay_server

run()