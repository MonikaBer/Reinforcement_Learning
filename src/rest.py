##################################################
#  code based on ACME tutorial, but not working  #
##################################################

import copy
import sonnet as snt
import numpy as np
import acme
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.agents import tf
from acme import environment_loop
from acme.agents import agent
from acme.utils import loggers

# own modules
from utils.display import createDisplay, render, saveVideo
from utils.environment import createEnv
from utils.server import createServer, createExperienceBuffer, collectExperience


def createAgent(env_spec, exploration_sigma = 0.3):
    #net = snt.Sequential(base + [networks.ClippedGaussian(exploration_sigma),
    #                    networks.ClipToSpec(envSpec.actions)])
    base = snt.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape = (batch_size, img_height, img_width, 3))
    ])
    base += networks.DQNAtariNetwork(env_spec.actions[0].num_values)
    return acme.agents.tf.dqn.agent.DQN(env_spec, base)


def loopAgent(env, agent, num_episodes):
    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes)


def rest(env):
    env_spec = acme.make_environment_spec(env)

    #TODO: define policy network

    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            observation_network = tf2_utils.batch_concat,
            action_network = tf.identity,
            critic_network = networks.LayerNormMLP(
                layer_sizes = (400, 300),
                activate_final = True
            )
        ),
        # Value-head gives a 51-atomed delta distribution over state-action values.
        networks.DiscreteValuedHead(vmin = -150., vmax = 150., num_atoms = 51)
    ])

    # Create the target networks
    target_policy_network = copy.deepcopy(policy_network)
    target_critic_network = copy.deepcopy(critic_network)

    # We must create the variables in the networks before passing them to learner.
    tf2_utils.create_variables(
        network = policy_network,
        input_spec = [env_spec.observations]
    )

    tf2_utils.create_variables(
        network = critic_network,
        input_spec = [env_spec.observations, env_spec.actions]
    )

    tf2_utils.create_variables(
        network = target_policy_network,
        input_spec = [env_spec.observations]
    )

    tf2_utils.create_variables(
        network = target_critic_network,
        input_spec = [env_spec.observations, env_spec.actions]
    )

    learner = d4pg.D4PGLearner(
        policy_network = policy_network,
        critic_network = critic_network,
        target_policy_network = target_policy_network,
        target_critic_network = target_critic_network,
        dataset_iterator = dataset,
        discount = 0.99,
        target_update_period = 100,
        policy_optimizer = snt.optimizers.Adam(1e-4),
        critic_optimizer = snt.optimizers.Adam(1e-4),
        # Log learner updates to console every 10 seconds.
        logger = loggers.TerminalLogger(time_delta = 10.),
        checkpoint = False
    )

    num_training_episodes =  10             # @param {type: "integer"}
    min_actor_steps_before_learning = 1000  # @param {type: "integer"}
    num_actor_steps_per_iteration = 100     # @param {type: "integer"}
    num_learner_steps_per_iteration = 1     # @param {type: "integer"}

    learner_steps_taken = 0
    actor_steps_taken = 0
    for episode in range(num_training_episodes):

        timestep = env.reset()
        actor.observe_first(timestep)
        episode_return = 0

        while not timestep.last():
            # Get an action from the agent and step in the environment.
            action = actor.select_action(timestep.observation)
            next_timestep = env.step(action)

            # Record the transition.
            actor.observe(action = action, next_timestep = next_timestep)

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
            episode, episode_return, learner_steps_taken, actor_steps_taken)
    )


    d4pg_agent = agent.Agent(
        actor = actor,
        learner = learner,
        min_observations = 1000,
        observations_per_step = 8.
    )

    # This may be necessary if any of the episodes were cancelled above.
    adder.reset()

    # We also want to make sure the logger doesn't write to disk because that can
    # cause issues in colab on occasion.
    logger = loggers.TerminalLogger(time_delta = 10.)

    loop = environment_loop.EnvironmentLoop(env, d4pg_agent, logger = logger)
    loop.run(num_episodes = 50)

    # Run the actor in the environment for desired number of steps.
    frames = []
    num_steps = 500
    timestep = env.reset()

    for _ in range(num_steps):
        frames.append(render(env))
        action = d4pg_agent.select_action(timestep.observation)
        timestep = env.step(action)

    # Save video of the behaviour.
    display_video(np.array(frames))

    del replay_server


def run():
    envName = 'Assault-v4'
    display = createDisplay()
    env = createEnv(envName)
    envSpec = env.getEnvSpec(env)
    #print(envSpec)
    agent = createAgent(envSpec)
    loopAgent(env, agent, 500)

    '''
    server, address = createServer()
    buffer = createExperienceBuffer(address)
    collectExperience(env, buffer, agent, 4)

    saveVideo(buffer)
    '''
