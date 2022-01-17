import copy
import pyvirtualdisplay
import imageio
import base64
import IPython

import acme
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

import gym
import atari_py

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
import functools

import pdb


FLAGS = {
    'num_episodes': 1
}

Images = tf.Tensor
QValues = tf.Tensor


class MyDQNAtariNetwork(networks.DQNAtariNetwork):
    def __init__(self, num_actions: int):
        networks.DQNAtariNetwork.__init__(self, num_actions)

    def __call__(self, inputs: Images) -> QValues:
        inputs = tf.cast(x = inputs, dtype = tf.float32)
        return self._network(inputs)

##
##
##
## Pod spodem podaję funkcje przydatne podczas debugowania.
## Przy niektórych z nich pozostawiam również komentarz wyjaśniający niektóre sprawy
##
##
##


def printAttrs(obj):
    print(vars(obj))
    #
    # Dla Assault-v0 env to
    #{'env': <gym.envs.atari.atari_env.AtariEnv object at 0x7fdc4d858c10>, 'action_space': Discrete(7),
    # 'observation_space': Box(210, 160, 3), 'reward_range': (-inf, inf), 'metadata': {'render.modes': ['human', 'rgb_array']},
    # '_max_episode_seconds': None, '_max_episode_steps': 10000, '_elapsed_steps': 0, '_episode_started_at': None}
    #
    # Z tego wynika, że nie można używać tutaj wrapera od Atari, bo nie ma obiektu 'ale' czyli liczby żyć. Jest jedynie wynik.
    # specs.make_environment_spec(env) to pobiera.


def printSpec(env):
    print(getEnvSpec(env))

# https://stackoverflow.com/questions/67656740/exception-rom-is-missing-for-ms-pacman-see-https-github-com-openai-atari-py

# trzeba w sieci zrobić wstępną obróbkę danych, przerobić obserwacje z int8 do float [0, 1] dla conv
# dla ConvND self._dtype = inputs.dtype
# print(getEnvSpec(env))
# observations=BoundedArray(shape=(210, 160, 3), dtype=dtype('uint8'), name='observation', minimum=[[[0 0 0]
# na początku modelu trzeba dodać odpowiednią warstwę, może z tf bezpośrednio?

##
##  koniec
##


def createDisplay(x = 160, y = 210):
	return pyvirtualdisplay.Display(visible = 0, size = (x, y)).start()


def createEnv(envName):
    env = gym.make(envName)

    #print(env.ale.lives())
    #print(gym.envs.registry.all())
    #env = gym_wrapper.GymWrapper(env)
    #env = wrappers.CanonicalSpecWrapper(env)
    #env = wrappers.AtariWrapper(env, to_float=True)
    #env = gym_wrapper.GymAtariAdapter(env)
    #env = wrappers.CanonicalSpecWrapper(env, clip=True)
    #env = wrappers.SinglePrecisionWrapper(env)
    #env = atari_wrapper.AtariWrapper(env)
    #env = wrappers.CanonicalSpecWrapper(env, clip=True)

    env = gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)

    '''wrapper_list = [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            to_float=True,
            max_episode_len=1000,
            zero_discount_on_life_loss=True,
        ),
        wrappers.SinglePrecisionWrapper,
        wrappers.ObservationActionRewardWrapper,
    ]
    env = wrappers.wrap_all(env, wrapper_list)'''

    timestep = env.reset()
    return env


def getEnvSpec(env):
    return specs.make_environment_spec(env)


def printEnv(envSpec):
    print('actions:\n', envSpec.actions, '\n')
    print('observations:\n', envSpec.observations, '\n')
    print('rewards:\n', envSpec.rewards, '\n')
    print('discounts:\n', envSpec.discounts, '\n')


def createAgent(envSpec, explorationSigma = 0.3):
    base = networks.DQNAtariNetwork(envSpec.actions[0].num_values)
    #net = snt.Sequential(base + [networks.ClippedGaussian(explorationSigma),
    #                    networks.ClipToSpec(envSpec.actions)])
    base = snt.Sequential([tf.keras.layers.Rescaling(1./255, input_shape=(batch_size, img_height, img_width, 3))]) + base
    return acme.agents.tf.dqn.agent.DQN(envSpec, base)


def saveVideo(frames, filename = 'temp.mp4'):
    if(not isinstance(frames, np.ndarray)):
        frames = np.array(frames)
    """Save and display video."""
    # Write video
    with imageio.get_writer(filename, fps = 60) as video:
        for frame in frames:
            video.append_data(frame)
    # Read video and display the video
    video = open(filename, 'rb').read()
    b64_video = base64.b64encode(video)
    video_tag = ('<video  width="160" height="210" controls alt="test" '
                'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    return IPython.display.HTML(video_tag)


def render(env):
    return env.environment.render(mode = 'rgb_array')


def collectFrames(environment, actor, steps = 500):
    frames = []
    timestep = environment.reset()

    for _ in range(steps):
        frames.append(render(environment))
        action = actor.select_action(timestep.observation)
        timestep = environment.step(action)
    return frames


def createServer(env_spec):
    replay_buffer = reverb.Table(
        name = adders.DEFAULT_PRIORITY_TABLE,
        max_size = 1000000,
        remover = reverb.selectors.Fifo(),
        sampler = reverb.selectors.Uniform(),
        rate_limiter = reverb.rate_limiters.MinSize(min_size_to_sample = 1),
        signature = adders.NStepTransitionAdder.signature(env_spec)
    )

    replay_server = reverb.Server([replay_buffer], port = None)
    replay_server_address = 'localhost:%d' % replay_server.port
    return replay_server, replay_server_address


def createExperienceBuffer(serverAddress):
    adder = adders.NStepTransitionAdder(
        client = reverb.Client(serverAddress),
        n_step = 5,
        discount = 0.99
    )
    return adder


def collectExperience(environment, expBuffer, actor, episodes = 2):
    for episode in range(episodes):
        timestep = environment.reset()
        expBuffer.add_first(timestep)

        while not timestep.last():
            action = actor.select_action(timestep.observation)
            timestep = environment.step(action)
            expBuffer.add(action=action, next_timestep = timestep)


def createReporter(serverAddress):
    dataset = iter(datasets.make_dataset(
        server_address = serverAddress,
        batch_size = 256,
        transition_adder = True,
        prefetch_size = 4)
    )
    return dataset


def loopAgent(env, agent, episodes):
    loop = acme.EnvironmentLoop(env, agent)
    loop.run(episodes)


def rest():
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
        input_spec = [environment_spec.observations]
    )

    tf2_utils.create_variables(
        network = critic_network,
        input_spec = [environment_spec.observations, environment_spec.actions]
    )

    tf2_utils.create_variables(
        network = target_policy_network,
        input_spec = [environment_spec.observations]
    )

    tf2_utils.create_variables(
        network = target_critic_network,
        input_spec = [environment_spec.observations, environment_spec.actions]
    )

    learner = d4pg.D4PGLearner(
        policy_network = policy_network,
        critic_network = critic_network,
        target_policy_network = target_policy_network,
        target_critic_network = target_critic_network,
        dataset_iterator = dataset,
        discount = 0.99,
        target_update_period=100,
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

        timestep = environment.reset()
        actor.observe_first(timestep)
        episode_return = 0

        while not timestep.last():
            # Get an action from the agent and step in the environment.
            action = actor.select_action(timestep.observation)
            next_timestep = environment.step(action)

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

    loop = environment_loop.EnvironmentLoop(environment, d4pg_agent, logger = logger)
    loop.run(num_episodes = 50)

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


def test():
    environment_name = 'Assault-v4'
    env = createEnv(environment_name)
    env_spec = acme.make_environment_spec(env)

    #network = networks.DQNAtariNetwork(env_spec.actions.num_values)
    network = MyDQNAtariNetwork(env_spec.actions.num_values)

    #print(env_spec)
    agent = dqn.DQN(env_spec, network)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(FLAGS['num_episodes'])


    server, address = createServer(env_spec)
    buffer = createExperienceBuffer(address)
    collectExperience(env, buffer, agent, 4)
    #saveVideo(buffer)

    frames = collectFrames(env, agent)
    saveVideo(frames)


def run():
    environment_name = 'Assault-v4'
    display = createDisplay()
    env = createEnv(environment_name)
    espec = getEnvSpec(env)
    print(espec)
    agent = createAgent(espec)

    #loopAgent(env, agent, 500)
    '''
    server, address = createServer()
    buffer = createExperienceBuffer(address)
    collectExperience(env, buffer, agent, 4)

    saveVideo(buffer)
    '''

#run()
#print(type(np.array([0])))

test()
