
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
from acme.agents.tf import d4pg
from acme.agents import agent
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import gym 
import dm_env
import matplotlib.pyplot as plt
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

environment_name = 'gym_mountaincar'  # @param ['dm_cartpole', 'gym_mountaincar']
# task_name = 'balance'  # @param ['swingup', 'balance']

def make_environment(domain_name='cartpole', task='balance'):
  from dm_control import suite
  env = suite.load(domain_name, task)
  env = wrappers.SinglePrecisionWrapper(env)
  return env

if 'dm_cartpole' in environment_name:
  environment = make_environment('cartpole')
  def render(env):
    return env._physics.render(camera_id=0)  #pylint: disable=protected-access

elif 'gym_mountaincar' in environment_name:
  environment = gym_wrapper.GymWrapper(gym.make('MountainCarContinuous-v0'))
  environment = wrappers.SinglePrecisionWrapper(environment)
  def render(env):
    return env.environment.render(mode='rgb_array')
else:
  raise ValueError('Unknown environment: {}.'.format(environment_name))

# Show the frame.
frame = render(environment)
plt.imshow(frame)
plt.axis('off')

environment_spec = specs.make_environment_spec(environment)

print('actions:\n', environment_spec.actions, '\n')
print('observations:\n', environment_spec.observations, '\n')
print('rewards:\n', environment_spec.rewards, '\n')
print('discounts:\n', environment_spec.discounts, '\n')

# Calculate how big the last layer should be based on total # of actions.
action_spec = environment_spec.actions
action_size = np.prod(action_spec.shape, dtype=int)
exploration_sigma = 0.3

# In order the following modules:
# 1. Flatten the observations to be [B, ...] where B is the batch dimension.
# 2. Define a simple MLP which is the guts of this policy.
# 3. Make sure the output action matches the spec of the actions.
policy_modules = [
    tf2_utils.batch_concat,
    networks.LayerNormMLP(layer_sizes=(300, 200, action_size)),
    networks.TanhToSpec(spec=environment_spec.actions)]

policy_network = snt.Sequential(policy_modules)

# We will also create a version of this policy that uses exploratory noise.
behavior_network = snt.Sequential(
    policy_modules + [networks.ClippedGaussian(exploration_sigma),
                      networks.ClipToSpec(action_spec)])

actor = actors.FeedForwardActor(policy_network)

def display_video(frames, filename='temp.mp4'):
  """Save and display video."""
  # Write video
  with imageio.get_writer(filename, fps=60) as video:
    for frame in frames:
      video.append_data(frame)
  # Read video and display the video
  video = open(filename, 'rb').read()
  b64_video = base64.b64encode(video)
  video_tag = ('<video  width="320" height="240" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
  return IPython.display.HTML(video_tag)

# Run the actor in the environment for desired number of steps.
frames = []
num_steps = 500
timestep = environment.reset()

for _ in range(num_steps):
  frames.append(render(environment))
  action = actor.select_action(timestep.observation)
  timestep = environment.step(action)

# Save video of the behaviour.
display_video(np.array(frames))

# Create a table with the following attributes:
# 1. when replay is full we remove the oldest entries first.
# 2. to sample from replay we will do so uniformly at random.
# 3. before allowing sampling to proceed we make sure there is at least
#    one sample in the replay table.
# 4. we use a default table name so we don't have to repeat it many times below;
#    if we left this off we'd need to feed it into adders/actors/etc. below.
replay_buffer = reverb.Table(
    name=adders.DEFAULT_PRIORITY_TABLE,
    max_size=1000000,
    remover=reverb.selectors.Fifo(),
    sampler=reverb.selectors.Uniform(),
    rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1))

# Get the server and address so we can give it to the modules such as our actor
# that will interact with the replay buffer.
replay_server = reverb.Server([replay_buffer], port=None)
replay_server_address = 'localhost:%d' % replay_server.port


# Create a 5-step transition adder where in between those steps a discount of
# 0.99 is used (which should be the same discount used for learning).
adder = adders.NStepTransitionAdder(
    client=reverb.Client(replay_server_address),
    n_step=5,
    discount=0.99)

num_episodes = 2  #@param

for episode in range(num_episodes):
  timestep = environment.reset()
  adder.add_first(timestep)

  while not timestep.last():
    action = actor.select_action(timestep.observation)
    timestep = environment.step(action)
    adder.add(action=action, next_timestep=timestep)

actor = actors.FeedForwardActor(policy_network=behavior_network, adder=adder)

num_episodes = 2  #@param

for episode in range(num_episodes):
  timestep = environment.reset()
  actor.observe_first(timestep)  # Note: observe_first.

  while not timestep.last():
    action = actor.select_action(timestep.observation)
    timestep = environment.step(action)
    actor.observe(action=action, next_timestep=timestep)  # Note: observe.

# This connects to the created reverb server; also note that we use a transition
# adder above so we'll tell the dataset function that so that it knows the type
# of data that's coming out.
dataset = datasets.make_dataset(
    server_address=replay_server_address,
    batch_size=256,
    environment_spec=environment_spec,
    transition_adder=True)

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
                           dataset=dataset,
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