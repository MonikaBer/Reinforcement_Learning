
# https://towardsdatascience.com/deepminds-reinforcement-learning-framework-acme-87934fa223bf

# reinforcement learning
import acme
from acme import types
from acme.wrappers import gym_wrapper
from acme.environment_loop import EnvironmentLoop
from acme.utils.loggers import TerminalLogger, InMemoryLogger

# environments
import gym
import dm_env

# other
import numpy as np
import pandas as pd


env = acme.wrappers.GymWrapper(gym.make('Blackjack-v1'))
# env = acme.wrappers.SinglePrecisionWrapper(env)

# print env specs
env_specs = env.observation_space, env.action_space, env.reward_range # env.observation_spec()
print('Observation Spec:', env.observation_space)
print('Action Spec:', env.action_space)
print('Reward Spec:', env.reward_range)


# show a timestep
env.reset()


class RandomAgent(acme.Actor):
    """A random agent for the Black Jack environment."""
    
    def __init__(self):
        
        # init action values, will not be updated by random agent
        self.Q = np.zeros((32,11,2,2))
        
        # specify the behavior policy
        self.behavior_policy = lambda q_values: np.random.choice(2)
        
        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None
        
    def select_action(self, observation):
        "Choose an action according to the behavior policy."
        return self.behavior_policy(self.Q[observation])    

    def observe_first(self, timestep):
        "Observe the first timestep." 
        self.timestep = timestep

    def observe(self, action, next_timestep):
        "Observe the next timestep."
        self.action = action
        self.next_timestep = next_timestep
        
    def update(self, wait = False):
        "Update the policy."
        # no updates occur here, it's just a random policy
        self.timestep = self.next_timestep

agent = RandomAgent()

# make first observation
timestep = env.reset()
agent.observe_first(timestep)

# run an episode
while not timestep.last():
    
    # generate an action from the agent's policy and step the environment
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)

    # have the agent observe the timestep and let the agent update itself
    agent.observe(action, next_timestep=timestep)
    agent.update()

# or use Acme training loop
loop = EnvironmentLoop(env, agent, logger=InMemoryLogger())
loop.run_episode()
loop.run(100)



# uniform random policy
def random_policy(q_values):
    return np.random.choice(len(q_values))

# greedy policy
def greedy(q_values):
    return np.argmax(q_values)

# epsilon greedy policy
def epsilon_greedy(q_values, epsilon):
    if epsilon < np.random.random():
        return np.argmax(q_values)
    else:
        return np.random.choice(len(q_values))


class SarsaAgent(acme.Actor):
    
    def __init__(self, env_specs=None, epsilon=0.1, step_size=0.1):
        
        
        # setting initial action values
        self.Q = np.zeros((32,11,2,2))
        
        # epsilon for policy and step_size for TD learning
        self.epsilon = epsilon
        self.step_size = step_size
        
        # set behavior policy
        # self.policy = None
        self.behavior = lambda q_values: epsilon_greedy(q_values, self.epsilon)
        
        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None

    def transform_state(self, state):
        # this is specifally required for the blackjack environment
        state = *map(int, state),
        return state
    
    def select_action(self, observation):
        state = self.transform_state(observation)
        return self.behavior(self.Q[state])

    def observe_first(self, timestep):
        self.timestep = timestep

    def observe(self, action, next_timestep):
        self.action = action
        self.next_timestep = next_timestep
        
    def update(self):
        
        # get variables for convenience
        state = self.timestep.observation
        _, reward, discount, next_state = self.next_timestep
        action = self.action
        
        # turn states into indices
        state = self.transform_state(state)
        next_state = self.transform_state(next_state)
        
        # sample a next action
        next_action = self.behavior(self.Q[next_state])

        # compute and apply the TD error
        td_error = reward + discount * self.Q[next_state][next_action] - self.Q[state][self.action]
        self.Q[state][action] += self.step_size * td_error
        
        # finally, set timestep to next_timestep
        self.timestep = self.next_timestep
        
sarsa = SarsaAgent()



loop = EnvironmentLoop(env, sarsa, logger=InMemoryLogger())
loop.run(50000)

df = pd.DataFrame(loop._logger._data)
print(df.head())
print(df.tail())