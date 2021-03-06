# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop."""

import operator
import time
from typing import Optional

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import signals

import dm_env
from dm_env import specs
import numpy as np
import tree
import copy

class CircularList():
    """
        Cykliczny bufor / lista. Może przechowywać ona wartości lub obiekty.
        W przypadku obiektów, niektóre metody mogą nie zadziałać. 
    """
    class CircularListIter():
        """
            Iterator dla cyklicznej listy. Dzięki takiej implementacji można iterować po 
            cyklicznej liście jednoczeście.
            Iteracja następuje od najstarszej wartości do najnowszej.
        """
        def __init__(self, circularList):
            self.circularList = circularList
            self.__iter__()

        def __iter__(self):
            lo = list(range(self.circularList.arrayIndex))
            hi = list(range(self.circularList.arrayIndex, len(self.circularList.array)))
            lo.reverse()
            hi.reverse()
            self.indexArray = lo + hi
            return self

        def __next__(self):
            if(self.indexArray):
                idx = self.indexArray.pop(0)
                return self.circularList.array[idx]
            else:
                raise StopIteration


    def __init__(self, maxCapacity):
        self.array = []
        self.arrayIndex = 0
        self.arrayMax = maxCapacity

    def pushBack(self, value):
        """
            Dodaje na koniec listy cyklicznej wartość lub obiekt. 
            Wstawiana wartość posiada największy indeks.
        """
        if(self.arrayIndex < len(self.array)):
            del self.array[self.arrayIndex] # trzeba usunąć, inaczej insert zachowa w liście obiekt
        self.array.insert(self.arrayIndex, value)
        self.arrayIndex = (self.arrayIndex + 1) % self.arrayMax

    def getAverage(self, startAt=0):
        """
            Zwraca średnią.
            Argument startAt mówi o tym, od którego momentu w kolejce należy liczyć średnią.
            Najstarsza wartość ma indeks 0.

            Można jej użyć tylko do typów, które wspierają dodawanie, które
            powinny implementować metody __copy__(self) oraz __deepcopy__(self)
        """
        l = len(self.array)
        if(startAt == 0):
            return sum(self.array) / l if l else 0
        if(l <= startAt):
            return 0
        l -= startAt
        tmpSum = None

        for i, (obj) in enumerate(iter(self)):
            if(i < startAt):
                continue
            tmpSum = copy.deepcopy(obj) # because of unknown type
            break

        for i, (obj) in enumerate(iter(self)):
            if(i < startAt + 1):
                continue
            tmpSum += obj

        return tmpSum / l

    def getStdMean(self):
        """
            Zwraca tuple(std, mean) z całego cyklicznego bufora.
        """
        return self.getStd(), self.getMean()

    def getMean(self):
        return numpy.mean(self.array)

    def getStd(self):
        return numpy.std(self.array)

    def __setstate__(self):
        self.__dict__.update(state)
        self.arrayIndex = self.arrayIndex % self.arrayMax

    def reset(self):
        """
            Usuwa cały cykliczny bufor, zastępując go nowym.
        """
        del self.array
        self.array = []
        self.arrayIndex = 0

    def __iter__(self):
        return CircularList.CircularListIter(self)

    def __len__(self):
        return len(self.array)

    def get(self, idx):
        """
            Najstarsza wartość ma indeks 0.
        """
        return self.array[(self.arrayIndex + idx) % self.arrayMax]

    def getMin(self):
        """
            Zwraca minimalną wartość / obiekt z całego bufora cyklicznego.
        """
        return min(self.array)
    
    def getMax(self):
        """
            Zwraca maksymalną wartość / obiekt z całego bufora cyklicznego.
        """
        return max(self.array)


class MyEnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      maxSteps: int,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
      bufferCap = 6
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)
    self._should_update = should_update
    self._maxSteps = maxSteps
    self._allSteps = 0
    self._circularBuffer = CircularList(bufferCap)

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    timestep = self._environment.reset()

    # Make the first observation.
    self._actor.observe_first(timestep)

    # Run an episode.
    while not timestep.last():
      if self._allSteps >= self._maxSteps:
          break
      # Generate an action from the agent's policy and step the environment.
      action = self._actor.select_action(timestep.observation)

      timestep = self._environment.step(action)
      timestep = timestep._replace(
        reward = convertReward(timestep.reward, self._circularBuffer, action)
      )

      # Have the agent observe the timestep and let the actor update itself.
      self._actor.observe(action, next_timestep=timestep)
      if self._should_update:
        self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)
      self._allSteps += 1

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }
    result.update(counts)
    return result

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count, step_count = 0, 0
    with signals.runtime_terminator():
      while (not should_terminate(episode_count, step_count) ) and self._allSteps < self._maxSteps:
        result = self.run_episode()
        episode_count += 1
        step_count += result['episode_length']
        # Log the given episode results.
        self._logger.write(result)


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)

def isActionMainShoot(action):
  return action == 2

def isActionSideShoot(action):
  return action == 5 or action == 6

def customReward(bufferAction: CircularList, reward, action):
  if(reward is None):
    return np.array(-100.0, dtype=np.float32)
  tmp = 0.6 * float(isActionMainShoot(action)) + float(isActionSideShoot(action))
  bufferAction.pushBack(tmp)
  ret = reward - bufferAction.getAverage()
  return np.array(ret, dtype=np.float32)

def convertReward(nestedValueReward, bufferAction, action):
  def _convertSingleValueReward(bufferAction: CircularList, reward, action):
    return customReward(bufferAction, reward, action)
  return tree.map_structure(_convertSingleValueReward, bufferAction, nestedValueReward, action)