import reverb
from acme.adders import reverb as adders
from acme.datasets import reverb as datasets
import numpy as np


def createServer(envSpec):
    replayBuffer = reverb.Table(
        name = adders.DEFAULT_PRIORITY_TABLE,
        max_size = 1000,
        remover = reverb.selectors.Fifo(),
        sampler = reverb.selectors.Uniform(),
        rate_limiter = reverb.rate_limiters.MinSize(min_size_to_sample = 1),
        signature = adders.NStepTransitionAdder.signature(envSpec)
    )

    replayServer = reverb.Server([replayBuffer], port = None)
    replayServerAddress = f'localhost:{replayServer.port}'
    return replayServer, replayServerAddress


def createExperienceBuffer(serverAddress):
    adder = adders.NStepTransitionAdder(
        client = reverb.Client(serverAddress),
        n_step = 5,
        discount = 0.99
    )
    return adder


def collectExperience(env, agent, numSteps = 500):
    '''
    for episode in range(numEpisodes):
        timestep = env.reset()
        expBuffer.add_first(timestep)

        while not timestep.last():
            action = actor.select_action(timestep.observation)
            timestep = env.step(action)
            expBuffer.add(action = action, next_timestep = timestep)
    '''
    frames = []
    timestep = env.reset()

    for _ in range(numSteps):
        frames.append(env.environment.render(mode='rgb_array'))
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)
    return np.array(frames)


def createReporter(serverAddress):
    dataset = iter(
        datasets.make_dataset(
            server_address = serverAddress,
            batch_size = 64,
            transition_adder = True,
            prefetch_size = 4
        )
    )
    return dataset
