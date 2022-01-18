import reverb
from acme.adders import reverb as adders
from acme.datasets import reverb as datasets
import numpy as np
import pandas as pd


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

def dfapend(olddframe, action, timestep, idx, envReset):
    newdframe = pd.DataFrame({
        'akcja': str(action),
        'nagroda': str(0.0 if timestep.reward is None else timestep.reward),
        'reset': str(int(envReset))
    }, index = [idx])
    return olddframe.append(newdframe)

def collectExperience(env, agent, fname, numSteps = 500, saveCsv = False):
    frames = []
    timestep = env.reset()

    dframe = pd.DataFrame(columns = ['akcja', 'nagroda', 'reset'])

    for i in range(numSteps):
        frames.append(env.environment.render(mode = 'rgb_array'))
        action = agent.select_action(timestep.observation)

        timestep = env.step(action)
        if(timestep.observation.reward is None):
            dframe = dfapend(dframe, action=action, timestep=timestep, idx=i, envReset=True)
            timestep = env.reset()
        else:
            dframe = dfapend(dframe, action=action, timestep=timestep, idx=i, envReset=False)

    if saveCsv:
        dframe.to_csv(fname, sep = ';', index=False)
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
