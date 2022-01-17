import reverb
from acme.adders import reverb as adders
from acme.datasets import reverb as datasets


def createServer(envSpec):
    replayBuffer = reverb.Table(
        name = adders.DEFAULT_PRIORITY_TABLE,
        max_size = 5000,
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


def collectExperience(env, expBuffer, actor, numEpisodes = 2):
    for episode in range(numEpisodes):
        timestep = env.reset()
        expBuffer.add_first(timestep)

        while not timestep.last():
            action = actor.select_action(timestep.observation)
            timestep = env.step(action)
            expBuffer.add(action = action, next_timestep = timestep)


def createReporter(serverAddress):
    dataset = iter(
        datasets.make_dataset(
            server_address = serverAddress,
            batch_size = 256,
            transition_adder = True,
            prefetch_size = 4
        )
    )
    return dataset
