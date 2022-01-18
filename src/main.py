from argparse import ArgumentParser
import acme
from acme.agents.tf import actors as actors
from acme.agents.tf import dqn
from acme.agents.jax import impala
from acme.tf import networks as net
import time
import pandas as pd

# own modules
from algorithms.dqn import MyDQNAtariNetwork
#from algorithms.impala import MyImpalaAtariNetwork
from utils.server import createServer, createExperienceBuffer, collectExperience
from utils.display import collectFrames, saveVideo
from utils.environment import createEnv

FLAGS = {
    'env_name' : 'Assault-v4',  # v0 vs v4 ???
}


def createAgent(envSpec, algType):
    if algType == 'dqn':
        #network = MyDQNAtariNetwork(envSpec.actions.num_values)
        network = net.DQNAtariNetwork(envSpec.actions.num_values)
        agent = dqn.DQN(envSpec, network)
    elif (algType == 'impala'):
        config = impala.IMPALAConfig(
            batch_size = 16,
            sequence_period = 5,
            seed = 111,
        )

        networks = impala.make_atari_networks(envSpec)
        #networks = MyImpalaAtariNetwork(envSpec)

        agent = impala.IMPALAFromConfig(
            environment_spec = envSpec,
            forward_fn = networks.forward_fn,
            unroll_init_fn = networks.unroll_init_fn,
            unroll_fn = networks.unroll_fn,
            initial_state_init_fn = networks.initial_state_init_fn,
            initial_state_fn = networks.initial_state_fn,
            config = config,
        )

        ser = agent._server
    else:
        raise Exception("Unknown parameter.")

    return agent

def generateName(args):
    return str(args.algType) + "_" + str(args.numEpisodes) + "_" + str(time.time()) + ".csv"

def execute(args):
    env = createEnv(FLAGS['env_name'], args.algType)
    envSpec = acme.make_environment_spec(env)
    #print(envSpec)

    agent = createAgent(envSpec, args.algType)

    loop = acme.EnvironmentLoop(env, agent)
    #loop.run(num_steps=60000)
    loop.run(num_steps=1)
    fname = generateName(args)

    if args.algType == 'impala':
        #server, address = createServer(envSpec)
        server = agent._server
        address = f'localhost:{server.port}'

        #buffer = createExperienceBuffer(address)

        #frames = collectExperience(env, agent, args, 7500)
        frames = collectExperience(env, agent, fname, 10)
        saveVideo(frames, args.videoName)
    elif(args.algType == 'dqn'):
        frames = collectFrames(env, agent, fname, 7500)
        saveVideo(frames, args.videoName)
    else:
        raise Exception("Unknown argument.")

def main():
    parser = ArgumentParser()
    parser.add_argument('--num_episodes', type = int, required = True, default = 500, dest = 'numEpisodes',
                        help = 'Number of training episodes')
    parser.add_argument('--alg', type = str, required = True, dest = 'algType', choices=['dqn', 'impala'],
                        help = 'Type of algorithm (dqn/impala)')
    parser.add_argument('--lr', type = float, required = False, choices=[0.001, 0.0001],
                        help = 'Learning rate')
    parser.add_argument('--disc', type = float, required = False, choices=[0.99, 0.95, 0.8],
                        help = 'Discount')
    parser.add_argument('--tupdateper', type = int, required = False, choices=[75, 400],
                        help = 'Target update period')
    parser.add_argument('--entropycost', type = int, required = False, choices=[0.01, 0.1],
                        help = 'Entropy cost')
    parser.add_argument('--maxabsr', type = float, required = False, default=None,
                        help = 'Max absolute reward. None == np.inf')
    parser.add_argument('--video_name', type = str, required = True, dest = 'videoName',
                        help = 'Filename for video saving')
    args = parser.parse_args()

    execute(args)


if __name__ == '__main__':
    exit(main())