from argparse import ArgumentParser
import acme
from acme.agents.tf import actors as actors
from acme.agents.tf import dqn
from acme.agents.jax import impala
from acme.tf import networks as net

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
            sequence_period = 10,
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


def execute(args):
    env = createEnv(FLAGS['env_name'], args.algType)
    envSpec = acme.make_environment_spec(env)
    #print(envSpec)

    agent = createAgent(envSpec, args.algType)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(args.numEpisodes)

    if args.algType == 'impala':
        #server, address = createServer(envSpec)
        server = agent._server
        address = f'localhost:{server.port}'

        buffer = createExperienceBuffer(address)
        collectExperience(env, buffer, agent, 4)
        saveVideo(buffer)
    elif(args.algType == 'dqn'):
        frames = collectFrames(env, agent)
        saveVideo(frames, args.videoName)
    else:
        raise Exception("Unknown argument.")

def main():
    parser = ArgumentParser()
    parser.add_argument('--num_episodes', type = int, required = True, default = 500, dest = 'numEpisodes',
                        help = 'Number of training episodes')
    parser.add_argument('--alg', type = str, required = True, dest = 'algType', choices=['dqn', 'impala'],
                        help = 'Type of algorithm (dqn/impala)')
    parser.add_argument('--video_name', type = str, required = False, dest = 'videoName',
                        help = 'Filename for video saving')
    args = parser.parse_args()

    execute(args)


if __name__ == '__main__':
    exit(main())
