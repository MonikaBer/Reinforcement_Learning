from argparse import ArgumentParser
import time
import tensorflow
import acme
from acme.agents.tf import actors as actors
from acme.agents.tf import dqn
from acme.agents.jax import impala
from acme.tf import networks as net
from MyEnvLoop import MyEnvironmentLoop as MyLoop

# own modules
#from algorithms.dqn import MyDQNAtariNetwork
#from algorithms.impala import MyImpalaAtariNetwork
from utils.server import collectExperience
from utils.display import saveVideo
from utils.environment import createEnv

FLAGS = {
    'env_name' : 'Assault-v4',  # v0 vs v4 ???
}


def createAgent(envSpec, args):
    if args.algType == 'dqn':
        #network = MyDQNAtariNetwork(envSpec.actions.num_values)
        network = net.DQNAtariNetwork(envSpec.actions.num_values)
        agent = dqn.DQN(
            envSpec,
            network,
            batch_size = 128,
            samples_per_insert = 16,
            checkpoint = False,
            min_replay_size = 100,
            max_replay_size = 500,
            learning_rate = args.lr,
            discount = args.discount,
            target_update_period = args.targetUpdatePeriod
        )
    elif args.algType == 'impala':
        config = impala.IMPALAConfig(
            batch_size = 16,
            sequence_period = 5,
            seed = 111,
            learning_rate = args.lr,
            discount = args.discount,
            entropy_cost = args.entropyCost
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

        serv = agent._server
    else:
        raise Exception("Unknown algorithm type")

    return agent


def generateName(args):
    fname = "csv/" + \
            str(time.time()) + "_" + \
            args.algType + "_" + \
            str(args.lr) + "_" + \
            str(args.discount) + "_"

    if args.algType == 'dqn':
        fname += str(args.targetUpdatePeriod)
    else:
        fname += str(args.entropyCost)

    return fname + ".csv"


def execute(args):
    env = createEnv(FLAGS['env_name'], args.algType)
    envSpec = acme.make_environment_spec(env)
    #print(envSpec)

    agent = createAgent(envSpec, args)

    loop = MyLoop(env, agent, 45000)
    loop._logger._to._to._to[1]._flush_every = 1
    loop.run(num_episodes = args.numEpisodes)
    fname = generateName(args)

    if args.algType == 'impala':
        #server, address = createServer(envSpec)
        server = agent._server
        address = f'localhost:{server.port}'
        #buffer = createExperienceBuffer(address)
        frames = collectExperience(env = env, agent = agent, agentType = "impala",
                                    fname = fname, numSteps = 7500, saveCsv = args.saveCsv)
    elif args.algType == 'dqn':
        frames = collectExperience(env = env, agent = agent, agentType = "dqn",
                                    fname = fname, numSteps = 7500, saveCsv = args.saveCsv)
    else:
        raise Exception("Unknown algorithm type")

    if args.saveVideo:
        saveVideo(frames, args.videoName)


def main():
    parser = ArgumentParser()
    parser.add_argument('--num_episodes', type = int, required = False, default = 100, dest = 'numEpisodes',
                        help = 'Number of training episodes')
    parser.add_argument('--gpu', type = int, required = False, default = 1, choices = [0, 1], dest = 'gpu',
                        help = 'Enable GPU')
    parser.add_argument('--alg', type = str, required = True, dest = 'algType', choices = ['dqn', 'impala'],
                        help = 'Type of algorithm (dqn/impala)')
    parser.add_argument('--save_video', type = int, required = False, default = 0, choices = [0, 1], dest = 'saveVideo',
                        help = 'Save video from model evaluation')
    parser.add_argument('--video_name', type = str, required = False, default = 'temp.mp4', dest = 'videoName',
                        help = 'Filename for video saving')
    parser.add_argument('--save_csv', type = int, required = False, default = 0, choices = [0, 1], dest = 'saveCsv',
                        help = 'Save csv from model evaluation')
    parser.add_argument('--lr', type = float, required = False, default = 1e-3, dest = 'lr',
                        help = 'Learning rate')
    parser.add_argument('--discount', type = float, required = False, default = 0.99, dest = 'discount',
                        help = 'Discount')
    parser.add_argument('--target_update_period', type = int, required = False, default = 100, dest = 'targetUpdatePeriod',
                        help = 'Target update period (for DQN)')
    parser.add_argument('--entropy_cost', type = float, required = False, default = 0.01, dest = 'entropyCost',
                        help = 'Entropy cost (for IMPALA)')
    args = parser.parse_args()

    if not args.gpu:
        tensorflow.config.set_visible_devices([], 'GPU')

    execute(args)


if __name__ == '__main__':
    exit(main())
