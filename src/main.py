from argparse import ArgumentParser
import time
import tensorflow
import acme
from acme.agents.tf import actors as actors
from acme.agents.tf import dqn
from acme.agents.jax import impala
from acme.tf import networks as net
from MyEnvLoop import MyEnvironmentLoop as MyLoop

from utils.server import collectExperience, collectExperienceRandom
from utils.display import saveVideo
from utils.environment import createEnv

import pathlib
import os

FLAGS = {
    'env_name' : 'Assault-v0',
}

CSV_FOLDER = "./csv/"
VIDEO_FOLDER = "./video/"
LOGS_FOLDER = "./logs/"

def createAgent(envSpec, args):
    batchs = args.batch_size
    if args.algType == 'dqn':
        if(batchs is None):
            batchs = 128
        network = net.DQNAtariNetwork(envSpec.actions.num_values)
        agent = dqn.DQN(
            envSpec,
            network,
            batch_size = batchs,
            samples_per_insert = 16,
            checkpoint = False,
            min_replay_size = 100,
            max_replay_size = 500,
            learning_rate = args.lr,
            discount = args.discount,
            target_update_period = args.targetUpdatePeriod
        )
    elif args.algType == 'impala':
        if(batchs is None):
            batchs = 16
        config = impala.IMPALAConfig(
            batch_size = batchs,
            sequence_period = 5,
            seed = 111,
            learning_rate = args.lr,
            discount = args.discount,
            entropy_cost = args.entropyCost
        )

        networks = impala.make_atari_networks(envSpec)

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

def createFolders(args):
    pathlib.Path(os.path.join(VIDEO_FOLDER, args.algType)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(CSV_FOLDER, args.algType)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(LOGS_FOLDER).mkdir(parents=True, exist_ok=True)

def generateVideoName(args):
    fname = VIDEO_FOLDER + args.algType + '/'

    if(args.videoName is not None):
        return fname + args.videoName + ".mp4"

    fname += str(time.time()) + "_" + \
            args.algType + "_" + \
            str(args.lr) + "_" + \
            str(args.discount) + "_"

    if args.algType == 'dqn':
        fname += str(args.targetUpdatePeriod)
    else:
        fname += str(args.entropyCost)

    return fname + ".mp4"


def generateCsvName(args):
    fname = CSV_FOLDER + args.algType + '/' + \
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

    fname = generateCsvName(args)

    if(args.algType == 'random'):
        frames = collectExperienceRandom(env = env, fname = fname, numSteps = args.collect_frames, saveCsv = args.saveCsv)
    else:
        agent = createAgent(envSpec, args)

        loop = MyLoop(env, agent, args.max_steps)
        loop._logger._to._to._to[1]._flush_every = 1
        loop.run(num_episodes = args.numEpisodes)

        if args.algType == 'impala':
            server = agent._server
            address = f'localhost:{server.port}'
            frames = collectExperience(env = env, agent = agent, agentType = "impala",
                                        fname = fname, numSteps = args.collect_frames, saveCsv = args.saveCsv)
        elif args.algType == 'dqn':
            frames = collectExperience(env = env, agent = agent, agentType = "dqn",
                                        fname = fname, numSteps = args.collect_frames, saveCsv = args.saveCsv)
        else:
            raise Exception("Unknown algorithm type")

    if args.saveVideo:
        videoName = generateVideoName(args)
        saveVideo(frames, videoName)


def main():
    parser = ArgumentParser()
    parser.add_argument('--num_episodes', type = int, required = False, default = 100, dest = 'numEpisodes',
                        help = 'Number of training episodes')
    parser.add_argument('--gpu', type = int, required = False, default = 1, choices = [0, 1], dest = 'gpu',
                        help = 'Enable GPU')
    parser.add_argument('--alg', type = str, required = True, dest = 'algType', choices = ['dqn', 'impala', 'random'],
                        help = 'Type of algorithm (dqn/impala)')
    parser.add_argument('--save_video', type = int, required = False, default = 0, choices = [0, 1], dest = 'saveVideo',
                        help = 'Save video from model evaluation')
    parser.add_argument('--video_name', type = str, required = False, dest = 'videoName',
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
    parser.add_argument('--max_steps', type = int, required = False, default = 100000,
                        help = 'Maximum number of steps in any network.')
    parser.add_argument('--batch_size', type = int, required = False, default = None,
                        help = 'Batch size. By default for DQN - 128; IMPALA - 16')
    parser.add_argument('--collect_frames', type = int, required = False, default = 20000,
                        help = 'Collect frames for provided number of iterations. It is used outside train loop.')
    args = parser.parse_args()

    createFolders(args)

    text = '\n\nAttributes:\n'
    text += f'num_episodes: {args.numEpisodes}, gpu: {args.gpu}, alg: {args.algType}, save_video: {args.saveVideo}, ' + \
            f'video_name: {args.videoName}, save_csv: {args.saveCsv}, lr: {args.lr}, discount: {args.discount}, '
    if args.algType == 'dqn':
        text += f'target_update_period: {args.targetUpdatePeriod}\n\n'
    else:
        text += f'entropy_cost: {args.entropyCost}\n\n'
    print(text)

    if not args.gpu:
        tensorflow.config.set_visible_devices([], 'GPU')

    execute(args)


if __name__ == '__main__':
    exit(main())
