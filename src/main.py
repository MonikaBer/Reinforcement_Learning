from argparse import ArgumentParser
import acme
from acme.agents.tf import actors as actors
from acme.agents.tf import dqn

# own modules
from algorithms.dqn import createEnvForDQN, MyDQNAtariNetwork
from utils.server import createServer, createExperienceBuffer, collectExperience
from utils.display import collectFrames, saveVideo

FLAGS = {
    'env_name' : 'Assault-v4',
}


def main():
    parser = ArgumentParser()
    parser.add_argument('--num_episodes', type = int, required = True, dest = 'numEpisodes',
                        help = 'Number of training episodes')
    args = parser.parse_args()

    env = createEnvForDQN(FLAGS['env_name'])
    envSpec = acme.make_environment_spec(env)
    #print(envSpec)

    network = MyDQNAtariNetwork(envSpec.actions.num_values)
    agent = dqn.DQN(envSpec, network)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(args.numEpisodes)

    server, address = createServer(envSpec)
    buffer = createExperienceBuffer(address)
    collectExperience(env, buffer, agent, 4)
    #saveVideo(buffer)

    frames = collectFrames(env, agent)
    saveVideo(frames)


if __name__ == '__main__':
    exit(main())
