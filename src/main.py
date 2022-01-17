from argparse import ArgumentParser
import acme
from acme.agents.tf import actors as actors
from acme import agents

# own modules
import utils, algorithms

FLAGS = {
    'env_name' : 'Assault-v4',
}


def main():
    parser = ArgumentParser()
    parser.add_argument('--num_episodes', type = int, required = True, dest = 'numEpisodes',
                        help = 'Number of training episodes')
    args = parser.parse_args()

    env = algorithms.dqn.createEnvForDQN(FLAGS['env_name'])
    envSpec = acme.make_environment_spec(env)
    #print(envSpec)

    network = algorithms.dqn.MyDQNAtariNetwork(envSpec.actions.num_values)
    agent = agents.dqn.DQN(envSpec, network)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(args.numEpisodes)

    server, address = utils.server.createServer(envSpec)
    buffer = utils.server.createExperienceBuffer(address)
    utils.server.collectExperience(env, buffer, agent, 4)
    #utils.display.saveVideo(buffer)

    frames = utils.display.collectFrames(env, agent)
    utils.display.saveVideo(frames)


if __name__ == '__main__':
    exit(main())
