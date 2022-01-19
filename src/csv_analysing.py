import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class Experiment:
    def __init__(self, file, steps, algType):
        self.filepath = file
        self.filename = file.name
        self.name = (' '.join(self.filename.split('_')[1:])).rstrip('.csv')
        self.steps = steps
        self.algType = algType
        self.getStats()

    def getStats(self):
        self.maxRewardsSums = []
        self.episodesLen = []
        episodeLen = 0
        for step in self.steps:
            episodeLen += 1
            if step.reset:
                self.maxRewardsSums.append(step.rewardsSum)
                self.episodesLen.append(episodeLen)
                episodeLen = 0


        # number of episodes
        self.episodesNr = len(self.episodesLen)

        if len(self.maxRewardsSums) != self.episodesNr:
            raise RuntimeError('number of maxRewardsSums and number of episodes differ!')

        # mean, min, max of max rewards sums
        sum = 0.0
        min = -1.0
        max = -1.0
        for i in self.maxRewardsSums:
            sum += i
            if min == -1 or i < min:
                min = i
            if max == -1 or i > max:
                max = i
        self.maxRewardsSumMean = round(sum / float(len(self.maxRewardsSums)), 0)
        self.maxRewardsSumMin = min
        self.maxRewardsSumMax = max

        # mean, min, max of episode lengths
        sum = 0.0
        min = -1.0
        max = -1.0
        for i in self.episodesLen:
            sum += i
            if min == -1 or i < min:
                min = i
            if max == -1 or i > max:
                max = i
        self.episodesLenMean = round(sum / float(len(self.episodesLen)), 0)
        self.episodesLenMin = min
        self.episodesLenMax = max

    def plotRewardsSumAndGetData(self):
        x = []
        y = []
        for step in self.steps:
            x.append(step.id)
            y.append(step.rewardsSum)

        xpoints = np.array(x)
        ypoints = np.array(y)
        plt.xlim(min(x),max(x))
        plt.ylim(min(y),max(y))
        plt.xlabel('krok')
        plt.ylabel('suma nagród')
        plt.title('Ewaluacja modelu')

        plt.plot(xpoints, ypoints)
        plt.savefig(f'plots/{self.algType}/rewards_sum/' + self.filename + '.png')
        plt.clf()
        return (self.name, x, y)

    def plotEpisodesLenAndGetData(self):
        x = []
        y = []
        count = 0
        id = 0
        for episodeLen in self.episodesLen:
            for i in range(episodeLen):
                id += 1
                x.append(id)
                count += 1
                y.append(count)
            count = 0

        xpoints = np.array(x)
        ypoints = np.array(y)
        plt.xlim(min(x),max(x))
        plt.ylim(min(y),max(y))
        plt.xlabel('krok')
        plt.ylabel('długość epizodu')
        plt.title('Ewaluacja modelu')

        plt.plot(xpoints, ypoints)
        plt.savefig(f'plots/{self.algType}/episodes_len/' + self.filename + '.png')
        plt.clf()
        return (self.name, x, y)


class Step:
    def __init__(self, id, action, reward, rewardsSum, reset):
        self.id = id
        self.action = action
        self.reward = reward
        self.rewardsSum = rewardsSum
        self.reset = reset


def getAllSteps(filePath):
    steps = []
    id = -1
    with open(filePath) as f:
        for line in f:
            id += 1
            if id == 0:
                continue

            splitted = line.strip('\n').split(';')
            action = int(splitted[0])
            reward = float(splitted[1])
            rewardsSum = float(splitted[2])
            reset = int(splitted[3])
            steps.append(Step(id, action, reward, rewardsSum, reset))
    return steps


def plotRewardsSum(exps, algType):
    data = dict()
    for exp in exps:
        (id, x, y) = exp.plotRewardsSumAndGetData()
        data[id] = (x, y)

    plt.figure(figsize = (20, 15))

    for key, value in data.items():
        plt.plot(value[0], value[1], label = f"exp {key}")

    plt.xlabel('krok')
    plt.ylabel('suma nagród')
    plt.xlim(min(x),max(x))
    plt.title('Ewaluacja modeli')
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.savefig(f'plots/{algType}/rewards_sum/all.png', dpi = 100, bbox_inches = 'tight')
    plt.clf()
    return data


def plotEpisodesLen(exps, algType):
    data = dict()
    for exp in exps:
        (id, x, y) = exp.plotEpisodesLenAndGetData()
        data[id] = (x, y)

    plt.figure(figsize = (20, 15))

    for key, value in data.items():
        plt.plot(value[0], value[1], label = f"exp {key}")

    plt.xlabel('krok')
    plt.ylabel('długość epizodu')
    plt.xlim(min(x),max(x))
    plt.title('Ewaluacja modeli')
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.savefig(f'plots/{algType}/episodes_len/all.png', dpi = 100, bbox_inches = 'tight')
    plt.clf()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str, required = True,
                        help = 'Path to directory with csv files for analysing')
    parser.add_argument('--alg', type = str, required = True, choices = ['dqn', 'impala'], dest = 'algType',
                        help = 'Algorithm name (dqn/impala)')
    args = parser.parse_args()

    # create paths if not exist
    Path("plots/impala/rewards_sum").mkdir(parents = True, exist_ok = True)
    Path("plots/dqn/rewards_sum").mkdir(parents = True, exist_ok = True)
    Path("plots/impala/episodes_len").mkdir(parents = True, exist_ok = True)
    Path("plots/dqn/episodes_len").mkdir(parents = True, exist_ok = True)

    # load exps results from csv files
    exps = []
    for file in Path(args.path).iterdir():
        if file.is_file():
            #print(file)

            steps = getAllSteps(file)
            exps.append(Experiment(file, steps, args.algType))

    # print statistics
    print('\n\tSTATISTICS\n\n')
    print('Legend:\n')
    print('exp - exp name')
    print('enr - episodes number')
    print('emin - min episode len')
    print('emax - max episode len')
    print('em - mean episode len')
    print('rmax - max result')
    print('rmin - min result')
    print('rm - mean result\n')

    for exp in exps:
        print(f'exp:{exp.name}, enr:{exp.episodesNr}, emin:{exp.episodesLenMin}, emax:{exp.episodesLenMax}, em:{exp.episodesLenMean}, rmax:{exp.maxRewardsSumMax}, rmin:{exp.maxRewardsSumMin}, rm:{exp.maxRewardsSumMean}\n')

    # plot statistics
    plotRewardsSum(exps, args.algType)
    plotEpisodesLen(exps, args.algType)


if __name__ == '__main__':
    exit(main())
