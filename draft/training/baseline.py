# import argparse
# import glob
# import itertools
# import os

# import numpy as np

# from draft import stats
# from draft.training import data_ingestion


# class TopSynergyBaseline:
#     def __init__(self, stats_):
#         self.stats = stats_

#     def predict(self, x):
#         max_radiant = max(
#             self.stats.Synergy(h1, h2)
#             for h1, h2 in itertools.combinations(x[:5], 2)
#         )
#         max_dire = max(
#             self.stats.Synergy(h1, h2)
#             for h1, h2 in itertools.combinations(x[5:], 2)
#         )
#         return int(max_radiant > max_dire)


# class TopMatchupBaseline:
#     def __init__(self, stats_):
#         self.stats = stats_

#     def predict(self, x):
#         advs = [
#             self.stats.Advantage(h1, h2)
#             for h1, h2 in itertools.product(x[:5], x[5:])
#         ]
#         return int(max(advs) > (1 - min(advs)))


# class TeamSynergyBaseline:
#     def __init__(self, stats_):
#         self.stats = stats_

#     def predict(self, x):
#         radiant_synergies = self.team_synergy(x[:5])
#         dire_synergies = self.team_synergy(x[5:])
#         return int(radiant_synergies > dire_synergies)

#     def team_synergy(self, team):
#         max_synergies = np.zeros(5)
#         for i in range(5):
#             for j in range(i + 1, 5):
#                 synergy = self.stats.Synergy(team[i], team[j])
#                 max_synergies[i] = max(max_synergies[i], synergy)
#                 max_synergies[j] = max(max_synergies[j], synergy)
#         return np.mean(max_synergies)

# class TeamMatchupsBaseline:
#     def __init__(self, stats_):
#         self.stats = stats_

#     def predict(self, x):
#         radiant_matchups, dire_matchups = self.matchups(x[:5], x[5:])
#         return int(radiant_matchups.mean() > dire_matchups.mean())

#     def matchups(self, team, opponent):
#         team_best_matchups = np.zeros(5)
#         team_worst_matchups = np.ones(5)
#         opponent_best_matchups = np.zeros(5)
#         opponent_worst_matchups = np.ones(5)
#         for i in range(5):
#             for j in range(5):
#                 adv = self.stats.Advantage(team[i], opponent[j])
#                 team_best_matchups[i] = max(team_best_matchups[i], adv)
#                 team_worst_matchups[i] = min(team_worst_matchups[i], adv)
#                 opponent_best_matchups[j] = max(team_best_matchups[j], 1 - adv)
#                 opponent_worst_matchups[j] = min(team_worst_matchups[j], 1 - adv)
        
#         return (
#             (team_best_matchups + team_worst_matchups) / 2,
#             (opponent_best_matchups + opponent_worst_matchups) / 2
#         )


# class MixBaseline:
#     def __init__(self, stats_):
#         self.stats = stats_

#     def predict(self, x):
#         radiant_synergies = self.team_synergy(x[:5])
#         dire_synergies = self.team_synergy(x[5:])
#         radiant_matchups, dire_matchups = self.matchups(x[:5], x[5:])
#         return int(sum(radiant_synergies + radiant_matchups) > 
#                    sum(dire_synergies + dire_matchups))

#     def team_synergy(self, team):
#         max_synergies = np.zeros(5)
#         for i in range(5):
#             for j in range(i + 1, 5):
#                 synergy = self.stats.Synergy(team[i], team[j])
#                 max_synergies[i] = max(max_synergies[i], synergy)
#                 max_synergies[j] = max(max_synergies[j], synergy)
#         return max_synergies

#     def matchups(self, team, opponent):
#         team_best_matchups = np.zeros(5)
#         team_worst_matchups = np.ones(5)
#         opponent_best_matchups = np.zeros(5)
#         opponent_worst_matchups = np.ones(5)
#         for i in range(5):
#             for j in range(5):
#                 adv = self.stats.Advantage(team[i], opponent[j])
#                 team_best_matchups[i] = max(team_best_matchups[i], adv)
#                 team_worst_matchups[i] = min(team_worst_matchups[i], adv)
#                 opponent_best_matchups[j] = max(team_best_matchups[j], 1 - adv)
#                 opponent_worst_matchups[j] = min(team_worst_matchups[j], 1 - adv)
        
#         return (
#             (team_best_matchups + team_worst_matchups) / 2,
#             (opponent_best_matchups + opponent_worst_matchups) / 2
#         )




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train model to predict the winning team.')
#     parser.add_argument('--data', required=True, help='Data glob for evaluating')
#     parser.add_argument('--stats_dir', required=False, help='Path to the stats directory')
#     args = parser.parse_args()

#     ## Data
#     batch_size = 256
#     files = glob.glob(args.data)

#     normal_ingester = data_ingestion.MatchIngester()
#     datasets = [normal_ingester.DatasetFromFile(f) for f in files]

#     # loader = data_ingestion.DataLoaderFromDatasets(
#     #     datasets, batch_size=batch_size, num_workers=4,
#     #     pin_memory=True, shuffle=False)

#     # # Display
#     # for x, y in training_loader:
#     #     y_hat = model(x)
#     #     print(x)
#     #     print(y, np.exp(y_hat.detach().numpy())) 

#     stats_ = stats.Stats.LoadFromJson(args.stats_dir)
#     baselines = [
#         TopSynergyBaseline(stats_),
#         TopMatchupBaseline(stats_),
#         TeamSynergyBaseline(stats_),
#         TeamMatchupsBaseline(stats_),
#         MixBaseline(stats_),
#     ]

#     ## Validate
#     baseline_correct = {
#         str(baseline): 0
#         for baseline in baselines
#     }
#     count = 0
#     for dataset in datasets:
#         for x, y in dataset:
#             for baseline in baselines:
#                 out = baseline.predict(x.detach().numpy())
#                 baseline_correct[str(baseline)] += y.narrow(0, 0, 1).detach().numpy() == out
#             count += 1

#     for baseline in baselines:
#         print(str(baseline))
#         print('-- Accuracy: %.2f' % (baseline_correct[str(baseline)] / count))

