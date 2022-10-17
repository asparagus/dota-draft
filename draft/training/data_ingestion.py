import enum
import json
import numpy as np
import torch
from torch.utils import data
from draft import stats


def DataLoaderFromDatasets(datasets, batch_size, num_workers=4, pin_memory=True, shuffle=True):
    concat_dataset = data.ConcatDataset(datasets)
    return data.DataLoader(
        concat_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )


class MatchIngester:
    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing

    def DatasetFromFile(self, file):
        with open(file, 'r') as f:
            instances = list(
                filter(None, [self._MatchFromLine(line)
                              for line in f.read().split('\n')
                              if line]))
            matches, labels = zip(*instances)
            return data.TensorDataset(torch.tensor(matches, dtype=torch.long),
                                      torch.tensor(labels, dtype=torch.float))

    def _MatchFromLine(self, line):
        match = json.loads(line)
        draft = _ParseDraft(match['radiant_team'] + ',' + match['dire_team'])
        label = _Opposites(self.label_smoothing + match['radiant_win'] * (1 - self.label_smoothing))
        if draft and len(draft) == 10 and all(draft):
            return draft, label


class WinningTeamIngester():
    def DatasetFromFile(self, file):
        with open(file, 'r') as f:
            instances = list(
                filter(None, [self._WinningTeamFromLine(line)
                              for line in f.read().split('\n')
                              if line]))
            return data.TensorDataset(torch.tensor(instances, dtype=torch.long))

    @classmethod
    def _WinningTeamFromLine(cls, line):
        match = json.loads(line)
        winning_team = (_ParseDraft(match['radiant_team'])
                        if match['radiant_win']
                        else _ParseDraft(match['dire_team']))
        if winning_team and len(winning_team) == 5 and all(winning_team):
            return (winning_team,)


class WinRateIngester():
    def __init__(self, single_heroes=True, hero_pairs=True, hero_matchups=True):
        self.single_heroes = single_heroes
        self.hero_pairs = hero_pairs
        self.hero_matchups = hero_matchups

    def DatasetFromFile(self, file):
        stats_data = stats.Stats.LoadFromJson(file)
        n_heroes = len(stats_data.hero_picks)

        drafts = []
        labels = []
        if self.single_heroes:
            single_hero_drafts = np.zeros((n_heroes, 10))
            single_hero_winrates = np.zeros((n_heroes, 2))
            for i, hero in enumerate(stats_data.hero_picks):
                single_hero_drafts[i][0] = hero
                single_hero_winrates[i, :] = _Opposites(stats_data.WinRate(hero))
            drafts.append(single_hero_drafts)
            labels.append(single_hero_winrates)

        if self.hero_pairs:
            n_pairs = len(stats_data.hero_pair_picks)
            hero_pair_drafts = np.zeros((n_pairs, 10))
            hero_pair_winrates = np.zeros((n_pairs, 2))
            for i, (hero_a, hero_b) in enumerate(stats_data.hero_pair_picks):
                hero_pair_drafts[i][0] = hero_a
                hero_pair_drafts[i][1] = hero_b
                hero_pair_winrates[i, :] = _Opposites(stats_data.Synergy(hero_a, hero_b))
            drafts.append(hero_pair_drafts)
            labels.append(hero_pair_winrates)

        if self.hero_matchups:
            n_matchups = len(stats_data.hero_matchup_picks)
            hero_matchup_drafts = np.zeros((n_matchups, 10))
            hero_matchup_winrates = np.zeros((n_matchups, 2))
            for i, (hero_a, hero_b) in enumerate(stats_data.hero_matchup_picks):
                hero_matchup_drafts[i][0] = hero_a
                hero_matchup_drafts[i][9] = hero_b
                hero_matchup_winrates[i, :] = _Opposites(stats_data.Advantage(hero_a, hero_b))
            drafts.append(hero_matchup_drafts)
            labels.append(hero_matchup_winrates)

        drafts = np.concatenate(drafts)
        labels = np.concatenate(labels)
        return data.TensorDataset(torch.tensor(drafts, dtype=torch.long),
                                  torch.tensor(labels, dtype=torch.float))


def _ParseDraft(team_str):
    return [int(i) for i in team_str.split(',')]


def _Opposites(label):
    return [label, 1 - label]
