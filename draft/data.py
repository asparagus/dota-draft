import json
import os
from typing import List

import numpy as np
import torch
import torch_geometric.data as geodata

import draft.stats as draft_stats


class MatchDataset(geodata.InMemoryDataset):
    def __init__(self, stats: draft_stats.Stats, root: str,
                 transform: bool = None, pre_transform: bool = None):
        self.stats = stats
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.fixed_edges = self.match_edges()

    @property
    def raw_file_names(self):
        return [
            'json/0-ancient.json',
            'json/1-ancient.json',
            'json/2-ancient.json',
            'json/3-ancient.json',
            'json/4-ancient.json',
            'json/5-ancient.json',
            'json/6-ancient.json',
            'json/7-ancient.json',
            'json/8-ancient.json',
        ]

    @classmethod
    def match_edges(cls):
        teammate_edges = ([
            # Radiant
            [i, j]
            for i in range(0, 5)
            for j in range(0, 5)
        ] + [
            # Dire
            [i, j]
            for i in range(5, 10)
            for j in range(5, 10)
        ])
        opponent_edges = ([
            # Radiant v/s Dire
            [i, j]
            for i in range(0, 5)
            for j in range(5, 10)
        ] + [
            # Dire v/s Radiant
            [i, j]
            for i in range(5, 10)
            for j in range(0, 5)
        ])
        return torch.Tensor(
            teammate_edges + opponent_edges,
            dtype=torch.long
        ).t().contiguous()

    @classmethod
    def match_edge_features(
            cls, radiant: List[int], dire: List[int],
            stats: draft_stats.Stats):
        teammate_edges = ([
            # Radiant
            [stats.Synergy(radiant[i], radiant[j])]
            for i in range(0, 5)
            for j in range(0, 5)
        ] + [
            # Dire
            [stats.Synergy(dire[i], dire[j])]
            for i in range(0, 5)
            for j in range(0, 5)
        ])
        opponent_edges = ([
            # Radiant v/s Dire
            [stats.Advantage(radiant[i], dire[j])]
            for i in range(0, 5)
            for j in range(0, 5)
        ] + [
            # Dire v/s Radiant
            [stats.Advantage(dire[i], radiant[j])]
            for i in range(0, 5)
            for j in range(0, 5)
        ])
        return torch.Tensor(
            teammate_edges + opponent_edges,
            dtype=torch.long
        )

    @property
    def processed_file_names(self):
        return ['data.pt']

    def parse(self, line):
        match = json.loads(line)
        radiant_heroes = [
            int(h) for h in match['radiant_team'].split(',')
        ]
        dire_heroes = [
            int(h) for h in match['dire_team'].split(',')
        ]
        onehot = np.zeros([10, 130])
        for i, h in enumerate(radiant_heroes):
            onehot[i][h] = 1
        for i, h in enumerate(dire_heroes):
            onehot[i + 5][h] = 1
        nodes = torch.Tensor(onehot, dtype=torch.float)
        edges = self.fixed_edges
        edge_attrs = self.match_edge_features(
            radiant_heroes, dire_heroes, self.stats)
        label = float(match['radiant_win'])
        return geodata.Data(nodes, edges, edge_attrs, label)

    def process(self):
        data_list = []
        for filename in self.raw_file_names:
            filepath = os.path.join(self.root, filename)
            with open(filepath, 'r') as f:
                data_list.extend([
                    self.parse(match_str)
                    for match_str in f.read().split('\n')
                ])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])