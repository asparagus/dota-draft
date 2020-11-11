"""Module containing the Stats class that accumulates pick & win statistics."""
from ast import literal_eval

import json
import os

from typing import Dict, List


class Stats:
    """Class for keeping track of pick and win rate statistics."""

    def __init__(self):
        self.hero_picks = {}
        self.hero_wins = {}
        self.hero_pair_picks = {}
        self.hero_pair_wins = {}
        self.hero_matchup_picks = {}
        self.hero_matchup_wins = {}
        self.num_matches = 0

    def _AccumulateSingle(self, hero: int, win: bool):
        """Accumulate information for a single pick and win.
        
        Args:
            hero: A single hero id.
            win: Whether that hero won.
        """
        self.hero_picks[hero] = self.hero_picks.get(hero, 0) + 1
        if win:
            self.hero_wins[hero] = self.hero_wins.get(hero, 0) + 1
    
    def _AccumulatePair(self, hero_1: int, hero_2: int, win: bool):
        """Accumulate information for a pair of picks and win.
        
        Args:
            hero_1: A hero id.
            hero_2: Another hero id.
            win: Whether the heroes won.
        """
        sorted_ids = tuple(sorted([hero_1, hero_2]))
        self.hero_pair_picks[sorted_ids] = self.hero_pair_picks.get(sorted_ids, 0) + 1
        if win:
            self.hero_pair_wins[sorted_ids] = self.hero_pair_wins.get(sorted_ids, 0) + 1

    def _AccumulateMatchup(self, radiant_hero: int, dire_hero: int, radiant_win: bool):
        """Accumulate information for an opposing pair of picks and win.

        Args:
            radiant_hero: The id for the radiant hero.
            dire_hero: The id for the dire hero.
            radiant_win: Whether the radiant hero won.
        """
        sorted_ids = tuple(sorted([radiant_hero, dire_hero]))
        win = radiant_win if sorted_ids[0] == radiant_hero else not radiant_win
        self.hero_matchup_picks[sorted_ids] = self.hero_matchup_picks.get(sorted_ids, 0) + 1
        if win:
            self.hero_matchup_wins[sorted_ids] = self.hero_matchup_wins.get(sorted_ids, 0) + 1

    def Add(self, radiant_picks: List[int], dire_picks: List[int], radiant_win: bool):
        """Add new information.
        
        Args:
            radiant_picks: List of picks from radiant.
            dire_picks: List of picks from dire.
            radiant_win: Whether radiant won the match.
        """
        for i, hero in enumerate(radiant_picks):
            self._AccumulateSingle(hero, radiant_win)
            for other_hero in radiant_picks[i + 1:]:
                self._AccumulatePair(hero, other_hero, radiant_win)

        for i, hero in enumerate(dire_picks):
            self._AccumulateSingle(hero, not radiant_win)
            for other_hero in dire_picks[i + 1:]:
                self._AccumulatePair(hero, other_hero, not radiant_win)

        for radiant_hero in radiant_picks:
            for dire_hero in dire_picks:
                self._AccumulateMatchup(radiant_hero, dire_hero, radiant_win)

        self.num_matches += 1

    def WinRate(self, hero_id: int):
        """Get the win rate for a given hero."""
        return self.hero_wins.get(hero_id, 0) / self.hero_picks.get(hero_id, 1)

    def Synergy(self, hero_1: int, hero_2: int):
        """Get the joint win rate of two heroes."""
        sorted_ids = tuple(sorted([hero_1, hero_2]))
        if sorted_ids not in self.hero_pair_picks:
            raise IndexError('Pair not found in existing data: %s' % str(sorted_ids))
        return self.hero_pair_wins.get(sorted_ids, 0) / self.hero_pair_picks.get(sorted_ids, 1)

    def Advantage(self, hero_1: int, hero_2: int):
        """Get the win rate of one hero against another."""
        sorted_ids = tuple(sorted([hero_1, hero_2]))
        if sorted_ids not in self.hero_matchup_picks:
            raise IndexError('Pair not found in existing data: %s' % str(sorted_ids))
        res = self.hero_matchup_wins.get(sorted_ids, 0) / self.hero_matchup_picks.get(sorted_ids, 1)
        return res if sorted_ids[0] == hero_1 else (1 - res)

    def Save(self, path: str):
        """Save all statistics to a given path."""
        os.makedirs(path)
        def SaveToJson(data: Dict, subpath: str):
            with open(os.path.join(path, subpath), 'w') as f:
                json.dump({str(key): value for key, value in data.items()}, f)

        SaveToJson({0: self.num_matches}, 'num_matches.json')
        SaveToJson(self.hero_picks, 'hero_picks.json')
        SaveToJson(self.hero_wins, 'hero_wins.json')
        SaveToJson(self.hero_pair_picks, 'hero_pair_picks.json')
        SaveToJson(self.hero_pair_wins, 'hero_pair_wins.json')
        SaveToJson(self.hero_matchup_picks, 'hero_matchup_picks.json')
        SaveToJson(self.hero_matchup_wins, 'hero_matchup_wins.json')

    @classmethod
    def LoadFromJson(cls, path: str):
        """Load all statistics previously saved to a given path."""
        def LoadJson(subpath: str):
            with open(os.path.join(path, subpath), 'r') as f:
                dict_data = json.load(f)
                return {literal_eval(key): values for key, values in dict_data.items()}

        stats = Stats()
        stats.num_matches = LoadJson('num_matches.json')[0]
        stats.hero_picks = LoadJson('hero_picks.json')
        stats.hero_wins = LoadJson('hero_wins.json')
        stats.hero_pair_picks = LoadJson('hero_pair_picks.json')
        stats.hero_pair_wins = LoadJson('hero_pair_wins.json')
        stats.hero_matchup_picks = LoadJson('hero_matchup_picks.json')
        stats.hero_matchup_wins = LoadJson('hero_matchup_wins.json')
        return stats
