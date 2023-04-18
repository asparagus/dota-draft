"""Module that defines filters for matches."""
from typing import Optional
import abc

import datetime

from draft.data.match import Match


class MatchFilter(abc.ABC):
    """Base class to implement a filter for stored matches.
    
    Filters must implement a __call__ function that returns a bool
    indicating whether the match should be kept.
    """

    @abc.abstractmethod
    def __call__(self, match: Match) -> bool:
        """Runs on the match and determines whether it should be kept.
        
        Args:
            match: The match to evaluate
        """
        raise NotImplementedError()

    def __or__(self, other: 'MatchFilter') -> 'MatchFilter':
        """Defines the or (|) operator between this and another filter."""
        return Disjunction(self, other)

    def __ror__(self, other) -> 'MatchFilter':
        """Defines the reverse or (|) operator between this and another filter."""
        return self.__or__(other)

    def __and__(self, other: 'MatchFilter') -> 'MatchFilter':
        """Defines the and (&) operator between this and another filter."""
        return Conjunction(self, other)

    def __invert__(self) -> 'MatchFilter':
        """Defines the invert (~) operator for this filter."""
        return Negation(self)


class Disjunction(MatchFilter):
    """A disjunction between two filters, will keep the match when any is true."""

    def __init__(self, filter_a: MatchFilter, filter_b: MatchFilter):
        """Initialize the disjunction with both filters.

        Args:
            filter_a: One of the filters
            filter_b: The other filter
        """
        self.filter_a = filter_a
        self.filter_b = filter_b

    def __call__(self, match: Match) -> bool:
        """Evaluates both filters and returns the or between them.

        Args:
            match: The match to evaluate
        """
        return self.filter_a(match) or self.filter_b(match)


class Conjunction(MatchFilter):
    """A conjunction between two filters, will keep the match when both are true."""

    def __init__(self, filter_a: MatchFilter, filter_b: MatchFilter):
        """Initialize the conjunction with both filters.

        Args:
            filter_a: One of the filters
            filter_b: The other filter
        """
        self.filter_a = filter_a
        self.filter_b = filter_b

    def __call__(self, match: Match) -> bool:
        """Evaluates both filters and returns the and between them.

        Args:
            match: The match to evaluate
        """
        return self.filter_a(match) and self.filter_b(match)


class Negation(MatchFilter):
    """A negation of a filter, will keep the match when the filter is false."""

    def __init__(self, filter: MatchFilter):
        """Initialize the negation of the filter.

        Args:
            filter: The filter to negate
        """
        self.filter = filter

    def __call__(self, match: Match) -> bool:
        """Evaluates the filter and returns the opposite result.

        Args:
            match: The match to evaluate
        """
        return not self.filter(match)


class ValidMatchFilter(MatchFilter):
    """A filter for removing invalid matches where the draft did not finish."""

    def __call__(self, match: Match) -> bool:
        """Evaluate whether the match had 5 heroes on both sides.

        Args:
            match: The match to evaluate
        """
        return (
            len(match.radiant_heroes) == 5 and
            len(match.dire_heroes) == 5
        )


class HighRankMatchFilter(MatchFilter):
    """A filter for removing matches below a certain rank."""

    def __init__(self, minimum_rank: int):
        """Initialize the filter with a minimum rank value.

        Args:
            minimum_rank: The minimum acceptable rank, matches
                below this will be discarded"""
        self.minimum_rank = minimum_rank

    def __call__(self, match: Match) -> bool:
        """Evaluate whether the match has the required rank.

        Args:
            match: The match to evaluate
        """
        return (
            match.avg_rank_tier >= self.minimum_rank
        )


class DateFilter(MatchFilter):
    """A filter for a particular date."""

    def __init__(
            self,
            start_date: Optional[datetime.date] = None,
            end_date: Optional[datetime.date] = None,
        ):
        """Initialize the filter with optional start and end dates.

        Args:
            start_date: (Optional) earliest date acceptable
            end_date: (Optional) latest date acceptable
        """
        self.start_time = int(start_date.strftime('%s')) if start_date else None
        self.end_time = int(end_date.strftime('%s')) if end_date else None

    def __call__(self, match: Match) -> bool:
        """Evaluate whether the match is within the date range.

        Args:
            match: The match to evaluate
        """
        if self.start_time and match.start_time < self.start_time:
            return False
        if self.end_time and match.start_time > self.end_time:
            return False
        return True
