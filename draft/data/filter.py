"""Module that defines filters for matches."""
import abc

from draft.data.match import Match


class MatchFilter(abc.ABC):
    """Base class to implement a filter for stored matches.
    
    Filters must implement a __call__ function that returns a bool
    indicating whether the match should be kept.
    """

    @abc.abstractmethod
    def __call__(self, match: Match):
        """Runs on the match and determines whether it should be kept.
        
        Args:
            match: The match to evaluate
        """
        raise NotImplementedError()

    def __or__(self, other: 'MatchFilter'):
        """Defines the or operator between this and another filter."""
        return Conjunction(self, other)

    def __and__(self, other: 'MatchFilter'):
        """Defines the and operator between this and another filter."""
        return Conjunction(self, other)

    def __not__(self):
        """Defines the not operator for this filter."""
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

    def __call__(self, match: Match):
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

    def __call__(self, match: Match):
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

    def __call__(self, match: Match):
        """Evaluates the filter and returns the opposite result.

        Args:
            match: The match to evaluate
        """
        return not self.filter(match)


class ValidMatchFilter(MatchFilter):
    """A filter for removing invalid matches where the draft did not finish."""

    def __call__(self, match: Match):
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

    def __call__(self, match: Match):
        """Evaluate whether the match has the required rank.

        Args:
            match: The match to evaluate
        """
        return (
            match.avg_rank_tier >= self.minimum_rank
        )