"""Scoring state machine tests.

These run without a video, a model, or a GPU. If the reported score of a match
is ever wrong, these tests decide whether the bug is here or in the vision
layer.
"""

from __future__ import annotations

import pytest

from tennis.scoring import GameScore, Match, MatchFormat, TiebreakScore


def award(match: Match, sequence: str) -> None:
    """Feed a point sequence like '112121' - each char is the point winner."""
    for char in sequence:
        match.award_point(int(char))  # type: ignore[arg-type]


class TestGameScore:
    def test_love_game(self):
        game = GameScore()
        assert [game.award(1) for _ in range(4)] == [None, None, None, 1]

    def test_display_progression(self):
        game = GameScore()
        seen = [game.display()]
        for _ in range(3):
            game.award(1)
            seen.append(game.display())
        assert seen == ["0-0", "15-0", "30-0", "40-0"]

    def test_thirty_all(self):
        game = GameScore()
        game.award(1), game.award(2), game.award(1), game.award(2)
        assert game.display() == "30-30"
        assert not game.is_deuce

    def test_deuce_requires_two_clear_points(self):
        game = GameScore()
        for _ in range(3):
            game.award(1)
            game.award(2)
        assert game.is_deuce
        assert game.display() == "40-40"

        assert game.award(1) is None       # advantage, not game
        assert game.display() == "AD-40"
        assert game.award(2) is None       # back to deuce
        assert game.is_deuce
        assert game.award(2) is None       # advantage other side
        assert game.display() == "40-AD"
        assert game.award(2) == 2          # game

    def test_long_deuce_still_needs_margin_of_two(self):
        game = GameScore()
        for _ in range(3):
            game.award(1), game.award(2)
        for _ in range(20):                # twenty alternating points
            assert game.award(1) is None
            assert game.award(2) is None
        assert game.is_deuce
        game.award(1)
        assert game.award(1) == 1

    def test_forty_thirty_is_not_deuce(self):
        game = GameScore()
        game.award(1), game.award(1), game.award(1)
        game.award(2), game.award(2)
        assert game.display() == "40-30"
        assert game.advantage() is None
        assert game.award(1) == 1


class TestTiebreak:
    def test_seven_to_five(self):
        tb = TiebreakScore()
        for _ in range(5):
            tb.award(1), tb.award(2)
        assert tb.display() == "5-5"
        assert tb.award(1) is None
        assert tb.award(1) == 1

    def test_needs_margin_of_two(self):
        tb = TiebreakScore()
        for _ in range(6):
            tb.award(1), tb.award(2)
        assert tb.display() == "6-6"
        assert tb.award(1) is None   # 7-6 is not enough
        assert tb.award(2) is None   # 7-7
        assert tb.award(2) is None   # 8-7
        assert tb.award(2) == 2      # 9-7


class TestSetAndMatch:
    def test_six_love_set(self):
        match = Match()
        award(match, "1111" * 6)
        assert match.sets_won[1] == 1
        assert match.sets[0].display() == "6-0"

    def test_five_all_goes_to_seven_five(self):
        match = Match()
        for _ in range(5):
            award(match, "1111")
            award(match, "2222")
        assert match.current_set.display() == "5-5"
        award(match, "1111")
        assert match.current_set.display() == "6-5"
        assert match.sets_won[1] == 0     # 6-5 does not win a set
        award(match, "1111")
        assert match.sets_won[1] == 1
        assert match.sets[0].display() == "7-5"

    def test_six_all_starts_tiebreak(self):
        match = Match()
        for _ in range(6):
            award(match, "1111")
            award(match, "2222")
        assert match.current_set.in_tiebreak
        assert "tiebreak" in match.scoreline()

        award(match, "1111111")
        assert match.sets_won[1] == 1
        assert match.sets[0].display() == "7-6"

    def test_best_of_three_ends_at_two_sets(self):
        match = Match(format=MatchFormat.BEST_OF_3)
        award(match, "1111" * 6)
        assert not match.is_over
        award(match, "1111" * 6)
        assert match.is_over
        assert match.winner == 1

    def test_best_of_five_needs_three_sets(self):
        match = Match(format=MatchFormat.BEST_OF_5)
        for _ in range(2):
            award(match, "1111" * 6)
        assert not match.is_over
        award(match, "1111" * 6)
        assert match.is_over

    def test_cannot_score_after_match_ends(self):
        match = Match()
        award(match, "1111" * 12)
        assert match.is_over
        with pytest.raises(RuntimeError, match="already complete"):
            match.award_point(1)

    def test_server_alternates_each_game(self):
        match = Match()
        assert match.server == 1
        award(match, "1111")
        assert match.server == 2
        award(match, "2222")
        assert match.server == 1


class TestReporting:
    def test_history_records_reason_and_confidence(self):
        match = Match()
        match.award_point(1, reason="double bounce far side", confidence=0.82,
                          start_frame=100, end_frame=340)
        event = match.history[0]
        assert event.winner == 1
        assert event.reason == "double bounce far side"
        assert event.confidence == 0.82
        assert event.start_frame == 100
        assert event.score_after == "0-0 | 15-0"

    def test_summary_counts_low_confidence_points(self):
        match = Match()
        match.award_point(1, confidence=0.9)
        match.award_point(2, confidence=0.4)
        match.award_point(1, confidence=0.55)
        summary = match.summary()
        assert summary["points_played"] == 3
        assert summary["low_confidence_points"] == 2

    def test_scoreline_across_a_realistic_set(self):
        match = Match()
        award(match, "1111")            # 1-0
        award(match, "2222")            # 1-1
        award(match, "1111")            # 2-1
        match.award_point(1)
        match.award_point(2)
        assert match.scoreline() == "2-1 | 15-15"
