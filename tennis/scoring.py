"""Tennis scoring state machine.

Deliberately contains no computer vision. It consumes "player X won the point"
and produces a score, which means the whole of tennis's scoring logic - deuce,
advantage, tiebreaks, set and match completion - is unit-testable in
milliseconds without a video file anywhere near it.

The vision half of this project is probabilistic and will always have an error
rate. This half is exact, and keeping the boundary sharp is what makes the
error rate measurable: any wrong score is either a wrong point attribution
(vision) or a bug in here (provable by test), never an ambiguous mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Player = Literal[1, 2]

POINT_NAMES = ["0", "15", "30", "40"]


class MatchFormat(Enum):
    """How many sets win the match, and how the final set ends."""

    BEST_OF_3 = 3
    BEST_OF_5 = 5


def _other(player: Player) -> Player:
    return 2 if player == 1 else 1


@dataclass
class GameScore:
    """Points within the current game. Handles deuce/advantage."""

    points: dict[Player, int] = field(default_factory=lambda: {1: 0, 2: 0})

    def display(self) -> str:
        p1, p2 = self.points[1], self.points[2]
        if p1 >= 3 and p2 >= 3:
            if p1 == p2:
                return "40-40"  # deuce
            return "AD-40" if p1 > p2 else "40-AD"
        return f"{POINT_NAMES[min(p1, 3)]}-{POINT_NAMES[min(p2, 3)]}"

    @property
    def is_deuce(self) -> bool:
        return self.points[1] >= 3 and self.points[1] == self.points[2]

    def advantage(self) -> Player | None:
        p1, p2 = self.points[1], self.points[2]
        if p1 >= 3 and p2 >= 3 and p1 != p2:
            return 1 if p1 > p2 else 2
        return None

    def award(self, player: Player) -> Player | None:
        """Award a point. Returns the game winner, or None if the game goes on."""
        self.points[player] += 1
        mine, theirs = self.points[player], self.points[_other(player)]
        # Win at 4+ points with a margin of 2. Covers both the ordinary 40-30
        # case and any length of deuce.
        if mine >= 4 and mine - theirs >= 2:
            return player
        return None


@dataclass
class TiebreakScore:
    """First to 7, win by 2. Used at 6-6."""

    points: dict[Player, int] = field(default_factory=lambda: {1: 0, 2: 0})
    target: int = 7

    def display(self) -> str:
        return f"{self.points[1]}-{self.points[2]}"

    def award(self, player: Player) -> Player | None:
        self.points[player] += 1
        mine, theirs = self.points[player], self.points[_other(player)]
        if mine >= self.target and mine - theirs >= 2:
            return player
        return None


@dataclass
class SetScore:
    games: dict[Player, int] = field(default_factory=lambda: {1: 0, 2: 0})
    tiebreak: TiebreakScore | None = None

    def display(self) -> str:
        return f"{self.games[1]}-{self.games[2]}"

    @property
    def in_tiebreak(self) -> bool:
        return self.tiebreak is not None

    def start_tiebreak_if_needed(self) -> None:
        if self.games[1] == 6 and self.games[2] == 6 and self.tiebreak is None:
            self.tiebreak = TiebreakScore()

    def award_game(self, player: Player) -> Player | None:
        """Award a game. Returns the set winner, or None."""
        self.games[player] += 1
        mine, theirs = self.games[player], self.games[_other(player)]
        # 6-4 or better, or 7-5, or 7-6 via tiebreak.
        if mine >= 6 and mine - theirs >= 2:
            return player
        if mine == 7 and theirs in (5, 6):
            return player
        self.start_tiebreak_if_needed()
        return None


@dataclass
class PointEvent:
    """One completed point, with the evidence behind it.

    ``confidence`` and ``reason`` come from the vision layer and are carried
    through untouched, so a reviewer can see not just the final score but which
    points the system was unsure about.
    """

    winner: Player
    reason: str = ""
    confidence: float = 1.0
    start_frame: int | None = None
    end_frame: int | None = None
    score_after: str = ""


@dataclass
class Match:
    """Full match state. Feed it point winners, read the score."""

    format: MatchFormat = MatchFormat.BEST_OF_3
    final_set_tiebreak: bool = True
    sets: list[SetScore] = field(default_factory=lambda: [SetScore()])
    sets_won: dict[Player, int] = field(default_factory=lambda: {1: 0, 2: 0})
    game: GameScore = field(default_factory=GameScore)
    history: list[PointEvent] = field(default_factory=list)
    winner: Player | None = None
    server: Player = 1

    @property
    def current_set(self) -> SetScore:
        return self.sets[-1]

    @property
    def sets_to_win(self) -> int:
        return self.format.value // 2 + 1

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def award_point(
        self,
        player: Player,
        reason: str = "",
        confidence: float = 1.0,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> PointEvent:
        """Award one point and advance every level of the score that follows."""
        if self.is_over:
            raise RuntimeError("match is already complete")

        current_set = self.current_set
        game_winner: Player | None

        if current_set.in_tiebreak:
            assert current_set.tiebreak is not None
            game_winner = current_set.tiebreak.award(player)
            if game_winner is not None:
                current_set.games[game_winner] += 1
                self._complete_set(game_winner)
        else:
            game_winner = self.game.award(player)
            if game_winner is not None:
                self.game = GameScore()
                self.server = _other(self.server)
                set_winner = current_set.award_game(game_winner)
                if set_winner is not None:
                    self._complete_set(set_winner)

        event = PointEvent(
            winner=player,
            reason=reason,
            confidence=confidence,
            start_frame=start_frame,
            end_frame=end_frame,
            score_after=self.scoreline(),
        )
        self.history.append(event)
        return event

    def _complete_set(self, set_winner: Player) -> None:
        self.sets_won[set_winner] += 1
        if self.sets_won[set_winner] >= self.sets_to_win:
            self.winner = set_winner
            return
        new_set = SetScore()
        # An unplayed final-set tiebreak is disabled by giving it an
        # unreachable target rather than by special-casing it downstream.
        if not self.final_set_tiebreak and len(self.sets) + 1 == self.format.value:
            new_set.tiebreak = None
        self.sets.append(new_set)
        self.game = GameScore()

    def scoreline(self) -> str:
        """Human-readable score, e.g. '6-4 3-2 | 40-30'."""
        sets_part = " ".join(s.display() for s in self.sets)
        if self.is_over:
            return f"{sets_part} | match to player {self.winner}"
        current_set = self.current_set
        if current_set.in_tiebreak:
            assert current_set.tiebreak is not None
            return f"{sets_part} | tiebreak {current_set.tiebreak.display()}"
        return f"{sets_part} | {self.game.display()}"

    def summary(self) -> dict:
        """Machine-readable state, for the JSON report."""
        return {
            "scoreline": self.scoreline(),
            "sets": [s.games.copy() for s in self.sets],
            "sets_won": self.sets_won.copy(),
            "current_game": self.game.display(),
            "server": self.server,
            "points_played": len(self.history),
            "winner": self.winner,
            "low_confidence_points": sum(
                1 for p in self.history if p.confidence < 0.6
            ),
        }
