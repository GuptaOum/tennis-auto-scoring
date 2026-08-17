"""Shot placement analytics: where the ball lands, and where it was hit from.

This is the analysis a coach actually asks for - depth, width, direction - and
almost all of it is already paid for. Bounce positions arrive in real court
metres from the homography, so what remains is classification, not estimation.

Two properties of the pipeline make these numbers trustworthy, and both are
worth stating because the obvious alternatives are not:

- **Landing positions are measured at ground contact.** A bounce lies on the
  court plane by definition, which is the one moment the homography is exact.
  Ball positions mid-flight are projected metres from the truth and are never
  used here.
- **Shot origin comes from the hitter's feet, not the ball.** A player stands
  on the plane, so their foot position is as reliable as a bounce. Using the
  ball's position at the moment of a hit would inherit the airborne error,
  since a struck ball is roughly a metre up.

Court frame: x runs 0 to 10.97 across the doubles court, y runs 0 to 23.77 from
the far baseline to the near one, net at y = 11.885.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis.bounce import BallEvent, EventType
from tennis.court import (
    ALLEY_WIDTH,
    COURT_LENGTH,
    NET_Y,
    SINGLES_WIDTH,
    is_inside_singles,
)
from tennis.rally import Rally

HALF_COURT = COURT_LENGTH / 2  # 11.885 m, net to baseline
SERVICE_LINE_DEPTH = 6.40      # metres from the net

# Depth bands as a fraction of the net-to-baseline distance. A ball landing in
# the last 25% is what a commentator calls "deep"; inside the service line is
# "short" and invites an attack.
DEEP_FROM = 0.75
SHORT_UNTIL = SERVICE_LINE_DEPTH / HALF_COURT   # ~0.54


@dataclass
class Landing:
    """One ball landing, classified."""

    frame: int
    court: np.ndarray
    side: str                 # which half it landed in
    depth_m: float            # distance from the net
    depth_band: str           # 'short' | 'mid' | 'deep'
    width_band: str           # 'left' | 'centre' | 'right'
    in_bounds: bool
    hit_by: int | None = None       # track id of the player who struck it
    direction: str | None = None    # 'cross-court' | 'down-the-line' | 'centre'
    confidence: float = 1.0

    def as_dict(self) -> dict:
        return {
            "frame": self.frame,
            "x_m": round(float(self.court[0]), 2),
            "y_m": round(float(self.court[1]), 2),
            "side": self.side,
            "depth_m": round(self.depth_m, 2),
            "depth_band": self.depth_band,
            "width_band": self.width_band,
            "in_bounds": self.in_bounds,
            "hit_by": self.hit_by,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
        }


def classify_landing(court_pt: np.ndarray, frame: int = 0) -> Landing:
    """Classify a single bounce by depth and width."""
    x, y = float(court_pt[0]), float(court_pt[1])
    side = "far" if y < NET_Y else "near"

    # Depth is measured from the net towards whichever baseline the ball landed
    # behind, so the two halves are directly comparable.
    depth = NET_Y - y if side == "far" else y - NET_Y
    fraction = depth / HALF_COURT

    if fraction >= DEEP_FROM:
        depth_band = "deep"
    elif fraction <= SHORT_UNTIL:
        depth_band = "short"
    else:
        depth_band = "mid"

    # Width normalised across the singles court, then flipped for the far side
    # so that "left" means the same physical side of the court to both players.
    u = (x - ALLEY_WIDTH) / SINGLES_WIDTH
    if side == "far":
        u = 1.0 - u
    if u < 1 / 3:
        width_band = "left"
    elif u > 2 / 3:
        width_band = "right"
    else:
        width_band = "centre"

    return Landing(
        frame=frame,
        court=np.asarray(court_pt, dtype=float),
        side=side,
        depth_m=depth,
        depth_band=depth_band,
        width_band=width_band,
        in_bounds=is_inside_singles(court_pt, margin=0.10),
    )


def shot_direction(origin: np.ndarray, landing: np.ndarray) -> str:
    """Cross-court, down-the-line, or through the centre.

    Judged by how far the ball moved laterally relative to the court's width.
    A shot that stays within an eighth of the court's width is down-the-line;
    one that crosses more than a third of it is cross-court.
    """
    lateral = abs(float(landing[0]) - float(origin[0]))
    if lateral < SINGLES_WIDTH / 8:
        return "down-the-line"
    if lateral > SINGLES_WIDTH / 3:
        return "cross-court"
    return "centre"


def landings_from_rallies(
    rallies: list[Rally],
    player_positions: dict[int, dict[int, np.ndarray]] | None = None,
) -> list[Landing]:
    """Extract and classify every in-play landing across all rallies.

    Each bounce is attributed to the last player to strike the ball, and its
    direction computed from that player's court position at the moment of the
    hit.
    """
    out: list[Landing] = []

    for rally in rallies:
        last_hit: BallEvent | None = None
        for event in rally.events:
            if event.type is EventType.HIT:
                last_hit = event
                continue
            if event.type is not EventType.BOUNCE:
                continue

            landing = classify_landing(event.court, frame=event.frame)
            landing.confidence = event.confidence

            if last_hit is not None and last_hit.by_player is not None:
                landing.hit_by = last_hit.by_player
                origin = _player_at(
                    player_positions, last_hit.by_player, last_hit.frame
                )
                if origin is not None:
                    landing.direction = shot_direction(origin, event.court)

            out.append(landing)

    return out


def _player_at(
    player_positions: dict[int, dict[int, np.ndarray]] | None,
    track_id: int,
    frame: int,
    search: int = 5,
) -> np.ndarray | None:
    """A player's court position at a frame, tolerating a few missing frames."""
    if not player_positions or track_id not in player_positions:
        return None
    by_frame = player_positions[track_id]
    for offset in range(search + 1):
        for candidate in (frame - offset, frame + offset):
            if candidate in by_frame:
                return np.asarray(by_frame[candidate], dtype=float)
    return None


def placement_grid(
    landings: list[Landing], side: str, rows: int = 3, cols: int = 3
) -> list[list[int]]:
    """Counts of landings per zone on one half of the court.

    Rows run net-to-baseline, columns left-to-right from that half's
    perspective, matching how the bands are named.
    """
    grid = [[0] * cols for _ in range(rows)]
    for landing in landings:
        if landing.side != side or not landing.in_bounds:
            continue
        row = int(np.clip(landing.depth_m / HALF_COURT * rows, 0, rows - 1))
        u = (float(landing.court[0]) - ALLEY_WIDTH) / SINGLES_WIDTH
        if side == "far":
            u = 1.0 - u
        col = int(np.clip(u * cols, 0, cols - 1))
        grid[row][col] += 1
    return grid


def render_grid(grid: list[list[int]]) -> list[str]:
    """Render a placement grid as shaded text, with the net at the top."""
    peak = max((max(row) for row in grid), default=0)
    if not peak:
        return ["(no landings)"]
    shades = " .:-=+*#%@"
    lines = []
    for row in grid:
        cells = "".join(
            shades[min(int(v / peak * (len(shades) - 1)), len(shades) - 1)] * 3
            for v in row
        )
        lines.append(f"|{cells}|")
    return lines


def summarise(
    rallies: list[Rally],
    player_positions: dict[int, dict[int, np.ndarray]] | None = None,
) -> dict:
    """Placement statistics, overall and per player."""
    landings = landings_from_rallies(rallies, player_positions)
    in_play = [l for l in landings if l.in_bounds]

    def stats_for(subset: list[Landing]) -> dict:
        if not subset:
            return {"landings": 0}
        depths = np.array([l.depth_m for l in subset])
        return {
            "landings": len(subset),
            "mean_depth_m": round(float(depths.mean()), 2),
            "depth_bands": {
                band: sum(1 for l in subset if l.depth_band == band)
                for band in ("short", "mid", "deep")
            },
            "width_bands": {
                band: sum(1 for l in subset if l.width_band == band)
                for band in ("left", "centre", "right")
            },
            "directions": {
                d: sum(1 for l in subset if l.direction == d)
                for d in ("cross-court", "down-the-line", "centre")
                if any(l.direction == d for l in subset)
            },
            "deep_share": round(
                sum(1 for l in subset if l.depth_band == "deep") / len(subset), 3
            ),
        }

    by_player: dict[str, dict] = {}
    for track_id in sorted({l.hit_by for l in in_play if l.hit_by is not None}):
        by_player[str(track_id)] = stats_for(
            [l for l in in_play if l.hit_by == track_id]
        )

    return {
        "total_landings": len(landings),
        "in_bounds": len(in_play),
        "out_of_bounds": len(landings) - len(in_play),
        "overall": stats_for(in_play),
        "by_player": by_player,
        "grids": {
            "far": placement_grid(in_play, "far"),
            "near": placement_grid(in_play, "near"),
        },
        "landings": [l.as_dict() for l in landings],
    }
