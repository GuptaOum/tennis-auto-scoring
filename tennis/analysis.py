"""Match analysis from tracked positions.

Everything here is derived from detections the system is confident about -
where each player stood, frame by frame, in metres. That makes these numbers
the reliable core of the output: they need no bounce to be found, no rally to
be segmented, and no point to be attributed. A video that defeats the scoring
layer still produces a full analysis.

Scoring sits on top as a best-effort layer. This does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis.court import COURT_LENGTH, DOUBLES_WIDTH, NET_Y


@dataclass
class PlayerStats:
    track_id: int
    side: str                    # which half they played from
    frames_tracked: int
    distance_m: float
    avg_speed_kmh: float
    top_speed_kmh: float
    avg_position: tuple[float, float]
    net_approaches: int
    time_at_net_s: float

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "side": self.side,
            "frames_tracked": self.frames_tracked,
            "distance_covered_m": round(self.distance_m, 1),
            "average_speed_kmh": round(self.avg_speed_kmh, 1),
            "top_speed_kmh": round(self.top_speed_kmh, 1),
            "average_position_m": [
                round(self.avg_position[0], 2),
                round(self.avg_position[1], 2),
            ],
            "net_approaches": self.net_approaches,
            "time_at_net_s": round(self.time_at_net_s, 1),
        }


# Beyond this speed a "player" is a tracking glitch, not an athlete. Usain Bolt
# peaked at 44.7 km/h; a tennis player never approaches that, so anything above
# is an ID switch teleporting the track across the court.
MAX_PLAUSIBLE_SPEED_KMH = 45.0

# Inside this distance from the net, in metres, counts as being at the net.
NET_ZONE_M = 5.0


def _speeds(positions: np.ndarray, frames: np.ndarray, fps: float) -> np.ndarray:
    """Frame-to-frame speed in km/h, with teleports removed.

    A tracker that swaps two players' IDs produces a single enormous step. Left
    in, one such glitch can add tens of metres to a distance total - which is
    how the baseline's stats drift. Discarding physically impossible steps
    costs a frame of data and saves the whole number.
    """
    if len(positions) < 2:
        return np.array([])

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)   # metres
    dt = np.diff(frames) / fps                                    # seconds
    with np.errstate(divide="ignore", invalid="ignore"):
        speeds = np.where(dt > 0, deltas / dt * 3.6, 0.0)
    return speeds[speeds <= MAX_PLAUSIBLE_SPEED_KMH]


def player_stats(
    track: list[dict], fps: float, track_id: int
) -> PlayerStats | None:
    """Summarise one player's movement. ``track`` is that player's rows only."""
    if len(track) < 2:
        return None

    frames = np.array([r["frame"] for r in track])
    positions = np.array([[r["x_m"], r["y_m"]] for r in track])

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(frames) / fps
    with np.errstate(divide="ignore", invalid="ignore"):
        step_speeds = np.where(dt > 0, deltas / dt * 3.6, 0.0)
    keep = step_speeds <= MAX_PLAUSIBLE_SPEED_KMH

    distance = float(deltas[keep].sum())
    speeds = step_speeds[keep]
    mean_position = positions.mean(axis=0)

    # Which half of the court they spent most of their time in.
    side = "near" if mean_position[1] > NET_Y else "far"

    # Distance from the net, on their own side.
    net_distance = np.abs(positions[:, 1] - NET_Y)
    at_net = net_distance < NET_ZONE_M
    # An approach is a transition into the net zone, not every frame spent in it.
    approaches = int(np.sum(np.diff(at_net.astype(int)) == 1))

    return PlayerStats(
        track_id=track_id,
        side=side,
        frames_tracked=len(track),
        distance_m=distance,
        avg_speed_kmh=float(speeds.mean()) if len(speeds) else 0.0,
        top_speed_kmh=float(speeds.max()) if len(speeds) else 0.0,
        avg_position=(float(mean_position[0]), float(mean_position[1])),
        net_approaches=approaches,
        time_at_net_s=float(at_net.sum()) / fps,
    )


def analyse_players(
    player_track: list[dict], fps: float, top_n: int = 2
) -> list[dict]:
    """Per-player movement stats, for the ``top_n`` most-tracked identities.

    Broadcast footage contains ball kids, line judges and crowd. Ranking by how
    many frames each identity was tracked for, then keeping the top two, picks
    out the players without needing to guess from frame 0 the way the baseline
    did - a choice it could never revise.
    """
    by_id: dict[int, list[dict]] = {}
    for row in player_track:
        if row.get("track_id") is not None:
            by_id.setdefault(row["track_id"], []).append(row)

    ranked = sorted(by_id.items(), key=lambda kv: len(kv[1]), reverse=True)
    stats = []
    for track_id, rows in ranked[:top_n]:
        computed = player_stats(sorted(rows, key=lambda r: r["frame"]), fps, track_id)
        if computed is not None:
            stats.append(computed.as_dict())
    return stats


def court_coverage(
    player_track: list[dict], track_id: int, bins: int = 6
) -> list[list[int]]:
    """A coarse occupancy grid of where one player spent their time.

    Rows run far baseline to near baseline, columns left to right. Small enough
    to print in a terminal, which is the point - it makes positioning legible
    without a plotting dependency.
    """
    grid = [[0] * bins for _ in range(bins)]
    for row in player_track:
        if row.get("track_id") != track_id:
            continue
        col = int(np.clip(row["x_m"] / DOUBLES_WIDTH * bins, 0, bins - 1))
        line = int(np.clip(row["y_m"] / COURT_LENGTH * bins, 0, bins - 1))
        grid[line][col] += 1
    return grid


def render_coverage(grid: list[list[int]]) -> list[str]:
    """Render an occupancy grid as shaded text rows."""
    peak = max((max(row) for row in grid), default=0)
    if not peak:
        return ["(no positions tracked)"]
    shades = " .:-=+*#%@"
    return [
        "".join(shades[min(int(value / peak * (len(shades) - 1)), len(shades) - 1)] * 2
                for value in row)
        for row in grid
    ]


def summarise(player_track: list[dict], ball_track: list[dict], fps: float) -> dict:
    players = analyse_players(player_track, fps)
    return {
        "players": players,
        "ball": {
            "frames_detected": len(ball_track),
            "mean_confidence": (
                round(float(np.mean([b["confidence"] for b in ball_track])), 3)
                if ball_track
                else 0.0
            ),
        },
    }
