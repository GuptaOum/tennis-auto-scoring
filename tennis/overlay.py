"""Broadcast-style annotated video.

The baseline's overlay drew boxes and a frame counter: enough to prove the
detector fires, not enough to show that anything was *understood*. This one
draws the interpretation - the live score, the rally in progress, where the
ball actually landed, who is covering what - because that is the part worth
watching, and the part that is hard.

**This requires a second pass over the video, and that is the whole point.**
The score after a point, the reason a rally ended, and whether a bounce landed
in cannot be known while the frame containing them is being detected: a bounce
is a local maximum in projected court y, so it is only identifiable once the
frames *after* it exist. The baseline drew during detection and was therefore
structurally incapable of showing a score. Re-decoding costs a few seconds of
CPU with no GPU work, which buys an overlay that can state the score at every
frame and mark a bounce at the moment it happened.

Court frame throughout: x 0 to 10.97 across, y 0 to 23.77 from the far
baseline, net at 11.885.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from tennis.bounce import BallEvent, EventType
from tennis.court import (
    COURT_LENGTH,
    COURT_LINES,
    COURT_MODEL,
    DOUBLES_WIDTH,
    NET_Y,
    SERVICE_LINE_FROM_NET,
    SINGLES_WIDTH,
    CourtCalibration,
)
from tennis.rally import Rally
from tennis.scoring import Match

# BGR. Players get stable colours so the eye can track them across the
# minimap, the boxes and the stat strip without reading a label each time.
P1 = (90, 220, 120)
P2 = (255, 170, 80)
BALL = (60, 240, 250)
INK = (245, 245, 245)
DIM = (170, 170, 170)
PANEL = (24, 26, 30)
LINE = (110, 120, 128)
COURT_FILL = (58, 48, 40)      # opaque slab behind the minimap plan
COURT_MARK = (150, 160, 168)   # its lines
KEYPOINT = (60, 60, 235)       # the 14 court landmarks, in red
GOOD = (110, 230, 140)
BAD = (90, 90, 240)
WARN = (70, 190, 250)

FONT = cv2.FONT_HERSHEY_SIMPLEX
ALLEY = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2

TRAIL_LENGTH = 14        # frames of ball history drawn behind the ball
EVENT_FLASH_FRAMES = 9   # how long a BOUNCE / HIT caption stays up
POINT_BANNER_SECONDS = 2.2


def _panel(canvas: np.ndarray, x: int, y: int, w: int, h: int,
           alpha: float = 0.72) -> None:
    """A translucent slab behind text, so the overlay stays readable on grass,
    clay and hard court alike rather than only on the one clip it was tuned on.
    """
    x2, y2 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return
    region = canvas[y:y2, x:x2]
    slab = np.full(region.shape, PANEL, dtype=np.uint8)
    cv2.addWeighted(slab, alpha, region, 1 - alpha, 0, region)
    cv2.rectangle(canvas, (x, y), (x2 - 1, y2 - 1), (60, 66, 74), 1)


def _text(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float,
          colour=INK, weight: int = 1, shadow: bool = True) -> None:
    if shadow:
        cv2.putText(canvas, text, (org[0] + 1, org[1] + 1), FONT, scale,
                    (0, 0, 0), weight + 1, cv2.LINE_AA)
    cv2.putText(canvas, text, org, FONT, scale, colour, weight, cv2.LINE_AA)


@dataclass
class FrameState:
    """Everything known about one frame, assembled before rendering starts."""

    scoreline: str = "0-0 | 0-0"
    server: int = 1
    points_played: int = 0
    rally_index: int | None = None
    rally_shots: int = 0
    rally_started: int | None = None
    event: BallEvent | None = None
    point_banner: tuple[str, str, float] | None = None  # text, detail, confidence
    bounces: list[BallEvent] = field(default_factory=list)


class Renderer:
    """Draws the annotated frames, given the whole run's results.

    Constructed once after detection, then called per frame. All the reasoning
    about what belongs on a given frame happens up front in
    :meth:`_build_states`, so the per-frame path is pure drawing.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        ball_track: list[dict],
        player_boxes: dict[int, dict[int, tuple]],
        player_court: dict[int, dict[int, np.ndarray]],
        events: list[BallEvent],
        rallies: list[Rally],
        match: Match,
        calibration: CourtCalibration | None,
        source_name: str = "",
    ) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.player_boxes = player_boxes
        self.player_court = player_court
        self.calibration = calibration
        self.source_name = source_name

        self.ball_by_frame = {b["frame"]: b for b in ball_track}
        self.ball_frames = sorted(self.ball_by_frame)
        self.events = events
        self.rallies = rallies
        self.match = match

        # Sized so the court plan is a fixed share of frame height rather than
        # a fixed pixel count: legible on 720p and 4K alike.
        self.minimap_scale = max(height * 0.38 / COURT_LENGTH, 8.0)
        self.states = self._build_states()
        self.last_state_frame = max(self.states, default=0)
        self.distance = self._cumulative_distance()

    # -- precomputation ---------------------------------------------------

    def _build_states(self) -> dict[int, FrameState]:
        """Assemble per-frame state: score, live rally, events, banners.

        The score is replayed point by point from ``match.history`` so that
        every frame carries the score *as it stood then*, not the final score.
        A video that shows the finishing score from frame one is a video that
        tells you nothing about when anything happened.
        """
        states: dict[int, FrameState] = {}
        last = max(
            [r.end_frame for r in self.rallies] + list(self.ball_by_frame) + [0]
        )

        # frame -> (scoreline, points_played, server) taking effect from there
        checkpoints: list[tuple[int, str, int, int]] = [(0, "0-0 | 0-0", 0, 1)]
        for index, point in enumerate(self.match.history, start=1):
            if point.end_frame is None:
                continue
            checkpoints.append(
                (point.end_frame, point.score_after or "", index, self.match.server)
            )

        banners: dict[int, tuple[str, str, float]] = {}
        banner_frames = int(POINT_BANNER_SECONDS * self.fps)
        for point in self.match.history:
            if point.end_frame is None:
                continue
            winner = f"POINT  -  PLAYER {point.winner}"
            banners[point.end_frame] = (
                winner, point.reason or "", point.confidence
            )

        event_by_frame = {e.frame: e for e in self.events}
        bounces_so_far: list[BallEvent] = []

        checkpoint_index = 0
        for frame in range(last + 1):
            while (
                checkpoint_index + 1 < len(checkpoints)
                and checkpoints[checkpoint_index + 1][0] <= frame
            ):
                checkpoint_index += 1
            _, scoreline, played, server = checkpoints[checkpoint_index]

            state = FrameState(
                scoreline=scoreline, server=server, points_played=played
            )

            for index, rally in enumerate(self.rallies):
                if rally.start_frame <= frame <= rally.end_frame:
                    state.rally_index = index
                    state.rally_started = rally.start_frame
                    state.rally_shots = sum(
                        1 for e in rally.events
                        if e.type is EventType.HIT and e.frame <= frame
                    )
                    break

            # An event caption persists for a few frames: at 30 fps a single
            # frame is 33 ms and would be invisible.
            for offset in range(EVENT_FLASH_FRAMES):
                candidate = event_by_frame.get(frame - offset)
                if candidate is not None:
                    state.event = candidate
                    break

            for offset in range(banner_frames):
                banner = banners.get(frame - offset)
                if banner is not None:
                    state.point_banner = banner
                    break

            # Bounces accumulate on the minimap within a rally and clear
            # between points, which is what makes the map readable.
            event = event_by_frame.get(frame)
            if event is not None and event.type is EventType.BOUNCE:
                bounces_so_far.append(event)
            if state.rally_index is None:
                bounces_so_far = []
            state.bounces = list(bounces_so_far)[-6:]

            states[frame] = state
        return states

    def _cumulative_distance(self) -> dict[int, dict[int, float]]:
        """Running metres covered per player, so the strip counts up live."""
        result: dict[int, dict[int, float]] = {}
        for track_id, positions in self.player_court.items():
            frames = sorted(positions)
            running = 0.0
            per_frame: dict[int, float] = {}
            for previous, current in zip(frames, frames[1:]):
                step = float(np.linalg.norm(positions[current] - positions[previous]))
                # Same guard the analysis module uses: a track-id swap teleports
                # a player, and unfiltered that inflates distance covered.
                if step < 2.0:
                    running += step
                per_frame[current] = running
            result[track_id] = per_frame
        return result

    def _distance_at(self, track_id: int, frame: int) -> float:
        per_frame = self.distance.get(track_id, {})
        if not per_frame:
            return 0.0
        keys = [f for f in per_frame if f <= frame]
        return per_frame[max(keys)] if keys else 0.0

    def _speed_at(self, track_id: int, frame: int, window: int = 8) -> float:
        """Instantaneous speed in km/h, over a short window."""
        positions = self.player_court.get(track_id, {})
        frames = [f for f in positions if frame - window <= f <= frame]
        if len(frames) < 2:
            return 0.0
        frames.sort()
        metres = float(np.linalg.norm(positions[frames[-1]] - positions[frames[0]]))
        seconds = (frames[-1] - frames[0]) / self.fps
        if seconds <= 0 or metres > 2.0 * len(frames):
            return 0.0
        return metres / seconds * 3.6

    # -- drawing ----------------------------------------------------------

    def state_for(self, index: int) -> FrameState:
        """State for a frame, carrying the last known one forward.

        Frames after the final rally still belong to the match: falling back to
        a fresh state there would redraw the score as 0-0 for the rest of the
        video, which is the worst possible failure - it reads as a real result.
        """
        if index in self.states:
            return self.states[index]
        if self.states and index > self.last_state_frame:
            return self.states[self.last_state_frame]
        return FrameState()

    def render(self, frame: np.ndarray, index: int) -> np.ndarray:
        canvas = frame.copy()
        state = self.state_for(index)

        self._draw_court(canvas)
        self._draw_players(canvas, index)
        self._draw_ball(canvas, index)
        self._draw_scoreboard(canvas, state)
        self._draw_minimap(canvas, index, state)
        self._draw_player_strip(canvas, index)
        self._draw_rally_strip(canvas, index, state)
        self._draw_event(canvas, state)
        self._draw_point_banner(canvas, state)
        return canvas

    def _draw_court(self, canvas: np.ndarray) -> None:
        """Court lines projected from the homography, not the raw keypoints.

        Drawing the fitted model rather than the regressed points means the
        lines are geometrically consistent - and visibly wrong when the
        calibration is wrong, which is the honest failure mode.
        """
        if self.calibration is None:
            return
        try:
            image_points = self.calibration.to_image(COURT_MODEL)
        except Exception:  # noqa: BLE001 - a bad matrix must not stop the video
            return
        for a, b in COURT_LINES:
            pa, pb = image_points[a].astype(int), image_points[b].astype(int)
            cv2.line(canvas, tuple(pa), tuple(pb), LINE, 1, cv2.LINE_AA)

        # The 14 court landmarks the keypoint model exists to find: doubles and
        # singles corners, service-line intersections and centre marks. Drawn
        # from the *fitted* homography rather than the raw regressed points, so
        # a dot sitting off its line is visible evidence that the calibration
        # has drifted - the honest failure mode, and the reason the median
        # reprojection error is 0.29 px when it is working.
        for point in image_points:
            centre = (int(point[0]), int(point[1]))
            cv2.circle(canvas, centre, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, 3, KEYPOINT, -1, cv2.LINE_AA)

    def _draw_players(self, canvas: np.ndarray, index: int) -> None:
        for track_id, boxes in self.player_boxes.items():
            box = boxes.get(index)
            if box is None:
                continue
            colour = P1 if track_id == 1 else P2
            x1, y1, x2, y2 = (int(v) for v in box)
            # Corner brackets rather than a full rectangle: they mark the box
            # without hiding the player's feet, which is where the court
            # position is measured.
            run = max(int((x2 - x1) * 0.28), 8)
            for cx, sx in ((x1, 1), (x2, -1)):
                for cy, sy in ((y1, 1), (y2, -1)):
                    cv2.line(canvas, (cx, cy), (cx + sx * run, cy), colour, 2,
                             cv2.LINE_AA)
                    cv2.line(canvas, (cx, cy), (cx, cy + sy * run), colour, 2,
                             cv2.LINE_AA)

            court = self.player_court.get(track_id, {}).get(index)
            label = f"P{track_id}"
            if court is not None:
                label += f"  {court[0]:.1f}, {court[1]:.1f} m"
            speed = self._speed_at(track_id, index)
            if speed > 0.5:
                label += f"   {speed:.0f} km/h"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            _panel(canvas, x1, max(y1 - th - 10, 0), tw + 12, th + 9, 0.6)
            _text(canvas, label, (x1 + 6, max(y1 - 8, th)), 0.5, colour, 1, False)

    def _draw_ball(self, canvas: np.ndarray, index: int) -> None:
        """The ball, with a fading trail of where it has just been.

        The trail is the cheapest way to make a 15-pixel object legible at
        speed, and it doubles as a sanity check on the detector: a physically
        impossible zigzag is a false positive, visible at a glance.
        """
        trail = [
            self.ball_by_frame[f]
            for f in range(max(index - TRAIL_LENGTH, 0), index + 1)
            if f in self.ball_by_frame
        ]
        for position, record in enumerate(trail[:-1]):
            weight = (position + 1) / len(trail)
            point = (int(record["image"][0]), int(record["image"][1]))
            colour = tuple(int(c * (0.35 + 0.65 * weight)) for c in BALL)
            cv2.circle(canvas, point, max(int(2 + 3 * weight), 2), colour, -1,
                       cv2.LINE_AA)

        current = self.ball_by_frame.get(index)
        if current is None:
            return
        centre = (int(current["image"][0]), int(current["image"][1]))
        cv2.circle(canvas, centre, 11, BALL, 2, cv2.LINE_AA)
        cv2.circle(canvas, centre, 3, BALL, -1, cv2.LINE_AA)
        _text(canvas, f"{current['confidence']:.2f}",
              (centre[0] + 15, centre[1] - 10), 0.42, BALL, 1)

    def _draw_scoreboard(self, canvas: np.ndarray, state: FrameState) -> None:
        """The live score - the one thing the baseline could not show."""
        x, y, w = 18, 18, 330
        _panel(canvas, x, y, w, 92)

        _text(canvas, "SCORE", (x + 14, y + 24), 0.44, DIM, 1)
        parts = state.scoreline.split("|")
        sets_part = parts[0].strip() or "0-0"
        game_part = parts[1].strip() if len(parts) > 1 else ""
        _text(canvas, sets_part, (x + 14, y + 58), 0.95, INK, 2)
        if game_part:
            _text(canvas, game_part, (x + 150, y + 58), 0.95, GOOD, 2)

        _text(canvas, f"points {state.points_played}", (x + 14, y + 80), 0.42, DIM, 1)
        server = f"serving  P{state.server}"
        _text(canvas, server, (x + 150, y + 80), 0.42,
              P1 if state.server == 1 else P2, 1)
        cv2.circle(canvas, (x + 140, y + 76), 4,
                   P1 if state.server == 1 else P2, -1, cv2.LINE_AA)

    def _minimap_geometry(self) -> tuple[int, int, float]:
        """Origin and scale, kept fully inside the frame.

        The caption sits below the plan and the panel border outside it, so the
        margin has to account for both - at 1080p an earlier version clipped
        "COURT VIEW" off the right edge.
        """
        scale = self.minimap_scale
        margin = 26
        width = int(DOUBLES_WIDTH * scale)
        origin_x = max(self.width - width - margin, 0)
        return origin_x, 32, scale

    def _draw_minimap(self, canvas: np.ndarray, index: int,
                      state: FrameState) -> None:
        """A scale plan of the court with live positions in real metres.

        This is the view the homography buys, and the reason it is worth
        having: a bounce marker here is a measurement in metres, not a guess
        about pixels.
        """
        ox, oy, scale = self._minimap_geometry()
        w, h = int(DOUBLES_WIDTH * scale), int(COURT_LENGTH * scale)
        _panel(canvas, ox - 12, oy - 12, w + 24, h + 44, 0.9)

        def to_px(x: float, y: float) -> tuple[int, int]:
            return int(ox + x * scale), int(oy + y * scale)

        # The playing surface is filled opaque. A translucent plan over a
        # bright crowd is unreadable - this is a diagram, not a tint.
        cv2.rectangle(canvas, to_px(0, 0), to_px(DOUBLES_WIDTH, COURT_LENGTH),
                      COURT_FILL, -1)
        cv2.rectangle(canvas, to_px(0, 0), to_px(DOUBLES_WIDTH, COURT_LENGTH),
                      INK, 1, cv2.LINE_AA)
        for x in (ALLEY, DOUBLES_WIDTH - ALLEY):
            cv2.line(canvas, to_px(x, 0), to_px(x, COURT_LENGTH), COURT_MARK, 1,
                     cv2.LINE_AA)
        for y in (NET_Y - SERVICE_LINE_FROM_NET, NET_Y + SERVICE_LINE_FROM_NET):
            cv2.line(canvas, to_px(ALLEY, y), to_px(DOUBLES_WIDTH - ALLEY, y),
                     COURT_MARK, 1, cv2.LINE_AA)
        cv2.line(canvas,
                 to_px(DOUBLES_WIDTH / 2, NET_Y - SERVICE_LINE_FROM_NET),
                 to_px(DOUBLES_WIDTH / 2, NET_Y + SERVICE_LINE_FROM_NET),
                 COURT_MARK, 1, cv2.LINE_AA)
        cv2.line(canvas, to_px(-0.4, NET_Y), to_px(DOUBLES_WIDTH + 0.4, NET_Y),
                 INK, 2, cv2.LINE_AA)

        for bounce in state.bounces:
            colour = GOOD if bounce.in_bounds else BAD
            point = to_px(float(bounce.court[0]), float(bounce.court[1]))
            cv2.drawMarker(canvas, point, colour, cv2.MARKER_CROSS, 9, 2,
                           cv2.LINE_AA)

        for track_id, positions in self.player_court.items():
            court = positions.get(index)
            if court is None:
                continue
            colour = P1 if track_id == 1 else P2
            cv2.circle(canvas, to_px(float(court[0]), float(court[1])), 6,
                       colour, -1, cv2.LINE_AA)

        ball = self.ball_by_frame.get(index)
        if ball is not None:
            # Drawn hollow to say what it is: an airborne ball's court
            # coordinate is projected metres beyond the truth, so this marker
            # is the pierce point, not a ground position. Bounces - the
            # crosses - are the ones that are exact.
            cv2.circle(canvas, to_px(ball["court"][0], ball["court"][1]), 4,
                       BALL, 1, cv2.LINE_AA)

        _text(canvas, "COURT VIEW  (metres)", (ox, oy + h + 24), 0.4, DIM, 1)

    def _draw_player_strip(self, canvas: np.ndarray, index: int) -> None:
        rows = sorted(self.player_court)
        if not rows:
            return
        h = 26 * len(rows) + 30
        x, y = 18, self.height - h - 18
        _panel(canvas, x, y, 300, h)
        _text(canvas, "PLAYERS", (x + 14, y + 22), 0.44, DIM, 1)
        for offset, track_id in enumerate(rows):
            colour = P1 if track_id == 1 else P2
            line_y = y + 46 + offset * 26
            cv2.circle(canvas, (x + 20, line_y - 4), 5, colour, -1, cv2.LINE_AA)
            _text(
                canvas,
                f"P{track_id}   {self._distance_at(track_id, index):5.1f} m"
                f"   {self._speed_at(track_id, index):4.1f} km/h",
                (x + 34, line_y), 0.48, INK, 1,
            )

    def _draw_rally_strip(self, canvas: np.ndarray, index: int,
                          state: FrameState) -> None:
        x, y = 18, 18 + 92 + 12
        if state.rally_index is None:
            _panel(canvas, x, y, 330, 34, 0.6)
            _text(canvas, "between points", (x + 14, y + 23), 0.48, DIM, 1)
            return
        _panel(canvas, x, y, 330, 34, 0.66)
        elapsed = (index - (state.rally_started or index)) / self.fps
        cv2.circle(canvas, (x + 20, y + 17), 5, BAD, -1, cv2.LINE_AA)
        _text(
            canvas,
            f"RALLY {state.rally_index + 1}   shot {state.rally_shots}"
            f"   {elapsed:.1f}s",
            (x + 34, y + 23), 0.48, INK, 1,
        )

    def _draw_event(self, canvas: np.ndarray, state: FrameState) -> None:
        if state.event is None:
            return
        event = state.event
        if event.type is EventType.BOUNCE:
            text = f"BOUNCE  {'IN' if event.in_bounds else 'OUT'}"
            colour = GOOD if event.in_bounds else BAD
            detail = (f"{event.side} court   "
                      f"{event.court[0]:.1f}, {event.court[1]:.1f} m")
        else:
            text = "HIT"
            colour = WARN
            detail = f"player {event.by_player}" if event.by_player else event.side

        (tw, _), _ = cv2.getTextSize(text, FONT, 0.8, 2)
        x = self.width // 2 - tw // 2 - 16
        y = 22
        _panel(canvas, x, y, tw + 32, 58, 0.74)
        _text(canvas, text, (x + 16, y + 30), 0.8, colour, 2)
        _text(canvas, detail, (x + 16, y + 50), 0.42, DIM, 1)

    def _draw_point_banner(self, canvas: np.ndarray, state: FrameState) -> None:
        if state.point_banner is None:
            return
        text, detail, confidence = state.point_banner
        (tw, _), _ = cv2.getTextSize(text, FONT, 1.0, 2)
        width = max(tw + 40, 420)
        x = self.width // 2 - width // 2
        y = self.height // 2 - 60
        _panel(canvas, x, y, width, 108, 0.82)
        _text(canvas, text, (x + 20, y + 46), 1.0, INK, 2)
        _text(canvas, detail, (x + 20, y + 76), 0.55, GOOD, 1)
        # Confidence is shown on the banner, not buried in the report. A point
        # the system was unsure about should look unsure on the video too.
        flag = "low confidence" if confidence < 0.6 else f"confidence {confidence:.2f}"
        _text(canvas, flag, (x + 20, y + 98), 0.44,
              WARN if confidence < 0.6 else DIM, 1)


def render_video(
    source: str | Path,
    destination: str | Path,
    renderer: Renderer,
    *,
    start: int = 0,
    limit: int | None = None,
    progress_every: int = 100,
) -> int:
    """Second pass: re-decode the source and write the annotated video."""
    from tennis import video

    info = video.probe(source)
    writer = video.VideoWriter(destination, info.fps, info.width, info.height)
    written = 0
    try:
        for index, frame in video.frames(source, start, limit):
            writer.write(renderer.render(frame, index))
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"  rendered {written} frames", flush=True)
    finally:
        writer.close()
    return written
