"""Two-index-finger piano and falling tiles. Run: python piano_game_over.py.
Uses the same MediaPipe Solutions API as the supplied project.
"""
import math
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pygame

KEYBOARD_HEIGHT = 0.28
PEACE_HOLD = 0.9
PEACE_RELEASE = 0.35
# Travel from above the screen to the middle line in 1.15 seconds.
TILE_TRAVEL = 1.15
BEAT_SECONDS = 0.40
HIT_WINDOW = 0.16
# Mary Had a Little Lamb: (white-key index, duration in beats).
MELODY = [(2,1),(1,1),(0,1),(1,1),(2,1),(2,1),(2,2),
          (1,1),(1,1),(1,2),(2,1),(4,1),(4,2),
          (2,1),(1,1),(0,1),(1,1),(2,1),(2,1),(2,1),(2,1),
          (1,1),(1,1),(2,1),(1,1),(0,2)]
NOTE_LANES = {0: 2, 1: 3, 2: 4, 4: 5}
SAMPLE_RATE = 44100
NOTES = [60, 62, 64, 65, 67, 69, 71, 72]
LABELS = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
WINDOW = 'Index Piano'


def contains(rect, point):
    x, y = point
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


def layout(width, height):
    left, right = int(width * .04), int(width * .96)
    bottom = height - 12
    top = bottom - int(height * KEYBOARD_HEIGHT)
    edges = [round(left + i * (right - left) / 8) for i in range(9)]
    whites = [(edges[i], top, edges[i + 1], bottom) for i in range(8)]
    black_width = int((right - left) / 8 * .56)
    blacks = [(edges[i] - black_width // 2, top,
               edges[i] + black_width // 2, top + int((bottom - top) * .61))
              for i in (1, 2, 4, 5, 6)]
    return whites, blacks


def hit_key(whites, blacks, point):
    # Decorative black keys are silent, including the white area underneath.
    if any(contains(rect, point) for rect in blacks):
        return None
    return next((i for i, rect in enumerate(whites) if contains(rect, point)), None)


class Audio:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(64)
        self.channels = [pygame.mixer.Channel(i) for i in range(64)]
        self.ages = [0] * 64
        self.serial = 0
        self.sounds = []
        t = np.arange(SAMPLE_RATE * 16, dtype=np.float64) / SAMPLE_RATE
        for note in NOTES:
            frequency = 440 * 2 ** ((note - 69) / 12)
            data = np.zeros_like(t)
            for harmonic, amplitude in enumerate([1, .6, .3, .15, .08, .04], 1):
                data += amplitude * np.sin(2 * np.pi * frequency * harmonic * t) * np.exp(
                    -t * (.65 + .22 * (harmonic - 1)))
            data *= np.minimum(t / .003, 1)
            data[-4410:] *= np.linspace(1, 0, 4410)
            data *= .10 / max(np.max(np.abs(data)), 1e-12)
            mono = (data * 32767).astype(np.int16)
            self.sounds.append(pygame.sndarray.make_sound(
                np.ascontiguousarray(np.column_stack((mono, mono)))))

    def play(self, note):
        index = next((i for i, channel in enumerate(self.channels)
                      if not channel.get_busy()), None)
        if index is None:
            # Only at capacity: replace the oldest of 64 natural-decay voices.
            index = min(range(len(self.ages)), key=self.ages.__getitem__)
        self.serial += 1
        self.ages[index] = self.serial
        self.channels[index].play(self.sounds[note], loops=0)


class PeaceToggle:
    def __init__(self):
        self.started = None
        self.released = None
        self.armed = True

    def update(self, peace, now):
        if not peace:
            self.started = None
            if self.released is None:
                self.released = now
            if now - self.released >= PEACE_RELEASE:
                self.armed = True
            return False, 0.0
        self.released = None
        if not self.armed:
            return False, 0.0
        if self.started is None:
            self.started = now
        progress = min(1.0, (now - self.started) / PEACE_HOLD)
        if progress >= 1:
            self.armed = False
            self.started = None
            return True, 0.0
        return False, progress


def extended(points, tip):
    base, joint, end = points[tip - 3], points[tip - 2], points[tip]
    first, second = base - joint, end - joint
    cosine = np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-9)
    return cosine < -.65 and np.linalg.norm(end - points[0]) > np.linalg.norm(joint - points[0])


def is_peace(points):
    return (extended(points, 8) and extended(points, 12)
            and not extended(points, 16) and not extended(points, 20))


@dataclass
class Tile:
    lane: int
    note: int
    target: float

    def rectangle(self, whites, height, now):
        left, _, right, _ = whites[self.lane]
        tile_height = max(36, int(height * .09))
        line = height // 2
        speed = (line + tile_height / 2) / TILE_TRAVEL
        center = line + (now - self.target) * speed
        top = round(center - tile_height / 2)
        return left + 5, top, right - 5, top + tile_height


class Game:
    def __init__(self):
        self.active = False
        self.tiles = []
        self.hits = self.misses = 0
        self.next_phrase = 0
        self.feedback = ''
        self.feedback_until = 0
        self.over = False
        self.score = 0

    def toggle(self, now):
        self.active = True if self.over else not self.active
        self.over = False
        self.score = 0
        self.tiles.clear()
        self.hits = self.misses = 0
        self.feedback = ''
        self.next_phrase = now + TILE_TRAVEL + .5

    def update(self, now, whites, height, fingers):
        played = []
        if not self.active or self.over:
            return played
        if now >= self.next_phrase - TILE_TRAVEL:
            # Absolute target times preserve the melody's rhythm despite frame jitter.
            target = max(self.next_phrase, now + TILE_TRAVEL)
            for note, beats in MELODY:
                self.tiles.append(Tile(NOTE_LANES[note], note, target))
                target += beats * BEAT_SECONDS
            self.next_phrase = target + 2 * BEAT_SECONDS
        remaining = []
        used_fingers = set()
        for tile in self.tiles:
            error = now - tile.target
            rect = tile.rectangle(whites, height, now)
            band = max(16, round(height * .045))
            finger = next((i for i, point in enumerate(fingers)
                           if i not in used_fingers
                           and -HIT_WINDOW <= error <= 0
                           and abs(point[1] - height // 2) <= band
                           and contains(rect, point)), None)
            if finger is not None:
                used_fingers.add(finger)
                played.append(tile.note)
                self.hits += 1
                self.score += 100 if abs(error) < .07 else 70
                self.feedback = 'Perfect!' if abs(error) < .07 else 'Good!'
                self.feedback_until = now + .4
            elif error > 0:
                # The tile's centre crossing the line is the visible deadline.
                # No late hits are accepted, even on a slow camera frame.
                self.misses += 1
                self.over = True
                self.tiles.clear()
                return played
            else:
                remaining.append(tile)
        self.tiles = remaining
        return played


def text(frame, message, x, y, scale=.55, color=(255, 255, 255)):
    length = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    scale *= min(1, (frame.shape[1] - x - 8) / max(length, 1))
    for thickness, shade in ((1, color),):
        cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, shade, thickness, cv2.LINE_AA)


def draw_keyboard(frame, whites, blacks, pressed):
    left, top, _, bottom = whites[0]
    right = whites[-1][2]
    cv2.rectangle(frame, (left - 6, top - 8), (right + 6, bottom + 6), (35, 35, 140), -1)
    for i, (x1, y1, x2, y2) in enumerate(whites):
        color = (190, 235, 245) if i in pressed else (245, 248, 250)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2), (65, 65, 65), 1)
        cv2.line(frame, (x1 + 2, y2 - 4), (x2 - 3, y2 - 4), (185, 185, 185), 3)
        cv2.putText(frame, LABELS[i], (x1 + 7, y2 - 13), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (40, 40, 40), 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in blacks:
        cv2.rectangle(frame, (x1 + 3, y1), (x2 + 3, y2 + 4), (100, 100, 100), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 23), -1)
        cv2.line(frame, (x1 + 2, y1 + 3), (x1 + 2, y2 - 4), (85, 85, 88), 2)
        cv2.line(frame, (x1 + 3, y2 - 3), (x2 - 3, y2 - 3), (65, 65, 70), 2)


def draw_result(frame, game, progress):
    """Quiet, regular-weight results overlay over the live camera."""
    height, width = frame.shape[:2]
    shade = np.full_like(frame, (23, 22, 20))
    cv2.addWeighted(frame, .22, shade, .78, 0, frame)
    left, right = int(width * .13), int(width * .87)
    top, bottom = int(height * .14), int(height * .88)
    cv2.rectangle(frame, (left, top), (right, bottom), (37, 35, 32), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (80, 76, 70), 1)

    def centered(message, fraction, scale=.55, color=(232, 232, 232)):
        scale *= min(width / 760, height / 520)
        size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        cv2.putText(frame, message, ((width - size[0]) // 2, int(height * fraction)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    centered('Game over', .27, 1.1)
    centered('A note crossed the line.', .35, .52, (170, 170, 170))
    cv2.line(frame, (left + 24, int(height * .40)),
             (right - 24, int(height * .40)), (72, 69, 65), 1)
    centered(f'Score  {game.score}     Notes played  {game.hits}', .49, .60)
    completion = 100 * game.hits / max(1, game.hits + game.misses)
    centered(f'Hit rate  {completion:.0f}%     Best combo  {game.hits}', .56, .48,
             (180, 180, 180))
    button = (int(width * .29), int(height * .64), int(width * .71), int(height * .74))
    cv2.rectangle(frame, button[:2], button[2:], (100, 105, 105), 1)
    centered('Hold V to try again', .70, .53)
    if progress:
        cv2.line(frame, (button[0], button[3]),
                 (button[0] + int((button[2] - button[0]) * progress), button[3]),
                 (170, 210, 190), 2)
    centered('Q  Back to piano    /    ESC  Exit', .82, .43, (165, 165, 165))


def main():
    cap = hands = None
    try:
        if not hasattr(mp, 'solutions'):
            raise RuntimeError('This program needs MediaPipe with mp.solutions, as in your original environment.')
        audio = Audio()
        hands = mp.solutions.hands.Hands(max_num_hands=2,
                    min_detection_confidence=.7, min_tracking_confidence=.7)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Check camera permission and close other camera apps.')
        game, toggle = Game(), PeaceToggle()
        previous = set()
        blocked = set()
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError('Webcam stopped returning frames.')
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.perf_counter()
            whites, blacks = layout(width, height)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = result.multi_hand_landmarks or []
            fingers, peace_flags = [], []
            for hand in landmarks:
                points = np.array([(p.x * width, p.y * height, p.z * width) for p in hand.landmark])
                peace = is_peace(points)
                peace_flags.append(peace)
                if not peace:
                    fingers.append(tuple(points[8, :2].astype(int)))
            changed, progress = toggle.update(any(peace_flags), now)
            occupied = {key for point in fingers if (key := hit_key(whites, blacks, point)) is not None}
            if changed:
                game.toggle(now)
                previous.clear()
                blocked = occupied.copy()
            if game.active and not game.over:
                if not changed:
                    for note in game.update(now, whites, height, fingers):
                        audio.play(note)
                for index in range(2, 7):
                    x = whites[index][0]
                    cv2.line(frame, (x, 95), (x, height), (100, 100, 100), 1)
                for tile in game.tiles:
                    if tile.target - now > TILE_TRAVEL:
                        continue
                    x1, y1, x2, y2 = tile.rectangle(whites, height, now)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 243, 250), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 155, 205), 2)
                    if y2 > 20:
                        cv2.putText(frame, LABELS[tile.note], (x1 + 8, y2 - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1, cv2.LINE_AA)
                text(frame, f'GAME | Hits: {game.hits}   Missed: {game.misses}', 12, 26)
                text(frame, 'Mary Had a Little Lamb | Touch tiles at the middle line', 12, 49, .45)
                line = height // 2
                cv2.line(frame, (whites[2][0], line), (whites[5][2], line), (60, 220, 255), 2)
                text(frame, 'PLAY HERE', 12, line + 5, .45, (60, 220, 255))
                if now < game.feedback_until:
                    text(frame, game.feedback, 12, line + 35, .5)
                for note, lane in NOTE_LANES.items():
                    text(frame, LABELS[note], whites[lane][0] + 8, height - 15, .42)
            elif not game.over:
                blocked &= occupied
                current = occupied - blocked
                for note in current - previous:
                    audio.play(note)
                previous = current
                draw_keyboard(frame, whites, blacks, current)
                text(frame, 'PIANO | White keys play; black keys are decorative', 12, 26)
                text(frame, 'Use only your two index fingers', 12, 49, .45)
            action = 'RETRY' if game.over else ('STOP' if game.active else 'START')
            text(frame, f'Hold V to {action} | Release V before repeating | Q: piano | ESC: exit', 12, 72, .42)
            if progress:
                cv2.rectangle(frame, (12, 80), (12 + int(progress * (width - 24)), 86), (90, 245, 160), -1)
            for hand in landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            for point in fingers:
                cv2.circle(frame, point, 8, (60, 255, 100), -1, cv2.LINE_AA)
            if game.over:
                draw_result(frame, game, progress)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 255
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('q'), ord('Q')) and game.active:
                game.active = False
                game.over = False
                game.tiles.clear()
                previous.clear()
                blocked = occupied.copy()
    except (RuntimeError, pygame.error) as error:
        print(f'Piano could not continue: {error}')
    finally:
        if cap is not None:
            cap.release()
        if hands is not None:
            hands.close()
        pygame.mixer.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
