"""
INVISIBLE INSTRUMENTS - GAME OF MUSIC  (OpenCV + MediaPipe + pygame)

Play a piano in the air with your fingertips, or play the 3-level tile game.

Controls
  Touch a button and hold for 1 second (or press SPACE) ... select
  H or M ... back to the welcome screen        F ... free play
  1 / 2  ... Pentatonic / Major scale (free play only)     ESC ... exit
"""
import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame


def get_screen_size(fallback_width, fallback_height):
    """Return the Windows desktop size, with a safe fallback elsewhere."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if width > 0 and height > 0:
            return width, height
    except (AttributeError, OSError):
        pass
    return fallback_width, fallback_height

# In-memory audio avoids stale WAV caches. Held notes decay, never loop.
SAMPLE_RATE = 44100
NOTE_DURATION = 16.0          # free-play notes (original value)
GAME_NOTE_DURATION = 8.0      # organ notes for the Dracula level
RELEASE_MS = 90
VOICE_COUNT = 32
BASE_FREQ = 261.63            # middle C

# Each scale = semitone steps above middle C + key labels (+ sound type).
SCALES = {
    'Pentatonic': {'steps': [0, 2, 4, 7, 9, 12, 14, 16],
                   'names': ['C', 'D', 'E', 'G', 'A', 'C2', 'D2', 'E2'],
                   'organ': False, 'duration': NOTE_DURATION},
    'Major':      {'steps': [0, 2, 4, 5, 7, 9, 11, 12],
                   'names': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2'],
                   'organ': False, 'duration': NOTE_DURATION},
    # C-minor keyboard with an organ sound for the Beethoven's 5th level.
    'Epic':       {'steps': [0, 2, 3, 5, 7, 8, 10, 12],
                   'names': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb', 'C2'],
                   'organ': True, 'duration': GAME_NOTE_DURATION},
    # D-minor style keyboard with an organ sound for the Dracula level.
    'Dracula':    {'steps': [1, 2, 4, 5, 7, 9, 10, 14],
                   'names': ['C#', 'D', 'E', 'F', 'G', 'A', 'Bb', 'D2'],
                   'organ': True, 'duration': GAME_NOTE_DURATION},
}

# ─────────────────────────────────────────
# GAME SETTINGS (easy to tune live)
# ─────────────────────────────────────────
DWELL_SECONDS = 1.0       # hold a fingertip on a button this long to click it
COUNTDOWN_SECONDS = 3.0
HIT_WINDOW = 0.65         # +/- seconds around a tile that still counts as a hit
PERFECT_WINDOW = 0.42     # inside this = PERFECT (bigger = easier)
PASS_RATIO = 0.40         # need to hit 40% of the tiles to pass a level
SPEED_FACTOR = 0.22       # slower tile speed for relaxed, beginner-friendly play

# Songs are plain text: NOTE[:beats]  (R = rest, default 1 beat).
# Bach's Toccata (the classic "Dracula" organ theme), Twinkle Twinkle and
# Beethoven's 5th are all public-domain melodies.
_FIFTH = "G G G Eb:4 R:1 F F F D:4 R:2 "
_TOCCATA = "A G A:2 R:1 G F E D C#:2 D:4 "
SONGS = [
{'title': "Beethoven's 5th", 'subtitle': 'Fate knocks at the door - C minor',
     'scale': 'Epic', 'bpm': 88,
     'notes': _FIFTH + _FIFTH + "C Eb G C2:4"},
{'title': 'Jana Gana Mana', 'subtitle': "India's National Anthem",
     'scale': 'Major', 'bpm': 96,
     'notes': "G A G E:2  G A B C2:2  B A G:2  A G E:2  "
              "D E G:2  G A G E:2  G A B C2:2  B A G:2  "
              "A G E:2  D E G:2  A A B C2:2  B A G:2  "
              "E G A:2  G E D:2  E G A:2  G E D:2  "
              "G A B C2:2  B A G:2  A G E:2  D E G:4"},
{'title': 'Dracula Theme', 'subtitle': 'Toccata in D minor (Bach)',
     'scale': 'Dracula', 'bpm': 84,
     'notes': _TOCCATA + "R:1 " + _TOCCATA + "R:1 D F A D2:4"}
]


def generate_piano_note(frequency, duration=NOTE_DURATION, organ=False):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    data = np.zeros_like(t)
    if organ:
        # Organ: strong harmonics, slow decay, soft attack.
        for harmonic, amplitude in enumerate([1., .7, .5, .4, .25, .15], 1):
            data += amplitude * np.sin(2*np.pi*frequency*harmonic*t) * np.exp(-t*.25)
        data *= np.minimum(t/.03, 1.)
    else:
        for harmonic, amplitude in enumerate([1., .6, .3, .15, .08, .04], 1):
            # Higher harmonics decay faster, gradually softening the tone.
            data += amplitude * np.sin(2*np.pi*frequency*harmonic*t) * np.exp(-t*(.65+.22*(harmonic-1)))
        data *= np.minimum(t/.003, 1.)
    n = int(SAMPLE_RATE*.1)
    data[-n:] *= np.linspace(1., 0., n)
    data *= .10 / max(float(np.max(np.abs(data))), 1e-12)
    mono = (data*32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack((mono, mono))))


class PianoAudio:
    """Track each strike separately so rapid re-strikes can overlap release tails."""
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(VOICE_COUNT)
        self.channels = [pygame.mixer.Channel(i) for i in range(VOICE_COUNT)]
        # Pre-generate every sound bank at startup so switching is instant.
        self.banks = {
            name: [generate_piano_note(BASE_FREQ * 2**(s/12), sc['duration'], sc['organ'])
                   for s in sc['steps']]
            for name, sc in SCALES.items()
        }
        self.scale_name = 'Major'
        self.sounds = self.banks[self.scale_name]
        self.held = {}
        self.voices = {}
        self.serial = 0

    def set_scale(self, name):
        """Silence everything, then swap to another sound bank."""
        pygame.mixer.stop()
        self.held.clear()
        self.voices.clear()
        self.scale_name = name
        self.sounds = self.banks[name]

    def release(self, key):
        i = self.held.pop(key, None)
        if i is not None:
            self.channels[i].fadeout(RELEASE_MS)
            self.voices[i] = (self.voices[i][0], True)

    def press(self, key):
        if key in self.held:
            return
        i = next((i for i, c in enumerate(self.channels) if not c.get_busy()), None)
        if i is None:
            # Steal oldest released voice first; keep held chords when possible.
            i = min(self.voices, key=lambda i: (not self.voices[i][1], self.voices[i][0]))
            self.channels[i].stop()
        # Remove any stale owner before reusing a finished or stolen channel.
        for old_key, old_i in list(self.held.items()):
            if old_i == i:
                del self.held[old_key]
        self.serial += 1
        self.voices[i] = (self.serial, False)
        self.held[key] = i
        self.channels[i].set_volume(1.)
        self.channels[i].play(self.sounds[key], loops=0)

    def update(self, current, previous):
        for key in previous-current:
            self.release(key)
        for key in current-previous:
            self.press(key)

    def strike(self, key):
        """Play a one-shot note only for a successful falling-tile hit."""
        i = next((i for i, c in enumerate(self.channels) if not c.get_busy()), None)
        if i is None:
            i = min(self.voices, key=lambda i: (not self.voices[i][1], self.voices[i][0]))
            self.channels[i].stop()
        self.serial += 1
        self.voices[i] = (self.serial, True)
        self.channels[i].set_volume(1.)
        self.channels[i].play(self.sounds[key], loops=0)

    def close(self):
        pygame.mixer.stop()
        pygame.mixer.quit()


# ─────────────────────────────────────────
# GAME LOGIC (no drawing in here)
# ─────────────────────────────────────────
def parse_song(text, names, bpm):
    """'C C G:2 R:1' -> [(key_index, start_seconds, length_seconds), ...]"""
    beat = 60.0 / bpm
    t, events = 0.0, []
    for token in text.split():
        name, _, b = token.partition(':')
        beats = float(b) if b else 1.0
        if name != 'R':
            events.append((names.index(name), t, beats * beat))
        t += beats * beat
    return events, t


class Game:
    def __init__(self, frame_h, y_hit):
        self.speed = SPEED_FACTOR * frame_h           # pixels per second
        self.y_hit = y_hit                            # the white "hit line"
        self.lead = y_hit / self.speed + 0.3          # first tile starts offscreen
        self.levels = [parse_song(s['notes'], SCALES[s['scale']]['names'], s['bpm'])[0]
                       for s in SONGS]
        self.level = 0
        self.notes, self.popups = [], []
        self.score = self.combo = self.best = self.hits = self.misses = 0
        self.t0 = self.end_T = 0.0

    def start(self, level, t0):
        self.level, self.t0 = level, t0
        self.notes = [{'key': k, 'T': t0 + self.lead + on, 'len': ln, 'state': 'pending'}
                      for k, on, ln in self.levels[level]]
        self.end_T = self.notes[-1]['T']
        self.popups = []
        self.score = self.combo = self.best = self.hits = self.misses = 0

    @property
    def total(self):
        return len(self.notes)

    def on_press(self, key, now):
        """Score the earliest matching tile; return True when it is hit."""
        for n in self.notes:
            if n['state'] == 'pending' and n['key'] == key and abs(now - n['T']) <= HIT_WINDOW:
                n['state'] = 'hit'
                # A successful tile hit is always PERFECT, making the game
                # forgiving and consistent when the player touches the falling bar.
                self.hits += 1
                self.combo += 1
                self.best = max(self.best, self.combo)
                self.score += 100 + 5 * min(self.combo, 10)
                self.popups.append({'text': 'PERFECT!',
                                    'key': key, 'born': now})
                return True
        return False

    def update(self, now):
        for n in self.notes:
            if n['state'] == 'pending' and now > n['T'] + HIT_WINDOW:
                n['state'] = 'miss'
                self.misses += 1
                self.combo = 0
                self.popups.append({'text': 'MISS', 'key': n['key'], 'born': now})
        self.popups = [p for p in self.popups if now - p['born'] < 0.8]

    def finished(self, now):
        return all(n['state'] != 'pending' for n in self.notes) and now > self.end_T + 1.0

    @property
    def accuracy(self):
        return self.hits / max(self.total, 1)

    @property
    def passed(self):
        return self.accuracy >= PASS_RATIO

    @property
    def stars(self):
        if not self.passed:
            return 0
        return 3 if self.accuracy >= 0.9 else 2 if self.accuracy >= 0.65 else 1


class DwellButtons:
    """Fingertip 'click': keep a fingertip inside a button for DWELL_SECONDS."""
    def __init__(self):
        self.start = {}

    def update(self, buttons, fingers, now, enabled=True):
        clicked = None
        for b in buttons:
            inside = enabled and any(b['x1'] < fx < b['x2'] and b['y1'] < fy < b['y2']
                                     for fx, fy in fingers)
            if inside:
                t0 = self.start.setdefault(b['id'], now)
                b['progress'] = min((now - t0) / DWELL_SECONDS, 1.0)
                if now - t0 >= DWELL_SECONDS and clicked is None:
                    clicked = b['id']
            else:
                self.start.pop(b['id'], None)
                b['progress'] = 0.0
        if clicked:
            self.start.clear()
        return clicked


# ─────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────
def hue_bgr(h, s=255, v=255):
    return tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[int(h) % 180, s, v]]]), cv2.COLOR_HSV2BGR)[0][0])


def put_text(frame, text, x, y, scale=1.0, color=(255, 255, 255), thick=2, center=False):
    s = frame.shape[1] / 640.0
    font = cv2.FONT_HERSHEY_DUPLEX
    th = max(1, int(thick * s))
    (tw, _), _ = cv2.getTextSize(text, font, scale * s, th)
    if center:
        x = x - tw // 2
    cv2.putText(frame, text, (int(x) + 2, int(y) + 2), font, scale * s, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (int(x), int(y)), font, scale * s, color, th, cv2.LINE_AA)
    return tw


def blend_rect(frame, x1, y1, x2, y2, color, alpha):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    colored = np.empty_like(roi)
    colored[:] = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, colored, alpha, 0)


def draw_star(frame, cx, cy, r, color):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((int(cx + rad * math.cos(ang)), int(cy + rad * math.sin(ang))))
    cv2.fillPoly(frame, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def draw_button(frame, b, now):
    x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
    blend_rect(frame, x1, y1, x2, y2, (0, 0, 0), 0.6)
    blend_rect(frame, x1, y1, x1 + int((x2 - x1) * b['progress']), y2, (0, 220, 255), 0.7)
    cv2.rectangle(frame, (x1, y1), (x2, y2), hue_bgr(now * 80), 3)
    s = frame.shape[1] / 640.0
    scale = 1.2 * (y2 - y1) / (80.0 * s)            # start from the box height
    while scale > 0.3:                              # shrink until the text fits the box
        (tw, th_), _ = cv2.getTextSize(b['label'], cv2.FONT_HERSHEY_DUPLEX, scale * s,
                                       max(1, int(2 * s)))
        if tw <= 0.85 * (x2 - x1) and th_ <= 0.6 * (y2 - y1):
            break
        scale *= 0.92
    put_text(frame, b['label'], (x1 + x2) // 2, (y1 + y2) // 2 + int((y2 - y1) * 0.16),
             scale, (255, 255, 255), 2, center=True)


def keys_touched(keys, fingers, pad_x=0, pad_up=0, pad_down=0):
    """Keys touched by any fingertip. The pads make the invisible touch zone bigger
    than the drawn key (sideways, upwards, downwards) so hits are easier."""
    pressed = set()
    for fx, fy in fingers:
        for i, (x1, y1, x2, y2) in enumerate(keys):
            if x1 - pad_x < fx < x2 + pad_x and y1 - pad_up < fy < y2 + pad_down:
                pressed.add(i)
    return pressed


def draw_keys(frame, keys, pressed, key_colors):
    for i, (x1, y1, x2, y2) in enumerate(keys):
        touching = i in pressed
        color = key_colors[i] if touching else (255, 255, 255)
        blend_rect(frame, x1, y1, x2, y2, color, 0.55 if touching else 0.25)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color if touching else (255, 255, 255), 2)


def draw_labels(frame, keys, names):
    for i, (x1, y1, x2, y2) in enumerate(keys):
        put_text(frame, names[i], (x1 + x2) // 2, y2 - 12, 0.7, (255, 255, 255), 2, center=True)


def make_confetti(w, h, count=120):
    pieces = []
    for _ in range(count):
        pieces.append({'x': random.uniform(0, w), 'y': random.uniform(-h, 0),
                       'vy': random.uniform(80, 220), 's': random.randint(4, 9),
                       'c': hue_bgr(random.randint(0, 179))})
    return pieces


class FlashyBackground:
    """Cycling rainbow gradient that pulses to the beat (cheap: built tiny, then upscaled)."""
    def __init__(self, w, h):
        self.w, self.h = w, h
        sw, sh = max(w // 4, 1), max(h // 4, 1)
        ys, xs = np.mgrid[0:sh, 0:sw]
        self.base = (xs / sw * 70 + ys / sh * 50).astype(np.float32)
        self.hsv = np.empty((sh, sw, 3), np.uint8)
        self.hsv[..., 1] = 255

    def render(self, frame, t, pulse, alpha=0.10):
        # Keep the live webcam image (and the player) clearly visible.
        # A very light, steady tint replaces the distracting rainbow animation.
        tint = np.zeros_like(frame)
        tint[:] = (42, 48, 54)
        return cv2.addWeighted(frame, 1 - alpha, tint, alpha, 0)


def main():
    audio = hands = cap = None
    try:
        print("Loading sounds, please wait a few seconds...")
        audio = PianoAudio()
        # ─────────────────────────────────────────
        # MEDIAPIPE INIT (Tracking up to 2 hands)
        # ─────────────────────────────────────────
        mp_hands = mp.solutions.hands
        hands    = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        mp_draw = mp.solutions.drawing_utils

        # ─────────────────────────────────────────
        # WEBCAM + KEY LAYOUTS
        # ─────────────────────────────────────────
        cap = cv2.VideoCapture(0)
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            raise RuntimeError("Could not read from webcam 0.")
        FRAME_H, FRAME_W = test_frame.shape[:2]
        SCREEN_W, SCREEN_H = get_screen_size(FRAME_W, FRAME_H)

        # Create a true fullscreen window and scale each displayed camera frame
        # to the desktop size. Game and hand-tracking coordinates still use
        # the original camera resolution.
        cv2.namedWindow("Virtual Piano", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Virtual Piano", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        S = FRAME_W / 640.0

        NUM_KEYS    = 8
        GAP         = 6
        KEY_W       = (FRAME_W - GAP * (NUM_KEYS + 1)) // NUM_KEYS
        total_width = NUM_KEYS * KEY_W + (NUM_KEYS - 1) * GAP
        START_X     = (FRAME_W - total_width) // 2

        def build_keys(y, h):
            return [(START_X + i * (KEY_W + GAP), y, START_X + i * (KEY_W + GAP) + KEY_W, y + h)
                    for i in range(NUM_KEYS)]

        keys_free = build_keys(20, 160)                        # free play: keys at top
        GAME_KEY_H = int(FRAME_H * 0.34)           # taller keys = bigger target
        GAME_KEY_Y = FRAME_H - GAME_KEY_H - 20
        keys_game = build_keys(GAME_KEY_Y, GAME_KEY_H)         # game: keys at bottom
        key_colors = [hue_bgr(180 * i / NUM_KEYS) for i in range(NUM_KEYS)]

        fingertip_ids = [8]      # index fingertip only; one index per hand = two controls total

        game = Game(FRAME_H, GAME_KEY_Y + int(GAME_KEY_H * 0.4))
        background = FlashyBackground(FRAME_W, FRAME_H)
        dwell = DwellButtons()
        sparkles = [[random.uniform(0, FRAME_W), random.uniform(0, FRAME_H),
                     random.uniform(20, 70), random.randint(1, 3)] for _ in range(45)]
        confetti = []
        mid_x = FRAME_W // 2

        def button(bid, label, cy, w, h):
            return {'id': bid, 'label': label, 'progress': 0.0,
                    'x1': mid_x - w // 2, 'y1': cy - h // 2,
                    'x2': mid_x + w // 2, 'y2': cy + h // 2}

        welcome_buttons = [button('play', 'PLAY', int(FRAME_H * .62), int(300 * S), int(80 * S)),
                           button('free', 'FREE PLAY', int(FRAME_H * .82), int(220 * S), int(50 * S))]
        result_buttons = []
        home_button = {'id': 'home', 'label': 'BACK TO HOME', 'progress': 0.0,
                       'x1': FRAME_W - int(205 * S), 'y1': int(16 * S),
                       'x2': FRAME_W - int(12 * S), 'y2': int(60 * S)}

        # ─────────────────────────────────────────
        # STATE: welcome | free | countdown | playing | result
        # ─────────────────────────────────────────
        state = 'welcome'
        now = time.perf_counter()
        state_t = prev_now = now
        level = 0
        last_pressed_keys = set()
        audio.set_scale('Major')

        def enter(new_state, t):
            nonlocal state, state_t
            state, state_t = new_state, t
            dwell.start.clear()

        def start_level(n, t):
            nonlocal level, last_pressed_keys
            level = n
            audio.set_scale(SONGS[n]['scale'])
            last_pressed_keys = set()
            enter('countdown', t)

        def go_home(t):
            nonlocal last_pressed_keys
            audio.set_scale('Major')
            last_pressed_keys = set()
            enter('welcome', t)

        def go_free(t):
            nonlocal last_pressed_keys
            audio.set_scale('Pentatonic')
            last_pressed_keys = set()
            enter('free', t)

        def handle(action, t):
            if action in ('play', 'again'):
                start_level(0, t)
            elif action == 'next':
                start_level(level + 1, t)
            elif action == 'retry':
                start_level(level, t)
            elif action == 'home':
                go_home(t)
            elif action == 'free':
                go_free(t)

        # ─────────────────────────────────────────
        # MAIN LOOP
        # ─────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            dt = min(now - prev_now, 0.1)
            prev_now = now

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            finger_positions = []
            hand_list = results.multi_hand_landmarks or []
            for hand_landmarks in hand_list:
                for tip_id in fingertip_ids:
                    tip = hand_landmarks.landmark[tip_id]
                    finger_positions.append((int(tip.x * FRAME_W), int(tip.y * FRAME_H)))

            # ── Which keys are touched? (only where keys exist) ──
            current_pressed_keys = set()
            if state == 'free':
                current_pressed_keys = keys_touched(keys_free, finger_positions)
            elif state in ('countdown', 'playing'):
                current_pressed_keys = keys_touched(keys_game, finger_positions,
                                                    pad_x=GAP // 2 + 1,
                                                    pad_up=int(GAME_KEY_H * 0.45),
                                                    pad_down=FRAME_H)
            new_presses = current_pressed_keys - last_pressed_keys

            # Free Play sounds every touched key. In song-game mode, only a
            # press matching a falling tile makes a note; stray lane touches
            # and fingers resting on unused keys stay silent.
            if state == 'free':
                audio.update(current_pressed_keys, last_pressed_keys)
            elif last_pressed_keys:
                audio.update(set(), last_pressed_keys)
            last_pressed_keys = current_pressed_keys

            # ── State logic ──
            clicked = None
            if state == 'welcome':
                clicked = dwell.update(welcome_buttons, finger_positions, now, now - state_t > 0.8)
            elif state == 'result':
                clicked = dwell.update(result_buttons, finger_positions, now, now - state_t > 1.5)
            elif state == 'countdown':
                if now - state_t >= COUNTDOWN_SECONDS:
                    game.start(level, now)
                    enter('playing', now)
            elif state == 'playing':
                for k in new_presses:
                    if game.on_press(k, now):
                        audio.strike(k)
                game.update(now)
                if game.finished(now):
                    if game.passed:
                        confetti = make_confetti(FRAME_W, FRAME_H)
                        main_id, main_label = (('next', 'NEXT LEVEL') if level < len(SONGS) - 1
                                               else ('again', 'PLAY AGAIN'))
                    else:
                        confetti = []
                        main_id, main_label = 'retry', 'TRY AGAIN'
                    result_buttons = [button(main_id, main_label, int(FRAME_H * .76), int(300 * S), int(70 * S)),
                                      button('home', 'BACK TO HOME', int(FRAME_H * .92), int(210 * S), int(44 * S))]
                    enter('result', now)
            if clicked:
                handle(clicked, now)

            # The persistent Back to Home control is available during free play and levels.
            if state in ('free', 'countdown', 'playing'):
                home_clicked = dwell.update([home_button], finger_positions, now, True)
                if home_clicked:
                    handle('home', now)

            # ── Drawing ──
            def draw_hands():
                for hl in hand_list:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                for fx, fy in finger_positions:
                    cv2.circle(frame, (fx, fy), int(9 * S), (0, 255, 0), -1)

            if state == 'free':
                draw_hands()
                draw_keys(frame, keys_free, current_pressed_keys, key_colors)
                draw_labels(frame, keys_free, SCALES[audio.scale_name]['names'])
                put_text(frame, "Virtual Piano  [" + audio.scale_name + "]", 10, FRAME_H - 50, 1.0, (0, 200, 255), 2)
                put_text(frame, "1 = Pentatonic, 2 = Major | M = menu | ESC = exit", 10, FRAME_H - 20,
                         0.55, (180, 180, 180), 1)
            else:
                bpm = SONGS[level]['bpm'] if state in ('countdown', 'playing') else 110
                pulse = 0.5 + 0.5 * math.sin(2 * math.pi * now * bpm / 60.0)
                frame = background.render(frame, now, pulse)
                for sp in sparkles:
                    sp[1] = (sp[1] + sp[2] * dt) % FRAME_H
                    cv2.circle(frame, (int(sp[0]), int(sp[1])), sp[3], (255, 255, 255), -1)
                cv2.rectangle(frame, (3, 3), (FRAME_W - 4, FRAME_H - 4), (150, 165, 175), 2)

                if state == 'welcome':
                    put_text(frame, "WELCOME TO THE", mid_x, int(FRAME_H * .15), 1.0, (255, 255, 255), 2, center=True)
                    put_text(frame, "GAME OF MUSIC", mid_x, int(FRAME_H * .31), 2.2, (210, 220, 230), 5, center=True)
                    put_text(frame, "Play famous tunes with just your fingers!", mid_x, int(FRAME_H * .41),
                             0.7, (255, 255, 255), 1, center=True)
                    put_text(frame, "Use either index finger to hit a tile at the white line",
                             mid_x, int(FRAME_H * .48), 0.5, (230, 230, 230), 1, center=True)
                    for b in welcome_buttons:
                        draw_button(frame, b, now)
                    put_text(frame, "Hold a fingertip on a button for 1 second  |  or press SPACE",
                             mid_x, int(FRAME_H * .96), 0.5, (255, 255, 255), 1, center=True)
                    draw_hands()

                elif state in ('countdown', 'playing'):
                    names = SCALES[SONGS[level]['scale']]['names']
                    for i, (x1, y1, x2, y2) in enumerate(keys_game):
                        blend_rect(frame, x1, 0, x2, GAME_KEY_Y, key_colors[i], 0.15)   # lanes
                    draw_keys(frame, keys_game, current_pressed_keys, key_colors)
                    cv2.line(frame, (START_X, game.y_hit), (START_X + total_width, game.y_hit),
                             (255, 255, 255), 2)

                    if state == 'playing':
                        for n in game.notes:
                            yb = int(game.y_hit + (now - n['T']) * game.speed)
                            yt = yb - max(28, int(n['len'] * game.speed) - 10)
                            if yb < 0 or yt > GAME_KEY_Y + GAME_KEY_H:
                                continue
                            x1, _, x2, _ = keys_game[n['key']]
                            fill = {'pending': key_colors[n['key']], 'hit': (255, 255, 255),
                                    'miss': (80, 80, 80)}[n['state']]
                            cv2.rectangle(frame, (x1 + 4, max(yt, 0)), (x2 - 4, min(yb, FRAME_H)), fill, -1)
                            cv2.rectangle(frame, (x1 + 4, max(yt, 0)), (x2 - 4, min(yb, FRAME_H)), (0, 0, 0), 2)
                        for p in game.popups:
                            age = now - p['born']
                            x1, _, x2, _ = keys_game[p['key']]
                            col = (0, 80, 255) if p['text'] == 'MISS' else (0, 255, 255)
                            put_text(frame, p['text'], (x1 + x2) // 2, int(game.y_hit - 15 - 50 * age),
                                     0.6, col, 2, center=True)
                    draw_labels(frame, keys_game, names)
                    draw_hands()

                    if state == 'countdown':
                        left = COUNTDOWN_SECONDS - (now - state_t)
                        put_text(frame, "LEVEL " + str(level + 1), mid_x, int(FRAME_H * .16), 1.6,
                                 (210, 220, 230), 4, center=True)
                        put_text(frame, SONGS[level]['title'], mid_x, int(FRAME_H * .27), 1.0,
                                 (255, 255, 255), 2, center=True)
                        put_text(frame, SONGS[level]['subtitle'], mid_x, int(FRAME_H * .34), 0.6,
                                 (230, 230, 230), 1, center=True)
                        put_text(frame, "Use either index finger to hit a tile at the white line", mid_x,
                                 int(FRAME_H * .42), 0.6, (255, 255, 255), 1, center=True)
                        put_text(frame, str(max(int(math.ceil(left)), 1)), mid_x, int(FRAME_H * .62),
                                 3.5, (0, 255, 255), 6, center=True)
                    else:
                        put_text(frame, "LEVEL %d: %s" % (level + 1, SONGS[level]['title']), 10, 28,
                                 0.6, (255, 255, 255), 1)
                        put_text(frame, "SCORE %d" % game.score, FRAME_W - int(150 * S), 56, 0.7,
                                 (0, 255, 255), 2)
                        if game.combo >= 2:
                            put_text(frame, "COMBO x%d" % game.combo, 10, 56, 0.7, (0, 255, 120), 2)
                        frac = min(max((now - game.t0) / max(game.end_T - game.t0, 1e-6), 0.0), 1.0)
                        cv2.rectangle(frame, (0, 0), (int(FRAME_W * frac), 6), (0, 255, 255), -1)

                elif state == 'result':
                    n_total = len(SONGS)
                    if game.passed:
                        put_text(frame, "CONGRATULATIONS!", mid_x, int(FRAME_H * .17), 1.7,
                                 (210, 220, 230), 4, center=True)
                        put_text(frame, "You Passed Level %d!" % (level + 1), mid_x, int(FRAME_H * .29),
                                 1.2, (255, 255, 255), 2, center=True)
                    else:
                        put_text(frame, "SO CLOSE!", mid_x, int(FRAME_H * .17), 1.7, (0, 160, 255), 4, center=True)
                        put_text(frame, "Hit at least %d%% of the tiles to pass" % int(PASS_RATIO * 100),
                                 mid_x, int(FRAME_H * .29), 0.8, (255, 255, 255), 2, center=True)
                    for s_i in range(3):
                        draw_star(frame, mid_x + (s_i - 1) * int(70 * S), int(FRAME_H * .43), int(26 * S),
                                  (0, 215, 255) if s_i < game.stars else (90, 90, 90))
                    put_text(frame, "Accuracy %d%%   Score %d   Best combo %d" %
                             (int(game.accuracy * 100), game.score, game.best),
                             mid_x, int(FRAME_H * .58), 0.65, (255, 255, 255), 1, center=True)
                    if game.passed and level == n_total - 1:
                        put_text(frame, "YOU COMPLETED ALL %d LEVELS - MUSIC MASTER!" % n_total,
                                 mid_x, int(FRAME_H * .66), 0.65, (210, 220, 230), 2, center=True)
                    for b in result_buttons:
                        draw_button(frame, b, now)
                    for c in confetti:
                        c['y'] += c['vy'] * dt
                        if c['y'] > FRAME_H:
                            c['y'] = -10
                        cv2.rectangle(frame, (int(c['x']), int(c['y'])),
                                      (int(c['x']) + c['s'], int(c['y']) + c['s']), c['c'], -1)
                    draw_hands()

            if state in ('free', 'countdown', 'playing'):
                draw_button(frame, home_button, now)

            display_frame = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Virtual Piano", display_frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 27:  # ESC key
                break
            elif key_pressed in (32, 13):          # SPACE / ENTER
                if state == 'welcome':
                    handle('play', now)
                elif state == 'result':
                    handle(result_buttons[0]['id'], now)
            elif key_pressed in (ord('m'), ord('h')):
                go_home(now)
            elif key_pressed == ord('f'):
                go_free(now)
            elif state == 'free' and key_pressed in (ord('1'), ord('2')):
                audio.set_scale('Pentatonic' if key_pressed == ord('1') else 'Major')
                last_pressed_keys = set()
    finally:
        if cap is not None:
            cap.release()
        if hands is not None:
            hands.close()
        if audio is not None:
            audio.close()
        elif pygame.mixer.get_init():
            pygame.mixer.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
