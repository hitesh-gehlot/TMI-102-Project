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
import time

# In-memory audio avoids stale WAV caches. Held notes decay, never loop.
SAMPLE_RATE = 44100
NOTE_DURATION = 16.0
RELEASE_MS = 90
VOICE_COUNT = 32
FREQUENCIES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]


def generate_piano_note(frequency):
    t = np.arange(int(SAMPLE_RATE * NOTE_DURATION)) / SAMPLE_RATE
    data = np.zeros_like(t)
    for harmonic, amplitude in enumerate([1., .6, .3, .15, .08, .04], 1):
        # Higher harmonics decay faster, gradually softening the tone.
        data += amplitude * np.sin(2*np.pi*frequency*harmonic*t) * np.exp(-t*(.65+.22*(harmonic-1)))
    data *= np.minimum(t/.003, 1.)
    n = int(SAMPLE_RATE*.1)
    data[-n:] *= np.linspace(1., 0., n)
    data *= .10 / max(float(np.max(np.abs(data))), 1e-12)
    mono = (data*32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack((mono, mono))))


DRUM_NAMES = ['Kick', 'Snare', 'Hat', 'OpenHat', 'LowTom', 'HiTom', 'Ride', 'Crash']


def generate_drum(kind):
    """Synthesize eight percussion sounds without external sample files."""
    duration = [0.7, 0.45, 0.16, 0.8, 0.65, 0.5, 0.35, 1.5][kind]
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    noise = np.random.default_rng(100 + kind).uniform(-1, 1, len(t))
    bright = noise - np.concatenate(([0.], noise[:-1]))
    if kind == 0:
        phase = 2*np.pi*(48*t + 100*.025*(1-np.exp(-t/.025)))
        data = np.sin(phase)*np.exp(-t*9) + .12*noise*np.exp(-t*100)
    elif kind == 1:
        data = .7*bright*np.exp(-t*17) + .4*np.sin(2*np.pi*180*t)*np.exp(-t*24)
    elif kind in (2, 3, 6, 7):
        rate = {2: 45, 3: 7, 6: 5, 7: 3.8}[kind]
        metal = sum(np.sin(2*np.pi*f*t) for f in [3171, 4283, 5711, 6823])/4
        data = (.65*bright + .35*metal)*np.exp(-t*rate)
    elif kind in (4, 5):
        freq = 100 if kind == 4 else 170
        phase = 2*np.pi*(freq*t + 45*.04*(1-np.exp(-t/.04)))
        data = np.sin(phase)*np.exp(-t*10) + .08*noise*np.exp(-t*35)
    data *= np.minimum(t/.001, 1.)
    n = min(len(t), int(SAMPLE_RATE*.015))
    data[-n:] *= np.linspace(1., 0., n)
    mono = (data * (.18*32767/max(np.max(np.abs(data)), 1e-12))).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack((mono, mono))))


# Eight pads in two rows. Drawing and touch detection share these rectangles.
def drum_layout(width, height):
    margin = max(12, int(width*.04))
    gap = max(8, int(width*.018))
    top, bottom = 25, height-110
    usable_w = width-2*margin
    usable_h = max(2, bottom-top)
    pad_w = (usable_w-3*gap)//4
    pad_h = (usable_h-gap)//2
    # Common performance layout: cymbals above, kick/snare/toms below.
    order = [2, 3, 6, 7, 0, 1, 4, 5]
    labels = ['KICK', 'SNARE', 'HI-HAT', 'OPEN HAT', 'LOW TOM', 'HIGH TOM', 'RIDE', 'CRASH']
    colors = [(205,120,65), (150,80,215), (65,175,220), (65,175,220),
              (190,145,70), (190,145,70), (70,185,140), (70,185,140)]
    pads = []
    for slot, key in enumerate(order):
        row, col = divmod(slot,4)
        x = margin+col*(pad_w+gap)
        y = top+row*(pad_h+gap)
        pads.append(dict(key=key,label=labels[key],color=colors[key],
                         rect=(x,y,x+pad_w,y+pad_h)))
    return pads


def drum_at(x, y, pads):
    for pad in pads:
        x1,y1,x2,y2 = pad['rect']
        if x1 <= x < x2 and y1 <= y < y2:
            return pad['key']
    return None


def draw_drumpads(frame, pads, current, hit_times, now):
    layer = frame.copy()
    aa = cv2.LINE_AA
    for pad in pads:
        x1,y1,x2,y2 = pad['rect']
        active = pad['key'] in current or now-hit_times.get(pad['key'],-10.) < .14
        accent = pad['color']
        fill = tuple(min(255,int(c*.85)+65) for c in accent) if active else (38,32,32)
        cv2.rectangle(layer,(x1+3,y1+5),(x2+3,y2+5),(12,12,12),-1)
        cv2.rectangle(layer,(x1,y1),(x2,y2),fill,-1)
        cv2.rectangle(layer,(x1,y1),(x2,y2),(0,245,255) if active else accent,3,aa)
        cv2.rectangle(layer,(x1+7,y1+7),(x2-7,y2-7),accent,1,aa)
        scale = min(.65,(x2-x1)/220.)
        label = pad['label']
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
        cx,cy = (x1+x2)//2,(y1+y2)//2
        cv2.putText(layer,label,(cx-tw//2,cy+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX,scale,(245,245,245),2,aa)
        cv2.circle(layer,(cx,y2-20),4,(0,245,255) if active else accent,-1,aa)
    cv2.addWeighted(layer,.88,frame,.12,0,frame)


def is_switch_gesture(landmarks, width, height):
    """Recognize I-love-you: thumb/index/pinky out, middle/ring curled.

    Joint angles and wrist distances work for either hand and image rotation.
    Use a palm facing the camera so curled fingers remain visible.
    """
    pts = np.array([(p.x*width, p.y*height) for p in landmarks.landmark])

    def joint_cosine(a, b, c):
        u, v = pts[a]-pts[b], pts[c]-pts[b]
        norm = np.linalg.norm(u)*np.linalg.norm(v)
        return None if norm < 1e-6 else float(np.dot(u, v)/norm)

    palm = np.linalg.norm(pts[5]-pts[17])
    if palm < 10:
        return False
    for mcp, pip, tip in [(5,6,8), (17,18,20)]:
        angle = joint_cosine(mcp, pip, tip)
        if angle is None or angle >= -.85:
            return False
        if np.linalg.norm(pts[tip]-pts[0]) < 1.15*np.linalg.norm(pts[pip]-pts[0]):
            return False
    for mcp, pip, tip in [(9,10,12), (13,14,16)]:
        angle = joint_cosine(mcp, pip, tip)
        if angle is None or angle < -.45:
            return False
        if np.linalg.norm(pts[tip]-pts[0]) > 1.10*np.linalg.norm(pts[pip]-pts[0]):
            return False
    thumb_angle = joint_cosine(1, 2, 4)
    return (thumb_angle is not None and thumb_angle < -.75
            and np.linalg.norm(pts[4]-pts[5]) > .55*palm)


class GestureSwitch:
    HOLD_SECONDS = .8
    RESET_SECONDS = .35

    def __init__(self):
        self.started = None
        self.latched = False
        self.absent_since = None
        self.progress = 0.

    def update(self, switch_pose, now):
        if switch_pose:
            self.absent_since = None
            if self.latched:
                return False
            if self.started is None:
                self.started = now
            self.progress = min((now-self.started)/self.HOLD_SECONDS, 1.)
            if self.progress >= 1.:
                self.latched = True
                return True
        else:
            self.started = None
            self.progress = 0.
            if self.absent_since is None:
                self.absent_since = now
            if now-self.absent_since >= self.RESET_SECONDS:
                self.latched = False
        return False


class InstrumentAudio:
    """Track each strike separately so rapid re-strikes can overlap release tails."""
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(VOICE_COUNT)
        self.channels = [pygame.mixer.Channel(i) for i in range(VOICE_COUNT)]
        self.sounds = [generate_piano_note(f) for f in FREQUENCIES]
        self.drums = [generate_drum(i) for i in range(8)]
        self.mode = 'piano'
        self.held = {}
        self.voices = {}
        self.serial = 0

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
        self.voices[i] = (self.serial, self.mode == 'drums')
        if self.mode == 'piano':
            self.held[key] = i
        self.channels[i].set_volume(1.)
        self.channels[i].play((self.sounds if self.mode == 'piano' else self.drums)[key], loops=0)

    def update(self, current, previous):
        for key in previous-current:
            self.release(key)
        for key in current-previous:
            self.press(key)

    def toggle_mode(self):
        for i, channel in enumerate(self.channels):
            channel.fadeout(60)
            if i in self.voices:
                self.voices[i] = (self.voices[i][0], True)
        self.held.clear()
        self.mode = 'drums' if self.mode == 'piano' else 'piano'

    def close(self):
        pygame.mixer.stop()
        pygame.mixer.quit()


def main():
    audio = hands = cap = None
    try:
        audio = InstrumentAudio()
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
        # WEBCAM + KEY LAYOUT (Top Placement)
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
        cv2.namedWindow("Virtual Piano and Drum Pads", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Virtual Piano and Drum Pads", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        NUM_KEYS    = 8
        GAP         = 6
        KEY_H       = 160
        KEY_W       = (FRAME_W - GAP * (NUM_KEYS + 1)) // NUM_KEYS
        KEY_Y       = 20  # Positioned at top of screen
        total_width = NUM_KEYS * KEY_W + (NUM_KEYS - 1) * GAP
        START_X     = (FRAME_W - total_width) // 2

        NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']

        keys = []
        for i in range(NUM_KEYS):
            x1 = START_X + i * (KEY_W + GAP)
            x2 = x1 + KEY_W
            keys.append((x1, KEY_Y, x2, KEY_Y + KEY_H))

        # ─────────────────────────────────────────
        # STATE
        # ─────────────────────────────────────────
        last_pressed_keys = set()
        drum_pads = drum_layout(FRAME_W, FRAME_H)
        hit_times = {}
        gesture = GestureSwitch()
        wait_for_clear = False

        # ─────────────────────────────────────────
        # MAIN LOOP
        # ─────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            finger_positions = []
            switch_pose = False
            fingertip_ids = [8]  # Index fingertip ID on each hand

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    switch_pose |= is_switch_gesture(hand_landmarks, FRAME_W, FRAME_H)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for tip_id in fingertip_ids:
                        tip = hand_landmarks.landmark[tip_id]
                        fx, fy = int(tip.x * FRAME_W), int(tip.y * FRAME_H)
                        finger_positions.append((fx, fy))
                        cv2.circle(frame, (fx, fy), 9, (0, 255, 0), -1)

            now = time.monotonic()
            if gesture.update(switch_pose, now):
                audio.toggle_mode()
                last_pressed_keys = set()
                hit_times.clear()

            current_pressed_keys = set()
            for fx, fy in finger_positions:
                if audio.mode == 'drums':
                    hit = drum_at(fx, fy, drum_pads)
                    if hit is not None:
                        current_pressed_keys.add(hit)
                else:
                    for i, (x1,y1,x2,y2) in enumerate(keys):
                        if x1 < fx < x2 and y1 < fy < y2:
                            current_pressed_keys.add(i)

            # Gesturing silences piano and blocks pad hits. Require all tips to
            # leave the pads before playing resumes, preventing accidental hits.
            if switch_pose or gesture.latched:
                wait_for_clear = True
            if wait_for_clear and not current_pressed_keys and not switch_pose and not gesture.latched:
                wait_for_clear = False
            if wait_for_clear:
                current_pressed_keys = set()
            for hit in current_pressed_keys-last_pressed_keys:
                hit_times[hit] = now
            audio.update(current_pressed_keys, last_pressed_keys)

            last_pressed_keys = current_pressed_keys

            if audio.mode == 'drums':
                draw_drumpads(frame, drum_pads, current_pressed_keys, hit_times, now)
                # Keep finger markers visible above the kit graphics.
                for fx, fy in finger_positions:
                    cv2.circle(frame, (fx,fy), 9, (0,255,0), -1)
            else:
                # ── Draw Keys at Top ──
                for i, (x1, y1, x2, y2) in enumerate(keys):
                    touching = i in current_pressed_keys
                    color    = (0, 255, 255) if touching else (255, 255, 255)

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    label = NOTE_NAMES[i]
                    cv2.putText(frame, label,
                                (x1 + 2, y2 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

            title = 'Piano' if audio.mode == 'piano' else 'Drum Pads'
            cv2.putText(frame, 'Virtual Instruments - ' + title, (10, FRAME_H-75),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 200, 255), 2)
            status = 'ILY sign (thumb+index+pinky): hold 0.8s | ESC: exit'
            if switch_pose and not gesture.latched:
                status = f'Switching... {int(gesture.progress*100)}%'
            elif wait_for_clear:
                status = 'Release gesture; move index fingers outside pads to play'
            cv2.putText(frame, status, (10, FRAME_H-45),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (180, 255, 180), 1)
            cv2.putText(frame, 'Index fingers: enter pads to play; leave and re-enter to repeat',
                        (10, FRAME_H-20), cv2.FONT_HERSHEY_SIMPLEX, .45, (180,180,180), 1)
            display_frame = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Virtual Piano and Drum Pads', display_frame)

            key_pressed = cv2.waitKey(1)
            if key_pressed == 27:  # ESC key
                break
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
