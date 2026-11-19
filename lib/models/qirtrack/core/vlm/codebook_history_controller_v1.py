"""Only _evidence trusted retrieval changes; original W1 admission remains intact."""
from lib.config.paths import configured_path
import os, sys, json, base64, subprocess, selectors, time
from collections import deque
from pathlib import Path
import cv2, numpy as np
from lib.models.qirtrack.core.vlm.mllm_recovery_search_r1 import RecoverySearchController
from lib.models.qirtrack.core.vlm.representative_history_v1 import RepresentativeHistory

def tight(image, box):
    (x, y, w, h) = map(float, box)
    (H, W) = image.shape[:2]
    visible = max(0.0, min(x + w, W) - max(x, 0)) * max(0.0, min(y + h, H) - max(y, 0)) / (w * h)
    (l, t) = (max(0, int(np.ceil(x))), max(0, int(np.ceil(y))))
    (r, b) = (min(W, int(np.floor(x + w))), min(H, int(np.floor(y + h))))
    if visible < 0.5 or min(r - l, b - t) < 2:
        return None
    return cv2.resize(image[t:b, l:r], (384, 384), interpolation=cv2.INTER_LINEAR)

class ClipWorker:

    def __init__(self, folder):
        self.count = 0
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=False)
        self.err = (self.folder / 'worker.stderr.log').open('x')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        self.p = subprocess.Popen([os.environ.get('QIRTRACK_CLIP_PYTHON', sys.executable), '-u', str(Path(__file__).with_name('codebook_clip_worker_v1.py'))], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, env=env)
        try:
            assert json.loads(self.read(120))['ready']
        except Exception:
            self.p.terminate();self.p.wait(timeout=10);self.err.close();raise

    def read(self, timeout):
        sel = selectors.DefaultSelector()
        sel.register(self.p.stdout, selectors.EVENT_READ)
        ready = sel.select(timeout)
        sel.close()
        if not ready:
            raise RuntimeError('CPU CLIP worker timeout')
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('CPU CLIP worker failed')
        return line

    def encode(self, image):
        self.count += 1
        (ok, data) = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert ok
        name = f'crop_{self.count:05d}.png'
        (self.folder / name).write_bytes(data.tobytes())
        self.p.stdin.write(json.dumps({'id': self.count, 'png': base64.b64encode(data).decode()}) + '\n')
        self.p.stdin.flush()
        row = json.loads(self.read(60))
        assert row['id'] == self.count
        return (row['vector'], str(self.folder / name))

    def close(self):
        try:
            if self.p.poll() is None:
                try:
                    self.p.stdin.write('{"close":true}\n')
                    self.p.stdin.flush()
                    self.p.wait(timeout=30)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    if self.p.poll() is None:
                        self.p.terminate()
                    self.p.wait(timeout=10)
        finally:
            self.err.close()

class HistoryController(RecoverySearchController):

    def __init__(self, *args, history_choice, worker, history_log, **kwargs):
        super().__init__(*args, **kwargs)
        if history_choice not in ['fifo', 'representative']:
            raise ValueError(history_choice)
        self.choice = history_choice
        self.worker = worker
        self.history_log = Path(history_log)
        self.metadata = {}
        self.encoded = []
        crop = tight(args[2], args[3])
        self.bank = RepresentativeHistory(worker.encode(crop)[0]) if crop is not None else None
        self.log(dict(event='initialize', frame=0, bank_available=self.bank is not None, arm=self.choice))

    def log(self, row):
        with self.history_log.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def _evidence(self, image, previous, current, past_limit=3, include_trusted=False, search_region=None):
        original = self.trusted
        fifo = list(original)
        selected = fifo
        if include_trusted and self.choice == 'representative' and (self.bank is not None):
            ids = self.bank.query(self._frame)
            if len(ids) == len(fifo) and all((t in self.metadata for t in ids)):
                selected = [self.metadata[t] for t in ids]
        if include_trusted:
            assert len(selected) == len(fifo) <= 4 and all((x['frame'] < self._frame for x in selected))
            self.log(dict(event='retrieve', frame=self._frame, fifo_frames=[x['frame'] for x in fifo], selected_frames=[x['frame'] for x in selected], changed=[x['frame'] for x in selected] != [x['frame'] for x in fifo]))
        try:
            self.trusted = deque(selected, maxlen=4)
            return super()._evidence(image, previous, current, past_limit, include_trusted, search_region)
        finally:
            self.trusted = original

    def step(self, frame, image, previous, raw, q_raw, tracker):
        row = super().step(frame, image, previous, raw, q_raw, tracker)
        if row.get('trusted_added'):
            item = self.trusted[-1]
            audit = row['periodic_audit']
            if audit.get('status') != 'ok':
                raise ValueError('Unsuccessful audit cannot admit')
            p = float(audit['conditional_probs_uncalibrated']['YES'])
            crop = tight(image, raw)
            if crop is None or self.bank is None:
                self.log(dict(event='embedding_unavailable_fifo_fallback', frame=frame))
            else:
                (v, path) = self.worker.encode(crop)
                request = str(audit['evidence_request_id'])
                ids = self.bank.add(frame, v, request, float(q_raw), float(p))
                self.metadata[frame] = dict(item)
                self.metadata = {t: x for (t, x) in self.metadata.items() if t in ids}
                self.log(dict(event='admitted_original_W1', frame=frame, request_id=request, q_raw=q_raw, p_yes=p, tight_crop=path, bank_frames=ids, bank_state=self.bank.export()))
        return row
