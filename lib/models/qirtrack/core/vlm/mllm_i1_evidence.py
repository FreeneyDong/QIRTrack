"""I1 evidence/prompt ablation on the frozen G1 controller.

No proposal, quality, acceptance, budget or native tracker policy is changed.
The legacy arm preserves image bytes and both prompts exactly; strict_prompt
changes only the user prompt. Request files are research outputs, never Git.
"""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import cv2
import numpy as np
from lib.models.qirtrack.core.vlm.mllm_backend import MLLMBackend
from lib.models.qirtrack.core.vlm.mllm_full_controller import FullMLLMController
from lib.models.qirtrack.core.vlm.mllm_input_contract import CONTRACT_VERSION, validate_manifest, validate_payload
TARGET_CONTRACT = prompt_text('mllm_i1_evidence_e21770bf36')

def crop_geometry(region, marked, factor, side, shape):
    """Match sample_target's square/rounding, including off-image padding."""
    (x, y, w, h) = map(float, region)
    size = math.ceil(math.sqrt(w * h) * factor)
    (left, top) = (round(x + w / 2 - size / 2), round(y + h / 2 - size / 2))
    scale = side / size
    (bx, by, bw, bh) = map(float, marked)
    return {'source_hw': list(shape[:2]), 'display_hw': [side, side], 'crop_xywh_full': [left, top, size, size], 'full_to_display': {'sx': scale, 'sy': scale, 'tx': -left * scale, 'ty': -top * scale}, 'bbox_pixels_in_display': [(bx - left) * scale, (by - top) * scale, bw * scale, bh * scale], 'bbox_pixels_full': [bx, by, bw, bh], 'padding_ltrb': [max(0, -left), max(0, -top), max(0, left + size - shape[1] + 1), max(0, top + size - shape[0] + 1)]}

class RequestArchive:
    """Durable pre-request payload metadata and exact bytes, then raw response."""

    def __init__(self, root: Path, domain: str, sequence: str, arm: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'images').mkdir()
        self.identity = {'domain': domain, 'sequence': sequence, 'arm': arm}
        self.context = {}
        self.last_id = None
        self.count = 0

    def begin(self, payload):
        validate_payload(payload, dict(self.context, request_identity=self.identity))
        self.count += 1
        request_id = f'request_{self.count:06d}'
        body = copy.deepcopy(payload)
        images = []
        for message in body['messages']:
            for item in message.get('content', []) if isinstance(message.get('content'), list) else []:
                if item.get('type') != 'image_url':
                    continue
                url = item['image_url']['url']
                if not url.startswith('data:image/jpeg;base64,'):
                    raise ValueError(prompt_text('mllm_i1_evidence_f0ba621d0d'))
                blob = base64.b64decode(url.split(',', 1)[1], validate=True)
                digest = hashlib.sha256(blob).hexdigest()
                rel = f'images/{digest}.jpg'
                path = self.root / rel
                if not path.exists():
                    with path.open('xb') as handle:
                        handle.write(blob)
                item['image_url']['url'] = rel
                images.append({'sha256': digest, 'path': rel, 'bytes': len(blob)})
        expected = self.context.get('ordered_images', [])
        if [i['sha256'] for i in images] != [i['sha256'] for i in expected]:
            raise RuntimeError(prompt_text('mllm_i1_evidence_3caef481a8'))
        entry = {'schema': 'I1ActualRequest-v1', 'input_contract': CONTRACT_VERSION, 'input_contract_validated': True, 'request_id': request_id, **self.identity, **copy.deepcopy(self.context), 'payload': body, 'actual_images': images, 'status': 'pending', 'input_reconstructed': False}
        self._write(request_id, entry)
        self.last_id = request_id
        return (request_id, entry)

    def _write(self, request_id, entry):
        path = self.root / f'{request_id}.json'
        temp = path.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8') as handle:
            json.dump(entry, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)

    def finish(self, request_id, entry, body, latency, error):
        entry.update(status='response' if error is None else 'request_error', raw_response=body, latency_sec=latency, error=error)
        self._write(request_id, entry)
        with (self.root / 'requests.jsonl').open('a', encoding='utf-8') as handle:
            json.dump({'request_id': request_id, **self.identity, 'frame': entry['frame'], 'stage': entry['stage'], 'status': entry['status'], 'file': f'{request_id}.json'}, handle)
            handle.write('\n')

class RecordedBackend(MLLMBackend):

    def __init__(self, backend: MLLMBackend, archive: RequestArchive):
        super().__init__(backend.kind, backend.model, backend.url, backend.timeout_sec)
        object.__setattr__(self, 'archive', archive)

    def _post(self, payload):
        (request_id, entry) = self.archive.begin(payload)
        try:
            (body, latency, error) = super()._post(payload)
        except Exception as exc:
            self.archive.finish(request_id, entry, None, None, f'{type(exc).__name__}: {exc}')
            raise
        self.archive.finish(request_id, entry, body, latency, error)
        return (body, latency, error)

class I1Controller(FullMLLMController):

    def __init__(self, *args, input_arm: str, archive: RequestArchive, **kwargs):
        if input_arm not in {'legacy', 'strict_prompt'}:
            raise ValueError(input_arm)
        super().__init__(*args, **kwargs)
        (self.input_arm, self.archive) = (input_arm, archive)
        self._frame = None
        self._q_raw = None
        self._stage = None
        self._manifests = {}
        self._evidence_stages = {}
        self._recent_request_ids = []
        self._local_boxes = []

    def _register(self, blobs, entries):
        if len(blobs) != len(entries):
            raise RuntimeError('Wrong evidence manifest size')
        rows = []
        for (index, (blob, entry)) in enumerate(zip(blobs, entries), 1):
            decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError('Invalid evidence JPEG')
            rows.append({**copy.deepcopy(entry), 'image_id': f'image_{index}', 'sha256': hashlib.sha256(blob).hexdigest(), 'encoded_hw': list(decoded.shape[:2])})
        validate_manifest(rows, self._frame)
        self._manifests[tuple((row['sha256'] for row in rows))] = rows
        return rows

    def _evidence(self, image, previous, current, past_limit=3, include_trusted=False, search_region=None):
        blobs = super()._evidence(image, previous, current, past_limit, include_trusted, search_region)
        scale = min(1.0, 640 / max(image.shape[:2]))
        (dw, dh) = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        (sx, sy) = (dw / image.shape[1], dh / image.shape[0])
        current_role = 'CURRENT_CANDIDATE' if search_region is not None else 'CURRENT_RAW'
        entries = [{'role': 'ANCHOR_TARGET', 'frame': 0, 'box_drawn': False, **crop_geometry(self.init_box, self.init_box, 2.0, 192, self.first.shape)}, {'role': current_role + '_FULL', 'frame': self._frame, 'box_drawn': True, 'source_hw': list(image.shape[:2]), 'display_hw': [dh, dw], 'full_to_display': {'sx': sx, 'sy': sy, 'tx': 0, 'ty': 0}, 'bbox_pixels_full': list(map(float, current)), 'bbox_pixels_in_display': [current[0] * sx, current[1] * sy, current[2] * sx, current[3] * sy]}, {'role': current_role + '_SEARCH', 'frame': self._frame, 'box_drawn': True, **crop_geometry(previous if search_region is None else search_region, current, 5.0, 384, image.shape)}]
        if past_limit:
            selected = list(self.past)[-past_limit:]
            if selected:
                if any((item['frame'] >= self._frame for item in selected)):
                    raise RuntimeError('Noncausal recent evidence')
                entries.append({'role': 'RECENT_RAW', 'box_drawn': False, 'tiles': [{'index': i, 'frame': item['frame'], 'tile_xywh_in_sheet': [i % 4 * 160, i // 4 * 160, 160, 160], 'age': self._frame - item['frame'], 'tracker_score': item['score'], 'q_raw': item.get('q_raw'), 'legacy_overlay': f't-{len(selected) - i}', **crop_geometry(item['box'], item['box'], 2.0, 160, item['image'].shape)} for (i, item) in enumerate(selected)]})
        if include_trusted and self.trusted:
            if any((item['frame'] >= self._frame for item in self.trusted)):
                raise RuntimeError('Noncausal trusted evidence')
            entries.append({'role': 'TRUSTED', 'box_drawn': False, 'screening': prompt_text('mllm_i1_evidence_8b2a8444ee'), 'tiles': [{'index': i, 'frame': item['frame'], 'tile_xywh_in_sheet': [i % 4 * 160, i // 4 * 160, 160, 160], 'age': self._frame - item['frame'], 'q_raw': item['q'], 'bbox_pixels_full': item['box'], 'verifier_model': item['verifier_model'], **item.get('crop_geometry', {})} for (i, item) in enumerate(self.trusted)]})
        manifest = self._register(blobs, entries)
        key = tuple((row['sha256'] for row in manifest))
        self._evidence_stages[key] = 'temporal_verifier' if include_trusted or past_limit > 3 else 'periodic_audit' if self._q_raw >= self.config.q_trusted else 'verifier'
        return blobs

    def _local_candidates(self, *args):
        self._local_boxes = super()._local_candidates(*args)
        return self._local_boxes

    def _proposal_task(self):
        return prompt_text('mllm_i1_evidence_8565c48ab6')

    def _request(self, method, *args, bypass_budget=False, **kwargs):
        if not bypass_budget and self.budgeted_requests >= self.config.max_mllm_requests:
            return super()._request(method, *args, bypass_budget=bypass_budget, **kwargs)
        blobs = args[0]
        key = tuple((hashlib.sha256(blob).hexdigest() for blob in blobs))
        if method == 'choose_candidate' and key not in self._manifests:
            factor = min(1.0, 640 / max(self._current_shape[:2]))
            dh = max(1, round(self._current_shape[0] * factor))
            dw = max(1, round(self._current_shape[1] * factor))
            entries = [{'role': 'ANCHOR_TARGET', 'frame': 0, 'box_drawn': False, **crop_geometry(self.init_box, self.init_box, 2.0, 192, self.first.shape)}, {'role': 'CURRENT_FULL_UNMARKED', 'frame': self._frame, 'box_drawn': False, 'source_hw': list(self._current_shape[:2]), 'display_hw': [dh, dw], 'full_to_display': {'sx': dw / self._current_shape[1], 'sy': dh / self._current_shape[0], 'tx': 0, 'ty': 0}}, {'role': 'CANDIDATE_GALLERY', 'frame': self._frame, 'box_drawn': False, 'tiles': [{'index': i, 'candidate_id': f'C{i}', 'tile_xywh_in_sheet': [i % 4 * 160, i // 4 * 160, 160, 160], **crop_geometry(box, box, 2.0, 160, self._current_shape)} for (i, box) in enumerate(self._local_boxes)]}]
            manifest = self._register(blobs, entries)
        else:
            manifest = self._manifests[key]
        validate_manifest(manifest, self._frame, blobs)
        stage = self._stage or ('local_selection' if method == 'choose_candidate' else self._evidence_stages[key])
        original_prompt = args[-1]
        if self.input_arm == 'strict_prompt':
            short_manifest = [{k: v for (k, v) in row.items() if k not in {'sha256', 'padding_ltrb'}} for row in manifest]
            task = {'score': prompt_text('mllm_i1_evidence_847e8d91fd'), 'propose': self._proposal_task(), 'choose_candidate': prompt_text('mllm_i1_evidence_962797eb50')}[method]
            compact = []
            for item in short_manifest:
                row = {k: item[k] for k in ('image_id', 'role', 'frame', 'box_drawn') if k in item}
                if 'frame' in row:
                    row['frame_number_one_based'] = row['frame'] + 1
                if 'meaning' in item:
                    row['meaning'] = item['meaning']
                if item['role'] == 'ANCHOR_TARGET':
                    row['target_xywh_pixels'] = item['bbox_pixels_in_display']
                    row['image_hw'] = item['encoded_hw']
                if 'tiles' in item:
                    row['tiles_left_to_right_then_top_to_bottom'] = [{k: tile[k] for k in ('index', 'frame', 'candidate_id') if k in tile} for tile in item['tiles']]
                compact.append(row)
            prompt = TARGET_CONTRACT + '\nORDERED IMAGE ROLES: ' + json.dumps(compact) + '\nTASK: ' + task
            args = (*args[:-1], prompt)
        self.archive.context = {'frame': self._frame, 'stage': stage, 'method': method, 'ordered_images': manifest, 'q_raw': self._q_raw, 'legacy_prompt': original_prompt, 'image_policy': getattr(self, 'evidence_policy', 'byte-identical legacy evidence'), 'target_contract': 'legacy' if self.input_arm == 'legacy' else 'strict-unit-v1'}
        if hasattr(self, 'sequence_source'):
            self.archive.context.update(sequence_source=copy.deepcopy(self.sequence_source), frame_source=copy.deepcopy(self.frame_source))
        result = super()._request(method, *args, bypass_budget=bypass_budget, **kwargs)
        request_id = self.archive.last_id
        if request_id is not None:
            self._recent_request_ids.append(request_id)
            result = dict(result, evidence_request_id=request_id, evidence_request_file=str(self.archive.root / f'{request_id}.json'))
        return result

    def _postcheck(self, *args, **kwargs):
        (before, self._stage) = (self._stage, 'candidate_postcheck')
        try:
            return super()._postcheck(*args, **kwargs)
        finally:
            self._stage = before

    def _recover(self, *args, **kwargs):
        (before, self._stage) = (self._stage, 'global_recovery')
        try:
            return super()._recover(*args, **kwargs)
        finally:
            self._stage = before

    def step(self, frame, image, previous, raw, q_raw, tracker):
        (self._frame, self._q_raw, self._current_shape) = (frame, float(q_raw), image.shape)
        self._manifests.clear()
        self._evidence_stages.clear()
        self._recent_request_ids = []
        row = super().step(frame, image, previous, raw, q_raw, tracker)
        row['evidence_request_ids'] = self._recent_request_ids[:]
        if row.get('trusted_added'):
            self.trusted[-1]['crop_geometry'] = crop_geometry(raw, raw, 2.0, 160, image.shape)
            self.trusted[-1]['crop_geometry']['screened_by_response'] = row['periodic_audit'].get('evidence_request_id')
        return row

    def remember(self, frame, image, raw, score, entropy):
        super().remember(frame, image, raw, score, entropy)
        if frame != self._frame or not math.isfinite(self._q_raw):
            raise ValueError('Missing same-frame frozen PureIoU Q')
        self.past[-1]['q_raw'] = self._q_raw
