"""Provider-independent VLM recovery scaffold for the ARTrackV2 student gate.

The router, external calls, and recovery application are separately gated by
environment variables.  All are disabled by default so importing this module
cannot change tracker output or incur external cost.
"""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text
from lib.models.qirtrack.mllm.prompts import guarded_urlopen
import base64
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import cv2
import numpy as np
RISK_THRESHOLDS = {'safety': -0.9754552841186523, 'balanced': -0.6461024284362793, 'low_false_trigger': -0.2736161947250366}
RECOVERY_SCHEMA = {'type': 'object', 'properties': {'target_present': {'type': 'boolean'}, 'bbox_xywh_norm': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'evidence': {'type': 'string'}}, 'required': ['target_present', 'bbox_xywh_norm', 'confidence', 'evidence'], 'additionalProperties': False}
SYSTEM_PROMPT = prompt_text('artrackv2_vlm_recovery_840ce1d1a4')
USER_PROMPT = prompt_text('artrackv2_vlm_recovery_e28c301016')

def _env_flag(name: str, default: bool=False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

@dataclass(frozen=True)
class VLMRecoveryConfig:
    router_enabled: bool
    call_enabled: bool
    apply_recovery: bool
    profile: str
    bad_risk_threshold: float
    severe_probability_threshold: float
    cooldown_frames: int
    trusted_bad_probability_max: float
    qwen_model: str
    ollama_url: str
    gpt_model: str
    openai_url: str
    openai_key_env: str
    timeout: float
    max_attempts: int
    max_output_tokens: int
    image_detail: str
    jpeg_quality: int
    current_max_side: int
    output_root: Path
    save_images: bool

    @classmethod
    def from_env(cls) -> 'VLMRecoveryConfig':
        profile = os.getenv('ARTRACK_VLM_TRIGGER_PROFILE', 'balanced').strip().lower()
        if profile not in RISK_THRESHOLDS:
            raise ValueError(f'Unknown ARTRACK_VLM_TRIGGER_PROFILE: {profile}')
        return cls(router_enabled=_env_flag('ARTRACK_VLM_ROUTER_ENABLE', False), call_enabled=_env_flag('ARTRACK_VLM_CALL_ENABLE', False), apply_recovery=_env_flag('ARTRACK_VLM_APPLY_RECOVERY', False), profile=profile, bad_risk_threshold=_env_float('ARTRACK_VLM_BAD_RISK_THRESHOLD', RISK_THRESHOLDS[profile]), severe_probability_threshold=min(1.0, max(0.0, _env_float('ARTRACK_VLM_SEVERE_PROB_THRESHOLD', 0.5))), cooldown_frames=max(0, _env_int('ARTRACK_VLM_COOLDOWN_FRAMES', 15)), trusted_bad_probability_max=min(1.0, max(0.0, _env_float('ARTRACK_VLM_TRUSTED_BAD_PROB_MAX', 0.2))), qwen_model=os.getenv('ARTRACK_VLM_QWEN_MODEL', 'qwen3.6:35b'), ollama_url=os.getenv('ARTRACK_VLM_OLLAMA_URL', 'http://127.0.0.1:11434/api/chat'), gpt_model=os.getenv('ARTRACK_VLM_GPT_MODEL', 'gpt-5.4-nano'), openai_url=os.getenv('ARTRACK_VLM_OPENAI_URL', 'https://api.openai.com/v1/responses'), openai_key_env=os.getenv('ARTRACK_VLM_OPENAI_KEY_ENV', 'OPENAI_API_KEY'), timeout=max(1.0, _env_float('ARTRACK_VLM_TIMEOUT', 90.0)), max_attempts=max(1, _env_int('ARTRACK_VLM_MAX_ATTEMPTS', 2)), max_output_tokens=max(64, _env_int('ARTRACK_VLM_MAX_OUTPUT_TOKENS', 300)), image_detail=os.getenv('ARTRACK_VLM_IMAGE_DETAIL', 'high'), jpeg_quality=min(100, max(40, _env_int('ARTRACK_VLM_JPEG_QUALITY', 90))), current_max_side=max(256, _env_int('ARTRACK_VLM_CURRENT_MAX_SIDE', 1280)), output_root=Path(os.getenv('ARTRACK_VLM_OUTPUT_ROOT', 'output/uncertainty_gate/vlm_recovery_events')), save_images=_env_flag('ARTRACK_VLM_SAVE_IMAGES', True))

def sanitize_bbox(bbox: Any) -> Optional[list[float]]:
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return values if all((math.isfinite(value) for value in values)) else None

def sanitize_normalized_bbox(bbox: Any, target_present: bool) -> Optional[list[float]]:
    values = sanitize_bbox(bbox)
    if values is None:
        return None
    if not target_present:
        return [0.0, 0.0, 0.0, 0.0]
    (x, y, w, h) = values
    if min(x, y, w, h) < 0 or x > 1 or y > 1 or (w <= 0) or (h <= 0):
        return None
    if x + w > 1.0001 or y + h > 1.0001:
        return None
    return [min(1.0, x), min(1.0, y), min(1.0, w), min(1.0, h)]

def normalized_to_pixels(bbox: Sequence[float], width: int, height: int) -> list[float]:
    (x, y, w, h) = bbox
    return [x * width, y * height, w * width, h * height]

def pixels_to_normalized(bbox: Any, width: int, height: int) -> Optional[list[float]]:
    values = sanitize_bbox(bbox)
    if values is None or width <= 0 or height <= 0:
        return None
    (x, y, w, h) = values
    return [x / width, y / height, w / width, h / height]

def _safe_sequence_name(name: str) -> str:
    value = re.sub('[^A-Za-z0-9_.-]+', '_', str(name)).strip('._')
    return value or 'sequence'

def _draw_bbox(image_rgb: np.ndarray, bbox: Any, color=(255, 255, 0), label='tracker') -> np.ndarray:
    output = np.asarray(image_rgb).copy()
    values = sanitize_bbox(bbox)
    if values is None:
        return output
    (x, y, w, h) = values
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(output, p1, p2, color, 3)
    cv2.putText(output, label, (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output

def _context_crop(image_rgb: np.ndarray, bbox: Any, factor=2.0) -> tuple[np.ndarray, list[float]]:
    values = sanitize_bbox(bbox)
    if values is None:
        return (np.asarray(image_rgb).copy(), [0, 0, image_rgb.shape[1], image_rgb.shape[0]])
    (x, y, w, h) = values
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    side_w = max(2.0, w * factor)
    side_h = max(2.0, h * factor)
    x1 = max(0, int(math.floor(cx - 0.5 * side_w)))
    y1 = max(0, int(math.floor(cy - 0.5 * side_h)))
    x2 = min(image_rgb.shape[1], int(math.ceil(cx + 0.5 * side_w)))
    y2 = min(image_rgb.shape[0], int(math.ceil(cy + 0.5 * side_h)))
    crop = np.asarray(image_rgb)[y1:y2, x1:x2].copy()
    return (crop, [x - x1, y - y1, w, h])

def _resize_max_side(image_rgb: np.ndarray, max_side: int) -> np.ndarray:
    (height, width) = image_rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1:
        return np.asarray(image_rgb).copy()
    return cv2.resize(image_rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)

def _jpeg_bytes(image_rgb: np.ndarray, quality: int) -> bytes:
    (ok, buffer) = cv2.imencode('.jpg', cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('Failed to encode VLM image')
    return buffer.tobytes()

def _parse_json_text(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    match = re.search('\\{.*\\}', text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None

def _validate_response(parsed: Optional[dict]) -> dict:
    if parsed is None:
        return {'status': 'invalid_json'}
    target_present = bool(parsed.get('target_present', False))
    bbox = sanitize_normalized_bbox(parsed.get('bbox_xywh_norm'), target_present)
    try:
        confidence = float(parsed.get('confidence', 0.0))
    except (TypeError, ValueError):
        confidence = float('nan')
    if bbox is None or not math.isfinite(confidence) or (not 0 <= confidence <= 1):
        return {'status': 'invalid_output', 'raw_parsed': parsed}
    return {'status': 'completed', 'target_present': target_present, 'bbox_xywh_norm': bbox, 'confidence': confidence, 'evidence': str(parsed.get('evidence', ''))[:1000]}

class OpenAIResponsesAdapter:

    def __init__(self, config: VLMRecoveryConfig):
        self.config = config

    def build_payload(self, images: Sequence[np.ndarray], prompt: str) -> dict:
        content = [{'type': 'input_text', 'text': SYSTEM_PROMPT}, {'type': 'input_text', 'text': prompt}]
        for image in images:
            encoded = base64.b64encode(_jpeg_bytes(image, self.config.jpeg_quality)).decode('ascii')
            content.append({'type': 'input_image', 'image_url': f'data:image/jpeg;base64,{encoded}', 'detail': self.config.image_detail})
        return {'model': self.config.gpt_model, 'input': [{'role': 'user', 'content': content}], 'max_output_tokens': self.config.max_output_tokens, 'text': {'format': {'type': 'json_schema', 'name': 'tracking_recovery_bbox', 'strict': True, 'schema': RECOVERY_SCHEMA}}}

    @staticmethod
    def extract(response: dict) -> Optional[dict]:
        candidates = []
        if isinstance(response.get('output_text'), str):
            candidates.append(response['output_text'])
        for item in response.get('output', []):
            if isinstance(item, dict):
                for content in item.get('content', []):
                    if isinstance(content, dict) and isinstance(content.get('text'), str):
                        candidates.append(content['text'])
        for text in candidates:
            parsed = _parse_json_text(text)
            if parsed is not None:
                return parsed
        return None

    def call(self, images: Sequence[np.ndarray], prompt: str) -> dict:
        key = os.getenv(self.config.openai_key_env)
        if not key:
            return {'status': 'missing_api_key', 'key_env': self.config.openai_key_env}
        return self._post(self.build_payload(images, prompt), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}, extract=self.extract)

    def _post(self, payload: dict, headers: dict, extract) -> dict:
        started = time.time()
        errors = []
        encoded = json.dumps(payload).encode('utf-8')
        for attempt in range(1, self.config.max_attempts + 1):
            request = urllib.request.Request(self.config.openai_url, data=encoded, headers=headers, method='POST')
            try:
                with guarded_urlopen(request, timeout=self.config.timeout) as response:
                    body = json.loads(response.read().decode('utf-8'))
                result = _validate_response(extract(body))
                result.update({'provider': 'openai', 'model': self.config.gpt_model, 'attempts': attempt, 'latency_sec': time.time() - started})
                return result
            except urllib.error.HTTPError as exc:
                error = exc.read().decode('utf-8', errors='replace')[:2000]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.config.max_attempts:
                    errors.append(f'http_{exc.code}')
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                return {'status': 'http_error', 'provider': 'openai', 'http_status': exc.code, 'error': error, 'attempts': attempt, 'latency_sec': time.time() - started}
            except Exception as exc:
                if attempt < self.config.max_attempts:
                    errors.append(str(exc)[:300])
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                return {'status': 'request_error', 'provider': 'openai', 'error': str(exc)[:2000], 'attempts': attempt, 'retry_errors': errors, 'latency_sec': time.time() - started}

class OllamaVisionAdapter:

    def __init__(self, config: VLMRecoveryConfig):
        self.config = config

    def build_payload(self, images: Sequence[np.ndarray], prompt: str) -> dict:
        encoded = [base64.b64encode(_jpeg_bytes(image, self.config.jpeg_quality)).decode('ascii') for image in images]
        return {'model': self.config.qwen_model, 'stream': False, 'think': False, 'format': RECOVERY_SCHEMA, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt, 'images': encoded}], 'options': {'temperature': 0}}

    def call(self, images: Sequence[np.ndarray], prompt: str) -> dict:
        started = time.time()
        errors = []
        encoded = json.dumps(self.build_payload(images, prompt)).encode('utf-8')
        for attempt in range(1, self.config.max_attempts + 1):
            request = urllib.request.Request(self.config.ollama_url, data=encoded, headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with guarded_urlopen(request, timeout=self.config.timeout) as response:
                    body = json.loads(response.read().decode('utf-8'))
                text = body.get('message', {}).get('content', '')
                result = _validate_response(_parse_json_text(text))
                result.update({'provider': 'ollama', 'model': self.config.qwen_model, 'attempts': attempt, 'latency_sec': time.time() - started, 'prompt_eval_count': body.get('prompt_eval_count'), 'eval_count': body.get('eval_count')})
                return result
            except Exception as exc:
                if attempt < self.config.max_attempts:
                    errors.append(str(exc)[:300])
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                return {'status': 'request_error', 'provider': 'ollama', 'error': str(exc)[:2000], 'attempts': attempt, 'retry_errors': errors, 'latency_sec': time.time() - started}

class VLMRecoveryController:

    def __init__(self, config: Optional[VLMRecoveryConfig]=None):
        self.config = config or VLMRecoveryConfig.from_env()
        self.qwen = OllamaVisionAdapter(self.config)
        self.gpt = OpenAIResponsesAdapter(self.config)
        self.reset(None, None, 'sequence')

    def reset(self, initial_image: Optional[np.ndarray], initial_bbox: Any, sequence_name: str):
        self.sequence_name = _safe_sequence_name(sequence_name)
        self.template_image = None
        self.reference_image = None if initial_image is None else np.asarray(initial_image).copy()
        self.reference_bbox = sanitize_bbox(initial_bbox)
        self.last_trigger_frame = -10 ** 9
        self.call_index = 0

    def set_template(self, template_image: np.ndarray):
        self.template_image = np.asarray(template_image).copy()

    def decision(self, uncertainty: dict, frame_id: int) -> dict:
        raw = np.asarray(uncertainty.get('risk_logits_raw', []), dtype=np.float64).reshape(-1)
        prob = np.asarray(uncertainty.get('risk_probabilities', []), dtype=np.float64).reshape(-1)
        if len(raw) < 3 or len(prob) < 3 or (not np.isfinite(raw[:3]).all()) or (not np.isfinite(prob[:3]).all()):
            return {'triggered': False, 'status': 'invalid_uncertainty'}
        threshold = self.config.bad_risk_threshold
        above = float(raw[1]) >= threshold
        cooldown = frame_id - self.last_trigger_frame < self.config.cooldown_frames
        triggered = bool(above and (not cooldown))
        tier = 'complex' if float(prob[2]) >= self.config.severe_probability_threshold else 'simple'
        return {'triggered': triggered, 'status': 'trigger' if triggered else 'continue', 'profile': self.config.profile, 'bad_score': float(raw[1]), 'threshold': threshold, 'risk_probabilities': prob[:3].tolist(), 'tier': tier, 'provider': 'openai' if tier == 'complex' else 'ollama', 'cooldown_active': bool(cooldown)}

    def step(self, uncertainty: dict, frame_id: int, current_image: np.ndarray, tracker_bbox: Any) -> dict:
        decision = self.decision(uncertainty, frame_id)
        probabilities = decision.get('risk_probabilities', [1, 1, 1])
        if not decision.get('triggered'):
            if len(probabilities) >= 2 and probabilities[1] <= self.config.trusted_bad_probability_max:
                self.reference_image = np.asarray(current_image).copy()
                self.reference_bbox = sanitize_bbox(tracker_bbox)
            return {'decision': decision, 'call': {'status': 'not_called'}}
        self.last_trigger_frame = int(frame_id)
        self.call_index += 1
        if not self.config.call_enabled:
            event = {'decision': decision, 'call': {'status': 'call_disabled'}}
            self._save_event(frame_id, tracker_bbox, event, None)
            return event
        if self.template_image is None or self.reference_image is None or self.reference_bbox is None:
            event = {'decision': decision, 'call': {'status': 'missing_reference'}}
            self._save_event(frame_id, tracker_bbox, event, None)
            return event
        (reference_crop, reference_crop_bbox) = _context_crop(self.reference_image, self.reference_bbox, factor=2.0)
        reference_crop = _draw_bbox(reference_crop, reference_crop_bbox, color=(0, 255, 0), label='trusted')
        current = _draw_bbox(current_image, tracker_bbox, color=(255, 255, 0), label='tracker hint')
        current = _resize_max_side(current, self.config.current_max_side)
        images = [self.template_image, reference_crop, current]
        (height, width) = current_image.shape[:2]
        prompt = USER_PROMPT.format(bbox_hint_norm=pixels_to_normalized(tracker_bbox, width, height), risks=probabilities)
        adapter = self.gpt if decision['tier'] == 'complex' else self.qwen
        call_result = adapter.call(images, prompt)
        if call_result.get('status') == 'completed' and call_result.get('target_present'):
            call_result['bbox_xywh_pixels'] = normalized_to_pixels(call_result['bbox_xywh_norm'], width, height)
        event = {'decision': decision, 'call': call_result}
        self._save_event(frame_id, tracker_bbox, event, images)
        return event

    def accept_recovery(self, image: np.ndarray, bbox: Any):
        self.reference_image = np.asarray(image).copy()
        self.reference_bbox = sanitize_bbox(bbox)

    def _save_event(self, frame_id: int, tracker_bbox: Any, event: dict, images: Optional[Sequence[np.ndarray]]):
        directory = self.config.output_root / self.sequence_name
        directory.mkdir(parents=True, exist_ok=True)
        stem = f'frame_{int(frame_id):06d}_call_{self.call_index:03d}'
        image_records = []
        if images is not None:
            for (index, image) in enumerate(images):
                encoded = _jpeg_bytes(image, self.config.jpeg_quality)
                record = {'index': index, 'sha256': hashlib.sha256(encoded).hexdigest(), 'shape': list(image.shape)}
                if self.config.save_images:
                    path = directory / f'{stem}_image_{index}.jpg'
                    path.write_bytes(encoded)
                    record['path'] = str(path)
                image_records.append(record)
        payload = {'sequence': self.sequence_name, 'frame': int(frame_id), 'tracker_bbox': sanitize_bbox(tracker_bbox), 'event': event, 'images': image_records, 'config': {'profile': self.config.profile, 'severe_probability_threshold': self.config.severe_probability_threshold, 'call_enabled': self.config.call_enabled, 'apply_recovery': self.config.apply_recovery, 'qwen_model': self.config.qwen_model, 'gpt_model': self.config.gpt_model}}
        (directory / f'{stem}.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
