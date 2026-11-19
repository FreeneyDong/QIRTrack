"""Comparable local vision backends for the ARTrack MLLM study.

All probabilities are conditional on canonical answer tokens, not calibrated
identity probabilities.  This module never reads tracking ground truth.
"""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text, require_prompts
from lib.models.qirtrack.mllm.prompts import guarded_urlopen
import base64
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Sequence
from lib.models.qirtrack.core.vlm.identity_verifier_logprobs import LABELS, _system_prompt, parse_label_logprobs
from lib.models.qirtrack.core.vlm.artrackv2_vlm_recovery import RECOVERY_SCHEMA, _parse_json_text, _validate_response
TIGHT_SYSTEM = prompt_text('mllm_backend_c9a9490d5d')

@dataclass(frozen=True)
class MLLMBackend:
    kind: str
    model: str
    url: str
    timeout_sec: float = 300.0

    @classmethod
    def ollama(cls) -> 'MLLMBackend':
        import os
        return cls('ollama', os.environ['QIRTRACK_MLLM_MODEL'], os.environ['QIRTRACK_MLLM_URL'])

    @classmethod
    def vllm(cls) -> 'MLLMBackend':
        import os
        return cls('vllm', os.environ['QIRTRACK_MLLM_MODEL'], os.environ['QIRTRACK_MLLM_URL'])

    def _post(self, payload: dict) -> tuple[dict | None, float, str | None]:
        require_prompts()
        started = time.perf_counter()
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with guarded_urlopen(request, timeout=self.timeout_sec) as handle:
                body = json.load(handle)
            if not isinstance(body, dict):
                raise ValueError('Non-object backend response')
            return (body, time.perf_counter() - started, None)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            return (None, time.perf_counter() - started, str(exc)[:400])

    def _messages(self, system: str, prompt: str, images: Sequence[bytes]) -> list[dict]:
        if self.kind == 'ollama':
            return [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt, 'images': [base64.b64encode(im).decode('ascii') for im in images]}]
        contents = [{'type': 'text', 'text': prompt}]
        contents.extend(({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(im).decode('ascii')}} for im in images))
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': contents}]

    def score(self, images: Sequence[bytes], variant: str, question: str) -> dict:
        if variant not in LABELS:
            raise ValueError(variant)
        messages = self._messages(_system_prompt(variant), question, images)
        if self.kind == 'ollama':
            payload = {'model': self.model, 'messages': messages, 'stream': False, 'think': False, 'options': {'temperature': 1, 'top_k': 0, 'top_p': 1, 'repeat_penalty': 1, 'num_predict': 1}, 'logprobs': True, 'top_logprobs': 20}
        else:
            payload = {'model': self.model, 'messages': messages, 'temperature': 1, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 20, 'chat_template_kwargs': {'enable_thinking': False}}
        (body, latency, error) = self._post(payload)
        if error:
            return {'status': 'request_error', 'model': self.model, 'error': error, 'latency_sec': latency}
        if self.kind == 'ollama':
            parsed = parse_label_logprobs(body, variant)
        else:
            try:
                choice = body['choices'][0]
                step = choice['logprobs']['content'][0]
                pseudo = {'logprobs': [{'token': step['token'], 'top_logprobs': step['top_logprobs']}], 'message': choice['message']}
                parsed = parse_label_logprobs(pseudo, variant)
            except (KeyError, IndexError, TypeError) as exc:
                parsed = {'status': 'invalid_logprobs', 'error': str(exc)}
        parsed.update(model=self.model, backend=self.kind, latency_sec=latency)
        return parsed

    def propose(self, images: Sequence[bytes], prompt: str) -> dict:
        messages = self._messages(TIGHT_SYSTEM, prompt, images)
        if self.kind == 'ollama':
            payload = {'model': self.model, 'messages': messages, 'stream': False, 'think': False, 'format': RECOVERY_SCHEMA, 'options': {'temperature': 0}}
        else:
            payload = {'model': self.model, 'messages': messages, 'temperature': 0, 'max_tokens': 256, 'response_format': {'type': 'json_object'}, 'chat_template_kwargs': {'enable_thinking': False}}
        (body, latency, error) = self._post(payload)
        if error:
            return {'status': 'request_error', 'model': self.model, 'error': error, 'latency_sec': latency}
        try:
            content = body['message']['content'] if self.kind == 'ollama' else body['choices'][0]['message']['content']
            parsed = _validate_response(_parse_json_text(content))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            parsed = {'status': 'invalid_output', 'error': str(exc)[:300]}
        parsed.update(model=self.model, backend=self.kind, latency_sec=latency)
        return parsed

    def choose_candidate(self, images: Sequence[bytes], count: int, prompt: str) -> dict:
        """Choose a displayed candidate ID or explicitly reject every candidate."""
        if count < 1:
            raise ValueError('Candidate count must be positive')
        system = f"{prompt_text('mllm_backend_d21f1e1d39')}{count - 1}{prompt_text('mllm_backend_f50f69c0fa')}"
        messages = self._messages(system, prompt, images)
        if self.kind == 'ollama':
            payload = {'model': self.model, 'messages': messages, 'stream': False, 'think': False, 'format': {'type': 'object', 'properties': {'selection': {'type': 'string'}, 'evidence': {'type': 'string'}}, 'required': ['selection', 'evidence'], 'additionalProperties': False}, 'options': {'temperature': 0}}
        else:
            payload = {'model': self.model, 'messages': messages, 'temperature': 0, 'max_tokens': 128, 'response_format': {'type': 'json_object'}, 'chat_template_kwargs': {'enable_thinking': False}}
        (body, latency, error) = self._post(payload)
        if error:
            return {'status': 'request_error', 'model': self.model, 'error': error, 'latency_sec': latency}
        try:
            content = body['message']['content'] if self.kind == 'ollama' else body['choices'][0]['message']['content']
            parsed = _parse_json_text(content)
            selection = parsed.get('selection') if isinstance(parsed, dict) else None
            if selection == 'REJECT_ALL':
                index = None
            elif isinstance(selection, str) and selection.startswith('C') and selection[1:].isdigit() and (0 <= int(selection[1:]) < count):
                index = int(selection[1:])
            else:
                raise ValueError('invalid candidate selection')
            return {'status': 'ok', 'selection': selection, 'selected_index': index, 'evidence': str(parsed.get('evidence', ''))[:300], 'model': self.model, 'backend': self.kind, 'latency_sec': latency}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return {'status': 'invalid_output', 'error': str(exc)[:300], 'model': self.model, 'backend': self.kind, 'latency_sec': latency}

def expanded_box(box: Sequence[float], image_width: int, image_height: int, area_factor: float=2.0) -> list[float]:
    """Expand around the MLLM box centre; clipping may reduce actual area."""
    (x, y, width, height) = (float(value) for value in box)
    if not all((math.isfinite(v) for v in (x, y, width, height))) or width <= 0 or height <= 0:
        raise ValueError('Invalid visible target box')
    scale = math.sqrt(area_factor)
    (cx, cy) = (x + width / 2, y + height / 2)
    (x1, y1) = (max(0.0, cx - scale * width / 2), max(0.0, cy - scale * height / 2))
    (x2, y2) = (min(float(image_width), cx + scale * width / 2), min(float(image_height), cy + scale * height / 2))
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
