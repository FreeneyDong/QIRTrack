"""First-token identity-label scoring with Ollama's native chat logprobs.

This is a shadow-only evidence interface. Its returned probabilities are
conditional on the canonical answer tokens, not calibrated identity risks.
"""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text
from lib.models.qirtrack.mllm.prompts import guarded_urlopen
import base64
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, Sequence
Variant = Literal['binary', 'ternary']
LABELS: dict[Variant, tuple[str, ...]] = {'binary': ('YES', 'NO'), 'ternary': ('YES', 'NO', 'UNKNOWN')}
IDENTITY_QUESTION = prompt_text('identity_verifier_logprobs_176584980d')

@dataclass(frozen=True)
class OllamaIdentityLogprobsConfig:
    model: str = 'qwen3.6:35b'
    url: str = 'http://127.0.0.1:11434/api/chat'
    timeout_sec: float = 300.0
    top_logprobs: int = 20
    keep_alive: str = '5m'

    def __post_init__(self) -> None:
        if not 1 <= self.top_logprobs <= 20:
            raise ValueError(prompt_text('identity_verifier_logprobs_53f8e02d20'))
        if self.timeout_sec <= 0:
            raise ValueError('timeout_sec must be positive')

def _system_prompt(variant: Variant) -> str:
    if variant == 'binary':
        return prompt_text('identity_verifier_logprobs_39c6fd13c0')
    if variant == 'ternary':
        return prompt_text('identity_verifier_logprobs_f4aac23348')
    raise ValueError(f'Unknown verifier variant: {variant}')

def build_payload(config: OllamaIdentityLogprobsConfig, images: Sequence[bytes], variant: Variant, *, user_prompt: str=IDENTITY_QUESTION) -> dict:
    """Build a score-only request; image bytes must be in fixed evidence order."""
    if variant not in LABELS:
        raise ValueError(f'Unknown verifier variant: {variant}')
    if not images or any((not isinstance(image, bytes) or not image for image in images)):
        raise ValueError(prompt_text('identity_verifier_logprobs_9af69ef895'))
    if not user_prompt.strip():
        raise ValueError('user_prompt must not be empty')
    return {'model': config.model, 'stream': False, 'think': False, 'messages': [{'role': 'system', 'content': _system_prompt(variant)}, {'role': 'user', 'content': user_prompt, 'images': [base64.b64encode(image).decode('ascii') for image in images]}], 'options': {'temperature': 1, 'top_k': 0, 'top_p': 1, 'repeat_penalty': 1, 'num_predict': 1}, 'logprobs': True, 'top_logprobs': config.top_logprobs, 'keep_alive': config.keep_alive}

def parse_label_logprobs(response: dict, variant: Variant) -> dict:
    """Require every canonical one-token label at the first generation step.

    Never infer missing logprobs from the generated answer, a model-provided
    confidence field, or the tail of a top-k distribution.
    """
    labels = LABELS[variant]
    steps = response.get('logprobs')
    if not isinstance(steps, list) or not steps or (not isinstance(steps[0], dict)):
        return {'status': 'missing_logprobs', 'variant': variant}
    first = steps[0]
    top = first.get('top_logprobs')
    if not isinstance(top, list):
        return {'status': 'missing_top_logprobs', 'variant': variant}
    scores: dict[str, float] = {}
    for entry in top:
        if not isinstance(entry, dict) or entry.get('token') not in labels:
            continue
        token = entry['token']
        value = entry.get('logprob')
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            return {'status': 'invalid_logprob', 'variant': variant, 'token': token}
        value = float(value)
        if not math.isfinite(value) or value > 1e-06 or token in scores:
            return {'status': 'invalid_logprob', 'variant': variant, 'token': token}
        scores[token] = value
    missing = [label for label in labels if label not in scores]
    if missing:
        return {'status': 'missing_label_logprobs', 'variant': variant, 'missing': missing, 'observed_labels': sorted(scores)}
    peak = max(scores.values())
    weights = {label: math.exp(scores[label] - peak) for label in labels}
    total = sum(weights.values())
    label_mass = sum((math.exp(scores[label]) for label in labels))
    return {'status': 'ok', 'variant': variant, 'raw_label_logprobs': scores, 'conditional_probs_uncalibrated': {label: weights[label] / total for label in labels}, 'canonical_label_mass': label_mass, 'generated_first_token': first.get('token'), 'generated_answer': response.get('message', {}).get('content'), 'prompt_eval_count': response.get('prompt_eval_count'), 'eval_count': response.get('eval_count')}

class OllamaIdentityLogprobsAdapter:

    def __init__(self, config: OllamaIdentityLogprobsConfig | None=None):
        self.config = config or OllamaIdentityLogprobsConfig()

    def score(self, images: Sequence[bytes], variant: Variant, *, user_prompt: str=IDENTITY_QUESTION) -> dict:
        payload = build_payload(self.config, images, variant, user_prompt=user_prompt)
        request = urllib.request.Request(self.config.url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with guarded_urlopen(request, timeout=self.config.timeout_sec) as handle:
                response = json.load(handle)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            return {'status': 'request_error', 'variant': variant, 'error': str(exc)[:300]}
        if not isinstance(response, dict):
            return {'status': 'invalid_response', 'variant': variant}
        if response.get('error'):
            return {'status': 'backend_error', 'variant': variant, 'error': str(response['error'])[:300]}
        result = parse_label_logprobs(response, variant)
        result['model'] = self.config.model
        return result

    def score_pair(self, images: Sequence[bytes], *, user_prompt: str=IDENTITY_QUESTION) -> dict[str, dict]:
        """Matched V2/V3 calls with the identical ordered evidence and question."""
        frozen_images = tuple(images)
        return {variant: self.score(frozen_images, variant, user_prompt=user_prompt) for variant in ('binary', 'ternary')}
