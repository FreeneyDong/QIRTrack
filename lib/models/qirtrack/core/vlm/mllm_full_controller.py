"""Causal, output-only MLLM tracking controller shared by full V2/V3 branches.

This file performs no annotation access.  It never mutates the native tracker;
the caller owns the native trajectory and records candidate-only counterfactuals.
"""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence
import cv2
import numpy as np
from lib.trackers.artrackv2.native.train.data.processing_utils import sample_target
from lib.models.qirtrack.core.localisation.ar_coordinate_proposals import decode_tokens_to_image_boxes, rank_coordinate_proposals
from lib.models.qirtrack.core.vlm.artrackv2_vlm_recovery import _draw_bbox
from lib.models.qirtrack.core.vlm.mllm_backend import MLLMBackend, expanded_box

@dataclass(frozen=True)
class FrameworkConfig:
    variant: str
    q_trigger: float = 0.4
    q_accept: float = 0.4
    candidate_acceptance_rule: str = 'absolute_high_yes'
    q_trusted: float = 0.8
    p_high: float = 0.8
    cooldown_frames: int = 15
    max_trigger_events: int = 6
    max_mllm_requests: int = 36
    uncapped_low_q: bool = False
    shadow_every: int = 90
    max_trusted: int = 4
    max_recent: int = 10
    ar_budget: int = 8
    geometry_budget: int = 4
    ar_top_k: int = 4
    ar_radius: int = 16
    coarse_area_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.variant not in {'binary', 'ternary'}:
            raise ValueError(self.variant)
        if self.candidate_acceptance_rule not in {'absolute_high_yes', 'relative_not_high_no'}:
            raise ValueError(self.candidate_acceptance_rule)
        if not 0 < self.q_trigger < self.q_trusted < 1 or not 0 < self.q_accept < 1:
            raise ValueError('Invalid Q thresholds')
        if not 0.5 < self.p_high < 1:
            raise ValueError('Invalid high-answer threshold')

def jpeg(rgb: np.ndarray) -> bytes:
    (ok, encoded) = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('Could not encode image')
    return encoded.tobytes()

def shrink(rgb: np.ndarray, max_side: int) -> np.ndarray:
    factor = min(1.0, max_side / max(rgb.shape[:2]))
    if factor == 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(rgb.shape[1] * factor)), max(1, round(rgb.shape[0] * factor))), interpolation=cv2.INTER_AREA)

def clip_box(box: Sequence[float], width: int, height: int) -> list[float]:
    (x, y, w, h) = (float(v) for v in box)
    (x1, y1) = (min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0))
    (x2, y2) = (min(max(x + w, x1 + 1), width), min(max(y + h, y1 + 1), height))
    return [x1, y1, x2 - x1, y2 - y1]

def _crop_with_mark(image: np.ndarray, region: Sequence[float], marked_box: Sequence[float], factor: float, side: int=384) -> np.ndarray:
    (crop, resize, _) = sample_target(image, region, factor, output_sz=side)
    square = math.ceil(math.sqrt(float(region[2] * region[3])) * factor)
    left = round(float(region[0] + region[2] / 2 - square / 2))
    top = round(float(region[1] + region[3] / 2 - square / 2))
    mapped = [(float(marked_box[0]) - left) * resize, (float(marked_box[1]) - top) * resize, float(marked_box[2]) * resize, float(marked_box[3]) * resize]
    return _draw_bbox(crop, mapped, color=(255, 255, 0), label='tracker')

def _contact_sheet(images: Sequence[np.ndarray], labels: Sequence[str], side: int=160, columns: int=4) -> np.ndarray:
    rows = math.ceil(len(images) / columns)
    result = np.zeros((rows * side, columns * side, 3), dtype=np.uint8)
    for (index, (im, label)) in enumerate(zip(images, labels)):
        tile = cv2.resize(im, (side, side), interpolation=cv2.INTER_AREA)
        (y, x) = divmod(index, columns)
        result[y * side:(y + 1) * side, x * side:(x + 1) * side] = tile
        cv2.putText(result, label, (x * side + 6, y * side + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return result

class FullMLLMController:

    def __init__(self, backend: MLLMBackend, config: FrameworkConfig, first: np.ndarray, init_box: Sequence[float], atom, rerun_and_score: Callable[[list[float]], tuple]):
        self.backend = backend
        self.config = config
        self.first = first.copy()
        self.init_box = list(init_box)
        (self.template, _, _) = sample_target(first, init_box, 2.0, output_sz=192)
        self.atom = atom
        self.rerun_and_score = rerun_and_score
        self.past: deque[dict] = deque(maxlen=max(16, config.max_recent))
        self.trusted: deque[dict] = deque(maxlen=config.max_trusted)
        self.triggers = 0
        self.requests = 0
        self.budgeted_requests = 0
        self.last_trigger = -10 ** 9

    def _request(self, method: str, *args, bypass_budget: bool=False, **kwargs) -> dict:
        if not bypass_budget and self.budgeted_requests >= self.config.max_mllm_requests:
            return {'status': 'budget_exhausted'}
        self.requests += 1
        if not bypass_budget:
            self.budgeted_requests += 1
        return getattr(self.backend, method)(*args, **kwargs)

    def _evidence(self, image: np.ndarray, previous: Sequence[float], current: Sequence[float], past_limit: int=3, include_trusted: bool=False, search_region: Sequence[float] | None=None) -> tuple[bytes, ...]:
        full = _draw_bbox(image, current, color=(255, 255, 0), label='current')
        search = _crop_with_mark(image, previous if search_region is None else search_region, current, 5.0)
        images = [shrink(self.template, 192), shrink(full, 640), shrink(search, 384)]
        if past_limit:
            selected = list(self.past)[-past_limit:]
            if selected:
                crops = [sample_target(item['image'], item['box'], 2.0, output_sz=160)[0] for item in selected]
                labels = [f't-{len(selected) - i}' for i in range(len(selected))]
                images.append(_contact_sheet(crops, labels))
        if include_trusted and self.trusted:
            crops = [item['crop'] for item in self.trusted]
            labels = [f"trusted {item['frame']}" for item in self.trusted]
            images.append(_contact_sheet(crops, labels))
        return tuple((jpeg(im) for im in images))

    @staticmethod
    def _question() -> str:
        return prompt_text('mllm_full_controller_ae4aaa98c7')

    def _classify(self, score: dict) -> str:
        if score.get('status') != 'ok':
            return 'uncertain'
        probs = score['conditional_probs_uncalibrated']
        if probs.get('YES', 0) >= self.config.p_high:
            return 'high_yes'
        if probs.get('NO', 0) >= self.config.p_high:
            return 'high_no'
        return 'uncertain'

    def _candidate_accepts(self, score: dict, q_raw: float, q_candidate: float) -> bool:
        """Apply exactly one frozen postcheck rule; never consult ground truth."""
        if not math.isfinite(q_candidate):
            return False
        if self.config.candidate_acceptance_rule == 'absolute_high_yes':
            return self._classify(score) == 'high_yes' and q_candidate >= self.config.q_accept
        if not math.isfinite(q_raw) or q_candidate <= q_raw:
            return False
        if score.get('status') != 'ok':
            return False
        probs = score.get('conditional_probs_uncalibrated')
        if not isinstance(probs, dict):
            return False
        (yes, no) = (probs.get('YES'), probs.get('NO'))
        if isinstance(yes, bool) or isinstance(no, bool) or (not isinstance(yes, (int, float))) or (not isinstance(no, (int, float))) or (not math.isfinite(yes)) or (not math.isfinite(no)) or (not 0.0 <= yes <= 1.0) or (not 0.0 <= no <= 1.0) or (not math.isclose(yes + no, 1.0, abs_tol=0.0001)):
            return False
        return no < self.config.p_high

    def _atom_shadow(self, image: np.ndarray, raw: Sequence[float]) -> dict:
        if self.atom is None:
            raise RuntimeError(prompt_text('mllm_full_controller_b9c1475a04'))
        try:
            refinement = self.atom.refine(image, np.asarray(raw, dtype=np.float32))
            candidate = refinement.get('refined_top3_box')
            return {'status': 'failed' if refinement.get('failed') else 'ok', 'candidate_box': None if candidate is None else np.asarray(candidate, dtype=np.float64).reshape(4).tolist()}
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            return {'status': 'failed', 'error': str(exc)[:300]}

    def _local_candidates(self, tracker, previous: Sequence[float], raw: Sequence[float], image_shape: Sequence[int]) -> list[list[float]]:
        diagnostics = tracker.last_diagnostics
        logits = np.asarray(diagnostics['coord_logits'])
        raw_tokens = np.asarray(diagnostics['coord_argmax_index'])
        rows = rank_coordinate_proposals(logits, raw_tokens, budget=self.config.ar_budget, top_k=self.config.ar_top_k, radius=self.config.ar_radius, beam_width=64)
        decoded = decode_tokens_to_image_boxes(rows, previous, image_shape, bins=int(tracker.bins), search_size=int(tracker.params.search_size), search_factor=float(tracker.params.search_factor)) if rows else []
        candidates = [list(raw)]
        for box in decoded:
            if all((np.max(np.abs(np.asarray(box) - np.asarray(old))) > 2 for old in candidates)):
                candidates.append([float(v) for v in box])
        (x, y, w, h) = (float(v) for v in raw)
        shifts = [(-0.15 * w, 0), (0.15 * w, 0), (0, -0.15 * h), (0, 0.15 * h)]
        for (dx, dy) in shifts[:self.config.geometry_budget]:
            box = clip_box([x + dx, y + dy, w, h], image_shape[1], image_shape[0])
            if all((np.max(np.abs(np.asarray(box) - np.asarray(old))) > 2 for old in candidates)):
                candidates.append(box)
        return candidates

    def _candidate_gallery(self, image: np.ndarray, candidates: Sequence[Sequence[float]]) -> bytes:
        crops = [sample_target(image, box, 2.0, output_sz=160)[0] for box in candidates]
        return jpeg(_contact_sheet(crops, [f'C{i}' for i in range(len(crops))]))

    def _propose(self, images: Sequence[bytes], prompt: str, image: np.ndarray, bypass_budget: bool=False) -> tuple[list[float] | None, dict]:
        proposal = self._request('propose', images, prompt, bypass_budget=bypass_budget)
        if proposal.get('status') != 'completed' or not proposal.get('target_present'):
            return (None, proposal)
        (x, y, w, h) = proposal['bbox_xywh_norm']
        if w * h >= 0.8:
            proposal['status'] = 'rejected_whole_frame_box'
            return (None, proposal)
        original = [x * image.shape[1], y * image.shape[0], w * image.shape[1], h * image.shape[0]]
        expanded = expanded_box(original, image.shape[1], image.shape[0], self.config.coarse_area_factor)
        proposal['tight_box_pixels'] = original
        proposal['expanded_box_pixels'] = expanded
        proposal['actual_area_factor'] = expanded[2] * expanded[3] / (original[2] * original[3])
        return (expanded, proposal)

    def _postcheck(self, image: np.ndarray, previous: Sequence[float], raw: Sequence[float], seed: list[float], row: dict, origin: str, bypass_budget: bool=False) -> bool:
        try:
            rerun_result = self.rerun_and_score(seed)
            if len(rerun_result) == 2:
                (candidate, q_candidate) = rerun_result
                q_candidate_seeds = None
            elif len(rerun_result) == 3:
                (candidate, q_candidate, q_candidate_seeds) = rerun_result
            else:
                raise ValueError(prompt_text('mllm_full_controller_51e81d2a4e'))
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            row['postcheck'] = {'status': 'tracker_rerun_error', 'error': str(exc)[:300]}
            row.setdefault('candidate_attempts', []).append({'origin': origin, 'seed': seed, 'status': 'tracker_rerun_error'})
            row['action'] = origin + '_rejected_no_op'
            return False
        candidate = clip_box(candidate, image.shape[1], image.shape[0])
        row['candidate_box'] = candidate
        row['candidate_q'] = q_candidate
        row['candidate_origin'] = origin
        attempt = {'origin': origin, 'seed': seed, 'box': candidate, 'q': q_candidate, 'q_raw': row.get('q_raw'), 'q_gain_pred': q_candidate - row['q_raw'], 'acceptance_rule': self.config.candidate_acceptance_rule, 'status': 'postcheck_pending'}
        if q_candidate_seeds is not None:
            values = [float(value) for value in q_candidate_seeds]
            if len(values) != 3 or not all((math.isfinite(value) for value in values)):
                raise ValueError(prompt_text('mllm_full_controller_8463e51906'))
            mean = sum(values) / len(values)
            if not math.isclose(mean, q_candidate, rel_tol=0.0, abs_tol=1e-07):
                raise ValueError(prompt_text('mllm_full_controller_f675d39ca9'))
            attempt['q_seeds'] = values
            attempt['q_mean'] = mean
            attempt['q_std'] = math.sqrt(sum(((value - mean) ** 2 for value in values)) / len(values))
            attempt['q_min'] = min(values)
            attempt['q_max'] = max(values)
        row.setdefault('candidate_attempts', []).append(attempt)
        evidence = self._evidence(image, previous, candidate, past_limit=3, include_trusted=True, search_region=candidate)
        score = self._request('score', evidence, self.config.variant, self._question(), bypass_budget=bypass_budget)
        row['postcheck'] = score
        attempt['verifier'] = score
        attempt['verifier_branch'] = self._classify(score)
        if self._candidate_accepts(score, row['q_raw'], q_candidate):
            attempt['status'] = 'accepted'
            row['final_box'] = candidate
            row['action'] = origin + '_accepted_output_only'
            return True
        attempt['status'] = 'rejected'
        row['action'] = origin + '_rejected_no_op'
        return False

    def _recover(self, image: np.ndarray, previous: Sequence[float], raw: Sequence[float], row: dict, temporal: bool=False, bypass_budget: bool=False) -> None:
        evidence = self._evidence(image, previous, raw, past_limit=self.config.max_recent if temporal else 3, include_trusted=temporal)
        row['recovery_stages'] = []
        prompt = prompt_text('mllm_full_controller_e4284c4bb3')
        (seed, proposal) = self._propose(evidence, prompt, image, bypass_budget=bypass_budget)
        row['recovery_stages'].append({'stage': 'global', 'proposal': proposal})
        if seed is not None and self._postcheck(image, previous, raw, seed, row, 'global', bypass_budget=bypass_budget):
            return
        row['action'] = 'recovery_exhausted_no_op'

    def step(self, frame: int, image: np.ndarray, previous: Sequence[float], raw: Sequence[float], q_raw: float, tracker) -> dict:
        row = {'frame': frame, 'q_raw': q_raw, 'raw_box': list(raw), 'final_box': list(raw), 'candidate_box': None, 'triggered': False, 'action': 'no_op', 'variant': self.config.variant, 'requests_before': self.requests}
        if q_raw >= self.config.q_trigger:
            if q_raw >= self.config.q_trusted and frame % self.config.shadow_every == 0:
                evidence = self._evidence(image, previous, raw, past_limit=3)
                audit = self._request('score', evidence, self.config.variant, self._question())
                row['periodic_audit'] = audit
                if self._classify(audit) == 'high_yes':
                    (crop, _, _) = sample_target(image, raw, 2.0, output_sz=160)
                    duplicate = any((np.mean(np.abs(crop.astype(np.float32) - item['crop'].astype(np.float32))) / 255 < 0.03 for item in self.trusted))
                    if not duplicate:
                        self.trusted.append({'frame': frame, 'crop': crop, 'box': list(raw), 'q': q_raw, 'verifier_variant': self.config.variant, 'verifier_model': self.backend.model})
                        row['trusted_added'] = True
                    else:
                        row['trusted_duplicate_rejected'] = True
            row['requests_after'] = self.requests
            return row
        if not self.config.uncapped_low_q and (frame - self.last_trigger < self.config.cooldown_frames or self.triggers >= self.config.max_trigger_events):
            row['action'] = 'cooldown_or_trigger_budget'
            row['requests_after'] = self.requests
            return row
        self.triggers += 1
        self.last_trigger = frame
        row['triggered'] = True
        evidence = self._evidence(image, previous, raw, past_limit=3)
        score = self._request('score', evidence, self.config.variant, self._question(), bypass_budget=self.config.uncapped_low_q)
        row['verifier'] = score
        branch = self._classify(score)
        if branch == 'uncertain':
            temporal = self._evidence(image, previous, raw, past_limit=self.config.max_recent, include_trusted=True)
            second = self._request('score', temporal, self.config.variant, self._question(), bypass_budget=self.config.uncapped_low_q)
            row['temporal_verifier'] = second
            branch = self._classify(second)
            if branch == 'uncertain':
                self._recover(image, previous, raw, row, temporal=True, bypass_budget=self.config.uncapped_low_q)
                row['requests_after'] = self.requests
                return row
        if branch == 'high_yes':
            row['atom_shadow'] = self._atom_shadow(image, raw)
            row['action'] = 'high_yes_atom_shadow_no_op'
            row['requests_after'] = self.requests
            return row
        candidates = self._local_candidates(tracker, previous, raw, image.shape)
        row['local_candidates'] = candidates
        if len(candidates) > 1:
            gallery = self._candidate_gallery(image, candidates)
            local_prompt = prompt_text('mllm_full_controller_4ce68fa199')
            selection = self._request('choose_candidate', (jpeg(shrink(self.template, 192)), jpeg(shrink(image, 640)), gallery), len(candidates), local_prompt, bypass_budget=self.config.uncapped_low_q)
            row['local_selection'] = selection
            selected = selection.get('selected_index')
            if isinstance(selected, int) and 0 < selected < len(candidates):
                if self._postcheck(image, previous, raw, candidates[selected], row, 'local_marginal', bypass_budget=self.config.uncapped_low_q):
                    row['requests_after'] = self.requests
                    return row
            elif selected == 0:
                row['action'] = 'local_selected_raw_no_op'
                row['requests_after'] = self.requests
                return row
        self._recover(image, previous, raw, row, bypass_budget=self.config.uncapped_low_q)
        row['requests_after'] = self.requests
        return row

    def remember(self, frame: int, image: np.ndarray, raw: Sequence[float], score: float, entropy: float) -> None:
        self.past.append({'frame': frame, 'image': image.copy(), 'box': np.asarray(raw, dtype=np.float32).copy(), 'score': float(score), 'entropy': float(entropy)})
