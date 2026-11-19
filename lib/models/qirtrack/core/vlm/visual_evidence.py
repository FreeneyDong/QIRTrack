"""I1 visual evidence arms; frozen G1 actions/thresholds are unchanged.

tight_anchor: strict prompt + distinct first-frame context and tight target.
split_history: tight_anchor + individual past images with actual frame numbers.
This does not add a local coordinate-generation stage or change search factors.
"""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text
import hashlib
import math
import cv2
import numpy as np
from lib.trackers.artrackv2.native.train.data.processing_utils import sample_target
from lib.models.qirtrack.core.vlm.controller import jpeg
from lib.models.qirtrack.core.vlm.evidence import I1Controller, crop_geometry

def tight_target(image, box, max_side=192):
    """Rectangular init-box crop, no context ring; keep aspect ratio.

    Integer enclosure contains <1 pixel extra on each boundary; expose its exact
    geometry instead of calling this segmentation or inventing pixels outside
    the image. Resampling occurs within this clipped rectangle.
    """
    (x, y, w, h) = map(float, box)
    if not all((math.isfinite(v) for v in (x, y, w, h))) or w <= 0 or h <= 0:
        raise ValueError('Invalid initialization rectangle')
    (left, top) = (max(0, math.floor(x)), max(0, math.floor(y)))
    (right, bottom) = (min(image.shape[1], math.ceil(x + w)), min(image.shape[0], math.ceil(y + h)))
    if right <= left or bottom <= top:
        raise ValueError(prompt_text('mllm_i1_visual_40e2e6fe50'))
    crop = image[top:bottom, left:right].copy()
    factor = max_side / max(crop.shape[:2])
    (dw, dh) = (max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor)))
    crop = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)
    (sx, sy) = (dw / (right - left), dh / (bottom - top))
    metadata = dict(role='ANCHOR_TARGET', frame=0, box_drawn=False, meaning=prompt_text('mllm_i1_visual_06dcc18700'), source_hw=list(image.shape[:2]), display_hw=[dh, dw], crop_xywh_full=[left, top, right - left, bottom - top], bbox_pixels_full=[x, y, w, h], bbox_pixels_in_display=[(x - left) * sx, (y - top) * sy, w * sx, h * sy], full_to_display=dict(sx=sx, sy=sy, tx=-left * sx, ty=-top * sy), integer_enclosure_extra_ltrb=[max(0, x - left), max(0, y - top), max(0, right - x - w), max(0, bottom - y - h)], context_expansion=1.0)
    return (jpeg(crop), metadata)

def context_target(image, box, area_factor=1.5, max_side=192):
    """Same centre/aspect as init bbox; modest rectangular context expansion."""
    if not math.isfinite(area_factor) or area_factor <= 1.0:
        raise ValueError(prompt_text('mllm_i1_visual_7e0d756273'))
    (x, y, w, h) = map(float, box)
    scale = math.sqrt(area_factor)
    (cx, cy) = (x + w / 2, y + h / 2)
    expanded = [cx - w * scale / 2, cy - h * scale / 2, w * scale, h * scale]
    (blob, entry) = tight_target(image, expanded, max_side)
    transform = entry['full_to_display']
    entry.update(role='ANCHOR_CONTEXT', meaning=prompt_text('mllm_i1_visual_72bb2a45e2'), bbox_pixels_full=[x, y, w, h], bbox_pixels_in_display=[x * transform['sx'] + transform['tx'], y * transform['sy'] + transform['ty'], w * transform['sx'], h * transform['sy']], context_expansion=area_factor, context_area_factor=area_factor, context_requested_xywh_full=expanded, context_centre_full=[cx, cy], context_side_factor=scale)
    return (blob, entry)

class VisualI1Controller(I1Controller):

    def __init__(self, *args, visual_arm, archive, anchor_context_area=1.5, **kwargs):
        if visual_arm not in {'tight_anchor', 'split_history'}:
            raise ValueError(visual_arm)
        super().__init__(*args, input_arm='strict_prompt', archive=archive, **kwargs)
        self.visual_arm = visual_arm
        self.evidence_policy = visual_arm + '-rect-context-v2'
        self.anchor_context_area = float(anchor_context_area)
        (self._tight_blob, self._tight_entry) = tight_target(self.first, self.init_box)
        (self._context_blob, self._context_meta) = context_target(self.first, self.init_box, self.anchor_context_area)

    def _evidence(self, image, previous, current, past_limit=3, include_trusted=False, search_region=None):
        originals = super()._evidence(image, previous, current, past_limit, include_trusted, search_region)
        oldkey = tuple((hashlib.sha256(b).hexdigest() for b in originals))
        oldrows = self._manifests[oldkey]
        blobs = [self._context_blob, self._tight_blob, *originals[1:3]]
        entries = [self._context_meta, self._tight_entry, *oldrows[1:3]]
        if self.visual_arm == 'tight_anchor':
            blobs.extend(originals[3:])
            entries.extend(oldrows[3:])
        else:
            selected = list(self.past)[-past_limit:] if past_limit else []
            for item in selected:
                if item['frame'] >= self._frame:
                    raise RuntimeError('Noncausal individual history')
                (crop, _, _) = sample_target(item['image'], item['box'], 2.0, output_sz=160)
                blobs.append(jpeg(crop))
                entries.append(dict(role='RECENT_RAW', frame=item['frame'], box_drawn=False, meaning=prompt_text('mllm_i1_visual_79462d23d9'), q_raw=item.get('q_raw'), tracker_score=item['score'], age=self._frame - item['frame'], **crop_geometry(item['box'], item['box'], 2.0, 160, item['image'].shape)))
            if include_trusted:
                for item in self.trusted:
                    if item['frame'] >= self._frame:
                        raise RuntimeError('Noncausal individual trusted history')
                    blobs.append(jpeg(item['crop']))
                    entries.append(dict(**item.get('crop_geometry', {})) | dict(role='TRUSTED', frame=item['frame'], box_drawn=False, meaning=prompt_text('mllm_i1_visual_9405558c25'), q_raw=item['q'], bbox_pixels_full=item['box'], verifier_model=item['verifier_model']))
        rows = self._register(tuple(blobs), entries)
        key = tuple((row['sha256'] for row in rows))
        self._evidence_stages[key] = self._evidence_stages[oldkey]
        return tuple(blobs)

    def _request(self, method, *args, bypass_budget=False, **kwargs):
        if not bypass_budget and self.budgeted_requests >= self.config.max_mllm_requests:
            return super()._request(method, *args, bypass_budget=bypass_budget, **kwargs)
        if method == 'choose_candidate':
            original = args[0]
            factor = min(1.0, 640 / max(self._current_shape[:2]))
            (dh, dw) = (max(1, round(self._current_shape[0] * factor)), max(1, round(self._current_shape[1] * factor)))
            entries = [self._context_meta, self._tight_entry, dict(role='CURRENT_FULL_UNMARKED', frame=self._frame, box_drawn=False, meaning=prompt_text('mllm_i1_visual_59e0624b7f'), source_hw=list(self._current_shape[:2]), display_hw=[dh, dw]), dict(role='CANDIDATE_GALLERY', frame=self._frame, box_drawn=False, tiles=[dict(index=i, candidate_id=f'C{i}', **crop_geometry(box, box, 2.0, 160, self._current_shape)) for (i, box) in enumerate(self._local_boxes)])]
            blobs = (self._context_blob, self._tight_blob, *original[1:])
            self._register(blobs, entries)
            args = (blobs, *args[1:])
        if method == 'propose' and self._stage != 'global_recovery':
            raise RuntimeError(prompt_text('mllm_i1_visual_e41b3d5d92'))
        return super()._request(method, *args, bypass_budget=bypass_budget, **kwargs)
