"""Soft native-search prior for global localization only; no spatial gate.

Use the same current full-frame pixels twice: clean and search-outline copies.
No GT, tracker-state mutation, new search multiplier or extra backend request.
"""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text
import hashlib
import math
import cv2
from lib.models.qirtrack.core.vlm.mllm_full_controller import jpeg, shrink
from lib.models.qirtrack.core.vlm.mllm_i1_evidence import crop_geometry
from lib.models.qirtrack.core.vlm.mllm_i1_visual import VisualI1Controller

def global_search_views(image, previous, frame):
    """Outline the nominal native square, clipped to the visible full image.

    Coordinates are continuous image-edge coordinates, not target coordinates.
    Padding remains documented but is never drawn as fabricated image content.
    """
    geometry = crop_geometry(previous, previous, 5.0, 384, image.shape)
    (left, top, width, height) = geometry['crop_xywh_full']
    (ih, iw) = image.shape[:2]
    (x1, y1) = (max(0, left), max(0, top))
    (x2, y2) = (min(iw, left + width), min(ih, top + height))
    visible = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
    clean = shrink(image, 640)
    marked = clean.copy()
    (dh, dw) = clean.shape[:2]
    (sx, sy) = (dw / iw, dh / ih)
    rect = None
    if visible[2] > 0 and visible[3] > 0:
        rect = [max(0, min(dw - 1, math.floor(x1 * sx))), max(0, min(dh - 1, math.floor(y1 * sy))), max(0, min(dw - 1, math.ceil(x2 * sx) - 1)), max(0, min(dh - 1, math.ceil(y2 * sy) - 1))]
        cv2.rectangle(marked, tuple(rect[:2]), tuple(rect[2:]), (0, 255, 255), 2)
    common = dict(frame=frame, source_hw=[ih, iw], display_hw=[dh, dw], full_to_display=dict(sx=sx, sy=sy, tx=0.0, ty=0.0))
    entries = [dict(common, role='CURRENT_CLEAN_FULL', box_drawn=False, meaning=prompt_text('mllm_i1_global_hint_77100bfdbd')), dict(common, role='CURRENT_SEARCH_PRIORITY_FULL', box_drawn=rect is not None, meaning=prompt_text('mllm_i1_global_hint_4a8dbfad70'), native_search_factor=5.0, native_search_size=384, nominal_search_xywh_full=geometry['crop_xywh_full'], visible_search_xywh_full=visible, outline_ltrb_display=rect, padding_ltrb=geometry['padding_ltrb'], soft_prior_only=True)]
    return ((jpeg(clean), jpeg(marked)), entries)

class GlobalSearchPriorityController(VisualI1Controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, visual_arm='split_history', **kwargs)
        self.evidence_policy = 'split-history-global-native-search-soft-priority-v1'

    def _evidence(self, image, previous, current, past_limit=3, include_trusted=False, search_region=None):
        original = super()._evidence(image, previous, current, past_limit, include_trusted, search_region)
        if self._stage != 'global_recovery':
            return original
        if search_region is not None:
            raise RuntimeError(prompt_text('mllm_i1_global_hint_bbdc7f6b3e'))
        oldkey = tuple((hashlib.sha256(b).hexdigest() for b in original))
        (fulls, fullrows) = global_search_views(image, previous, self._frame)
        rows = self._manifests[oldkey]
        blobs = (*original[:2], *fulls, *original[3:])
        entries = [*rows[:2], *fullrows, *rows[3:]]
        registered = self._register(blobs, entries)
        key = tuple((r['sha256'] for r in registered))
        self._evidence_stages[key] = self._evidence_stages[oldkey]
        return blobs

    def _proposal_task(self):
        return prompt_text('mllm_i1_global_hint_a2801a1c8f')
