"""Reuse semantic/ATOM/PIC controller; replace AR-only proposal generation."""
from lib.models.qirtrack.core.vlm.quadrant_controller import ConflictFusedControllerV3 as AtomMainlineControllerV1

class NativeResponseController(AtomMainlineControllerV1):

    def _atom_shadow(self, *args, **kwargs):
        return {"status": "disabled_no_atom"}

    def _local_candidates(self, tracker, previous, raw, image_shape):
        (x, y, w, h) = map(float, raw)
        (height, width) = image_shape[:2]
        result = [list(raw)]
        for (dx, dy) in [(-0.15 * w, 0), (0.15 * w, 0), (0, -0.15 * h), (0, 0.15 * h)]:
            left = max(0.0, min(float(width) - 1.0, x + dx))
            top = max(0.0, min(float(height) - 1.0, y + dy))
            right = max(left + 1.0, min(float(width), x + dx + w))
            bottom = max(top + 1.0, min(float(height), y + dy + h))
            box = [left, top, right - left, bottom - top]
            if all((max((abs(a - b) for (a, b) in zip(box, old))) > 2.0 for old in result)):
                result.append(box)
        return result
