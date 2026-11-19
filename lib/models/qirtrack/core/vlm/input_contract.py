"""Fail-closed contracts for actual multimodal inputs; no model or GT access."""
from __future__ import annotations
from lib.models.qirtrack.mllm.prompts import prompt_text
import base64
import hashlib
import json
import math
import numbers
import cv2
import numpy as np
CONTRACT_VERSION = 'ordered-evidence-v2'
ROLES = frozenset(('ANCHOR_TARGET', 'ANCHOR_CONTEXT', 'RECENT_RAW', 'TRUSTED', 'CURRENT_RAW_FULL', 'CURRENT_RAW_SEARCH', 'CURRENT_CANDIDATE_FULL', 'CURRENT_CANDIDATE_SEARCH', 'CURRENT_CLEAN_FULL', 'CURRENT_SEARCH_PRIORITY_FULL', 'CURRENT_FULL_UNMARKED', 'CANDIDATE_GALLERY'))

class EvidenceContractError(ValueError):
    """Abort before transport or durable request creation on invalid evidence."""

def _require(condition, message):
    if not condition:
        raise EvidenceContractError(message)

def _integer(value):
    return isinstance(value, numbers.Integral) and (not isinstance(value, (bool, np.bool_)))

def _numbers(value, length, label, positive_tail=0):
    _require(isinstance(value, (list, tuple)) and len(value) == length, label + ' shape')
    _require(all((isinstance(x, numbers.Real) and (not isinstance(x, (bool, np.bool_))) and math.isfinite(x) for x in value)), label + ' nonfinite/type')
    if positive_tail:
        _require(all((x > 0 for x in value[-positive_tail:])), label + ' nonpositive size')

def _geometry(row, actual_hw=None):
    for field in ('source_hw', 'display_hw', 'encoded_hw'):
        if field in row:
            v = row[field]
            _require(isinstance(v, (list, tuple)) and len(v) == 2 and all((_integer(x) and x > 0 for x in v)), field + ' invalid')
    if actual_hw is not None and 'encoded_hw' in row:
        _require(list(actual_hw) == list(row['encoded_hw']), 'Encoded image size mismatch')
    if actual_hw is not None and 'display_hw' in row:
        _require(list(actual_hw) == list(row['display_hw']), 'Display image size mismatch')
    for field in ('bbox_pixels_full', 'crop_xywh_full'):
        if field in row:
            _numbers(row[field], 4, field, positive_tail=2)
    if 'bbox_pixels_in_display' in row:
        _numbers(row['bbox_pixels_in_display'], 4, 'bbox_pixels_in_display', positive_tail=2)
    if 'full_to_display' in row:
        transform = row['full_to_display']
        _require(isinstance(transform, dict) and set(transform) == {'sx', 'sy', 'tx', 'ty'}, 'Invalid transform')
        _numbers([transform[k] for k in ('sx', 'sy', 'tx', 'ty')], 4, 'transform')
        _require(transform['sx'] > 0 and transform['sy'] > 0, 'Nonpositive scale')
        if 'bbox_pixels_full' in row and 'bbox_pixels_in_display' in row:
            (x, y, w, h) = row['bbox_pixels_full']
            expected = [x * transform['sx'] + transform['tx'], y * transform['sy'] + transform['ty'], w * transform['sx'], h * transform['sy']]
            _require(np.allclose(expected, row['bbox_pixels_in_display'], rtol=1e-06, atol=0.0001), 'BBox coordinate transform mismatch')
    if 'padding_ltrb' in row:
        _require(len(row['padding_ltrb']) == 4 and all((_integer(x) and x >= 0 for x in row['padding_ltrb'])), 'Invalid padding')

def validate_manifest(rows, current_frame, blobs=None):
    _require(_integer(current_frame) and current_frame >= 0, 'Missing/invalid current frame')
    _require(isinstance(rows, list) and rows, 'Empty image manifest')
    if blobs is not None:
        _require(len(rows) == len(blobs), 'Wrong image count')
    _require([r.get('image_id') for r in rows] == [f'image_{i}' for i in range(1, len(rows) + 1)], prompt_text('mllm_input_contract_fe8ef27c71'))
    _require(sum((r.get('role') == 'ANCHOR_TARGET' for r in rows)) == 1, 'Exactly one target anchor required')
    for (position, row) in enumerate(rows):
        role = row.get('role')
        _require(role in ROLES, 'Unknown image role')
        actual_hw = row.get('encoded_hw')
        if blobs is not None:
            blob = blobs[position]
            _require(hashlib.sha256(blob).hexdigest() == row.get('sha256'), 'Image bytes/order mismatch')
            decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            _require(decoded is not None, 'Invalid JPEG image')
            actual_hw = list(decoded.shape[:2])
        _geometry(row, actual_hw)
        frame = row.get('frame')
        if role.startswith('ANCHOR_'):
            _require(_integer(frame) and frame == 0, prompt_text('mllm_input_contract_45efa27041'))
        elif role.startswith('CURRENT_') or role == 'CANDIDATE_GALLERY':
            _require(_integer(frame) and frame == current_frame, 'Current image/frame mismatch')
        else:
            tiles = row.get('tiles')
            histories = tiles if tiles is not None else [row]
            _require(isinstance(histories, list) and histories, 'Empty history evidence')
            for history in histories:
                f = history.get('frame')
                _require(_integer(f) and 0 <= f < current_frame, 'Current/future history forbidden')
                if 'age' in history:
                    _require(history['age'] == current_frame - f, 'History age mismatch')
                _geometry(history)
        if 'tiles' in row:
            tiles = row['tiles']
            _require([t.get('index') for t in tiles] == list(range(len(tiles))), 'Gallery/history tile order mismatch')
            for tile in tiles:
                _geometry(tile)
            if role == 'CANDIDATE_GALLERY':
                _require([t.get('candidate_id') for t in tiles] == [f'C{i}' for i in range(len(tiles))], 'Candidate IDs/order mismatch')
        if 'frame_number_one_based' in row:
            _require(row['frame_number_one_based'] == frame + 1, 'One-based frame mismatch')
    return rows

def _prompt_manifest(text):
    marker = '\nORDERED IMAGE ROLES: '
    _require(text.count(marker) == 1, 'Missing/duplicate actual prompt image manifest')
    try:
        (parsed, end) = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])
    except (ValueError, TypeError) as exc:
        raise EvidenceContractError('Invalid actual prompt manifest') from exc
    _require(isinstance(parsed, list), prompt_text('mllm_input_contract_cf1f649a79'))
    return parsed

def validate_payload(payload, context):
    """Check final payload, including compact structured prompts, before _post."""
    if 'sequence_source' in context:
        (source, frame_source) = (context['sequence_source'], context.get('frame_source', {}))
        identity = context.get('request_identity', context)
        _require(all((source.get(k) == identity.get(k) for k in ('domain', 'sequence'))), 'Request/archive dataset sequence mismatch')
        _require(frame_source.get('frame_index_zero_based') == context.get('frame') and frame_source.get('frame_number_one_based') == context.get('frame') + 1, 'Request/source frame mismatch')
    (blobs, texts) = ([], [])
    for message in payload.get('messages', []):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get('type') == 'text':
                texts.append(item['text'])
            elif item.get('type') == 'image_url':
                url = item.get('image_url', {}).get('url', '')
                _require(url.startswith('data:image/jpeg;base64,'), prompt_text('mllm_input_contract_f122e06d78'))
                try:
                    blobs.append(base64.b64decode(url.split(',', 1)[1], validate=True))
                except (ValueError, TypeError) as exc:
                    raise EvidenceContractError('Invalid image encoding') from exc
    rows = context.get('ordered_images', [])
    validate_manifest(rows, context.get('frame'), blobs)
    if context.get('target_contract') != 'legacy':
        prompt_rows = _prompt_manifest('\n'.join(texts))
        _require(len(prompt_rows) == len(rows), 'Prompt/image count mismatch')
        for (prompt, row) in zip(prompt_rows, rows):
            _require(all((prompt.get(k) == row.get(k) for k in ('image_id', 'role', 'frame', 'box_drawn'))), 'Actual prompt image identity/role/frame mismatch')
            if 'frame' in row:
                _require(prompt.get('frame_number_one_based') == row['frame'] + 1, 'Prompt frame numbering mismatch')
            if row['role'] == 'ANCHOR_TARGET':
                _require(prompt.get('target_xywh_pixels') == row.get('bbox_pixels_in_display') and prompt.get('image_hw') == row.get('encoded_hw'), 'Prompt target coordinates mismatch')
            if 'tiles' in row:
                expected = [{k: t[k] for k in ('index', 'frame', 'candidate_id') if k in t} for t in row['tiles']]
                _require(prompt.get('tiles_left_to_right_then_top_to_bottom') == expected, 'Prompt tile identity mismatch')
    return blobs
