"""Real OSTrack/SUTrack candidate transport for the shared BQP head.

No tracker or MLLM inference. Native maps are not AR coordinate logits. Spatial
features retain the existing qint8 reconstruction contract; this is not an
unquantised or bit-exact substitute for a fresh native forward.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
from lib.models.qirtrack.native import training_data as raw_data
from lib.models.qirtrack.native.native_response_bqp_v1 import NativeResponseU1, validate_weight_role
from lib.models.qirtrack.core.localisation.searchmap_candidate import PrPoolSearchMapCandidateIoU

class NativeCachedPastS9(PrPoolSearchMapCandidateIoU):
    """Same BQP parameter keys; actual full current template + pooled past."""

    def _encode_strict_past(self, batch):
        if '_frozen_strict_past' in batch:
            return batch['_frozen_strict_past']
        return super()._encode_strict_past(batch)

    def forward(self, batch):
        if 'current_template_tokens' in batch:
            batch = dict(batch, template_tokens=batch['current_template_tokens'][:, None])
        return super().forward(batch)

def file_sha(path):
    return raw_data.sha(path)

def current_npz(path, receipt):
    path = Path(path)
    assert file_sha(path) == receipt['files_sha256'][str(path)]
    with np.load(path, allow_pickle=False) as f:
        x = {k: f[k].copy() for k in f.files}
    for key in ('search', 'template'):
        q = x.pop(key + '_tokens_qint8')
        scale = x.pop(key + '_tokens_scale')
        assert q.dtype == np.int8 and scale.shape == (768,) and np.isfinite(scale).all()
        x[key + '_tokens'] = q.astype(np.float32) * scale
    assert x['search_tokens'].shape == (576, 768) and x['template_tokens'].shape in ((144, 768), (288, 768))
    assert all((not np.issubdtype(v.dtype, np.floating) or np.isfinite(v).all() for v in x.values()))
    return x

def recent_from_raw(raw, frame, policy):
    required = {'minimum_native_score', 'maximum_spatial_entropy', 'minimum_age', 'maximum_age'}
    assert set(policy) == required, 'Explicit fit-derived policy required; no hidden online threshold'
    ids = raw['history_frame_index']
    mask = raw['history_observation_valid'] & raw['history_roi_valid']
    evidence = raw['history_native_evidence']
    ages = frame - ids
    valid = mask & (ids < frame) & (ages >= policy['minimum_age']) & (ages <= policy['maximum_age'])
    valid &= (evidence[:, 0] >= policy['minimum_native_score']) & (evidence[:, 2] <= policy['maximum_spatial_entropy'])
    indices = np.flatnonzero(valid)
    if not len(indices):
        return (torch.zeros(1, 768), torch.tensor(0.0), torch.tensor(False), torch.tensor(0.0), dict(source_frame=None, trusted=False))
    index = int(indices[-1])
    tokens = raw['history_roi_pool'][index].astype(np.float32)[None]
    assert 0 <= int(ids[index]) < frame
    first = raw['template_pool'].astype(np.float32)
    v = tokens[0]
    similarity = float(np.dot(v, first) / max(float(np.linalg.norm(v) * np.linalg.norm(first)), 1e-12))
    return (torch.from_numpy(tokens), torch.tensor(float(ages[index])), torch.tensor(True), torch.tensor(similarity), dict(source_frame=int(ids[index]), trusted=False, source='genuine past native target-ROI pooled feature, NOT re-encoded image template', quality_policy='explicit native confidence/entropy only; no GT or test-trained Q'))

def geometry_current(x):
    crop = x['search_crop_actual_xywh'].astype(np.float64)
    origin = x['search_model_origin'].astype(np.float64)
    box = x['current_bbox_xywh'].astype(np.float64)
    assert (crop[2:] > 0).all() and (box[2:] > 0).all()
    normalized = np.r_[(box[:2] - origin) / crop[2:], box[2:] / crop[2:]].astype(np.float32)
    inverse = np.r_[normalized[:2] * crop[2:] + origin, normalized[2:] * crop[2:]]
    assert np.max(abs(inverse - box)) < 0.001
    validity = x['search_valid_fraction'].astype(np.float32)
    removed = x['search_ce_removed'].reshape(24, 24)
    assert ((validity >= 0) & (validity <= 1)).all()
    validity = validity * (~removed).astype(np.float32)
    return (normalized[None], validity)

def build_current(x, raw, frame, past, history_mask, policy):
    (normalized, validity) = geometry_current(x)
    (tokens, age, accepted, similarity, proof) = recent_from_raw(raw, frame, policy)
    result = dict(current_template_tokens=torch.from_numpy(x['template_tokens']), search_tokens_full=torch.from_numpy(x['search_tokens']), search_valid_fraction_full=torch.from_numpy(validity), candidate_search_xywh=torch.from_numpy(normalized), candidate_geometry=torch.zeros(1, 4), recent_template_tokens=tokens, recent_template_age=age, recent_template_valid=accepted, recent_template_similarity=similarity, history_mask=history_mask.clone(), _frozen_strict_past=past.clone())
    assert past.shape == (15, 128) and history_mask.shape == (16,)
    return (result, proof)

def prepare_sequence(kind, record, raw_model, policy):
    assert kind in ('ostrack', 'sutrack') and record['training_role'] == 'got10k_train'
    assert record['domain'] == 'got10k' and record['supervised_test_data_allowed'] is False
    root = raw_data.SOURCE[kind]
    name = record['sequence']
    rp = Path(record['source_record']['receipt'])
    assert file_sha(rp) == record['source_record']['receipt_sha256']
    receipt = json.loads(rp.read_text())
    assert receipt['complete'] and receipt['official_subset'] == 'train'
    assert receipt['checkpoint_sha256'] == raw_data.WEIGHTS[kind]
    lp = root / 'labels/got10k' / (name + '.json')
    assert file_sha(lp) == receipt['files_sha256'][str(lp)]
    labels = json.loads(lp.read_text())
    label_rows = {r['frame_index']: r for r in labels['rows']}
    (history, masked) = raw_data.sequence_inputs(kind, record)
    history = {t: (x, y) for (t, x, y) in history}
    current = {}
    past = {}
    pairs = []
    proofs = []
    for pair in labels['pairs']:
        frame = pair['frame_index']
        truth = label_rows[frame].get('gt_bbox')
        good = truth is not None and np.isfinite(truth).all() and (min(truth[2:]) > 0)
        if not pair['valid'] or not good or frame not in history:
            masked += 1
            continue
        assert np.isfinite([pair['raw_iou'], pair['candidate_iou'], pair['gain']]).all()
        assert abs(pair['gain'] - pair['candidate_iou'] + pair['raw_iou']) < 1e-10
        (raw_x, y) = history[frame]
        assert abs(y - pair['raw_iou']) < 1e-10
        raw = current.get(frame)
        if raw is None:
            raw = current_npz(pair['raw_input'], receipt)
            current[frame] = raw
            mask = raw_x['history_mask'][:-1]
            from lib.models.qirtrack.core.trigger_e2e.model import causal_box_geometry
            geom = causal_box_geometry(raw_x['pred_bbox'][:-1][None], mask[None])
            with torch.no_grad():
                past[frame] = raw_model.frame_encoder(raw_x['template_tokens'][:-1], raw_x['search_tokens'][:-1], raw_x['coord_logits'][:-1], raw_x['tracker_score'][:-1], geom.reshape(15, -1)).cpu()
        cand = current_npz(pair['candidate_input'], receipt)
        for key in ('history_frame_index', 'history_observation_valid', 'history_roi_pool', 'history_template_pool', 'history_native_evidence', 'history_boxes_xywh'):
            assert np.array_equal(raw[key][:-1], cand[key][:-1]), (name, frame, key, 'trial past changed')
        assert int(raw['history_frame_index'][-1]) == frame and int(cand['history_frame_index'][-1]) == frame
        pred_path = Path(pair['prediction_file'])
        assert file_sha(pred_path) == receipt['files_sha256'][str(pred_path)]
        pred = json.loads(pred_path.read_text())
        assert pred['gt_used_for_prediction'] is False and pred['committed'] is False
        assert np.allclose(cand['current_bbox_xywh'], pred['candidate_bbox'], atol=0.002, rtol=0)
        (a, proof) = build_current(raw, raw, frame, past[frame], raw_x['history_mask'], policy)
        (b, other) = build_current(cand, raw, frame, past[frame], raw_x['history_mask'], policy)
        assert proof == other and torch.equal(a['_frozen_strict_past'], b['_frozen_strict_past'])
        pairs.append(dict(sequence=name, frame=frame, origin=pair['source'], raw=a, candidate=b, raw_iou=pair['raw_iou'], candidate_iou=pair['candidate_iou'], true_gain=pair['gain']))
        proofs.append(proof)
    return (pairs, dict(sequence=name, tracker=kind, split=record['split'], valid_pairs=len(pairs), masked=masked, recent_sources=proofs, source_role='got10k_train', real_spatial_maps=True, current_template_tokens=144 if kind == 'ostrack' else 288, qint8_reconstruction=True, new_MLLM_requests=0, new_tracker_forwards=0, training_started=False))

def cpu_preflight():
    torch.set_num_threads(1)
    result = []
    policy = dict(minimum_native_score=0.0, maximum_spatial_entropy=1.0, minimum_age=1, maximum_age=128)
    for kind in ('ostrack', 'sutrack'):
        path = raw_data.ROOT / kind / 'checkpoints/step_004000.pt'
        ck = torch.load(path, map_location='cpu')
        validate_weight_role(ck, kind, 'got10k_train', raw_data.WEIGHTS[kind])
        assert not ck['full_S9']
        model = NativeResponseU1(kind)
        model.load_state_dict(ck['model'], strict=True)
        model.eval()
        for partition in ('fit', 'selection'):
            record = next((r for r in raw_data.scope(kind) if r['split'] == partition))
            (pairs, meta) = prepare_sequence(kind, record, model, policy)
            assert pairs
            assert any((p['source_frame'] is not None for p in meta['recent_sources']))
            assert all((not value.is_floating_point() or torch.isfinite(value).all() for p in pairs for item in (p['raw'], p['candidate']) for value in item.values()))
            result.append(meta)
    out = raw_data.DATA / 'native_tracker_candidate_input_preflight_v1_20261102.json'
    raw_data.save(out, dict(passed=True, results=result, real_cpu_inputs_only=True, fixture_policy=policy, fixture_policy_not_for_training_or_deployment=True, full_S9_training_started=False, real_CUDA_BQP_forward_backward_pending=True, native_equivalence_and_controller_integration_pending=True, script_sha256=file_sha(__file__)))
    print(json.dumps(dict(passed=True, report=str(out), cases=len(result))), flush=True)
