"""Two tracker-specific raw/U1 prerequisites on the frozen shared GOT scope.

Reads genuine pooled K16 observations and native unwindowed response maps.
No fake AR coordinate distributions, no repeated pseudo spatial tokens, no
test-supervised initialisation, no tracker or MLLM inference.
"""
from lib.config.paths import configured_path
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import numpy as np
import torch
from lib.models.qirtrack.native.native_response_bqp_v1 import NativeResponseU1, checked_history_responses
from lib.models.qirtrack.base_raw_model import IoUDistributionLoss, absolute_metrics
DATA = Path(configured_path('QIR_PATH_9C2792C454'))
SCOPE = DATA / 'native_tracker_got_fast_scope_v1_20261101'
ROOT = DATA / 'native_tracker_got_raw_budget_v2_20261101'
DEADLINE = datetime.fromisoformat('2026-11-06T21:00:00+09:00')
GOT_RAW = DATA / 'predictor_training/artrackv2_base256_seq/got10k_train/raw_u1_v1_20261101'
SOURCE = {'ostrack': DATA / 'ostrack_384_predictor_data_v1_20261027/main_train', 'sutrack': DATA / 'sutrack_l384_predictor_data_v1_20261027/main_train'}
WEIGHTS = {'ostrack': '8e6de3c6f10cbc21eaf1c34532358e23618f78a1a2bb9f0d9a2e55adc7af4894', 'sutrack': '2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4'}
KEYS = ['template_tokens', 'search_tokens', 'coord_logits', 'tracker_score', 'pred_bbox', 'history_mask']

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def save(p, j):
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), p
    with p.open('x') as f:
        json.dump(j, f, indent=2, allow_nan=False)

def event(kind, **j):
    row = dict(utc=time.strftime('%FT%TZ', time.gmtime()), tracker=kind, **j)
    with (ROOT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def scope(kind):
    return [json.loads(x) for x in (SCOPE / (kind + '_scope.jsonl')).read_text().splitlines() if x.strip()]

def sequence_inputs(kind, r):
    name = r['sequence']
    root = SOURCE[kind]
    rp = Path(r['source_record']['receipt'])
    assert sha(rp) == r['source_record']['receipt_sha256']
    receipt = json.loads(rp.read_text())
    assert receipt['complete'] and receipt['domain'] == 'got10k' and (receipt['sequence'] == name)
    assert receipt['checkpoint_sha256'] == WEIGHTS[kind]
    lp = root / 'labels/got10k' / (name + '.json')
    assert sha(lp) == receipt['files_sha256'][str(lp)]
    labels = {z['frame_index']: z for z in json.loads(lp.read_text())['rows']}
    responses = {}
    native = root / 'native/got10k' / name
    for p in sorted(native.glob('chunk_*.npz')):
        assert sha(p) == receipt['files_sha256'][str(p)]
        with np.load(p, allow_pickle=False) as f:
            ids = f['frame_index']
            maps = f['score_map']
            assert len(ids) == len(maps)
            for (t, response) in zip(ids, maps):
                t = int(t)
                assert t not in responses
                responses[t] = np.array(response, copy=True)
    assert sorted(responses) == list(range(receipt['first_tracking_index'], receipt['last_tracking_index'] + 1))
    records = []
    masked = 0
    for t in receipt['sampled_frame_indices']:
        label = labels[t]
        if not label['valid'] or label['true_iou'] is None:
            masked += 1
            continue
        y = float(label['true_iou'])
        assert np.isfinite(y) and 0 <= y <= 1
        p = root / 'inputs/got10k' / name / (str(t) + '_raw.npz')
        assert sha(p) == receipt['files_sha256'][str(p)]
        with np.load(p, allow_pickle=False) as f:
            indices = f['history_frame_index']
            valid = f['history_observation_valid']
            assert int(indices[-1]) == t and bool(valid[-1])
            evidence = f['history_native_evidence'].astype(np.float32)
            x = dict(template_tokens=torch.from_numpy(f['history_template_pool'].astype(np.float32)).unsqueeze(1), search_tokens=torch.from_numpy(f['history_roi_pool'].astype(np.float32)).unsqueeze(1), coord_logits=checked_history_responses(indices, valid, responses, f['score_map']), tracker_score=torch.from_numpy(evidence), pred_bbox=torch.from_numpy(f['history_boxes_xywh'].astype(np.float32)), history_mask=torch.from_numpy(valid.copy()))
        assert x['template_tokens'].shape == x['search_tokens'].shape == (16, 1, 768)
        assert x['coord_logits'].shape == (16, 1, 24, 24)
        assert all((torch.isfinite(v).all() for v in x.values() if v.is_floating_point()))
        records.append((t, x, y))
    return (records, masked)

def preflight():
    ROOT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result = []
    for kind in SOURCE:
        r = next((x for x in scope(kind) if x['split'] == 'fit'))
        (data, masked) = sequence_inputs(kind, r)
        assert data
        (t, x, y) = data[-1]
        model = NativeResponseU1(kind).train()
        loss = IoUDistributionLoss()(model({k: v.unsqueeze(0) for (k, v) in x.items()})['iou_logits'], torch.tensor([y]))['loss']
        loss.backward()
        assert torch.isfinite(loss)
        assert all((p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        result.append(dict(tracker=kind, sequence=r['sequence'], frame=t, loss=float(loss.detach()), receipt_sha256=r['source_record']['receipt_sha256'], masked=masked))
    save(ROOT / 'ACTUAL_RAW_CPU_PREFLIGHT.json', dict(passed=True, cpu_only=True, entries=result, temporary_model_weights_discarded=True, full_BQP_not_yet_qualified=True, pooled_history_tokens_per_frame=1, real_dense_response_history=True, script_sha256=sha(__file__)))
    save(ROOT / 'PROTOCOL.json', dict(training_role='got10k_train', scope=str(SCOPE), source_scope_sha256=sha(SCOPE / 'PROTOCOL.json'), script_sha256=sha(__file__), raw_updates_per_tracker=4000, seed=20261101, batch_sequences=4, frames_per_sequence=4, AdamW=dict(lr=0.0003, weight_decay=0.0001), gradient_clip=1.0, TF32=False, history='causal K16: real pooled ROI/template 1 token each, native response maps, three native scalars, xywh', missing_true_history='error; padding only masked', initialization='fresh per tracker', test_trained_weights_used=False, tracker_training=False, mllm_requests=0, candidate_stage_training=False, max_cached_bytes=24 * 1024 ** 3, memory_reserve_bytes=96 * 1024 ** 3, snapshots_and_BQP_actual_trial_qualification_pending=True))
    print(json.dumps(dict(passed=True, preflight=result)), flush=True)

def wait_gpu(kind):
    announced = False
    stable = {2: 0, 3: 0}
    while True:
        if datetime.now(timezone.utc) >= DEADLINE:
            event(kind, event='training_deadline_reached_no_GPU_updates_started')
            return None
        if not (GOT_RAW / 'TRAINING_COMPLETE.json').exists():
            if not announced:
                event(kind, event='waiting_GOT_raw_completion_before_GPU_updates')
                announced = True
            time.sleep(30)
            continue
        assert json.loads((GOT_RAW / 'TRAINING_COMPLETE.json').read_text())['complete']
        table = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        for line in table.splitlines():
            (gpu, free, util) = map(lambda x: int(x.strip()), line.split(','))
            if gpu not in stable:
                continue
            stable[gpu] = stable[gpu] + 1 if free >= 12288 and util <= 85 else 0
            if stable[gpu] >= 2:
                return gpu
        time.sleep(30)

def main(kind):
    assert json.loads((ROOT / 'ACTUAL_RAW_CPU_PREFLIGHT.json').read_text())['passed']
    assert json.loads((ROOT / 'PROTOCOL.json').read_text())['script_sha256'] == sha(__file__)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert kind in SOURCE
    for kind in [kind]:
        dest = ROOT / kind
        dest.mkdir(exist_ok=False)
        fit = []
        selection = []
        masked = 0
        resident = 0
        for (i, r) in enumerate(scope(kind), 1):
            if datetime.now(timezone.utc) >= DEADLINE:
                save(dest / 'DEADLINE_REACHED.json', dict(complete=False, stage='input_preparation', prepared=i - 1))
                return
            (rows, invalid) = sequence_inputs(kind, r)
            masked += invalid
            if not rows:
                continue
            x = {k: torch.stack([z[1][k] for z in rows]) for k in KEYS}
            resident += sum((v.numel() * v.element_size() for v in x.values()))
            assert resident <= 24 * 1024 ** 3
            available = next((int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
            assert available >= 96 * 1024 ** 3, 'Memory reserve; preserve state, do not allocate GPU'
            g = dict(sequence=r['sequence'], frames=[z[0] for z in rows], x=x, y=torch.tensor([z[2] for z in rows]))
            (fit if r['split'] == 'fit' else selection).append(g)
            if i % 100 == 0:
                event(kind, event='inputs_prepared', identities=i, resident_bytes=resident)
        assert fit and selection
        save(dest / 'DATA_READY.json', dict(complete=True, fit_sequences=len(fit), selection_sequences=len(selection), fit_frames=sum((len(g['frames']) for g in fit)), selection_frames=sum((len(g['frames']) for g in selection)), masked=masked, resident_bytes=resident, raw_stage_only=True))
        gpu = wait_gpu(kind)
        if gpu is None:
            save(dest / 'DEADLINE_REACHED.json', dict(complete=False, stage='GPU_wait'))
            return
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        random.seed(20261101)
        np.random.seed(20261101)
        torch.manual_seed(20261101)
        torch.cuda.manual_seed_all(20261101)
        model = NativeResponseU1(kind).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
        objective = IoUDistributionLoss()
        rng = random.Random(20261101)
        order = []
        cursor = 0
        (dest / 'checkpoints').mkdir()

        def checkpoint(step):
            p = dest / 'checkpoints' / ('step_%06d.pt' % step)
            assert not p.exists()
            torch.save(dict(model_type='NativeResponseU1', tracker_kind=kind, training_role='got10k_train', tracker_checkpoint_sha256=WEIGHTS[kind], model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, seed=20261101, protocol_sha256=sha(ROOT / 'PROTOCOL.json'), full_S9=False, history_tokens_per_frame=1, test_trained_diagnostic=False, rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())), p)
        checkpoint(0)
        start = time.monotonic()
        with (dest / 'steps.jsonl').open('x') as log, (dest / 'draw_schedule.jsonl').open('x') as draws:
            for step in range(1, 4001):
                if datetime.now(timezone.utc) >= DEADLINE:
                    if step - 1 not in [0, 1000, 2000, 4000]:
                        checkpoint(step - 1)
                    save(dest / 'DEADLINE_REACHED.json', dict(complete=False, stage='training', updates=step - 1))
                    return
                picked = []
                while len(picked) < 4:
                    if cursor >= len(order):
                        order = rng.sample(list(range(len(fit))), len(fit))
                        cursor = 0
                    k = order[cursor]
                    cursor += 1
                    if k not in picked:
                        picked.append(k)
                samples = [(k, rng.randrange(len(fit[k]['y']))) for k in picked for _ in range(4)]
                x = {key: torch.stack([fit[k]['x'][key][j] for (k, j) in samples]).cuda() for key in KEYS}
                y = torch.stack([fit[k]['y'][j] for (k, j) in samples]).cuda()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(x)['iou_logits'], y)['loss']
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert all((p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                optimizer.step()
                log.write(json.dumps(dict(step=step, loss=float(loss.detach()), seconds=time.monotonic() - start)) + '\n')
                draws.write(json.dumps(dict(step=step, frames=[(fit[k]['sequence'], fit[k]['frames'][j]) for (k, j) in samples])) + '\n')
                if step % 100 == 0:
                    log.flush()
                    draws.flush()
                    event(kind, event='raw_training', step=step)
                if step in [1000, 2000, 4000]:
                    checkpoint(step)
        model.eval()
        pred = []
        with torch.no_grad():
            for g in selection:
                for j in range(0, len(g['y']), 8):
                    x = {k: v[j:j + 8].cuda() for (k, v) in g['x'].items()}
                    q = model(x)['predicted_iou'].reshape(-1).cpu().numpy()
                    pred.extend((dict(sequence=g['sequence'], frame=g['frames'][j + n], predicted_iou=float(v), true_iou=float(g['y'][j + n])) for (n, v) in enumerate(q)))
        with (dest / 'selection_predictions.jsonl').open('x') as f:
            for r in pred:
                f.write(json.dumps(r) + '\n')
        save(dest / 'RAW_TRAINING_COMPLETE.json', dict(complete=True, updates=4000, tracker=kind, checkpoint=str(dest / 'checkpoints/step_004000.pt'), checkpoint_sha256=sha(dest / 'checkpoints/step_004000.pt'), selection_metrics=absolute_metrics([r['predicted_iou'] for r in pred], [r['true_iou'] for r in pred]), full_S9_complete=False, actual_trial_qualification_still_required=True))
        del model, optimizer, fit, selection
        torch.cuda.empty_cache()
        event(kind, event='raw_stage_complete')
