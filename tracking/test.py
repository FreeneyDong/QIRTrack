"""Run QIRTrack on ordered image frames with only the first target box supplied."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external/CLIP'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--sequence', required=True, help='Directory of ordered JPG/PNG frames')
    parser.add_argument('--init-box', type=float, nargs=4, required=True, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--output', required=True, help='New directory; existing output is never overwritten')
    parser.add_argument('--max-frames', type=int, help='Smoke-test limit; omit for the full sequence')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    kind = config['tracker']
    if kind not in ('artrackv2', 'ostrack', 'sutrack'):
        raise ValueError('Unsupported tracker: ' + kind)
    for key in ('native_checkpoint', 'native_config', 'predictor'):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(key + ': ' + config[key])
    for key, env in [('prompts', 'QIRTRACK_PROMPT_DIR'), ('clip', 'QIRTRACK_CLIP_PATH')]:
        os.environ[env] = str(Path(config[key]).expanduser().resolve())
    if config.get('clip_python'):
        os.environ['QIRTRACK_CLIP_PYTHON'] = config['clip_python']
    from lib.models.qirtrack.mllm.prompts import require_prompts
    require_prompts()  # Fail before CUDA allocation or any HTTP request.

    import cv2
    import numpy as np
    import torch
    from lib.models.qirtrack.core.vlm.mllm_backend import MLLMBackend
    from lib.models.qirtrack.core.vlm.evidence import RequestArchive, RecordedBackend
    from lib.models.qirtrack.core.vlm.controller import FrameworkConfig
    from lib.models.qirtrack.core.vlm.codebook_controller import ClipWorker
    from lib.models.qirtrack.core.vlm.online_codebook import SemanticCodebookPolicy, guarded_bbox_writeback
    from lib.trackers.feedback import apply_accepted_state_writeback

    class CheckedBackend(RecordedBackend):
        def _post(self, payload):
            body, latency, error = super()._post(payload)
            if error:
                raise RuntimeError('MLLM request failed; partial output preserved: ' + str(error))
            return body, latency, error

    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.manual_seed(config.get('seed', 20261027))
    np.random.seed(config.get('seed', 20261027))
    import random
    random.seed(config.get('seed', 20261027))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    files = sorted(
        [p for p in Path(args.sequence).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
        key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)])
    if args.max_frames is not None:
        if args.max_frames < 2:
            raise ValueError('--max-frames must be at least 2')
        files = files[:args.max_frames]
    if len(files) < 2:
        raise ValueError('At least two image frames are required')
    if not np.isfinite(args.init_box).all() or min(args.init_box[2:]) <= 0:
        raise ValueError('Invalid first-frame xywh box')

    def read_image(path):
        value = cv2.imread(str(path))
        if value is None:
            raise ValueError('Unreadable image: ' + str(path))
        return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = worker = None
    initialisation_start = time.monotonic()
    try:
        if kind == 'artrackv2':
            from lib.trackers.artrackv2.observation import Adapter
            from lib.trackers.artrackv2.controller import Controller
            runtime = Adapter(config['domain'], config['native_checkpoint'], config['native_config'],
                              config['predictor'], config['recent_policy'], files, read_image)
        else:
            from lib.trackers.native_runtime import NativeRuntime
            Adapter = importlib.import_module('lib.trackers.' + kind + '.observation').Adapter
            Controller = importlib.import_module('lib.trackers.' + kind + '.controller').NativeResponseController
            runtime = NativeRuntime(Adapter(config['domain'], config['native_checkpoint'], config['native_config']),
                                    kind, config['predictor'])
        first = read_image(files[0])
        runtime.initialize(first, args.init_box)
        archive = RequestArchive(output / 'evidence', config['domain'], Path(args.sequence).name, kind)
        backend = CheckedBackend(MLLMBackend('vllm', config['mllm_model'], config['mllm_url'],
                                             config.get('timeout', 300)), archive)
        worker = ClipWorker(output / 'clip')

        def rerun(seed):
            return runtime.trial(seed, float(getattr(rerun, 'trial_search_factor', 5.)))

        framework = FrameworkConfig(**{
            'variant': 'binary', 'q_trigger': config['q'], 'p_high': config.get('p_high', .7),
            'candidate_acceptance_rule': 'relative_not_high_no', 'uncapped_low_q': True,
            **config.get('framework', {})})
        controller = Controller(backend, framework, first, args.init_box, None, rerun,
            archive=archive, anchor_context_area=1.5, history_arm='countmatched_recent_trusted',
            trial_search_factor=3., codebook_policy=SemanticCodebookPolicy(route_enabled=False, rank_enabled=False),
            worker=worker, memory_log=output / 'memory.jsonl', quality_contract=sha256(config['predictor']),
            encoder_sha256=config['clip_sha256'])
        from dataclasses import asdict
        protocol = dict(config=config, framework=asdict(framework),
            checkpoint_sha256={k: sha256(config[k]) for k in ('native_checkpoint', 'predictor')},
            frames=[p.name for p in files], init_box=args.init_box, ATOM=False,
            feedback='guarded bbox-only; native recursive prompts/templates not replaced',
            noninitial_GT_read=False, cache_replay=False)
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        initialisation_seconds = time.monotonic() - initialisation_start
        start = time.monotonic()
        with (output / 'predictions.txt').open('x') as predictions, (output / 'decisions.jsonl').open('x') as log:
            predictions.write(','.join(map(str, args.init_box)) + '\n')
            for t, path in enumerate(files[1:], 1):
                image = read_image(path)
                previous, raw, quality, score, entropy = runtime.native_step(image, t)
                controller.frame_source = dict(frame_index_zero_based=t, frame_number_one_based=t + 1,
                                               image_file=str(path))
                row = controller.step(t, image, previous, raw, quality, runtime.t)
                runtime.assert_fallback()
                guarded_bbox_writeback(apply_accepted_state_writeback, runtime.t, row, True)
                effective = row['final_box'] if row['state_writeback']['applied'] else raw
                controller.remember(t, image, effective, score, entropy)
                if hasattr(runtime, 'advance'):
                    runtime.advance()
                predictions.write(','.join(map(str, row['final_box'])) + '\n')
                predictions.flush()
                log.write(json.dumps({**row, 'frame': t, 'q_raw': quality}, allow_nan=False) + '\n')
                log.flush()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        result = dict(complete=True, frames=len(files), processed_frames=len(files) - 1,
            logical_calls=controller.requests, http_requests=archive.count,
            initialisation_seconds=initialisation_seconds, online_wall_seconds=elapsed,
            fps_excluding_initialisation=(len(files) - 1) / elapsed)
        (output / 'completion.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps(dict(type=type(exc).__name__, error=str(exc))))
        raise
    finally:
        if worker is not None:
            worker.close()
        if runtime is not None:
            runtime.close()


if __name__ == '__main__':
    main()
