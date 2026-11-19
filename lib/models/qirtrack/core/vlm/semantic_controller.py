"""Opt-in online codebook controller. Existing W1/controller files stay untouched."""
from lib.models.qirtrack.mllm.prompts import prompt_text
import copy, hashlib, json, math
from pathlib import Path
from collections import deque
from dataclasses import asdict
import cv2, numpy as np
from lib.trackers.artrackv2.native.train.data.processing_utils import sample_target
from .recovery import RecoverySearchController
from .controller import jpeg, shrink
from .artrackv2_vlm_recovery import _draw_bbox
from .evidence import TARGET_CONTRACT, crop_geometry
from .unified_memory import Contract
from .online_codebook import OnlineIdentityCodebook, semantic_label
from .codebook_controller import tight

class SemanticCodebookController(RecoverySearchController):

    def __init__(self, *args, codebook_policy, worker, memory_log, quality_contract, encoder_sha256, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_policy = codebook_policy
        self.worker = worker
        self.memory_log = Path(memory_log)
        self.memory_log.parent.mkdir(parents=True, exist_ok=True)
        self.memory = None
        self._vectors = {}
        self._active_query = None
        self._alarm_this_frame = False
        self._trajectory_refs = set()
        self._used_refs = set()
        self._last_row = None
        self._admission_audit = False
        self._current_image = None
        self.contract = Contract('local-CLIP-ViT-B32', 'tight-visible>=.5-ceil-floor-resize384-CLIP224', encoder_sha256, quality_contract)
        if codebook_policy.memory_enabled:
            crop = tight(args[2], args[3])
            if crop is not None:
                (vector, path) = worker.encode(crop)
                anchor = dict(sequence=self.archive.identity['domain'] + '__' + self.archive.identity['sequence'], frame=0, kind='raw', vector=vector, crop_path=path, crop_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), contract=asdict(self.contract))
                self.memory = OnlineIdentityCodebook(anchor['sequence'], anchor, self.contract, codebook_policy)
        self.log('initialise', available=self.memory is not None, policy=asdict(codebook_policy), trainable_encoder=False, thresholds_validated=False)

    def log(self, event, **kw):
        row = dict(event=event, frame=self._frame, **kw)
        with self.memory_log.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def _encode(self, image, box):
        key = (self._frame, tuple((float(v) for v in box)))
        if key not in self._vectors:
            crop = tight(image, box)
            if crop is None:
                self._vectors[key] = (None, None)
            else:
                (v, path) = self.worker.encode(crop)
                self._vectors[key] = (v, path)
                self.log('encode', bbox_xywh_full=list(map(float, box)), crop_path=path, crop_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), frame_source=copy.deepcopy(getattr(self, 'frame_source', None)))
        return self._vectors[key]

    def _query(self, image, box):
        if self.memory is None:
            return dict(label='unavailable', score=None, matches=[], anchor_similarity=None, frozen=False, source='codebook_unavailable')
        (vector, _) = self._encode(image, box)
        return self.memory.query(self._frame, vector)

    def _item(self, o):
        crop = cv2.imread(o['reference_crop_path'], cv2.IMREAD_COLOR)
        if crop is None:
            raise ValueError('Missing actual stored reference crop')
        if hashlib.sha256(Path(o['reference_crop_path']).read_bytes()).hexdigest() != o['reference_crop_sha256']:
            raise ValueError('Changed historical reference crop')
        return dict(frame=o['frame'], crop=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), box=o['bbox_xywh_full'], q=o['q_raw'], verifier_model=o['verifier_model'], crop_geometry=o['crop_geometry'])

    def _evidence(self, image, previous, current, past_limit=3, include_trusted=False, search_region=None):
        if self.memory is None:
            return super()._evidence(image, previous, current, past_limit, include_trusted, search_region)
        (vector, _) = self._encode(image, current)
        self._active_query = self.memory.query(self._frame, vector)
        refs = [] if self._admission_audit else self.memory.references(self._frame, vector)
        (old_trusted, old_past) = (self.trusted, self.past)
        frozen = self.memory.safe.core.frozen and self.cb_policy.safe_updates
        try:
            self.trusted = deque([self._item(o) for o in refs], maxlen=4)
            self.past = deque([] if self._admission_audit or frozen else [p for p in old_past if p['frame'] not in self.memory.safe.invalid_frames], maxlen=old_past.maxlen)
            limit = 0 if self._admission_audit or frozen else past_limit
            blobs = super()._evidence(image, previous, current, limit, bool(refs), search_region)
        finally:
            (self.trusted, self.past) = (old_trusted, old_past)
        self._used_refs.update((o['frame'] for o in refs))
        self.log('retrieve', stage=self._stage, frames=[o['frame'] for o in refs], capacity=self.cb_policy.history_slots, query=self._active_query, anchor_only=frozen or self._admission_audit)
        return blobs

    def _request(self, method, *args, bypass_budget=False, **kwargs):
        before = self.archive.count
        before_ids = len(self._recent_request_ids)
        result = super()._request(method, *args, bypass_budget=bypass_budget, **kwargs)
        if self.archive.count == before:
            result = dict(result)
            result.pop('evidence_request_id', None)
            result.pop('evidence_request_file', None)
            self._recent_request_ids = self._recent_request_ids[:before_ids]
        return result

    def _score(self, image, previous, raw, stage, past_limit=3, anchor_only=False):
        (old_stage, old_admission) = (self._stage, self._admission_audit)
        self._stage = stage
        self._admission_audit = anchor_only
        try:
            evidence = self._evidence(image, previous, raw, past_limit, stage == 'temporal_verifier')
            return self._request('score', evidence, self.config.variant, self._question(), bypass_budget=self.config.uncapped_low_q and self._q_raw < self.config.q_trigger)
        finally:
            (self._stage, self._admission_audit) = (old_stage, old_admission)

    def _review_support(self, match):
        o = self.memory.safe.core.records.get(match['record'])
        if o is None:
            return dict(status='unavailable')
        if self.budgeted_requests >= self.config.max_mllm_requests:
            return dict(status='budget_exhausted')
        item = self._item(o)
        marked = _draw_bbox(item['crop'], item['crop_geometry']['bbox_pixels_in_display'], color=(255, 255, 0), label='PAST t=' + str(o['frame']))
        blobs = (self._context_blob, self._tight_blob, jpeg(marked))
        entries = [self._context_meta, self._tight_entry, dict(item['crop_geometry'], role='TRUSTED', frame=o['frame'], box_drawn=True, bbox_pixels_full=o['bbox_xywh_full'], meaning=prompt_text('semantic_codebook_controller_v1_9dd88cae36'))]
        manifest = self._register(blobs, entries)
        compact = []
        for im in manifest:
            m = {k: im[k] for k in ('image_id', 'role', 'frame', 'box_drawn', 'meaning') if k in im}
            m['frame_number_one_based'] = im['frame'] + 1
            if im['role'] == 'ANCHOR_TARGET':
                m.update(target_xywh_pixels=im['bbox_pixels_in_display'], image_hw=im['encoded_hw'])
            compact.append(m)
        prompt = TARGET_CONTRACT + '\nORDERED IMAGE ROLES: ' + json.dumps(compact) + '\nTASK: ' + prompt_text('semantic_codebook_controller_v1_a5d24ebe3a')
        context = self.archive.context
        self.archive.context = dict(frame=self._frame, stage='memory_revalidation', method='score', ordered_images=manifest, q_raw=self._q_raw, legacy_prompt=prompt, target_contract='strict-unit-v1', image_policy='anchor-only-past-memory-review', reviewed_frame=o['frame'], reviewed_record=match['record'])
        if hasattr(self, 'sequence_source'):
            self.archive.context.update(sequence_source=copy.deepcopy(self.sequence_source), frame_source=copy.deepcopy(self.frame_source))
        self.requests += 1
        self.budgeted_requests += 1
        try:
            score = self.backend.score(blobs, self.config.variant, prompt)
        finally:
            self.archive.context = context
        if self.archive.last_id is not None:
            self._recent_request_ids.append(self.archive.last_id)
            score = dict(score, evidence_request_id=self.archive.last_id, evidence_request_file=str(self.archive.root / (self.archive.last_id + '.json')))
        self.log('prototype_revalidation', record=match['record'], reviewed_frame=o['frame'], score=score)
        return score

    def _alarm(self, query, score):
        if self.memory is None or not self.cb_policy.safe_updates or self._alarm_this_frame or (query.get('label') != 'high') or (semantic_label(score, self.config.p_high) != 'high_no'):
            return
        self._alarm_this_frame = True
        self.memory.freeze(prompt_text('semantic_codebook_controller_v1_f211379610'))
        matches = []
        seen = set()
        for m in query.get('matches', []):
            if m['frame'] not in seen and m['similarity'] >= self.cb_policy.id_high:
                matches.append(m)
                seen.add(m['frame'])
            if len(matches) >= self.cb_policy.max_reviews_per_alarm:
                break
        bad = []
        proof = []
        for match in matches:
            review = self._review_support(match)
            if semantic_label(review, self.config.p_high) == 'high_no' and review.get('evidence_request_id'):
                bad.append(match['frame'])
                proof.append(review['evidence_request_id'])
        if bad:
            self.memory.reject_support(bad, 'Actual anchor-only historical revalidation highNO', proof)
        self.log('conflict_handled', bad_frames=bad, review_ids=proof, updates_frozen=self.memory.safe.core.frozen, contamination_confirmed=False, proof_is_model_judgement_not_ground_truth=True)

    def _local_candidates(self, tracker, previous, raw, image_shape):
        boxes = super()._local_candidates(tracker, previous, raw, image_shape)
        if self.memory is None or not self.cb_policy.rank_enabled:
            return boxes
        rows = []
        for (i, b) in enumerate(boxes):
            (v, _) = self._encode(self._current_image, b)
            rows.append(dict(id='raw' if i == 0 else str(i), vector=v))
        ranked = self.memory.rank(self._frame, rows)
        self._local_boxes = [boxes[i] for i in ranked['selected_positions']]
        self.log('pretrial_candidate_rank', **ranked, original_boxes=boxes, retained_boxes=self._local_boxes)
        return self._local_boxes

    def _postcheck(self, *args, **kwargs):
        accepted = super()._postcheck(*args, **kwargs)
        row = args[4] if len(args) > 4 else kwargs['row']
        if self.memory is not None:
            query = copy.deepcopy(self._active_query or {})
            row['candidate_attempts'][-1]['codebook'] = query
            self._alarm(query, row.get('postcheck', {}))
        return accepted

    def _route(self, row, image, previous, raw, tracker, score):
        bypass = self.config.uncapped_low_q and self._q_raw < self.config.q_trigger
        branch = self._classify(score)
        if branch == 'uncertain':
            second = self._score(image, previous, raw, 'temporal_verifier', self.config.max_recent)
            row['temporal_verifier'] = second
            branch = self._classify(second)
            self._alarm(self._raw_query, second)
            if branch == 'uncertain':
                self._recover(image, previous, raw, row, temporal=True, bypass_budget=bypass)
                return
        if branch == 'high_yes':
            row['atom_shadow'] = self._atom_shadow(image, raw)
            row['action'] = 'high_yes_atom_shadow_no_op'
            return
        candidates = self._local_candidates(tracker, previous, raw, image.shape)
        row['local_candidates'] = candidates
        if len(candidates) > 1:
            gallery = self._candidate_gallery(image, candidates)
            prompt = prompt_text('semantic_codebook_controller_v1_321420c155')
            selection = self._request('choose_candidate', (jpeg(shrink(self.template, 192)), jpeg(shrink(image, 640)), gallery), len(candidates), prompt, bypass_budget=bypass)
            row['local_selection'] = selection
            selected = selection.get('selected_index')
            if type(selected) is int and 0 < selected < len(candidates):
                if self._postcheck(image, previous, raw, candidates[selected], row, 'local_marginal', bypass_budget=bypass):
                    return
            elif selected == 0:
                row['action'] = 'local_selected_raw_no_op'
                return
        self._recover(image, previous, raw, row, bypass_budget=bypass)

    def _admission(self, row, image, raw):
        score = row.get('periodic_audit', {})
        if self.memory is None or not score.get('evidence_request_id'):
            return
        if semantic_label(score, self.config.p_high) != 'high_yes':
            return
        (vector, path) = self._encode(image, raw)
        if vector is None:
            return
        (crop, _, _) = sample_target(image, raw, 2.0, output_sz=160)
        blob = jpeg(crop)
        folder = self.memory_log.parent / 'reference_crops'
        folder.mkdir(exist_ok=True)
        rel = folder / ('frame_%08d.jpg' % self._frame)
        with rel.open('xb') as f:
            f.write(blob)
        obs = dict(sequence=self.memory.safe.core.sequence, frame=self._frame, kind='raw', vector=vector, crop_path=path, crop_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), contract=asdict(self.contract), reference_crop_path=str(rel), reference_crop_sha256=hashlib.sha256(blob).hexdigest(), bbox_xywh_full=list(map(float, raw)), q_raw=self._q_raw, verifier_model=self.backend.model, crop_geometry=crop_geometry(raw, raw, 2.0, 160, image.shape))
        request = json.loads(Path(score['evidence_request_file']).read_text())
        if request.get('status') != 'response' or request.get('frame') != self._frame or request.get('stage') != 'periodic_audit':
            raise ValueError('Admission response provenance mismatch')
        manifest = request['ordered_images']
        actual_aux = [r['frame'] for r in manifest if r['role'] == 'RECENT_RAW']
        refs = [r['frame'] for r in manifest if r['role'] == 'TRUSTED']
        e = dict(source='actual_mllm', score=score, request_id=score['evidence_request_id'], q_raw=self._q_raw, stage='periodic_audit', trusted_reference_frames=refs, auxiliary_raw_frames=sorted(set(actual_aux) | self._trajectory_refs), actual_auxiliary_image_frames=actual_aux, trajectory_dependency_frames=sorted(self._trajectory_refs))
        row['codebook_admission'] = self.memory.admit(obs, e)
        if row['codebook_admission'].get('trajectory_evidence_epoch_reset'):
            self._trajectory_refs.clear()

    def step(self, frame, image, previous, raw, q_raw, tracker):
        if type(frame) is not int or frame <= 0 or (not math.isfinite(q_raw)):
            raise ValueError('Invalid online frame/quality')
        (self._frame, self._q_raw, self._current_shape) = (frame, float(q_raw), image.shape)
        self._current_image = image
        self._vectors = {}
        self._used_refs = set()
        self._alarm_this_frame = False
        self._manifests.clear()
        self._evidence_stages.clear()
        self._recent_request_ids = []
        self._stage = None
        self._raw_query = self._query(image, raw)
        row = dict(frame=frame, q_raw=q_raw, raw_box=list(raw), final_box=list(raw), candidate_box=None, triggered=False, action='no_op', variant=self.config.variant, requests_before=self.requests)
        periodic = q_raw >= self.config.q_trusted and frame % self.config.shadow_every == 0
        audit = None
        if periodic:
            audit = self._score(image, previous, raw, 'periodic_audit', anchor_only=self.cb_policy.safe_updates)
            row['periodic_audit'] = audit
            self._alarm(self._raw_query, audit)
        low_q = q_raw < self.config.q_trigger
        memory_conflict = self.cb_policy.route_enabled and self._raw_query['label'] == 'low'
        need_route = low_q or memory_conflict or (self.cb_policy.route_enabled and audit is not None and (self._classify(audit) == 'high_no'))
        if need_route:
            if not self.config.uncapped_low_q and (frame - self.last_trigger < self.config.cooldown_frames or self.triggers >= self.config.max_trigger_events):
                row['action'] = 'cooldown_or_trigger_budget'
            else:
                self.triggers += 1
                self.last_trigger = frame
                row['triggered'] = True
                skip = low_q and self.cb_policy.route_enabled and (self._raw_query['label'] == 'high') and (not self._raw_query['frozen']) and (audit is None)
                if skip:
                    row['verifier_skipped'] = dict(source='codebook', similarity=self._raw_query['score'], probability=None, not_an_mllm_response=True)
                    row['atom_shadow'] = self._atom_shadow(image, raw)
                    row['action'] = 'codebook_high_id_atom_shadow_no_op'
                else:
                    score = audit if audit is not None else self._score(image, previous, raw, 'verifier')
                    row['verifier'] = score
                    self._alarm(self._raw_query, score)
                    self._route(row, image, previous, raw, tracker, score)
        self._admission(row, image, raw)
        frozen = bool(self.memory and self.memory.safe.core.frozen)
        row['codebook'] = dict(raw_query=self._raw_query, policy=asdict(self.cb_policy), updates_frozen=frozen, logical_mllm_requests=self.requests - row['requests_before'], writeback_guard=dict(blocked=self.cb_policy.guard_bbox_writeback and self.cb_policy.safe_updates and frozen, reason='unresolved_memory_conflict' if frozen else None))
        row['requests_after'] = self.requests
        row['evidence_request_ids'] = self._recent_request_ids[:]
        self._last_row = row
        self.log('decision', action=row['action'], triggered=row['triggered'], codebook=row['codebook'])
        self._current_image = None
        return row

    def remember(self, frame, image, raw, score, entropy):
        super().remember(frame, image, raw, score, entropy)
        if self._last_row and self._last_row.get('state_writeback', {}).get('applied'):
            self._trajectory_refs.update(self._used_refs)
        if self.memory:
            self._persist_state()

    def _persist_state(self):
        path = self.memory_log.parent / 'memory_state.json'
        tmp = path.with_suffix('.tmp')
        with tmp.open('x') as f:
            json.dump(self.memory.export(), f, allow_nan=False)
        tmp.replace(path)
