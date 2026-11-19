import math
class Capture:

    def __init__(self, tracker, a):
        self.t = tracker
        self.a = a
        self.current = {}
        self.full = True
        self.bb = tracker.network.backbone.forward
        self.net = tracker.network.forward
        tracker.network.backbone.forward = self.backbone
        tracker.network.forward = self.forward

    def backbone(self, *args, **kwargs):
        out = self.bb(*args, **kwargs)
        t = self.t
        s = self.a['setup']
        search = t.network.backbone.norm(out[3]).detach()
        self.current = dict(template_tokens=s.pool16(out[1].detach()).float().cpu(), search_tokens=s.pool16(search).float().cpu(), dynamic_tokens=s.pool16(out[2].detach()).float().cpu())
        if self.full:
            self.current['search_tokens_full'] = search.float().cpu()[0]
        return out

    def forward(self, *args, **kwargs):
        out = self.net(*args, **kwargs)
        torch = self.a['torch']
        logits = out['feat'][:4, 0, :800].detach().float().cpu()
        prob = torch.softmax(logits, -1)
        h = -(prob * prob.clamp_min(1e-12).log()).sum(-1) / math.log(800)
        self.current.update(coord_logits=logits, tracker_score=out['score'].detach().float().cpu().reshape(-1)[:1], coord_entropy_normalized=h, entropy=h.mean().item(), argmax=logits.argmax(-1).numpy())
        return out

    def step(self, image):
        self.current = {}
        box = self.a['np'].asarray(self.t.track(image)['target_bbox'], dtype='float32')
        return (box, dict(self.current))

def recent_source(native, t, pol, a):
    np = a['np']
    ids = np.arange(max(1, t - pol['maximum_age']), t - pol['minimum_age'] + 1, dtype=int)
    if not len(ids):
        return None
    valid = (native['tracker_score'][ids - 1, 0] >= pol['score_threshold']) & (native['coord_entropy_normalized'][ids - 1].mean(-1) <= pol['entropy_threshold'])
    return int(ids[valid][-1]) if valid.any() else None

def recent_features(tracker, cap, image, previous, before, files, native, t, pol, a):
    (torch, np) = (a['torch'], a['np'])
    source = recent_source(native, t, pol, a)
    if source is None:
        return (np.zeros((16, 768), np.float32), 0.0, False, -1.0, dict(source_frame=None, trusted=False))
    assert 0 < source < t
    old = a['base'].image(files[source])
    box = native['pred_bbox'][source - 1]
    (patch, _, mask) = a['sample_target'](old, box, tracker.params.template_factor, output_sz=tracker.params.template_size)
    recent = tracker.preprocessor.process(patch, mask).tensors
    (patch, resize, mask) = a['sample_target'](image, previous, tracker.params.search_factor, output_sz=tracker.params.search_size)
    search = tracker.preprocessor.process(patch, mask).tensors
    coords = []
    for box_old in before.values['store_result']:
        b = a['transform_image_to_crop'](torch.tensor(box_old), torch.tensor(previous), resize, torch.tensor([256, 256]), normalize=True)
        b[2:] += b[:2].clone()
        coords.append((b.clamp(-0.5, 1.5) + 0.5) * (tracker.bins - 1))
    seq = torch.cat(coords).unsqueeze(0)
    with torch.no_grad():
        tracker.network.forward(template=torch.stack([tracker.z_dict1.tensors, recent], dim=1), dz_feat=tracker.network.backbone.patch_embed(recent), search=search, seq_input=seq, stage='sequence', ce_template_mask=tracker.box_mask_z)
    current = cap.current
    similarity = torch.nn.functional.cosine_similarity(current['template_tokens'], current['dynamic_tokens'], dim=-1).mean().item()
    return (current['dynamic_tokens'].numpy(), float(t - source), True, float(similarity), dict(source_frame=source, source_box=box.tolist(), source_image=str(files[source]), trusted=False, policy='visual-preselection, not MLLM admission'))

def inputs_for(tracker, cap, native, t, image, previous, box, current, before, files, pol, a):
    (np, torch) = (a['np'], a['torch'])
    x = a['make_sample'](native, t)
    x = {k: v.float() if v.is_floating_point() else v for (k, v) in x.items()}
    for key in ['template_tokens', 'search_tokens', 'coord_logits', 'tracker_score']:
        x[key][-1] = current[key]
    x['pred_bbox'][-1] = torch.from_numpy(box.copy())
    (_, resize, invalid) = a['sample_target'](image, previous, tracker.params.search_factor, output_sz=tracker.params.search_size)
    geo = a['compute_search_crop_metadata'](image.shape, previous, tracker.params.search_factor, 256, 16, 4)
    a['validate_resize_factor'](geo, resize)
    (recent, age, valid, sim, proof) = recent_features(tracker, cap, image, previous, before, files, native, t, pol, a)
    x.update(search_tokens_full=current['search_tokens_full'], search_valid_fraction_full=torch.from_numpy(a['pooled_valid_fraction'](invalid, 16)), candidate_search_xywh=torch.from_numpy(a['xywh_image_to_search_normalized'](box, geo['search_crop_xywh_image'])[None]), candidate_geometry=torch.from_numpy(a['candidate_relative_geometry'](box[None], box)), recent_template_tokens=torch.from_numpy(recent), recent_template_age=torch.tensor(age), recent_template_valid=torch.tensor(valid), recent_template_similarity=torch.tensor(sim))
    return (x, {k: np.asarray(v).tolist() for (k, v) in geo.items()}, proof)
