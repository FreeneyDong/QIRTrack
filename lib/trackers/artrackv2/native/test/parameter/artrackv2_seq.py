# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

from lib.trackers.artrackv2.native.test.utils import TrackerParams
import os
from lib.trackers.artrackv2.native.test.evaluation.environment import env_settings
from lib.trackers.artrackv2.native.config.artrackv2_seq.config import cfg, update_config_from_file

def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    yaml_file = os.path.join(prj_dir, 'experiments/artrackv2_seq/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print('test config: ', cfg)
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = os.path.join(save_dir, 'checkpoints/train/artrackv2_seq/%s/ARTrackV2Seq_ep%04d.pth.tar' % (yaml_name, cfg.TEST.EPOCH))
    params.save_all_boxes = False
    return params
