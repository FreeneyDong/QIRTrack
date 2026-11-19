# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
from lib.trackers.sutrack.native.test.vot.sutrack_class import run_vot_exp
run_vot_exp('sutrack', 'sutrack_l384', vis=False, out_conf=True, channel_type='rgbd')
