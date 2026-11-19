# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from lib.config.paths import configured_path
import cv2
import torch
import sys
import time
import os
from lib.trackers.sutrack.native.test.evaluation import Tracker
import lib.trackers.sutrack.native.test.vot_rgb.vot as vot
from lib.trackers.sutrack.native.test.vot_rgb.vot_utils import *

class SUTrack(object):

    def __init__(self, tracker_name='sutrack', para_name='sutrack_b224'):
        tracker_info = Tracker(tracker_name, para_name, 'vot', None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, img_rgb, selection):
        (x, y, w, h) = selection
        bbox = [x, y, w, h]
        (self.H, self.W, _) = img_rgb.shape
        self.init_info = {'init_bbox': bbox}
        _ = self.tracker.initialize(img_rgb, self.init_info)

    def track(self, img_rgb):
        outputs = self.tracker.track(img_rgb, self.init_info)
        pred_bbox = outputs['target_bbox']
        return pred_bbox

def run_vot_exp(tracker_name, para_name, vis=False):
    torch.set_num_threads(1)
    save_root = os.path.join(configured_path('QIR_PATH_5848F30BCC'), para_name)
    if vis and (not os.path.exists(save_root)):
        os.makedirs(save_root)
    tracker = SUTrack(tracker_name=tracker_name, para_name=para_name)
    handle = vot.VOT('rectangle')
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        'for vis'
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root, seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    tracker.initialize(image, selection)
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        b1 = tracker.track(image)
        handle.report(vot.Rectangle(*b1))
        if vis:
            'Visualization'
            image_ori = image[:, :, ::-1].copy()
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])), (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg', '_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
