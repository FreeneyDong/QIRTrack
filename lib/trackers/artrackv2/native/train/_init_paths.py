# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import sys

def add_path(path):
    pass
this_dir = osp.dirname(__file__)
prj_path = osp.join(this_dir, '../..')
add_path(prj_path)
