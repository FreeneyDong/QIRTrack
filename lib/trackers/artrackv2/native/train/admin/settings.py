# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

from lib.trackers.artrackv2.native.train.admin.environment import env_settings

class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""

    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True
