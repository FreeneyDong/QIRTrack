from lib.models.qirtrack.core.vlm.conflict_fused_controller_v3 import ConflictFusedControllerV3

class Controller(ConflictFusedControllerV3):
    def _atom_shadow(self, *args, **kwargs):
        return {"status": "disabled_no_atom"}
