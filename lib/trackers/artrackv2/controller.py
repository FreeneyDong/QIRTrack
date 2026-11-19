from lib.models.qirtrack.core.vlm.quadrant_controller import ConflictFusedControllerV3

class Controller(ConflictFusedControllerV3):
    def _atom_shadow(self, *args, **kwargs):
        return {"status": "disabled_no_atom"}
