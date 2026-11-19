import numpy as np
def apply_accepted_state_writeback(tracker, row: dict, enabled: bool) -> bool:
    """Write an actually substituted output box into future tracker state."""
    if not enabled:
        row['state_writeback'] = {'applied': False, 'reason': 'disabled'}
        return False
    origin = row.get('candidate_origin')
    expected_action = f'{origin}_accepted_output_only'
    if not origin or row.get('action') != expected_action:
        row['state_writeback'] = {'applied': False, 'reason': 'no_accepted_replacement'}
        return False
    box = np.asarray(row.get('final_box'), dtype=np.float32).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
        raise ValueError('Accepted replacement box is invalid for writeback')
    before = np.asarray(tracker.state, dtype=np.float32).reshape(-1)
    if before.shape != (4,) or not np.isfinite(before).all():
        raise ValueError('Native tracker state is invalid before writeback')
    tracker.state = box.tolist()
    row['state_writeback'] = {'applied': True, 'origin': origin, 'state_before_xywh': before.tolist(), 'state_after_xywh': box.tolist()}
    row['action'] = f'{origin}_accepted_state_writeback'
    return True
