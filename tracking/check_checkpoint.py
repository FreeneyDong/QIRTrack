from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.models.qirtrack.checkpoint import load_legacy_diagnostic
m,c=load_legacy_diagnostic(ROOT/'checkpoints/s9_b256_seq_got_64k.pt')
print(json.dumps(dict(strict_load=True,step=c['step'],parameters=sum(p.numel() for p in m.parameters()),role=c['release_role'],GPU_inference_tested=False)))
