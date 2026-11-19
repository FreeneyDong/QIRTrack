from pathlib import Path
import argparse,ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ap=argparse.ArgumentParser();ap.add_argument('--offline',action='store_true');a=ap.parse_args()
files=list((ROOT/'lib').rglob('*.py'))
for p in files:ast.parse(p.read_text())
verified=[];missing=[]
for row in json.loads((ROOT/'checkpoints/manifest.json').read_text())['models']:
 p=ROOT/'checkpoints'/row['file']
 if not p.is_file():missing.append(row['file']);continue
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Checkpoint SHA256 mismatch: '+row['file'])
 verified.append(row['file'])
if not a.offline:
 from lib.models.qirtrack.mllm.prompts import require_prompts
 require_prompts()
print(json.dumps(dict(syntax_files=len(files),offline=a.offline,checkpoint_hash_verified=bool(verified) and not missing,verified_checkpoints=verified,missing_checkpoints=missing,online_benchmark_tested=False)))
