"""Create ignored local paths without modifying source files."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output-root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',default='configs/local.json');a=p.parse_args()
paths={k:str(Path(v).expanduser().resolve()) for k,v in [('data_root',a.data_root),('output_root',a.output_root),('checkpoint',a.checkpoint)]}
if not Path(paths['data_root']).is_dir():p.error('data-root must exist')
if not Path(paths['checkpoint']).is_file():p.error('checkpoint must exist')
dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open('x') as f:json.dump(dict(**paths,path_map={}),f,indent=2)
print(dest.resolve())
