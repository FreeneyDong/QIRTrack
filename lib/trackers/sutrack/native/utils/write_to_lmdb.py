# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

from lib.config.paths import configured_path
import lmdb
import os
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='generate lmdb')
args = parser.parse_args()
base_dir = configured_path('QIR_PATH_F432B53345')
result_file = configured_path('QIR_PATH_FE0CA89AA6')
namelist = [x.strip() for x in open(result_file).readlines()]
print('number:', len(namelist))
if base_dir.endswith('/'):
    lmdb_fname = base_dir[:-1] + '_lmdb'
else:
    lmdb_fname = base_dir + '_lmdb'
env = lmdb.open(lmdb_fname, map_size=1024 ** 4)
txn = env.begin(write=True)
for (i, t) in enumerate(tqdm(namelist)):
    if i % 100000 == 0:
        txn.commit()
        txn = env.begin(write=True)
    with open(os.path.join(base_dir, t), 'rb') as fin:
        txn.put(key=t.encode(), value=fin.read())
txn.commit()
env.close()
