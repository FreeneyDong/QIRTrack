# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

from lib.config.paths import configured_path
import os
import argparse
parser = argparse.ArgumentParser(description='generate namelist')
args = parser.parse_args()
base_dir = configured_path('QIR_PATH_F432B53345')
result_file = configured_path('QIR_PATH_FE0CA89AA6')
assert base_dir.endswith('/')
fnames = []
for (root, dirs, files) in os.walk(base_dir):
    for file in files:
        path = os.path.join(root, file)
        if not os.path.exists(path):
            print("%s doesn't exist.")
        rela_path = path.replace(base_dir, '')
        fnames.append(rela_path)
with open(result_file, 'w') as fout:
    for name in fnames:
        fout.write(name + '\n')
