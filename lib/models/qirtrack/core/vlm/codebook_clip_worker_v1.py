"""One persistent CPU encoder; separate established dependency environment."""
import sys,json,base64,io
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
import os
root=os.environ['QIRTRACK_CLIP_PATH']
torch.set_num_threads(2)
p=CLIPProcessor.from_pretrained(root,local_files_only=True)
m=CLIPModel.from_pretrained(root,local_files_only=True).eval().requires_grad_(False)
print(json.dumps({'ready':True,'root':root}),flush=True)
for line in sys.stdin:
    row=json.loads(line)
    if row.get('close'):break
    image=Image.open(io.BytesIO(base64.b64decode(row['png']))).convert('RGB')
    x=p(images=image,return_tensors='pt')
    # Explicit projection keeps the historical CLIP feature contract across API versions.
    with torch.inference_mode():v=m.visual_projection(m.vision_model(pixel_values=x['pixel_values']).pooler_output)[0].float().numpy()
    v=v/np.linalg.norm(v)
    if v.shape!=(512,) or not np.isfinite(v).all():raise ValueError('Invalid embedding')
    print(json.dumps({'id':row['id'],'vector':v.tolist()}),flush=True)
