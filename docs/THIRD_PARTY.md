# Third-party attribution and licence boundaries

Original QIRTrack source code is licensed under [Apache-2.0](../LICENSE).
See [NOTICE](../NOTICE) for the scope of that grant. Third-party materials and
portions adapted from them retain their existing terms. Apache-2.0 does not
override more specific third-party notices or restrictions.

| Component | Upstream root licence | Local licence |
| --- | --- | --- |
| ARTrack / ARTrackV2 | Apache-2.0 | `lib/trackers/artrackv2/LICENSE` |
| OSTrack | MIT | `lib/trackers/ostrack/LICENSE` |
| SUTrack | MIT | `lib/trackers/sutrack/LICENSE.txt` |
| PreciseRoIPooling | MIT | licence under `external/PreciseRoIPooling` |
| OpenAI CLIP implementation | MIT | `external/CLIP/LICENSE` |

Upstream: https://github.com/MIV-XJTU/ARTrack,
https://github.com/botaoye/OSTrack, https://github.com/chenxin-dlut/SUTrack,
https://github.com/vacancy/PreciseRoIPooling, https://github.com/openai/CLIP.
Keep these licences, copyright notices and acknowledgements in redistributions.
The namespaced native Python files carry modification notices for this export.
Later modifications include checkpoint namespace loading and resource paths;
they do not imply authorship of upstream neural networks.

## Nested provenance: do not infer all files are permissively licensed

The native trees include Meta/Facebook MAE-derived position-embedding and
schedule utilities. MAE's original licence is **CC BY-NC 4.0**, retained at
`external/licenses/MAE_CC_BY_NC_4.0.txt`, from
https://github.com/facebookresearch/mae/blob/main/LICENSE.
Relevant files include ARTrackV2 `native/models/mask_decoder/pos_embed.py` and
SUTrack `native/utils/{pos_embed,lr_sched,ema_sched}.py`. Their upstream notices
are retained. Treat these portions as non-commercial unless additional
permission or an applicable alternative licence is verified. Do not advertise
the aggregate tree as unrestricted commercial MIT/Apache software. This is a
conservative provenance boundary, not a resolution of upstream relicensing.

The COCO API-derived `native/train/dataset/COCO_tool.py` copies retain their
BSD copyright notices and `external/licenses/COCO_API_BSD.txt` from
https://github.com/cocodataset/cocoapi/blob/master/license.txt.

Model weights, datasets, paper figures, MLLM services and prompts have separate
rights and terms. The code licence does not authorise redistribution of
checkpoints or images. No native tracker/MLLM weights or dataset files are bundled.
Academic citation and copyright permission are different requirements: cite
the original papers, and also comply with the applicable code licences.
