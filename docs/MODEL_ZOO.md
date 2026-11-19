# MODEL ZOO

Only checkpoints/old_s9_b256_seq_got_diagnostic_64k.pt is included locally, ignored by Git. SHA256 and training scope are in checkpoints/manifest.json. It is the legacy ARTrackV2-Base256-seq-GOT S9, NOT final model-test weights. No OSTrack/SUTrack predictor or native tracker weights are bundled. No invented public download links. If later distributed, use release assets/model hosting and preserve this qualification.

## Native weight identities accepted by the online adapters

| Adapter | Native profile | SHA256 |
| --- | --- | --- |
| ARTrackV2 | Base256-seq-GOT | `6c2654caa850fce22ccde730e1040b72becb4364055cd044dea22322f677c604` |
| OSTrack | ViT-B/384 CE | `8e6de3c6f10cbc21eaf1c34532358e23618f78a1a2bb9f0d9a2e55adc7af4894` |
| SUTrack | L384 | `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4` |

Obtain native weights through their upstream repositories subject to their
terms. These hashes identify the integration profiles, not additional bundled
weights. The old AR S9 cannot be loaded into OSTrack or SUTrack. Matching native
response predictors used for internal qualification are not released here.
