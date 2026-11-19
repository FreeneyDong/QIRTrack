# MODEL ZOO

## QIRTrack predictor checkpoints

| Checkpoint | Compatible native profile | Download |
| --- | --- | --- |
| Legacy S9 (64k paired updates, 7.9 MB) | ARTrackV2-Base256-seq-GOT | [Google Drive (restricted access)](https://drive.google.com/file/d/1JjT9JF8ryTl1RcfIy_sV0NMC0oBHGSXI/view) |

Save the downloaded file as `checkpoints/s9_b256_seq_got_64k.pt`. Git ignores the binary; it does not come with the clone. Sign in to an account with Drive access to download it.

SHA256: `1e85329a3595dcd8b045b7fa159401c8d4fc38720b72c30a3f169e2e3b5f830e`. The [checkpoint manifest](../checkpoints/manifest.json) records the size, tracker profile and provenance.

This legacy S9 was trained with ARTrackV2-Base256-seq-GOT. It is a **diagnostic checkpoint, not the final benchmark model**. No native tracker weights or OSTrack/SUTrack predictor weights are bundled.

## Native weight identities accepted by the online adapters

| Adapter | Native profile | SHA256 |
| --- | --- | --- |
| ARTrackV2 | Base256-seq-GOT | `6c2654caa850fce22ccde730e1040b72becb4364055cd044dea22322f677c604` |
| OSTrack | ViT-B/384 CE | `8e6de3c6f10cbc21eaf1c34532358e23618f78a1a2bb9f0d9a2e55adc7af4894` |
| SUTrack | L384 | `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4` |

Obtain native weights through [ARTrackV2](https://github.com/MIV-XJTU/ARTrack),
[OSTrack](https://github.com/botaoye/OSTrack) or [SUTrack](https://github.com/chenxin-dlut/SUTrack), subject to their
terms. These hashes identify the integration profiles, not additional bundled
weights. The old AR S9 cannot be loaded into OSTrack or SUTrack. Matching native
response predictors used for internal qualification are not released here.
