# Integration status — 2026-11-18

## Verified on the existing Linux/CUDA environment

- All three native tracker adapters: 12 input frames each, real native model +
  corresponding S9 + CPU CLIP + fused history/controller, valid completion and
  strictly ordered output rows. These easy frames issued no MLLM requests.
- All three adapters: separate 3-frame live-service checks, **5 actual 4B HTTP
  requests each**, request archival, successful completion. For these interface
  checks only, periodic audit was set to every frame; AR Q was .79. This is not
  a scientific ablation or a claimed benchmark configuration.
- The live AR check accepted one global-trial replacement and applied guarded
  bbox feedback. The OS/SU checks exercised invalid-output handling and
  postcheck rejection, retaining the native fallback; model response quality
  is not claimed perfect.
- One independent trial per tracker: predicted box and candidate Q computed;
  native fallback state restored exactly under the snapshot contract.
- Synthetic accepted-box test: only bbox state changed; recursive appearance,
  coordinate history and native template fields were preserved. This verifies
  state isolation, not the accuracy of an acceptance decision.
- Earlier legacy paired CLI: one real CUDA training update and four evaluation
  pairs; original checkpoint strict loading. Full retraining was not repeated.

Runtime: Python 3.9 / Torch CUDA 11.3 environment for trackers; a separate
transformers-compatible CPU environment for CLIP; existing 4B multimodal
endpoint. Unit tests and syntax/preflight status are recorded in
`VALIDATION_SUMMARY.json`. No new package installation was forced on shared
research environments. A clean-machine environment matrix was not tested.

## Explicit limits

- No full benchmark rerun or cross-hardware bitwise-equivalence claim.
- Short shared-server timings are not benchmark FPS.
- Live tests used private author prompts, **not bundled** public prompts.
- OSTrack/SUTrack used their own qualified predictors, **not bundled** weights.
- The only locally bundled S9 is the old AR diagnostic model, not final weights.
- Original-code licence selection remains pending; third-party restrictions,
  including nested MAE-derived code, are recorded in THIRD_PARTY.md.

The full ordered-frame command is now implemented: see ONLINE_TRACKING.md.
It is not a video decoder, a dataset downloader, an automatic training split
selector, or an official benchmark submission client.
