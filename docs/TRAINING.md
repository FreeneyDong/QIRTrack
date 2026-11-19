# TRAINING

The release contains BQP model, loss and data-contract code, not the server-specific collection/scheduling fleet. The old checkpoint used 2,000 raw/U1 updates then 64,000 paired S9 updates, with raw/U1 frozen during pairing; its 430 LaSOT identities had no independent validation split. Loss: 50-bin absolute IoU supervision plus 10*SmoothL1(Qc-Qr, true gain). Never silently initialise OSTrack/SUTrack from this AR checkpoint. New full-training/replay CLIs are not claimed qualified by this extraction.

See [executable commands](QUICKSTART.md) for local path configuration, legacy paired fine-tuning and offline predictor evaluation. These are not full video benchmark launchers.
