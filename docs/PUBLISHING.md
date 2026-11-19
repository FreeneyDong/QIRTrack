# Publishing later (no upload performed by preparation scripts)

Follow the official GitHub instructions: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

1. Create an empty GitHub repository named QIRTrack only if no remote exists. Do not initialise a README, licence or gitignore there. Keep the repository private until the intended public materials and their distribution rights have been reviewed.
2. Authenticate on the machine that will push, using your own SSH key or HTTPS credential manager. Never place a token/password in source files or the remote URL.
3. From this repository, review and run the following yourself (replace YOUR_ACCOUNT):

```bash
git status --short
git add .gitignore README.md LICENSE NOTICE CITATION.cff requirements.txt environment.yml pyproject.toml assets checkpoints data docs experiments external lib prompts scripts tests tracking
git diff --cached --stat
git diff --cached --check
git diff --cached
git commit -m "Prepare QIRTrack research source release"
git remote add origin git@github.com:YOUR_ACCOUNT/QIRTrack.git
git remote -v
git push -u origin main
```

Inspect staged content before committing: no local configuration, credentials, data, request logs, outputs or final model weights. The legacy checkpoint is ignored; only its documentation and manifest are committed. Empty prompts are intentional. Preserve both LICENSE and NOTICE, which identify the Apache-2.0 grant for original QIRTrack source code only. If origin already exists, inspect it before any change rather than adding or replacing it blindly.

Publication is not evidence of reproducibility. Paired fine-tuning/offline predictor commands are separate from the qualified ordered-frame workflow. This repository is a source preview that may change; it is not the final implementation described in the paper. The legacy ARTrackV2-Base256-seq-GOT S9 is a diagnostic checkpoint, not the final benchmark model.

Before public release, review [THIRD_PARTY.md](THIRD_PARTY.md): identified MAE-derived files retain non-commercial terms. The original-code Apache-2.0 grant does not remove these terms or grant rights to hosted checkpoints or paper figures.
