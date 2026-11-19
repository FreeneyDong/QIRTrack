# Publishing later (no upload performed by preparation scripts)

Follow the official GitHub instructions: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

1. Create an empty GitHub repository named QIRTrack. Do not initialise a README, licence or gitignore there. A private repository is suitable while integration and licence selection remain unfinished.
2. Authenticate on the machine that will push, using your own SSH key or HTTPS credential manager. Never place a token/password in source files or the remote URL.
3. From this repository, review and run the following yourself (replace YOUR_ACCOUNT):

```bash
git status --short
git add .gitignore README.md LICENSE CITATION.cff requirements.txt environment.yml pyproject.toml assets checkpoints data docs experiments external lib prompts scripts tests tracking
git diff --cached --stat
git diff --cached --check
git diff --cached
git commit -m "Prepare QIRTrack research source release"
git remote add origin git@github.com:YOUR_ACCOUNT/QIRTrack.git
git remote -v
git push -u origin main
```

Inspect staged content before committing: no local configuration, credentials, data, request logs, outputs or final model weights. The legacy checkpoint is ignored; only its documentation/manifest is committed. Upload that checkpoint separately as a release asset if desired, and add its real URL only after upload. Empty prompts are intentional. The LICENSE file does not grant rights to QIRTrack-original code; do not label this an open-source-licensed release while that decision remains pending. If origin already exists, inspect it before any change rather than adding/replacing it blindly.

Publication is not evidence of reproducibility. Paired fine-tuning/offline predictor commands are separate from the qualified ordered-frame workflow. This repository is a source preview that may change; it is not the final implementation described in the paper. The legacy ARTrackV2-Base256-seq-GOT S9 is a diagnostic checkpoint, not the final benchmark model.

Before public release, review THIRD_PARTY.md: nested MAE-derived files carry a non-commercial provenance boundary. Choosing MIT/Apache for original additions does not remove this.
