# RIBR release checklist

This branch is prepared for review in a private repository. It is not approved
for public visibility.

## Completed technical checks

- [x] CPU-only import and synthetic training/test smoke path.
- [x] Existing data-view, derivative, and split defaults preserved.
- [x] Explicit external-evaluation checkpoint argument.
- [x] Portable device selection (`cpu`, `mps`, `cuda`, or `cuda:<index>`).
- [x] Pseudonymous per-sample output by default.
- [x] Runtime dependencies limited to imports used by this branch.
- [x] Repository-owned code and documentation marked CC BY-NC 4.0.
- [x] No data, local paths, sample identifiers, or checkpoints added by this
  release-preparation change.

## Required before public visibility

- [ ] Confirm ownership and provenance of every source file and copied
  implementation.
- [ ] Confirm that CC BY-NC 4.0 is appropriate for every repository-owned file.
- [ ] Review existing Git history for internal manifests, identifiers, paths,
  author metadata, and other sensitive information; rewrite history if needed.
- [ ] Verify that any released checkpoints may be redistributed and contain no
  sensitive metadata.
- [ ] Compare a fresh fixed-protocol run with the manuscript's expected metrics.
- [ ] Add the final paper citation and public paper link.
- [ ] Review generated CSV, log, visualization, and checkpoint metadata before
  publishing artifacts.
