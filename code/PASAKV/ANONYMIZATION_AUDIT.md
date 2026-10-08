# Anonymization and scope audit

- No Git history, user profiles, experiment ledgers, logs, datasets, or checkpoints were copied.
- No author, institution, email, Windows path, AFS username, development host, port, or storage credential is intentionally included.
- Upstream model code remains an external dependency and is not copied or relicensed here.
- Checkpoint paths use environment-variable names; the promised FastWAM .pt is not present.
- License is intentionally unset pending author choice and upstream compatibility review.

A recursive scan on 2026-09-27 for author/user names, local and AFS paths, host/storage identifiers, credentials, and email-shaped strings returned no matches in the staged source/config/docs.

This is a practical string scan, not a guarantee against every possible identifying metadata channel; review commit metadata, package metadata, and newly added files again before public release.
