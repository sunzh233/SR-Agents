# Changelog

## v1.0.1

- Restore the ToolQA ReAct parser behavior used in the original experiments.
- Reject incomplete evaluation coverage and publish completed summaries
  atomically.
- Resume inference by retaining successful rows and retrying failed, empty, or
  interrupted rows.
- Share canonical retrieval-query construction and add BGE-M3 support.
- Forward configured ReAct temperatures and use a one-hour request timeout for
  heavily queued model servers.
- Parse BigCodeBench subprocess results even when sandbox diagnostics precede
  the final JSON line.
- Reuse ToolQA SQLite tables across worker threads instead of copying the same
  read-only data into every environment.
- Add content-addressed, cross-process ToolQA embedding caches suitable for
  shared filesystems while preserving the release offline-model and cache-dir
  compatibility paths.
