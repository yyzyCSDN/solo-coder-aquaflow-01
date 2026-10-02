# AquaFlow
Dependency-free municipal drinking-water treatment and distribution operations platform.

## Residual-chlorine anomaly correlation casebook
`aquaflow.treatment.residual_anomaly` (also on `AquaFlowService`) joins
residual-chlorine, temperature and contact-time readings for one treatment
chain over a time window and manages trackable anomaly cases.

* **Readings by asset/interval** — `ingest_readings(...)` stores half-open
  `[start, end)` observations (`Reading`/mappings). The log is append-only.
* **Correlation** — `correlate_evidence(chain, start, end)` cuts the window
  into elementary slices and proves release over each one. Blockers:
  `missing_segment` (missing fragment), `conflicting_overlap` (contradictory
  readings on the same interval), `residual_out_of_range`,
  `temperature_out_of_range`, `insufficient_contact_time`. Identical duplicate
  overlaps are `duplicate_overlap` info findings. Limits live in
  `ReleaseSpec`.
* **Open when release cannot be proven** —
  `open_residual_case(...)` returns the correlation and creates a case only
  when proof fails; overlapping windows on a chain reuse the active case.
* **Operator decisions** — confirm / hold / release / close
  (`decide_residual_case`). Release is refused unless a fresh correlation is
  provable. Every decision requires a reason.
* **Replayable, no regression** — state is replayed from ordered events
  (`Case.replay`). Transitions are a forward allow-list with monotonic ranks:
  `open → confirmed → held → released → closed` (skips forward allowed, never
  backward); event time may not move backwards.
* **Audit** — `residual_case(id)` returns correlated `evidence`, blockers, the
  current `status_reason`, and a full ordered `audit` of every state change
  (seq, time, actor, reason, from/to status).
* **Late / corrected readings** — late readings are new events; evaluating at
  a past `as_of` still sees the old gap, so history is never silently edited.
  A correction uses `supersedes`; the original remains visible as
  `corrected_reading` and never counts as proof.

