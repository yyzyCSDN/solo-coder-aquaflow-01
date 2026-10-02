# AquaFlow
Dependency-free municipal drinking-water treatment and distribution operations platform.

## Capabilities
- Treatment, network, metering, laboratory and regulatory workflow services (see `aquaflow.workflow_rules`).
- Residual-chlorine anomaly correlation desk (`aquaflow.treatment.chlorine_desk`): ingests chlorine,
  temperature and contact-time readings by asset and time window, correlates evidence per treatment
  chain, opens a traceable case whenever release conditions cannot be proven (missing segments,
  conflicting overlaps, CT/residual shortfalls), and records operator decisions. Case states move
  forward only (opened → confirmed → held → released → closed), are replayable from the audit log,
  and late readings append new evidence revisions instead of rewriting history.
