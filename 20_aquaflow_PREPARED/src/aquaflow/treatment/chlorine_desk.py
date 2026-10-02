"""Residual-chlorine anomaly correlation desk.

Operators watch chlorine, temperature and contact-time readings across one
treatment chain.  When the readings for a time window cannot prove the
release conditions, the desk opens a traceable case and records every
operator decision.  Evidence lives in an append-only store: missing
segments, overlapping windows and late readings arrive as new evidence
revisions, never as silent rewrites.  Case states only move forward
(opened -> confirmed -> held -> released -> closed) and the full history
can be replayed from the audit log in event order.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from math import isfinite

from .chlorine_ct import Segment, effective_ct

METRICS = ("chlorine", "temperature", "contact_time")

# States only move to a strictly higher rank, so a regression cannot be
# expressed and replay order stays unambiguous.
STATE_RANK = {"opened": 0, "confirmed": 1, "held": 2, "released": 3, "closed": 4}
ACTION_STATE = {"confirm": "confirmed", "hold": "held", "release": "released", "close": "closed"}
DECIDED = ("released", "closed")


def _utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError("expected datetime or ISO string")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Policy:
    required_ct: float = 10.0
    residual_min: float = 0.2
    residual_max: float = 4.0
    baffling: float = 1.0
    tolerances: dict = field(default_factory=lambda: {"chlorine": 0.02, "temperature": 0.2, "contact_time": 0.5})

    def __post_init__(self):
        if self.required_ct <= 0:
            raise ValueError("required_ct must be positive")
        if self.residual_min < 0 or self.residual_max < self.residual_min:
            raise ValueError("invalid residual limits")
        if not 0 < self.baffling <= 1:
            raise ValueError("invalid baffling factor")

    def tolerance(self, metric: str) -> float:
        return float(self.tolerances.get(metric, 0.0))


@dataclass(frozen=True)
class Reading:
    chain: str
    asset: str
    metric: str
    start: datetime
    end: datetime
    value: float
    sequence: int = 0  # 0 = unassigned, the desk stamps ingest order
    received_at: datetime | None = None


@dataclass(frozen=True)
class Gap:
    metric: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class Conflict:
    metric: str
    asset: str
    start: datetime
    end: datetime
    values: tuple


@dataclass(frozen=True)
class Evaluation:
    chain: str
    start: datetime
    end: datetime
    provable: bool
    ct: float | None
    chlorine_min: float | None
    chlorine_max: float | None
    temperature_min: float | None
    contact_time_min: float | None
    gaps: tuple
    conflicts: tuple
    reasons: tuple
    reading_count: int


@dataclass(frozen=True)
class Case:
    case_id: str
    chain: str
    start: datetime
    end: datetime
    state: str
    reason: str
    opened_at: datetime
    updated_at: datetime
    related_case: str | None = None


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    kind: str
    at: datetime
    actor: str
    from_state: str | None
    to_state: str | None
    reason: str
    details: dict


@dataclass(frozen=True)
class RegisteredWindow:
    chain: str
    start: datetime
    end: datetime
    first_evaluated_at: datetime


def _coverage_gaps(spans, start, end):
    gaps = []
    cursor = start
    for s, e in sorted(spans):
        if e <= cursor:
            continue
        if s > cursor:
            gaps.append((cursor, min(s, end)))
        cursor = max(cursor, e)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return [(s, e) for s, e in gaps if s < e]


def _asset_conflicts(rows, tolerance):
    rows = sorted(rows, key=lambda r: (r.start, r.sequence))
    found = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if b.start >= a.end:
                break
            if abs(a.value - b.value) > tolerance:
                found.append(Conflict(a.metric, a.asset, max(a.start, b.start), min(a.end, b.end), (a.value, b.value)))
    return found


def evaluate_window(readings, chain, start, end, policy) -> Evaluation:
    """Prove the release conditions for one chain window from stored readings."""
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("invalid evaluation window")
    rows = [r for r in readings if r.chain == chain and r.start < end and r.end > start]
    gaps, conflicts, reasons = [], [], []
    values: dict[str, list[float]] = {}
    for metric in METRICS:
        mrows = [r for r in rows if r.metric == metric]
        values[metric] = [r.value for r in mrows]
        if not mrows:
            gaps.append(Gap(metric, start, end))
            continue
        spans = [(max(r.start, start), min(r.end, end)) for r in mrows]
        gaps.extend(Gap(metric, s, e) for s, e in _coverage_gaps(spans, start, end))
        for asset in sorted({r.asset for r in mrows}):
            conflicts.extend(_asset_conflicts([r for r in mrows if r.asset == asset], policy.tolerance(metric)))
    for metric in METRICS:
        if any(g.metric == metric for g in gaps):
            reasons.append(f"missing_coverage:{metric}")
    for c in conflicts:
        reasons.append(f"conflicting_readings:{c.metric}@{c.asset}")
    chlorine = values["chlorine"]
    chlorine_min = min(chlorine) if chlorine else None
    chlorine_max = max(chlorine) if chlorine else None
    temperature_min = min(values["temperature"]) if values["temperature"] else None
    contact_min = min(values["contact_time"]) if values["contact_time"] else None
    ct = None
    if chlorine_min is not None and temperature_min is not None and contact_min is not None:
        if contact_min <= 0 or chlorine_min < 0:
            reasons.append("invalid_readings")
        else:
            ct = effective_ct([Segment(contact_min, chlorine_min, policy.baffling, temperature_min)])
    if chlorine_min is not None and chlorine_min < policy.residual_min:
        reasons.append("residual_below_min")
    if chlorine_max is not None and chlorine_max > policy.residual_max:
        reasons.append("residual_above_max")
    if ct is not None and ct < policy.required_ct:
        reasons.append("ct_below_required")
    unique = tuple(dict.fromkeys(reasons))
    return Evaluation(chain, start, end, not unique, ct, chlorine_min, chlorine_max, temperature_min, contact_min, tuple(gaps), tuple(conflicts), unique, len(rows))


def gap_view(g: Gap) -> dict:
    return {"metric": g.metric, "start": g.start.isoformat(), "end": g.end.isoformat()}


def conflict_view(c: Conflict) -> dict:
    return {"metric": c.metric, "asset": c.asset, "start": c.start.isoformat(), "end": c.end.isoformat(), "values": list(c.values)}


def evaluation_view(evaluation: Evaluation, policy: Policy) -> dict:
    return {
        "chain": evaluation.chain,
        "window": {"start": evaluation.start.isoformat(), "end": evaluation.end.isoformat()},
        "provable": evaluation.provable,
        "ct": evaluation.ct,
        "required_ct": policy.required_ct,
        "chlorine_min": evaluation.chlorine_min,
        "chlorine_max": evaluation.chlorine_max,
        "temperature_min": evaluation.temperature_min,
        "contact_time_min": evaluation.contact_time_min,
        "gaps": [gap_view(g) for g in evaluation.gaps],
        "conflicts": [conflict_view(c) for c in evaluation.conflicts],
        "reasons": list(evaluation.reasons),
        "reading_count": evaluation.reading_count,
    }


def replay_events(events) -> list[dict]:
    """Fold audit events in sequence order into the state trail.

    Raises if the trail would go backwards in state rank or in time, so a
    corrupted history is rejected instead of replayed.
    """
    trail = []
    state = last_at = last_seq = None
    for e in sorted(events, key=lambda e: e.seq):
        if last_seq is not None and e.seq <= last_seq:
            raise ValueError("event order violation")
        if last_at is not None and e.at < last_at:
            raise ValueError("time reversal")
        if e.to_state is not None:
            if state is not None and STATE_RANK[e.to_state] <= STATE_RANK[state]:
                raise ValueError("state regression")
            state = e.to_state
            trail.append({"seq": e.seq, "state": state, "reason": e.reason, "actor": e.actor, "at": e.at.isoformat()})
        last_seq, last_at = e.seq, e.at
    return trail


def _same_evaluation(a: Evaluation, b: Evaluation) -> bool:
    return (a.provable, a.reasons, a.reading_count, a.ct) == (b.provable, b.reasons, b.reading_count, b.ct)


class ChlorineDesk:
    """Stateful correlation desk: append-only evidence, forward-only cases."""

    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy()
        self._readings: list[Reading] = []
        self._cases: dict[str, Case] = {}
        self._audit: dict[str, list[AuditEvent]] = {}
        self._snapshots: dict[str, list[Evaluation]] = {}
        self._windows: list[RegisteredWindow] = []
        self._last_eval: dict[tuple, Evaluation] = {}
        self._seq = 0
        self._case_seq = 0
        self._reading_seq = 0

    # -- evidence intake -------------------------------------------------

    def ingest(self, rows, received_at=None, chain=None) -> dict:
        stamp = _utc(received_at) if received_at is not None else datetime.now(timezone.utc)
        accepted, duplicates = [], []
        for raw in rows:
            reading = self._normalize(raw, chain, stamp)
            if self._is_duplicate(reading):
                duplicates.append(self._reading_key(reading))
                continue
            if reading.sequence == 0:
                reading = replace(reading, sequence=self._next_reading_seq())
            self._readings.append(reading)
            accepted.append(reading)
        late, revised, opened = [], [], []
        for window in self._affected_windows(accepted):
            late.extend(
                {
                    "asset": r.asset,
                    "metric": r.metric,
                    "start": r.start.isoformat(),
                    "end": r.end.isoformat(),
                    "received_at": r.received_at.isoformat(),
                    "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
                }
                for r in accepted
                if r.chain == window.chain and r.start < window.end and r.end > window.start and r.received_at > window.first_evaluated_at
            )
            revised_id, opened_id = self._refresh(window, stamp)
            if revised_id:
                revised.append(revised_id)
            if opened_id:
                opened.append(opened_id)
        return {"accepted": len(accepted), "duplicates": len(duplicates), "late": late, "cases_revised": revised, "cases_opened": opened}

    # -- evaluation and case lifecycle -----------------------------------

    def evaluate(self, chain, start, end, at=None) -> dict:
        at = _utc(at) if at is not None else datetime.now(timezone.utc)
        start, end = _utc(start), _utc(end)
        evaluation = evaluate_window(self._readings, chain, start, end, self.policy)
        self._register(chain, evaluation.start, evaluation.end, at)
        self._last_eval[(evaluation.chain, evaluation.start, evaluation.end)] = evaluation
        case = self._active_case(evaluation.chain, evaluation.start, evaluation.end)
        if not evaluation.provable:
            if case is None:
                decided = self._decided_cases(evaluation.chain, evaluation.start, evaluation.end)
                case = self._open(evaluation, at, related=decided[-1].case_id if decided else None, trigger="evaluation")
            else:
                self._attach(case, evaluation, at, trigger="evaluation")
        elif case is not None:
            self._attach(case, evaluation, at, trigger="evaluation")
        view = evaluation_view(evaluation, self.policy)
        view["case_id"] = case.case_id if case else None
        return view

    def decide(self, case_id, action, reason, actor="operator", at=None) -> dict:
        case = self._require(case_id)
        if action not in ACTION_STATE:
            raise ValueError("unknown action: " + str(action))
        if not reason or not str(reason).strip():
            raise ValueError("decision reason required")
        at = _utc(at) if at is not None else datetime.now(timezone.utc)
        if self._audit[case_id] and at < self._audit[case_id][-1].at:
            raise ValueError("time reversal")
        target = ACTION_STATE[action]
        if STATE_RANK[target] <= STATE_RANK[case.state]:
            raise ValueError("state regression: " + case.state + " -> " + target)
        self._cases[case_id] = replace(case, state=target, reason=str(reason), updated_at=at)
        self._emit(case_id, "decision", at, str(actor), case.state, target, str(reason), {"action": action})
        return self.case_view(case_id)

    # -- read models ------------------------------------------------------

    def case_view(self, case_id) -> dict:
        case = self._require(case_id)
        snapshots = self._snapshots[case_id]
        latest = snapshots[-1]
        evidence = evaluation_view(latest, self.policy)
        evidence["revision"] = len(snapshots)
        evidence["readings"] = [self._reading_view(r) for r in self._correlated(case)]
        return {
            "case_id": case.case_id,
            "chain": case.chain,
            "window": {"start": case.start.isoformat(), "end": case.end.isoformat()},
            "state": case.state,
            "reason": case.reason,
            "opened_at": case.opened_at.isoformat(),
            "updated_at": case.updated_at.isoformat(),
            "related_case": case.related_case,
            "allowed_actions": self.allowed_actions(case_id),
            "evidence": evidence,
            "audit": [self._event_view(e) for e in self._audit[case_id]],
        }

    def cases(self) -> list[dict]:
        return [
            {
                "case_id": c.case_id,
                "chain": c.chain,
                "state": c.state,
                "reason": c.reason,
                "window": {"start": c.start.isoformat(), "end": c.end.isoformat()},
                "opened_at": c.opened_at.isoformat(),
                "related_case": c.related_case,
            }
            for c in self._cases.values()
        ]

    def replay(self, case_id) -> list[dict]:
        self._require(case_id)
        return replay_events(self._audit[case_id])

    def allowed_actions(self, case_id) -> list[str]:
        case = self._require(case_id)
        return sorted(a for a, s in ACTION_STATE.items() if STATE_RANK[s] > STATE_RANK[case.state])

    # -- internals --------------------------------------------------------

    def _normalize(self, raw, default_chain, stamp) -> Reading:
        if isinstance(raw, dict):
            raw = Reading(
                chain=str(raw.get("chain") or ""),
                asset=str(raw["asset"]),
                metric=str(raw["metric"]),
                start=_utc(raw["start"]),
                end=_utc(raw["end"]),
                value=float(raw["value"]),
                sequence=int(raw.get("sequence") or 0),
                received_at=_utc(raw["received_at"]) if raw.get("received_at") else None,
            )
        if not isinstance(raw, Reading):
            raise TypeError("reading must be a Reading or a mapping")
        chain_name = raw.chain or default_chain
        if not chain_name:
            raise ValueError("reading chain required")
        if raw.metric not in METRICS:
            raise ValueError("unknown metric: " + str(raw.metric))
        start, end = _utc(raw.start), _utc(raw.end)
        if end <= start:
            raise ValueError("reading window must be positive")
        value = float(raw.value)
        if not isfinite(value):
            raise ValueError("reading value must be finite")
        received = _utc(raw.received_at) if raw.received_at is not None else stamp
        return Reading(str(chain_name), str(raw.asset), raw.metric, start, end, value, raw.sequence, received)

    def _is_duplicate(self, r: Reading) -> bool:
        key = (r.chain, r.asset, r.metric, r.start, r.end, r.value)
        return any((x.chain, x.asset, x.metric, x.start, x.end, x.value) == key for x in self._readings)

    def _next_reading_seq(self) -> int:
        self._reading_seq += 1
        return self._reading_seq

    @staticmethod
    def _reading_key(r: Reading) -> dict:
        return {"asset": r.asset, "metric": r.metric, "start": r.start.isoformat(), "end": r.end.isoformat(), "value": r.value}

    def _affected_windows(self, accepted) -> list[RegisteredWindow]:
        return [w for w in self._windows if any(r.chain == w.chain and r.start < w.end and r.end > w.start for r in accepted)]

    def _refresh(self, window: RegisteredWindow, at):
        evaluation = evaluate_window(self._readings, window.chain, window.start, window.end, self.policy)
        key = (window.chain, window.start, window.end)
        previous = self._last_eval.get(key)
        self._last_eval[key] = evaluation
        active = self._active_case(window.chain, window.start, window.end)
        if active is not None:
            attached = self._attach(active, evaluation, at, trigger="late_evidence")
            return (active.case_id if attached else None), None
        revised = None
        decided = self._decided_cases(window.chain, window.start, window.end)
        for case in decided:
            if self._attach(case, evaluation, at, trigger="late_evidence"):
                revised = case.case_id
        if not evaluation.provable:
            followup = self._open(evaluation, at, related=decided[-1].case_id if decided else None, trigger="late_evidence", previously=previous)
            return revised, followup.case_id
        return revised, None

    def _register(self, chain, start, end, at):
        if not any(w.chain == chain and w.start == start and w.end == end for w in self._windows):
            self._windows.append(RegisteredWindow(chain, start, end, at))

    def _active_case(self, chain, start, end):
        for case in self._cases.values():
            if case.chain == chain and case.start < end and case.end > start and case.state not in DECIDED:
                return case
        return None

    def _decided_cases(self, chain, start, end):
        return sorted(
            (c for c in self._cases.values() if c.chain == chain and c.start < end and c.end > start and c.state in DECIDED),
            key=lambda c: c.opened_at,
        )

    def _open(self, evaluation, at, related, trigger, previously=None) -> Case:
        self._case_seq += 1
        case_id = f"CL-{self._case_seq:04d}"
        reason = "; ".join(evaluation.reasons) or "release_unprovable"
        case = Case(case_id, evaluation.chain, evaluation.start, evaluation.end, "opened", reason, at, at, related)
        self._cases[case_id] = case
        self._snapshots[case_id] = [evaluation]
        self._audit[case_id] = []
        details = {"trigger": trigger, "related_case": related, "revision": 1}
        if previously is not None:
            details["previously_provable"] = previously.provable
        self._emit(case_id, "opened", at, "system", None, "opened", reason, details)
        return case

    def _attach(self, case: Case, evaluation, at, trigger) -> bool:
        snapshots = self._snapshots[case.case_id]
        if snapshots and _same_evaluation(snapshots[-1], evaluation):
            return False
        snapshots.append(evaluation)
        at = max(at, self._audit[case.case_id][-1].at)
        self._emit(case.case_id, "evidence", at, "system", case.state, None, trigger, {
            "trigger": trigger,
            "revision": len(snapshots),
            "provable": evaluation.provable,
            "reasons": list(evaluation.reasons),
            "contradicts_decision": case.state in DECIDED and not evaluation.provable,
        })
        return True

    def _emit(self, case_id, kind, at, actor, from_state, to_state, reason, details):
        self._seq += 1
        self._audit[case_id].append(AuditEvent(self._seq, kind, at, actor, from_state, to_state, reason, details))

    def _require(self, case_id) -> Case:
        try:
            return self._cases[case_id]
        except KeyError:
            raise KeyError("unknown case: " + str(case_id)) from None

    def _correlated(self, case: Case) -> list[Reading]:
        rows = [r for r in self._readings if r.chain == case.chain and r.start < case.end and r.end > case.start]
        return sorted(rows, key=lambda r: (r.metric, r.asset, r.start, r.sequence))

    def _is_late(self, r: Reading) -> bool:
        return any(
            w.chain == r.chain and r.start < w.end and r.end > w.start and r.received_at is not None and r.received_at > w.first_evaluated_at
            for w in self._windows
        )

    def _reading_view(self, r: Reading) -> dict:
        return {
            "asset": r.asset,
            "metric": r.metric,
            "start": r.start.isoformat(),
            "end": r.end.isoformat(),
            "value": r.value,
            "sequence": r.sequence,
            "received_at": r.received_at.isoformat() if r.received_at else None,
            "late": self._is_late(r),
        }

    @staticmethod
    def _event_view(e: AuditEvent) -> dict:
        return {
            "seq": e.seq,
            "kind": e.kind,
            "at": e.at.isoformat(),
            "actor": e.actor,
            "from_state": e.from_state,
            "to_state": e.to_state,
            "reason": e.reason,
            "details": dict(e.details),
        }
