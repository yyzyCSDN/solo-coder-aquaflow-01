"""Residual-chlorine anomaly correlation casebook.

Operators must look at residual chlorine, temperature and contact-time
readings on the same treatment chain at the same time.  When those readings
are contradictory (or incomplete), release cannot be *proven* and a
trackable anomaly case must be opened.

Design notes
------------
* Readings are organized by asset and half-open time interval [start, end).
* ``correlate`` joins evidence for one chain over one window by cutting the
  window into elementary slices.  Every blocker (a missing fragment, an
  unresolved overlap/conflict, an out-of-range value or insufficient contact
  time) means release is not provable.
* Cases are event-sourced.  Only events are stored (append-only); state is
  derived by replaying events in sequence.  Transitions are an explicit
  forward allow-list and event time must never move backwards, so history can
  neither be rewritten nor regressed.
* Late readings are simply new events: they never mutate stored readings or
  past evaluations.  A reading that corrects an earlier one *supersedes* it;
  the original stays on record and is flagged in every evidence view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable


class DomainError(ValueError):
    """A business rule was violated (bad reading, illegal transition, ...)."""


class Metric(str, Enum):
    RESIDUAL = "residual"          # free chlorine, mg/L
    TEMPERATURE = "temperature"    # water temperature, deg C
    CONTACT_TIME = "contact_time"  # effective contact time, minutes

    @staticmethod
    def coerce(value: "Metric | str") -> "Metric":
        if isinstance(value, Metric):
            return value
        try:
            return Metric(str(value))
        except ValueError:
            raise DomainError(f"unknown metric: {value!r}")


# ---------------------------------------------------------------------------
# Reading model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reading:
    """One observation for one asset, valid over [start, end).

    ``recorded_at`` is when the observation entered the system, which may be
    later than the measured interval (a late reading).  ``supersedes`` points
    at an earlier reading that this one corrects.
    """

    reading_id: str
    chain_id: str
    asset_id: str
    metric: Metric
    value: float
    start: datetime
    end: datetime
    recorded_at: datetime
    unit: str = ""
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if not self.reading_id:
            raise DomainError("reading_id is required")
        if not self.chain_id or not self.asset_id:
            raise DomainError("chain_id and asset_id are required")
        if self.value < 0:
            raise DomainError("reading value must be non-negative")
        if self.end <= self.start:
            raise DomainError("reading interval must have end > start")
        if not (self.recorded_at.tzinfo and self.start.tzinfo and self.end.tzinfo):
            raise DomainError("timestamps must be timezone-aware")


@dataclass(frozen=True)
class ReleaseSpec:
    """Conditions that must be provable before water may be released."""

    residual_min: float = 0.2          # mg/L free chlorine
    residual_max: float = 4.0          # mg/L
    temperature_min: float = 1.0       # deg C
    temperature_max: float = 30.0
    required_contact_minutes: float = 20.0

    def residual_ok(self, value: float) -> bool:
        return self.residual_min <= value <= self.residual_max

    def temperature_ok(self, value: float) -> bool:
        return self.temperature_min <= value <= self.temperature_max


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Finding:
    code: str                 # machine-readable reason code
    severity: str             # "blocker" or "info"
    start: datetime
    end: datetime
    message: str
    metric: Metric | None = None
    reading_ids: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "message": self.message,
            "metric": self.metric.value if self.metric else None,
            "reading_ids": list(self.reading_ids),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class Correlation:
    """Joined evidence for one chain/window and the proof result."""

    chain_id: str
    start: datetime
    end: datetime
    as_of: datetime
    spec: ReleaseSpec
    provable: bool
    findings: list[Finding]
    evidence: list[dict[str, Any]]
    late_count: int = 0
    superseded_count: int = 0

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    def reason(self) -> str:
        blockers = self.blockers
        if not blockers:
            return "release conditions provable for the full window"
        parts: list[str] = []
        for code in sorted({f.code for f in blockers}):
            if code == "missing_segment":
                missing = sorted({
                    m for f in blockers
                    if f.code == code for m in f.detail.get("missing", [])
                })
                parts.append("missing_segment(" + ",".join(missing) + ")")
            else:
                parts.append(code)
        return "release not provable: " + ",".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "as_of": self.as_of.isoformat(),
            "provable": self.provable,
            "reason": self.reason(),
            "findings": [f.to_dict() for f in self.findings],
            "blockers": [f.to_dict() for f in self.blockers],
            "evidence": list(self.evidence),
            "late_count": self.late_count,
            "superseded_count": self.superseded_count,
            "spec": {
                "residual_min": self.spec.residual_min,
                "residual_max": self.spec.residual_max,
                "temperature_min": self.spec.temperature_min,
                "temperature_max": self.spec.temperature_max,
                "required_contact_minutes": self.spec.required_contact_minutes,
            },
        }


def _dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _intersects_clipped(r: Reading, start: datetime, end: datetime) -> tuple[datetime, datetime] | None:
    lo, hi = max(r.start, start), min(r.end, end)
    return (lo, hi) if hi > lo else None


def _merge_findings(raw: list[Finding]) -> list[Finding]:
    """Merge adjacent slices carrying the same finding."""
    merged: list[Finding] = []
    for f in raw:
        key = (
            f.code,
            f.severity,
            f.metric,
            tuple(f.reading_ids),
            tuple(sorted(f.detail.items(), key=lambda kv: kv[0])),
            f.message,
        )
        if merged:
            prev = merged[-1]
            prev_key = (
                prev.code,
                prev.severity,
                prev.metric,
                tuple(prev.reading_ids),
                tuple(sorted(prev.detail.items(), key=lambda kv: kv[0])),
                prev.message,
            )
            if prev_key == key and prev.end == f.start:
                merged[-1] = Finding(
                    f.code, f.severity, prev.start, f.end, f.message,
                    f.metric, f.reading_ids, dict(f.detail),
                )
                continue
        merged.append(f)
    return merged


def correlate(
    chain_id: str,
    start: datetime | str,
    end: datetime | str,
    readings: Iterable[Reading],
    spec: ReleaseSpec | None = None,
    as_of: datetime | str | None = None,
) -> Correlation:
    """Join evidence on one treatment chain over [start, end).

    Only readings recorded at or before ``as_of`` count as proof.  Readings
    recorded later are returned in the evidence as ``late`` (they are a gap
    for this evaluation, never a silent edit).  Superseded readings are kept
    visible as ``superseded`` and never used as proof.
    """
    spec = spec or ReleaseSpec()
    start, end = _dt(start), _dt(end)
    if end <= start:
        raise DomainError("correlation window must have end > start")
    as_of = _dt(as_of) if as_of is not None else datetime.now(timezone.utc)

    late: list[Reading] = []
    superseded: list[Reading] = []
    active: list[tuple[Reading, datetime, datetime]] = []
    candidates: list[tuple[Reading, datetime, datetime]] = []
    for r0 in readings:
        r = r0 if isinstance(r0.metric, Metric) else Reading(
            r0.reading_id, r0.chain_id, r0.asset_id, Metric.coerce(r0.metric),
            float(r0.value), _dt(r0.start), _dt(r0.end), _dt(r0.recorded_at),
            r0.unit, r0.supersedes,
        )
        if r.chain_id != chain_id:
            continue
        clip = _intersects_clipped(r, start, end)
        if clip is None:
            continue
        candidates.append((r, clip[0], clip[1]))

    # A reading that some later reading points at is the corrected one: it
    # stays visible but never counts as proof.  The correction itself is the
    # active reading.
    superseded_ids = {r.supersedes for r, _, _ in candidates if r.supersedes}
    for r, lo, hi in candidates:
        if r.reading_id in superseded_ids:
            superseded.append(r)
        elif r.recorded_at > as_of:
            late.append(r)
        else:
            active.append((r, lo, hi))

    # Elementary slice boundaries from every active clipped interval.
    bounds = {start, end}
    for _, lo, hi in active:
        bounds.add(lo)
        bounds.add(hi)
    cuts = sorted(bounds)

    raw: list[Finding] = []
    proven_ids: set[str] = set()
    for a, b in zip(cuts, cuts[1:]):
        present: dict[Metric, list[Reading]] = {m: [] for m in Metric}
        for r, lo, hi in active:
            if lo <= a and b <= hi:
                present[r.metric].append(r)  # half-open cover of the slice

        for metric, covers in present.items():
            if len(covers) > 1:
                # Deterministic primary: earliest arrival, then smallest id.
                covers.sort(key=lambda r: (r.recorded_at, r.reading_id))
                primary, extras = covers[0], covers[1:]
                same = all(abs(x.value - primary.value) < 1e-9 for x in extras)
                ids = tuple(x.reading_id for x in covers)
                if same:
                    raw.append(Finding(
                        "duplicate_overlap", "info", a, b,
                        f"{metric.value} reported {len(covers)} times with the "
                        f"same value; {primary.reading_id} used",
                        metric, ids, {"value": primary.value},
                    ))
                else:
                    raw.append(Finding(
                        "conflicting_overlap", "blocker", a, b,
                        f"{len(covers)} contradictory {metric.value} readings "
                        f"over the same interval",
                        metric, ids,
                        {"values": {x.reading_id: x.value for x in covers}},
                    ))
            else:
                primary = covers[0] if covers else None
            if primary is None:
                continue
            proven_ids.add(primary.reading_id)
            if metric is Metric.RESIDUAL and not spec.residual_ok(primary.value):
                raw.append(Finding(
                    "residual_out_of_range", "blocker", a, b,
                    f"residual {primary.value} mg/L outside "
                    f"[{spec.residual_min}, {spec.residual_max}]",
                    metric, (primary.reading_id,),
                    {"value": primary.value},
                ))
            elif metric is Metric.TEMPERATURE and not spec.temperature_ok(primary.value):
                raw.append(Finding(
                    "temperature_out_of_range", "blocker", a, b,
                    f"temperature {primary.value} C outside "
                    f"[{spec.temperature_min}, {spec.temperature_max}]",
                    metric, (primary.reading_id,),
                    {"value": primary.value},
                ))
            elif metric is Metric.CONTACT_TIME and primary.value < spec.required_contact_minutes:
                raw.append(Finding(
                    "insufficient_contact_time", "blocker", a, b,
                    f"contact time {primary.value} min below required "
                    f"{spec.required_contact_minutes} min",
                    metric, (primary.reading_id,),
                    {"value": primary.value, "required": spec.required_contact_minutes},
                ))

        missing = tuple(m for m in Metric if not present[m])
        if missing:
            raw.append(Finding(
                "missing_segment", "blocker", a, b,
                "no reading for " + ", ".join(m.value for m in missing),
                None, (), {"missing": [m.value for m in missing]},
            ))

    findings = _merge_findings(raw)

    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r, lo, hi in sorted(active, key=lambda t: (t[0].metric.value, t[1], t[0].reading_id)):
        if r.reading_id in seen:
            continue
        seen.add(r.reading_id)
        evidence.append({
            "reading_id": r.reading_id,
            "asset_id": r.asset_id,
            "metric": r.metric.value,
            "value": r.value,
            "unit": r.unit,
            "start": r.start.isoformat(),
            "end": r.end.isoformat(),
            "recorded_at": r.recorded_at.isoformat(),
            "status": "proven" if r.reading_id in proven_ids else "overlap",
        })
    for r in late:
        evidence.append({
            "reading_id": r.reading_id,
            "asset_id": r.asset_id,
            "metric": r.metric.value,
            "value": r.value,
            "unit": r.unit,
            "start": r.start.isoformat(),
            "end": r.end.isoformat(),
            "recorded_at": r.recorded_at.isoformat(),
            "status": "late",
        })
    for r in sorted(superseded, key=lambda r: r.reading_id):
        evidence.append({
            "reading_id": r.reading_id,
            "asset_id": r.asset_id,
            "metric": r.metric.value,
            "value": r.value,
            "unit": r.unit,
            "start": r.start.isoformat(),
            "end": r.end.isoformat(),
            "recorded_at": r.recorded_at.isoformat(),
            "status": "corrected_reading",
        })

    return Correlation(
        chain_id=chain_id, start=start, end=end, as_of=as_of, spec=spec,
        provable=not any(f.severity == "blocker" for f in findings),
        findings=findings, evidence=evidence,
        late_count=len(late), superseded_count=len(superseded),
    )


# ---------------------------------------------------------------------------
# Event-sourced case
# ---------------------------------------------------------------------------
class CaseStatus(str, Enum):
    OPEN = "open"            # 立案：放行无法证明
    CONFIRMED = "confirmed"  # 确认：运行人员确认异常属实
    HELD = "held"            # 保持：维持不放行
    RELEASED = "released"    # 放行：放行条件已证明
    CLOSED = "closed"        # 关闭：终态


# action -> (event type, target status)
_ACTIONS: dict[str, tuple[str, CaseStatus]] = {
    "confirm": ("CaseConfirmed", CaseStatus.CONFIRMED),
    "hold": ("CaseHeld", CaseStatus.HELD),
    "release": ("CaseReleased", CaseStatus.RELEASED),
    "close": ("CaseClosed", CaseStatus.CLOSED),
}

# Forward-only allow-list.  Ranks enforce "never regress" even if the
# allow-list were edited carelessly.
_RANK = {
    CaseStatus.OPEN: 0,
    CaseStatus.CONFIRMED: 1,
    CaseStatus.HELD: 2,
    CaseStatus.RELEASED: 3,
    CaseStatus.CLOSED: 4,
}
_ALLOWED: dict[CaseStatus, frozenset[str]] = {
    CaseStatus.OPEN: frozenset({"confirm", "hold", "release", "close"}),
    CaseStatus.CONFIRMED: frozenset({"hold", "release", "close"}),
    CaseStatus.HELD: frozenset({"release", "close"}),
    CaseStatus.RELEASED: frozenset({"close"}),
    CaseStatus.CLOSED: frozenset(),
}


@dataclass(frozen=True)
class Event:
    seq: int
    at: datetime
    type: str
    actor: str
    reason: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Case:
    case_id: str
    chain_id: str
    start: datetime
    end: datetime
    status: CaseStatus = CaseStatus.OPEN
    version: int = 0
    opened_at: datetime | None = None
    updated_at: datetime | None = None
    last_reason: str = ""
    last_evaluation: dict[str, Any] | None = None
    linked_cases: tuple[str, ...] = ()

    @staticmethod
    def replay(events: list[Event]) -> "Case":
        if not events:
            raise DomainError("case has no events")
        first = events[0]
        if first.type != "CaseOpened":
            raise DomainError("case history must start with CaseOpened")
        case = Case(
            case_id=first.data["case_id"],
            chain_id=first.data["chain_id"],
            start=_dt(first.data["start"]),
            end=_dt(first.data["end"]),
            status=CaseStatus.OPEN,
            opened_at=first.at,
            updated_at=first.at,
            last_reason=first.reason,
            last_evaluation=first.data.get("evaluation"),
            linked_cases=tuple(first.data.get("linked_cases", ())),
        )
        expected_seq = 1
        prev_at = first.at
        for ev in events[1:]:
            if ev.seq != expected_seq:
                raise DomainError("event sequence gap")
            if ev.at < prev_at:
                raise DomainError("event time reversal")
            prev_at = ev.at
            if ev.type == "CaseReevaluated":
                case.last_evaluation = ev.data.get("evaluation")
                case.last_reason = ev.reason or case.last_reason
            elif ev.type == "CaseLinked":
                case.linked_cases = case.linked_cases + tuple(ev.data.get("linked_cases", ()))
            else:
                target = {
                    "CaseConfirmed": CaseStatus.CONFIRMED,
                    "CaseHeld": CaseStatus.HELD,
                    "CaseReleased": CaseStatus.RELEASED,
                    "CaseClosed": CaseStatus.CLOSED,
                }.get(ev.type)
                if target is None:
                    raise DomainError(f"unknown case event {ev.type}")
                action = _action_for(ev.type)
                if action not in _ALLOWED[case.status] or _RANK[target] < _RANK[case.status]:
                    raise DomainError(
                        f"illegal transition {case.status.value} -> {target.value}"
                    )
                case.status = target
                case.last_reason = ev.reason
            case.version += 1
            case.updated_at = ev.at
            expected_seq += 1
        return case


def _action_for(event_type: str) -> str:
    for action, (etype, _) in _ACTIONS.items():
        if etype == event_type:
            return action
    return event_type


# ---------------------------------------------------------------------------
# Append-only store
# ---------------------------------------------------------------------------
class AppendOnlyStore:
    """In-memory append-only log, one ordered stream per aggregate."""

    def __init__(self) -> None:
        self._streams: dict[str, list[Event]] = {}

    def append(
        self,
        stream: str,
        event_type: str,
        actor: str,
        reason: str,
        data: dict[str, Any] | None = None,
        at: datetime | None = None,
        expected_version: int | None = None,
    ) -> Event:
        events = self._streams.setdefault(stream, [])
        if expected_version is not None and len(events) != expected_version:
            raise DomainError(
                f"optimistic concurrency conflict on {stream}: "
                f"expected v{expected_version}, found v{len(events)}"
            )
        at = _dt(at) if at is not None else datetime.now(timezone.utc)
        if events and at < events[-1].at:
            raise DomainError("event time reversal in stream")
        ev = Event(len(events), at, event_type, actor, reason, dict(data or {}))
        events.append(ev)
        return ev

    def read(self, stream: str) -> list[Event]:
        return list(self._streams.get(stream, ()))

    def streams(self) -> list[str]:
        return sorted(self._streams)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class ResidualAnomalyService:
    """Application facade: ingest readings, correlate evidence, run cases."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self.store = AppendOnlyStore()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._commands: dict[str, str] = {}

    # -- readings ----------------------------------------------------------
    def ingest(self, readings: Iterable[Reading | dict[str, Any]]) -> list[str]:
        """Record readings (append-only).  Returns accepted reading ids."""
        accepted: list[str] = []
        for item in readings:
            r = self._coerce_reading(item)
            if r.supersedes:
                target = self._find_reading(r.chain_id, r.supersedes)
                if target is None:
                    raise DomainError(f"cannot supersede unknown reading {r.supersedes}")
                if (target.asset_id, target.metric) != (r.asset_id, r.metric):
                    raise DomainError("a correction must keep asset and metric")
            self.store.append(
                f"readings:{r.chain_id}", "ReadingRecorded", "system",
                "reading ingested", {"reading": _reading_data(r)}, at=r.recorded_at,
            )
            accepted.append(r.reading_id)
        return accepted

    def readings(self, chain_id: str) -> list[Reading]:
        out: list[Reading] = []
        for ev in self.store.read(f"readings:{chain_id}"):
            out.append(_reading_from_data(ev.data["reading"]))
        return out

    def correlate(
        self,
        chain_id: str,
        start: datetime | str,
        end: datetime | str,
        *,
        spec: ReleaseSpec | None = None,
        as_of: datetime | str | None = None,
    ) -> Correlation:
        return correlate(
            chain_id, start, end, self.readings(chain_id), spec=spec,
            as_of=as_of if as_of is not None else self._clock(),
        )

    # -- cases -------------------------------------------------------------
    def open_if_unprovable(
        self,
        chain_id: str,
        start: datetime | str,
        end: datetime | str,
        *,
        spec: ReleaseSpec | None = None,
        case_id: str | None = None,
        actor: str = "system",
        reason: str | None = None,
        at: datetime | str | None = None,
        command_id: str | None = None,
    ) -> dict[str, Any]:
        """Correlate the window; open (or reuse) a case unless release is provable."""
        if command_id and command_id in self._commands:
            return {"case_id": self._commands[command_id], "created": False, "reused": True}
        at = _dt(at) if at is not None else self._clock()
        result = self.correlate(chain_id, start, end, spec=spec, as_of=at)
        start, end = result.start, result.end

        active_overlap = self._find_active_case(chain_id, start, end)
        if active_overlap is not None:
            cid = active_overlap
            if not result.provable:
                self.store.append(
                    f"case:{cid}", "CaseReevaluated", actor,
                    reason or result.reason(),
                    {"evaluation": result.to_dict()}, at=at,
                )
            if command_id:
                self._commands[command_id] = cid
            return {"case_id": cid, "created": False, "reused": True,
                    "provable": result.provable, "correlation": result.to_dict()}

        if result.provable:
            return {"case_id": None, "created": False, "reused": False,
                    "provable": True, "correlation": result.to_dict()}

        import uuid

        cid = case_id or f"case-{uuid.uuid4().hex[:12]}"
        stream = f"case:{cid}"
        if self.store.read(stream):
            raise DomainError(f"case {cid} already exists")
        self.store.append(
            stream, "CaseOpened", actor, reason or result.reason(),
            {
                "case_id": cid,
                "chain_id": chain_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "evaluation": result.to_dict(),
                "linked_cases": [],
            },
            at=at,
        )
        self.store.append(
            "cases:index", "CaseIndexed", actor, "case opened",
            {"case_id": cid, "chain_id": chain_id,
             "start": start.isoformat(), "end": end.isoformat()}, at=at,
        )
        if command_id:
            self._commands[command_id] = cid
        return {"case_id": cid, "created": True, "reused": False,
                "provable": False, "correlation": result.to_dict()}

    def decide(
        self,
        case_id: str,
        action: str,
        actor: str,
        reason: str,
        *,
        at: datetime | str | None = None,
        spec: ReleaseSpec | None = None,
        command_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an operator decision: confirm / hold / release / close."""
        if command_id and command_id in self._commands:
            return self.case_view(self._commands[command_id])
        if not reason or not str(reason).strip():
            raise DomainError("decision reason is required")
        if action not in _ACTIONS:
            raise DomainError(f"unknown action {action!r}")
        stream = f"case:{case_id}"
        events = self.store.read(stream)
        if not events:
            raise DomainError(f"unknown case {case_id}")
        case = Case.replay(events)
        if action not in _ALLOWED[case.status]:
            raise DomainError(
                f"illegal transition {case.status.value} -> {action} "
                f"(rank cannot regress)"
            )
        at = _dt(at) if at is not None else self._clock()

        if action == "release":
            result = self.correlate(
                case.chain_id, case.start, case.end, spec=spec, as_of=at,
            )
            if not result.provable:
                raise DomainError("cannot release: " + result.reason())
            if _evaluation_signature(result.to_dict()) != _evaluation_signature(
                case.last_evaluation or {}
            ):
                self.store.append(
                    stream, "CaseReevaluated", actor,
                    "release proof re-established",
                    {"evaluation": result.to_dict()}, at=at,
                )
        event_type, target = _ACTIONS[action]
        self.store.append(
            stream, event_type, actor, reason,
            {"from_status": case.status.value, "to_status": target.value}, at=at,
        )
        if command_id:
            self._commands[command_id] = case_id
        return self.case_view(case_id)

    def confirm(self, case_id, actor, reason, **kw):
        return self.decide(case_id, "confirm", actor, reason, **kw)

    def hold(self, case_id, actor, reason, **kw):
        return self.decide(case_id, "hold", actor, reason, **kw)

    def release(self, case_id, actor, reason, **kw):
        return self.decide(case_id, "release", actor, reason, **kw)

    def close(self, case_id, actor, reason, **kw):
        return self.decide(case_id, "close", actor, reason, **kw)

    def reevaluate(
        self, case_id: str, *, actor: str = "system",
        at: datetime | str | None = None, spec: ReleaseSpec | None = None,
    ) -> dict[str, Any]:
        """Attach a fresh correlation (e.g. after late readings arrived)."""
        stream = f"case:{case_id}"
        events = self.store.read(stream)
        if not events:
            raise DomainError(f"unknown case {case_id}")
        case = Case.replay(events)
        at = _dt(at) if at is not None else self._clock()
        result = self.correlate(case.chain_id, case.start, case.end, spec=spec, as_of=at)
        self.store.append(
            stream, "CaseReevaluated", actor, result.reason(),
            {"evaluation": result.to_dict()}, at=at,
        )
        return self.case_view(case_id)

    # -- queries -----------------------------------------------------------
    def case_view(self, case_id: str) -> dict[str, Any]:
        events = self.store.read(f"case:{case_id}")
        if not events:
            raise DomainError(f"unknown case {case_id}")
        case = Case.replay(events)
        evaluation = case.last_evaluation or {}
        return {
            "case_id": case.case_id,
            "chain_id": case.chain_id,
            "window": {"start": case.start.isoformat(), "end": case.end.isoformat()},
            "status": case.status.value,
            "version": case.version,
            "opened_at": case.opened_at.isoformat() if case.opened_at else None,
            "updated_at": case.updated_at.isoformat() if case.updated_at else None,
            "status_reason": case.last_reason,
            "provable": bool(evaluation.get("provable")),
            "linked_cases": list(case.linked_cases),
            "evidence": evaluation.get("evidence", []),
            "blockers": evaluation.get("blockers", []),
            "evaluation": evaluation,
            "audit": [_audit_entry(ev, case.case_id) for ev in events],
        }

    def cases(self, chain_id: str | None = None, status: str | None = None) -> list[str]:
        out: list[str] = []
        for ev in self.store.read("cases:index"):
            if chain_id and ev.data["chain_id"] != chain_id:
                continue
            cid = ev.data["case_id"]
            if status and Case.replay(self.store.read(f"case:{cid}")).status.value != status:
                continue
            out.append(cid)
        return out

    # -- internals ---------------------------------------------------------
    def _find_active_case(self, chain_id: str, start: datetime, end: datetime) -> str | None:
        for ev in self.store.read("cases:index"):
            if ev.data["chain_id"] != chain_id:
                continue
            cid = ev.data["case_id"]
            case = Case.replay(self.store.read(f"case:{cid}"))
            if case.status is CaseStatus.CLOSED:
                continue
            if _dt(case.start) < end and start < _dt(case.end):  # half-open overlap
                return cid
        return None

    def _find_reading(self, chain_id: str, reading_id: str) -> Reading | None:
        for r in self.readings(chain_id):
            if r.reading_id == reading_id:
                return r
        return None

    @staticmethod
    def _coerce_reading(item: Reading | dict[str, Any]) -> Reading:
        if isinstance(item, Reading):
            return item
        if not isinstance(item, dict):
            raise DomainError("reading must be a Reading or mapping")
        try:
            return Reading(
                reading_id=str(item["reading_id"]),
                chain_id=str(item["chain_id"]),
                asset_id=str(item["asset_id"]),
                metric=Metric.coerce(item["metric"]),
                value=float(item["value"]),
                start=_dt(item["start"]),
                end=_dt(item["end"]),
                recorded_at=_dt(item.get("recorded_at") or item["end"]),
                unit=str(item.get("unit", "")),
                supersedes=item.get("supersedes"),
            )
        except KeyError as exc:
            raise DomainError(f"missing reading field: {exc.args[0]}")


def _reading_data(r: Reading) -> dict[str, Any]:
    return {
        "reading_id": r.reading_id,
        "chain_id": r.chain_id,
        "asset_id": r.asset_id,
        "metric": r.metric.value,
        "value": r.value,
        "start": r.start.isoformat(),
        "end": r.end.isoformat(),
        "recorded_at": r.recorded_at.isoformat(),
        "unit": r.unit,
        "supersedes": r.supersedes,
    }


def _reading_from_data(d: dict[str, Any]) -> Reading:
    return Reading(
        reading_id=d["reading_id"],
        chain_id=d["chain_id"],
        asset_id=d["asset_id"],
        metric=Metric(d["metric"]),
        value=float(d["value"]),
        start=_dt(d["start"]),
        end=_dt(d["end"]),
        recorded_at=_dt(d["recorded_at"]),
        unit=d.get("unit", ""),
        supersedes=d.get("supersedes"),
    )


def _evaluation_signature(evaluation: dict[str, Any]) -> tuple:
    """Content identity of an evaluation, ignoring the evaluation clock."""
    return (
        bool(evaluation.get("provable")),
        tuple((f["code"], f["start"], f["end"], tuple(f["reading_ids"]))
              for f in evaluation.get("findings", [])),
        tuple((e["reading_id"], e["status"], e["value"])
              for e in evaluation.get("evidence", [])),
        tuple(sorted((evaluation.get("spec") or {}).items())),
    )


def _audit_entry(ev: Event, case_id: str) -> dict[str, Any]:
    entry = {
        "seq": ev.seq,
        "at": ev.at.isoformat(),
        "event": ev.type,
        "actor": ev.actor,
        "reason": ev.reason,
    }
    if "from_status" in ev.data:
        entry["from_status"] = ev.data["from_status"]
        entry["to_status"] = ev.data["to_status"]
        entry["state_change"] = True
    else:
        entry["state_change"] = False
    return entry
