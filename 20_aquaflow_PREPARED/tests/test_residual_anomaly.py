import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.treatment.residual_anomaly import (
    Correlation,
    Case,
    CaseStatus,
    DomainError,
    Metric,
    Reading,
    ReleaseSpec,
    ResidualAnomalyService,
    correlate,
)

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)


def at(mins):
    return T0 + timedelta(minutes=mins)


def svc():
    clock = {"t": T0 + timedelta(hours=2)}
    s = ResidualAnomalyService(clock=lambda: clock["t"])
    s.clock = clock  # test helper
    return s


def full_window(s, prefix="r", recorded=None, residual=0.8, temp=20.0, ct=25.0):
    recorded = recorded or at(60)
    return s.ingest([
        {"reading_id": f"{prefix}-cl", "chain_id": "C1", "asset_id": "clearwell",
         "metric": "residual", "value": residual, "start": at(0), "end": at(30),
         "recorded_at": recorded, "unit": "mg/L"},
        {"reading_id": f"{prefix}-tp", "chain_id": "C1", "asset_id": "probe",
         "metric": "temperature", "value": temp, "start": at(0), "end": at(30),
         "recorded_at": recorded, "unit": "C"},
        {"reading_id": f"{prefix}-ct", "chain_id": "C1", "asset_id": "baffle",
         "metric": "contact_time", "value": ct, "start": at(0), "end": at(30),
         "recorded_at": recorded, "unit": "min"},
    ])


class CorrelationTests(unittest.TestCase):
    def test_provable_full_evidence_opens_no_case(self):
        s = svc()
        full_window(s)
        out = s.open_if_unprovable("C1", at(0), at(30), actor="ops")
        self.assertFalse(out["created"])
        self.assertTrue(out["provable"])
        self.assertIsNone(out["case_id"])

    def test_missing_fragment_is_blocker_and_opens_case(self):
        s = svc()
        # residual + temperature only; contact time arrives late
        s.ingest([
            Reading("cl", "C1", "w", Metric.RESIDUAL, 0.8, at(0), at(30), at(60)),
            Reading("tp", "C1", "p", Metric.TEMPERATURE, 20, at(0), at(30), at(60)),
        ])
        out = s.open_if_unprovable("C1", at(0), at(30), actor="ops")
        self.assertTrue(out["created"])
        view = s.case_view(out["case_id"])
        self.assertEqual(view["status"], "open")
        codes = {b["code"] for b in view["blockers"]}
        self.assertIn("missing_segment", codes)
        self.assertEqual(len(view["audit"]), 1)
        self.assertIn("contact_time", view["status_reason"])

    def test_late_reading_does_not_rewrite_history(self):
        s = svc()
        s.ingest([
            Reading("cl", "C1", "w", Metric.RESIDUAL, 0.8, at(0), at(30), at(60)),
            Reading("tp", "C1", "p", Metric.TEMPERATURE, 20, at(0), at(30), at(60)),
        ])
        opened = s.open_if_unprovable("C1", at(0), at(30), actor="ops", at=at(60))
        cid = opened["case_id"]
        # operator confirms based on the evidence available then
        s.confirm(cid, "may", "confirmed missing CT telemetry", at=at(61))
        self.assertEqual(s.case_view(cid)["status"], "confirmed")

        # contact-time reading arrives late (covers the old window)
        s.ingest([Reading("ct", "C1", "b", Metric.CONTACT_TIME, 25, at(0), at(30), at(120))])
        # as-of the old decision time, the proof is still missing: history unchanged
        old = s.correlate("C1", at(0), at(30), as_of=at(61))
        self.assertFalse(old.provable)
        self.assertEqual(old.late_count, 1)
        # a fresh evaluation sees the new evidence; release becomes possible
        view = s.reevaluate(cid, at=at(121))
        self.assertTrue(view["provable"])
        # confirmation audit entry remains in place (no rewrite)
        self.assertEqual([e["event"] for e in view["audit"]][:2],
                         ["CaseOpened", "CaseConfirmed"])
        s.release(cid, "may", "late CT proves 25 min contact", at=at(122))
        self.assertEqual(s.case_view(cid)["status"], "released")

    def test_residual_out_of_range_blocks_release(self):
        s = svc()
        full_window(s, residual=5.0)  # above 4.0
        out = s.open_if_unprovable("C1", at(0), at(30), actor="ops")
        self.assertTrue(out["created"])
        with self.assertRaises(DomainError):
            s.release(out["case_id"], "ops", "try anyway")

    def test_insufficient_contact_time(self):
        s = svc()
        full_window(s, ct=10.0)
        c = s.correlate("C1", at(0), at(30))
        self.assertFalse(c.provable)
        self.assertIn("insufficient_contact_time", {f.code for f in c.blockers})

    def test_conflicting_overlap_is_blocker_duplicate_is_info(self):
        s = svc()
        full_window(s)
        # contradictory residual overlapping the same interval
        s.ingest([Reading("cl2", "C1", "w2", Metric.RESIDUAL, 0.3, at(0), at(30), at(62))])
        c = s.correlate("C1", at(0), at(30))
        self.assertFalse(c.provable)
        self.assertIn("conflicting_overlap", {f.code for f in c.blockers})

        # identical duplicate overlap is informational only
        s2 = svc()
        full_window(s2)
        s2.ingest([Reading("cl2", "C1", "w2", Metric.RESIDUAL, 0.8, at(0), at(30), at(62))])
        c2 = s2.correlate("C1", at(0), at(30))
        self.assertTrue(c2.provable)
        self.assertIn("duplicate_overlap", {f.code for f in c2.findings})

    def test_supersession_keeps_original_on_record(self):
        s = svc()
        # bad residual first, plus valid temperature and contact time
        s.ingest([
            Reading("cl-bad", "C1", "w", Metric.RESIDUAL, 5.0, at(0), at(30), at(60)),
            Reading("tp", "C1", "p", Metric.TEMPERATURE, 20, at(0), at(30), at(60)),
            Reading("ct", "C1", "b", Metric.CONTACT_TIME, 25, at(0), at(30), at(60)),
        ])
        bad = s.open_if_unprovable("C1", at(0), at(30), actor="ops", at=at(61))
        cid = bad["case_id"]
        # corrected reading arrives; original reading is never deleted/edited
        s.ingest([Reading("cl-fix", "C1", "w", Metric.RESIDUAL, 0.8, at(0), at(30),
                          at(90), supersedes="cl-bad")])
        view = s.reevaluate(cid, at=at(91))
        statuses = {e["reading_id"]: e["status"] for e in view["evidence"]}
        self.assertEqual(statuses["cl-bad"], "corrected_reading")
        self.assertEqual(statuses["cl-fix"], "proven")
        self.assertTrue(view["provable"])
        with self.assertRaises(DomainError):
            s.ingest([Reading("x", "C1", "w", Metric.RESIDUAL, 0.8, at(0), at(1),
                              at(95), supersedes="does-not-exist")])


class CaseLifecycleTests(unittest.TestCase):
    def _open(self, s):
        s.ingest([
            Reading("cl", "C1", "w", Metric.RESIDUAL, 0.8, at(0), at(30), at(60)),
            Reading("tp", "C1", "p", Metric.TEMPERATURE, 20, at(0), at(30), at(60)),
        ])
        return s.open_if_unprovable("C1", at(0), at(30), actor="ops", at=at(61))["case_id"]

    def test_full_lifecycle_forward_only(self):
        s = svc()
        cid = self._open(s)
        s.confirm(cid, "may", "anomaly real", at=at(62))
        s.hold(cid, "may", "keep batch held", at=at(63))
        # still missing CT -> release refused
        with self.assertRaises(DomainError):
            s.release(cid, "may", "optimistic", at=at(64))
        s.ingest([Reading("ct", "C1", "b", Metric.CONTACT_TIME, 25, at(0), at(30), at(70))])
        s.release(cid, "may", "CT proven", at=at(71))
        s.close(cid, "may", "batch released and filed", at=at(72))
        view = s.case_view(cid)
        self.assertEqual(view["status"], "closed")
        self.assertEqual(
            [e["to_status"] for e in view["audit"] if e["state_change"]],
            ["confirmed", "held", "released", "closed"],
        )

    def test_status_cannot_regress(self):
        s = svc()
        cid = self._open(s)
        s.confirm(cid, "may", "real", at=at(62))
        for action in ("confirm",):
            with self.assertRaises(DomainError):
                s.decide(cid, action, "may", "again", at=at(63))
        s.close(cid, "may", "filed", at=at(80))
        for action in ("confirm", "hold", "release", "close"):
            with self.assertRaises(DomainError):
                s.decide(cid, action, "may", "post-close", at=at(81))

    def test_event_time_cannot_move_backwards(self):
        s = svc()
        cid = self._open(s)
        s.confirm(cid, "may", "real", at=at(70))
        with self.assertRaises(DomainError):
            s.hold(cid, "may", "earlier", at=at(69))

    def test_replay_rebuilds_state_and_audit(self):
        s = svc()
        cid = self._open(s)
        s.confirm(cid, "may", "real", at=at(62))
        s.hold(cid, "may", "held", at=at(63))
        events = s.store.read(f"case:{cid}")
        rebuilt = Case.replay(events)
        self.assertEqual(rebuilt.status, CaseStatus.HELD)
        self.assertEqual(rebuilt.version, 2)
        with self.assertRaises(DomainError):
            Case.replay(events[:1] + [events[2], events[1]])  # out of order

    def test_decision_requires_reason(self):
        s = svc()
        cid = self._open(s)
        with self.assertRaises(DomainError):
            s.confirm(cid, "may", "  ", at=at(62))

    def test_overlapping_window_reuses_active_case(self):
        s = svc()
        cid1 = self._open(s)
        again = s.open_if_unprovable("C1", at(10), at(40), actor="ops", at=at(65))
        self.assertFalse(again["created"])
        self.assertEqual(again["case_id"], cid1)

    def test_idempotent_command(self):
        s = svc()
        s.ingest([
            Reading("cl", "C1", "w", Metric.RESIDUAL, 0.8, at(0), at(30), at(60)),
            Reading("tp", "C1", "p", Metric.TEMPERATURE, 20, at(0), at(30), at(60)),
        ])
        a = s.open_if_unprovable("C1", at(0), at(30), command_id="cmd-1", at=at(61))
        b = s.open_if_unprovable("C1", at(0), at(30), command_id="cmd-1", at=at(62))
        self.assertEqual(a["case_id"], b["case_id"])

    def test_optimistic_concurrency(self):
        s = svc()
        cid = self._open(s)
        s.store.append(f"case:{cid}", "CaseConfirmed", "may", "x",
                       {"from_status": "open", "to_status": "confirmed"},
                       at=at(62), expected_version=1)
        with self.assertRaises(DomainError):
            s.store.append(f"case:{cid}", "CaseHeld", "may", "stale",
                           {"from_status": "confirmed", "to_status": "held"},
                           at=at(63), expected_version=1)


class ValidationTests(unittest.TestCase):
    def test_bad_reading_rejected(self):
        with self.assertRaises(DomainError):
            Reading("x", "C", "a", Metric.RESIDUAL, -1, at(0), at(1), at(1))
        with self.assertRaises(DomainError):
            Reading("x", "C", "a", Metric.RESIDUAL, 0.5, at(2), at(1), at(3))

    def test_bad_window_rejected(self):
        s = svc()
        with self.assertRaises(DomainError):
            correlate("C", at(1), at(0), [])


if __name__ == "__main__":
    unittest.main()
