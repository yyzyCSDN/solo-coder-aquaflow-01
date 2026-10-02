import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.api import AquaFlowService
from aquaflow.treatment.chlorine_desk import (
    AuditEvent,
    ChlorineDesk,
    Policy,
    Reading,
    evaluate_window,
    replay_events,
)

BASE = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
H = timedelta(hours=1)
M = timedelta(minutes=1)
END = BASE + H


def rd(chain="T1", asset="CLW-1", metric="chlorine", start=BASE, end=None, value=1.0, seq=0, received=None):
    return Reading(chain, asset, metric, start, end or start + H, value, seq, received)


def clean_set(received=None):
    return [
        rd(metric="chlorine", value=1.0, received=received),
        rd(metric="temperature", value=20.0, received=received),
        rd(metric="contact_time", value=60.0, received=received),
    ]


class EvaluationTests(unittest.TestCase):
    def test_clean_window_is_provable_without_case(self):
        desk = ChlorineDesk()
        desk.ingest(clean_set(), received_at=BASE + H + 5 * M)
        result = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        self.assertTrue(result["provable"])
        self.assertEqual(result["ct"], 60.0)
        self.assertIsNone(result["case_id"])
        self.assertEqual(desk.cases(), [])

    def test_missing_metric_opens_case_with_gap_evidence(self):
        desk = ChlorineDesk()
        desk.ingest(clean_set()[:2], received_at=BASE + H + 5 * M)  # no contact_time
        result = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        self.assertFalse(result["provable"])
        self.assertIn("missing_coverage:contact_time", result["reasons"])
        case = desk.case_view(result["case_id"])
        self.assertEqual(case["state"], "opened")
        self.assertIn("missing_coverage:contact_time", case["reason"])
        self.assertEqual(case["evidence"]["gaps"], [{"metric": "contact_time", "start": BASE.isoformat(), "end": END.isoformat()}])
        self.assertEqual(case["evidence"]["revision"], 1)

    def test_missing_segment_inside_window(self):
        desk = ChlorineDesk()
        rows = clean_set()
        rows[0] = rd(metric="chlorine", start=BASE, end=BASE + 30 * M, value=1.0)
        rows.append(rd(metric="chlorine", start=BASE + 45 * M, end=END, value=1.0))
        desk.ingest(rows, received_at=BASE + H + 5 * M)
        result = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        self.assertFalse(result["provable"])
        self.assertIn("missing_coverage:chlorine", result["reasons"])
        gaps = result["gaps"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["start"], (BASE + 30 * M).isoformat())
        self.assertEqual(gaps[0]["end"], (BASE + 45 * M).isoformat())

    def test_overlapping_conflict_opens_case(self):
        desk = ChlorineDesk()
        rows = clean_set()
        rows.append(rd(metric="chlorine", start=BASE + 30 * M, value=2.0, seq=2))
        desk.ingest(rows, received_at=BASE + H + 5 * M)
        result = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        self.assertFalse(result["provable"])
        self.assertIn("conflicting_readings:chlorine@CLW-1", result["reasons"])
        conflict = result["conflicts"][0]
        self.assertEqual(conflict["asset"], "CLW-1")
        self.assertEqual(conflict["start"], (BASE + 30 * M).isoformat())
        self.assertEqual(conflict["values"], [1.0, 2.0])

    def test_overlap_within_tolerance_is_not_conflict(self):
        desk = ChlorineDesk()
        rows = clean_set()
        rows.append(rd(metric="chlorine", start=BASE + 30 * M, value=1.01, seq=2))
        desk.ingest(rows, received_at=BASE + H + 5 * M)
        result = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        self.assertTrue(result["provable"])

    def test_ct_below_required(self):
        desk = ChlorineDesk()
        rows = clean_set()
        rows[2] = rd(metric="contact_time", value=5.0)
        desk.ingest(rows, received_at=BASE + H + 5 * M)
        result = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        self.assertFalse(result["provable"])
        self.assertIn("ct_below_required", result["reasons"])

    def test_residual_out_of_bounds(self):
        for value, reason in ((0.1, "residual_below_min"), (5.0, "residual_above_max")):
            desk = ChlorineDesk()
            rows = clean_set()
            rows[0] = rd(metric="chlorine", value=value)
            desk.ingest(rows, received_at=BASE + H + 5 * M)
            result = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
            self.assertFalse(result["provable"])
            self.assertIn(reason, result["reasons"])

    def test_pure_function_matches_desk(self):
        result = evaluate_window(clean_set(), "T1", BASE, END, Policy())
        self.assertTrue(result.provable)
        self.assertEqual(result.reading_count, 3)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.desk = ChlorineDesk()
        self.desk.ingest(clean_set()[:2], received_at=BASE + H + 5 * M)
        self.case_id = self.desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)["case_id"]

    def test_decision_flow_replays_in_order(self):
        t0 = BASE + H + 15 * M
        self.desk.decide(self.case_id, "confirm", "operator verified low contact-time signal", actor="op-1", at=t0)
        self.desk.decide(self.case_id, "hold", "basin outlet closed", actor="op-1", at=t0 + 5 * M)
        self.desk.decide(self.case_id, "release", "retest proves CT", actor="op-2", at=t0 + 10 * M)
        view = self.desk.decide(self.case_id, "close", "paperwork filed", actor="op-2", at=t0 + 15 * M)
        self.assertEqual(view["state"], "closed")
        self.assertEqual(view["reason"], "paperwork filed")
        trail = self.desk.replay(self.case_id)
        self.assertEqual([s["state"] for s in trail], ["opened", "confirmed", "held", "released", "closed"])
        self.assertEqual([s["reason"] for s in trail][-1], "paperwork filed")
        self.assertTrue(all(a["seq"] < b["seq"] for a, b in zip(trail, trail[1:])))
        decisions = [e for e in view["audit"] if e["kind"] == "decision"]
        self.assertEqual(len(decisions), 4)
        self.assertEqual(decisions[0]["from_state"], "opened")
        self.assertEqual(decisions[0]["to_state"], "confirmed")
        self.assertEqual(decisions[0]["actor"], "op-1")

    def test_state_cannot_regress(self):
        t0 = BASE + H + 15 * M
        self.desk.decide(self.case_id, "confirm", "confirmed", at=t0)
        with self.assertRaises(ValueError):
            self.desk.decide(self.case_id, "confirm", "again", at=t0 + M)
        self.desk.decide(self.case_id, "release", "release", at=t0 + 2 * M)
        with self.assertRaises(ValueError):
            self.desk.decide(self.case_id, "hold", "back to hold", at=t0 + 3 * M)
        self.desk.decide(self.case_id, "close", "close", at=t0 + 4 * M)
        with self.assertRaises(ValueError):
            self.desk.decide(self.case_id, "confirm", "reopen", at=t0 + 5 * M)
        with self.assertRaises(ValueError):
            self.desk.decide(self.case_id, "reopen", "bad action", at=t0 + 6 * M)
        self.assertEqual(self.desk.allowed_actions(self.case_id), [])

    def test_decision_requires_reason_and_forward_time(self):
        t0 = BASE + H + 15 * M
        with self.assertRaises(ValueError):
            self.desk.decide(self.case_id, "confirm", "  ", at=t0)
        self.desk.decide(self.case_id, "confirm", "ok", at=t0)
        with self.assertRaises(ValueError):
            self.desk.decide(self.case_id, "hold", "backwards clock", at=t0 - M)

    def test_unknown_case(self):
        with self.assertRaises(KeyError):
            self.desk.case_view("CL-9999")
        with self.assertRaises(KeyError):
            self.desk.decide("CL-9999", "confirm", "x")

    def test_replay_events_rejects_tampered_history(self):
        at = BASE + H
        opened = AuditEvent(1, "opened", at, "system", None, "opened", "r", {})
        confirmed = AuditEvent(2, "decision", at + M, "op", "opened", "confirmed", "r", {})
        regressed = AuditEvent(3, "decision", at + 2 * M, "op", "confirmed", "opened", "r", {})
        self.assertEqual([s["state"] for s in replay_events([opened, confirmed])], ["opened", "confirmed"])
        with self.assertRaises(ValueError):
            replay_events([opened, confirmed, regressed])
        backwards = AuditEvent(3, "decision", at - M, "op", "confirmed", "held", "r", {})
        with self.assertRaises(ValueError):
            replay_events([opened, confirmed, backwards])


class LateEvidenceTests(unittest.TestCase):
    def test_late_reading_opens_case_without_rewriting(self):
        desk = ChlorineDesk()
        desk.ingest(clean_set(received=BASE + H + 5 * M), received_at=BASE + H + 5 * M)
        clean = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        self.assertTrue(clean["provable"])
        report = desk.ingest([rd(metric="chlorine", value=0.05, received=BASE + H + 30 * M)], received_at=BASE + H + 30 * M)
        self.assertEqual(len(report["late"]), 1)
        self.assertEqual(len(report["cases_opened"]), 1)
        case = desk.case_view(report["cases_opened"][0])
        self.assertEqual(case["state"], "opened")
        self.assertIn("residual_below_min", case["reason"])
        self.assertTrue(case["audit"][0]["details"]["previously_provable"])
        self.assertTrue(clean["provable"])  # the original evaluation is untouched

    def test_late_reading_appends_revision_to_open_case(self):
        desk = ChlorineDesk()
        desk.ingest([r for r in clean_set(received=BASE + H + 5 * M) if r.metric != "contact_time"], received_at=BASE + H + 5 * M)
        case_id = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)["case_id"]
        report = desk.ingest([rd(metric="contact_time", value=60.0, received=BASE + H + 20 * M)], received_at=BASE + H + 20 * M)
        self.assertEqual(report["cases_revised"], [case_id])
        self.assertEqual(report["cases_opened"], [])
        case = desk.case_view(case_id)
        self.assertEqual(case["state"], "opened")  # evidence changed, state did not
        self.assertEqual(case["evidence"]["revision"], 2)
        self.assertTrue(case["evidence"]["provable"])
        self.assertEqual(case["reason"], "missing_coverage:contact_time")  # original open reason preserved
        self.assertEqual([e["kind"] for e in case["audit"]], ["opened", "evidence"])
        self.assertEqual(case["audit"][1]["details"]["revision"], 2)
        late_flags = {r["metric"]: r["late"] for r in case["evidence"]["readings"]}
        self.assertTrue(late_flags["contact_time"])
        self.assertFalse(late_flags["chlorine"])

    def test_late_contradiction_after_release_opens_followup(self):
        desk = ChlorineDesk()
        desk.ingest([r for r in clean_set(received=BASE + H + 5 * M) if r.metric != "contact_time"], received_at=BASE + H + 5 * M)
        case_id = desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)["case_id"]
        desk.ingest([rd(metric="contact_time", value=60.0, received=BASE + H + 20 * M)], received_at=BASE + H + 20 * M)
        desk.decide(case_id, "release", "evidence complete, CT proven", actor="op-1", at=BASE + H + 25 * M)
        report = desk.ingest([rd(metric="chlorine", value=0.05, start=BASE + 15 * M, end=BASE + 45 * M, received=BASE + H + 40 * M)], received_at=BASE + H + 40 * M)
        self.assertEqual(report["cases_revised"], [case_id])
        self.assertEqual(len(report["cases_opened"]), 1)
        original = desk.case_view(case_id)
        self.assertEqual(original["state"], "released")  # decided state is never rewritten
        self.assertEqual(original["evidence"]["revision"], 3)
        self.assertFalse(original["evidence"]["provable"])
        note = original["audit"][-1]
        self.assertEqual(note["kind"], "evidence")
        self.assertTrue(note["details"]["contradicts_decision"])
        followup = desk.case_view(report["cases_opened"][0])
        self.assertEqual(followup["related_case"], case_id)
        self.assertEqual(followup["state"], "opened")
        self.assertEqual([s["state"] for s in desk.replay(case_id)], ["opened", "released"])

    def test_duplicate_ingest_is_idempotent(self):
        desk = ChlorineDesk()
        first = desk.ingest(clean_set(), received_at=BASE + H + 5 * M)
        second = desk.ingest(clean_set(), received_at=BASE + H + 6 * M)
        self.assertEqual(first["accepted"], 3)
        self.assertEqual(second["accepted"], 0)
        self.assertEqual(second["duplicates"], 3)
        desk.evaluate("T1", BASE, END, at=BASE + H + 10 * M)
        third = desk.ingest(clean_set(), received_at=BASE + H + 30 * M)
        self.assertEqual(third["cases_revised"], [])
        self.assertEqual(third["cases_opened"], [])


class ServiceTests(unittest.TestCase):
    def test_service_surface_end_to_end(self):
        svc = AquaFlowService()
        svc.ingest_chlorine_readings(
            [
                {"asset": "CLW-1", "metric": "chlorine", "start": BASE, "end": END, "value": 1.0},
                {"asset": "CLW-1", "metric": "temperature", "start": BASE, "end": END, "value": 20.0},
            ],
            received_at=BASE + H + 5 * M,
            chain="T1",
        )
        result = svc.evaluate_chlorine_window("T1", BASE, END, at=BASE + H + 10 * M)
        case_id = result["case_id"]
        self.assertIsNotNone(case_id)
        view = svc.decide_chlorine_case(case_id, "confirm", "operator confirmed", actor="op-9", at=BASE + H + 15 * M)
        self.assertEqual(view["state"], "confirmed")
        self.assertEqual(svc.chlorine_case(case_id)["reason"], "operator confirmed")
        self.assertEqual([s["state"] for s in svc.replay_chlorine_case(case_id)], ["opened", "confirmed"])
        self.assertEqual(len(svc.chlorine_cases()), 1)
        with self.assertRaises(ValueError):
            svc.decide_chlorine_case(case_id, "confirm", "again", at=BASE + H + 16 * M)

    def test_iso_and_naive_timestamps(self):
        svc = AquaFlowService()
        report = svc.ingest_chlorine_readings(
            [{"asset": "A1", "metric": "chlorine", "start": "2026-10-01T08:00:00Z", "end": "2026-10-01T09:00:00+00:00", "value": 0.9}],
            received_at="2026-10-01T09:05:00Z",
            chain="T1",
        )
        self.assertEqual(report["accepted"], 1)
        naive = datetime(2026, 10, 1, 8, 0)
        svc.ingest_chlorine_readings([{"asset": "A1", "metric": "temperature", "start": naive, "end": naive + H, "value": 20.0}], chain="T1")
        result = svc.evaluate_chlorine_window("T1", "2026-10-01T08:00:00Z", "2026-10-01T09:00:00Z", at="2026-10-01T09:10:00Z")
        self.assertIn("missing_coverage:contact_time", result["reasons"])

    def test_workflow_release_proof(self):
        svc = AquaFlowService()
        payload = {
            "readings": [
                {"asset": "CLW-1", "metric": "chlorine", "start": "2026-10-01T08:00:00Z", "end": "2026-10-01T09:00:00Z", "value": 1.0},
                {"asset": "CLW-1", "metric": "temperature", "start": "2026-10-01T08:00:00Z", "end": "2026-10-01T09:00:00Z", "value": 20.0},
                {"asset": "CLW-1", "metric": "contact_time", "start": "2026-10-01T08:00:00Z", "end": "2026-10-01T09:00:00Z", "value": 60.0},
            ],
            "chain": "T1",
            "start": "2026-10-01T08:00:00Z",
            "end": "2026-10-01T09:00:00Z",
        }
        result = svc.run_workflow("chlorine_release_proof", payload)
        self.assertEqual(result["decision"], "release")
        self.assertEqual(result["value"], 60.0)
        self.assertEqual(result["audit"]["contract"], "v1")
        payload["readings"] = payload["readings"][:2]
        result = svc.run_workflow("chlorine_release_proof", payload)
        self.assertEqual(result["decision"], "open_case")
        self.assertIn("missing_coverage:contact_time", result["evidence"]["reasons"])

    def test_desk_policy_is_configurable(self):
        svc = AquaFlowService()
        desk = svc.chlorine_desk("strict", required_ct=100.0)
        self.assertEqual(desk.policy.required_ct, 100.0)
        self.assertIs(svc.chlorine_desk("strict"), desk)
        self.assertNotEqual(svc.chlorine_desk("other").policy.required_ct, 100.0)


if __name__ == "__main__":
    unittest.main()
