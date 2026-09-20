"""
05_investigation_intelligence.py
---------------------------------
C5: Security investigation - correlate evidence into a timeline, identify
    affected entities, recommend response actions.
C6: Security intelligence - define intelligence requirements, enrich
    evidence, produce operational + executive intelligence outputs.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # T13_SME_Security/
PROC = str(ROOT / "02_data" / "processed")
OUT = str(ROOT / "08_outputs")

alerts = pd.read_csv(f"{PROC}/alerts_scored.csv", parse_dates=["timestamp"])
auth = pd.read_csv(f"{OUT}/auth_anomaly_scored.csv", parse_dates=["timestamp"])
vulns = pd.read_csv(f"{PROC}/vuln_scans_clean.csv", parse_dates=["scan_date"])
tickets = pd.read_csv(f"{PROC}/tickets_scored.csv", parse_dates=["timestamp"])
client_anom = pd.read_csv(f"{PROC}/client_anomaly_scored.csv", parse_dates=["date"])

# ---------------------------------------------------------------------------
# C5: Build correlated incident timelines
# For each client-day flagged anomalous by the unsupervised model, pull
# together all evidence (high-risk alerts, anomalous logins, open vuln
# findings, related tickets) into one timeline entry.
# ---------------------------------------------------------------------------
RESPONSE_PLAYBOOK = {
    "malware":      "Isolate affected endpoint from network; run full AV/EDR scan; check backups before any restore.",
    "phishing":     "Force password reset for affected user; review mailbox rules for forwarding; notify all staff.",
    "access":       "Disable/suspend account pending verification; force MFA re-enrolment; review recent account activity.",
    "network":      "Review firewall logs for source; consider temporary IP block; validate rule change authorisation.",
    "vulnerability":"Prioritise patch/mitigation based on CVSS score and asset criticality; document compensating controls.",
    "false_alarm":  "Document as false positive; consider tuning rule threshold to reduce recurrence.",
}

incidents = []
flagged_days = client_anom[client_anom["is_anomalous_day"]]

for _, row in flagged_days.iterrows():
    cid, date = row["client_id"], pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    day_alerts = alerts[(alerts["client_id"] == cid) & (alerts["timestamp"].dt.date.astype(str) == date)]
    high_risk_alerts = day_alerts[day_alerts["tp_probability"] > 0.5].sort_values("tp_probability", ascending=False)
    day_auth = auth[(auth["client_id"] == cid) & (auth["timestamp"].dt.date.astype(str) == date) & (auth["is_anomalous_login"])]
    day_tickets = tickets[(tickets["client_id"] == cid) & (tickets["timestamp"].dt.date.astype(str) == date)]
    open_vulns = vulns[(vulns["client_id"] == cid) & (~vulns["patched"])]

    if len(high_risk_alerts) == 0 and len(day_auth) == 0:
        continue  # anomalous day but nothing concrete to correlate - skip for timeline

    # affected entities = union of src/dst IPs + assets from correlated evidence
    affected_entities = set(high_risk_alerts["src_ip"]).union(set(high_risk_alerts["dst_ip"]))
    if len(open_vulns):
        affected_entities.update(open_vulns["asset"].head(3).tolist())

    likely_category = day_tickets["predicted_category"].mode().iloc[0] if len(day_tickets) else "network"
    response = RESPONSE_PLAYBOOK.get(likely_category, "Escalate to senior analyst for manual triage.")

    incidents.append({
        "client_id": cid,
        "date": date,
        "anomaly_score": round(row["anomaly_score"], 3),
        "n_high_risk_alerts": len(high_risk_alerts),
        "top_alert_rules": high_risk_alerts["rule_id"].value_counts().head(3).to_dict(),
        "n_anomalous_logins": len(day_auth),
        "n_open_vulns_on_client": len(open_vulns),
        "n_related_tickets": len(day_tickets),
        "affected_entities_sample": list(affected_entities)[:5],
        "likely_category": likely_category,
        "recommended_response": response,
        "evidence_alert_ids": high_risk_alerts["alert_id"].head(10).tolist(),
    })

incident_timeline = pd.DataFrame(incidents)
if len(incident_timeline):
    incident_timeline = incident_timeline.sort_values("anomaly_score", ascending=False)
incident_timeline.to_csv(f"{OUT}/incident_timeline.csv", index=False)
print(f"=== C5: Investigation ===")
print(f"Correlated {len(incident_timeline)} incident-worthy client-days from {len(flagged_days)} anomalous days flagged.")
if len(incident_timeline):
    print(incident_timeline[["client_id","date","n_high_risk_alerts","n_anomalous_logins","likely_category"]].head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# C6: Security intelligence outputs
# Priority Intelligence Requirements (PIRs) -> operational + executive outputs
# ---------------------------------------------------------------------------
PIRS = [
    "Which clients show alert or login behaviour deviating from their own baseline today?",
    "Which unpatched vulnerabilities exist on assets belonging to clients with recent high-risk alerts?",
    "Which ticket categories are trending upward across the client base this period?",
]

# Operational output: ranked ticket/alert queue for the on-shift analyst
alerts_ranked = alerts.sort_values("tp_probability", ascending=False).head(30)
operational_queue = alerts_ranked[["alert_id","client_id","timestamp","description","severity",
                                     "mitre_tag","tp_probability"]].copy()
operational_queue["priority_rank"] = range(1, len(operational_queue) + 1)
operational_queue.to_csv(f"{OUT}/operational_priority_queue.csv", index=False)

# Executive output: per-client risk digest
exec_digest = alerts.groupby("client_id").agg(
    total_alerts=("alert_id","count"),
    mean_tp_probability=("tp_probability","mean"),
    high_severity_alerts=("severity", lambda s: (s=="high").sum()),
).reset_index()
exec_digest = exec_digest.merge(
    vulns[~vulns["patched"]].groupby("client_id").size().rename("open_vulns").reset_index(),
    on="client_id", how="left"
).fillna({"open_vulns": 0})
exec_digest = exec_digest.merge(
    incident_timeline.groupby("client_id").size().rename("flagged_incident_days").reset_index() if len(incident_timeline) else pd.DataFrame(columns=["client_id","flagged_incident_days"]),
    on="client_id", how="left"
).fillna({"flagged_incident_days": 0})

# Composite score normalised 0-1 per component (min-max across clients), then
# tiered by RANK relative to the other clients rather than fixed thresholds -
# with only 7 clients, fixed absolute cutoffs don't discriminate meaningfully.
def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0

composite = (
    minmax(exec_digest["mean_tp_probability"]) * 0.5
    + minmax(exec_digest["open_vulns"]) * 0.3
    + minmax(exec_digest["flagged_incident_days"]) * 0.2
)
exec_digest["composite_risk_score"] = composite.round(3)
exec_digest["risk_tier"] = pd.qcut(composite.rank(method="first"), q=3, labels=["Low", "Medium", "High"])
exec_digest.to_csv(f"{OUT}/executive_risk_digest.csv", index=False)

with open(f"{OUT}/priority_intelligence_requirements.json", "w") as f:
    json.dump({"PIRs": PIRS}, f, indent=2)

print(f"\n=== C6: Security Intelligence ===")
print("Operational priority queue (top 30 alerts) saved.")
print("Executive risk digest by client:")
print(exec_digest[["client_id","total_alerts","mean_tp_probability","open_vulns","flagged_incident_days","composite_risk_score","risk_tier"]].to_string(index=False))
