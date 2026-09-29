"""Reproducible local analysis of fixed-model binary test-set predictions.

No training, network access, or threshold selection. See --help and the schema
document next to this file. All proportions are on the 0--1 scale; paired
differences are also supplied in percentage points.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METRICS = ("ACC", "AUC", "Specificity", "Sensitivity", "BA", "PPV", "NPV", "F1", "AP")
TEST_METRICS = ("AUC", "ACC")
FAMILY_SIZE = 52
DEFAULT_SEED = 20260920
DEFAULT_BOOTSTRAP = 20000


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def resolve_path(base, path):
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "NA" if v is None else v for k, v in clean(row).items()})


def safe_divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


class MetricComputer:
    """Sort once, then evaluate record multiplicities in vectorized batches.

    Tied scores are aggregated before AUC and AP calculation. AP is the
    noninterpolated precision-weighted recall increment used by sklearn.
    """

    def __init__(self, y, score, pred_class):
        self.y = np.asarray(y, dtype=int)
        self.score = np.asarray(score, dtype=float)
        self.pred_class = np.asarray(pred_class, dtype=int)
        if not (self.y.ndim == self.score.ndim == self.pred_class.ndim == 1):
            raise ValueError("metric inputs must be one-dimensional")
        if not (len(self.y) == len(self.score) == len(self.pred_class)) or not len(self.y):
            raise ValueError("metric inputs must have the same nonzero length")
        self.order = np.argsort(self.score, kind="stable")
        sorted_score = self.score[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
        self.tp = (self.y == 1) & (self.pred_class == 1)
        self.tn = (self.y == 0) & (self.pred_class == 0)
        self.fp = (self.y == 0) & (self.pred_class == 1)
        self.fn = (self.y == 1) & (self.pred_class == 0)

    def evaluate(self, weights):
        weights = np.atleast_2d(np.asarray(weights, dtype=float))
        if weights.shape[1] != len(self.y) or (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError("invalid record weights")
        tp = weights[:, self.tp].sum(axis=1)
        tn = weights[:, self.tn].sum(axis=1)
        fp = weights[:, self.fp].sum(axis=1)
        fn = weights[:, self.fn].sum(axis=1)
        pos, neg = tp + fn, tn + fp
        ordered = weights[:, self.order]
        positive = np.add.reduceat(ordered * self.y[self.order], self.starts, axis=1)
        negative = np.add.reduceat(ordered * (1 - self.y[self.order]), self.starts, axis=1)
        auc = safe_divide((positive * (np.cumsum(negative, axis=1) - negative / 2)).sum(axis=1), pos * neg)
        p_desc, n_desc = positive[:, ::-1], negative[:, ::-1]
        tp_cumulative = np.cumsum(p_desc, axis=1)
        all_cumulative = np.cumsum(p_desc + n_desc, axis=1)
        # Score groups absent from a draw have zero recall increment. Their
        # precision is immaterial and is explicitly assigned zero here.
        precision = np.divide(tp_cumulative, all_cumulative, out=np.zeros_like(tp_cumulative), where=all_cumulative > 0)
        ap = safe_divide((p_desc * precision).sum(axis=1), pos)
        specificity, sensitivity = safe_divide(tn, neg), safe_divide(tp, pos)
        return np.column_stack((
            safe_divide(tp + tn, pos + neg), auc, specificity, sensitivity,
            (specificity + sensitivity) / 2, safe_divide(tp, tp + fp),
            safe_divide(tn, tn + fn), safe_divide(2 * tp, 2 * tp + fp + fn), ap,
        ))

    def point(self):
        return self.evaluate(np.ones(len(self.y)))[0]


def build_cohort(mapping, config):
    dataset = config.get("mapping_dataset", config["id"])
    split = config.get("mapping_split", "Test")
    selected = {
        key: value for key, value in mapping.items()
        if value.get("dataset") == dataset and value.get("split") == split
    }
    if "expected_ids" in config:
        expected = set(config["expected_ids"])
        if expected != set(selected):
            raise ValueError(f"cohort {config['id']}: expected_ids do not match selected mapping")
    if not selected:
        raise ValueError(f"cohort {config['id']}: mapping contains no selected records")
    ids = sorted(selected)
    y, record_groups, sources = [], [], []
    labels_by_group = {}
    for record_id in ids:
        item = selected[record_id]
        label = item.get("label")
        if label not in (0, 1):
            raise ValueError(f"mapping label invalid for {record_id}")
        group = item.get("patient_group") or item.get("patient_id")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"unmapped patient group for {record_id}")
        if group in labels_by_group and labels_by_group[group] != label:
            raise ValueError(f"conflicting labels in patient group {group}")
        labels_by_group[group] = label
        y.append(label)
        record_groups.append(group)
        sources.append(item.get("source", ""))
    if set(y) != {0, 1}:
        raise ValueError(f"cohort {config['id']}: both label strata required")
    groups = sorted(labels_by_group)
    lookup = {key: index for index, key in enumerate(groups)}
    return {
        "id": config["id"], "ids": ids, "y": np.asarray(y, dtype=int),
        "groups": groups, "group_y": np.asarray([labels_by_group[g] for g in groups]),
        "record_groups": record_groups, "record_group_index": np.asarray([lookup[g] for g in record_groups]),
        "sources": sources, "id_set_sha256": sha256_bytes("\n".join(ids).encode()),
    }


def cluster_weights(cohort, bootstrap, seed):
    """One multinomial multiplicity per patient, shared by all their scans."""
    rng = np.random.default_rng(seed)
    count = len(cohort["groups"])
    if count > np.iinfo(np.uint16).max:
        raise ValueError("cohort has too many clusters for uint16 storage")
    weights = np.zeros((bootstrap, count), dtype=np.uint16)
    for label in (0, 1):
        stratum = np.flatnonzero(cohort["group_y"] == label)
        draws = rng.multinomial(len(stratum), np.full(len(stratum), 1 / len(stratum)), size=bootstrap)
        weights[:, stratum] = draws
    return weights


def parse_binary(value, column, record_id):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {column} for {record_id}: {value!r}") from None
    if number not in (0, 1):
        raise ValueError(f"nonbinary {column} for {record_id}: {value!r}")
    return int(number)


def load_prediction(item, cohort, manifest_dir):
    path = resolve_path(manifest_dir, item["path"])
    raw = path.read_bytes()
    actual_hash = sha256_bytes(raw)
    expected_hash = item.get("sha256", "").lower()
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(f"SHA256 mismatch or absent: {path}")
    columns = {"id": "ID", "label": "idh_truth", "score": "pred", "class": "pred_class"}
    columns.update(item.get("columns", {}))
    if item.get("score_column"):
        columns["score"] = item["score_column"]
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not set(columns.values()).issubset(reader.fieldnames or []):
        raise ValueError(f"missing columns; required={columns}, actual={reader.fieldnames}")
    rows = list(reader)
    by_id = {}
    for row in rows:
        record_id = row[columns["id"]]
        if not record_id or record_id != record_id.strip():
            raise ValueError("blank or padded record ID")
        if record_id in by_id:
            raise ValueError(f"duplicate record ID: {record_id}")
        by_id[record_id] = row
    expected, present = set(cohort["ids"]), set(by_id)
    if expected != present:
        raise ValueError(f"record ID set mismatch: missing={sorted(expected-present)[:10]}, extra={sorted(present-expected)[:10]}; counts={len(expected-present)}/{len(present-expected)}")
    y, score, classes = [], [], []
    for index, record_id in enumerate(cohort["ids"]):
        row = by_id[record_id]
        label = parse_binary(row[columns["label"]], "label", record_id)
        if label != cohort["y"][index]:
            raise ValueError(f"label conflict with mapping for {record_id}")
        value = float(row[columns["score"]])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"score outside [0,1] or nonfinite for {record_id}")
        classification = parse_binary(row[columns["class"]], "pred_class", record_id)
        if classification != int(value > 0.5):
            raise ValueError(f"saved class conflicts with score>0.5 for {record_id}")
        if row.get("patient_group") and row["patient_group"] != cohort["record_groups"][index]:
            raise ValueError(f"patient_group conflict for {record_id}")
        if row.get("source") and row["source"] != cohort["sources"][index]:
            raise ValueError(f"source conflict for {record_id}")
        y.append(label)
        score.append(value)
        classes.append(classification)
    computer = MetricComputer(y, score, classes)
    audit = {
        "resolved_path": str(path), "sha256": actual_hash, "n_records": len(y),
        "n_patients": len(cohort["groups"]), "n_WT": int((cohort["y"] == 0).sum()),
        "n_MT": int((cohort["y"] == 1).sum()), "id_set_sha256": cohort["id_set_sha256"],
        "classification_rule": "pred_class == (score > 0.5)",
        "TN": int(computer.tn.sum()), "FP": int(computer.fp.sum()),
        "FN": int(computer.fn.sum()), "TP": int(computer.tp.sum()),
    }
    return computer, audit


def percentile_interval(samples):
    valid = np.asarray(samples)[np.isfinite(samples)]
    if not len(valid):
        return np.nan, np.nan, 0
    low, high = np.quantile(valid, [0.025, 0.975], method="linear")
    return float(low), float(high), len(valid)


def paired_summary(point_a, point_b, draws_a, draws_b):
    delta = float(point_a - point_b)
    values = np.asarray(draws_a) - np.asarray(draws_b)
    low, high, valid_n = percentile_interval(values)
    values = values[np.isfinite(values)]
    result = {"difference": delta, "ci_low": low, "ci_high": high, "valid_bootstrap": valid_n,
              "difference_pp": 100 * delta, "ci_low_pp": 100 * low, "ci_high_pp": 100 * high}
    if not np.isfinite(delta) or valid_n == 0:
        return dict(result, p_raw=None, test_status="undefined", test_note="undefined point estimate or no finite bootstrap differences")
    zero_variance = bool(np.ptp(values) < 1e-14)
    if zero_variance and abs(delta) > 1e-14:
        return dict(result, p_raw=None, test_status="degenerate_nonzero", test_note="constant nonzero bootstrap difference; no reliable centered-bootstrap p-value reported")
    if zero_variance and abs(delta) <= 1e-14:
        return dict(result, p_raw=1.0, test_status="identical_or_zero_variance_zero", test_note="zero observed and bootstrap differences")
    count = int(np.count_nonzero(np.abs(values - delta) >= abs(delta) - 1e-14))
    p = (1 + count) / (valid_n + 1)
    note = "centered paired patient-cluster bootstrap approximation; finite-sample +1 correction"
    if valid_n != len(draws_a):
        note += "; denominator uses finite paired replicates only"
    return dict(result, p_raw=float(p), test_status="ok", test_note=note)


def holm_adjust(p_values, family_size=FAMILY_SIZE):
    """Missing planned hypotheses consume slots (p=1) but remain NA in output."""
    if len(p_values) > family_size:
        raise ValueError("more supplied tests than prespecified Holm family")
    values = [1.0 if p is None or not np.isfinite(p) else float(p) for p in p_values]
    if any(not 0 <= p <= 1 for p in values):
        raise ValueError("p-value outside [0,1]")
    values += [1.0] * (family_size - len(values))
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(family_size)
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1.0, (family_size - rank) * values[index]))
        adjusted[index] = maximum
    return [None if p is None or not np.isfinite(p) else float(adjusted[i]) for i, p in enumerate(p_values)]


def analyze(manifest_path, output_dir, bootstrap=None, seed=None, batch_size=512, smoke=False):
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must be 1")
    bootstrap = int(bootstrap if bootstrap is not None else manifest.get("bootstrap", DEFAULT_BOOTSTRAP))
    seed = int(seed if seed is not None else manifest.get("seed", DEFAULT_SEED))
    if bootstrap < 2 or batch_size < 1:
        raise ValueError("at least 2 bootstrap replicates and batch_size>=1 required")
    if not smoke and (bootstrap != DEFAULT_BOOTSTRAP or seed != DEFAULT_SEED):
        raise ValueError("formal analysis requires 20000 replicates and seed 20260920; use --smoke for verification")
    models = manifest["planned_models"]
    model_ids = [m["model"] for m in models]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("duplicate planned model")
    ours = manifest.get("ours_model", "Ours")
    if ours not in model_ids:
        raise ValueError("ours_model must occur in planned_models")
    cohort_configs = manifest["cohorts"]
    cohort_ids = [c["id"] for c in cohort_configs]
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("duplicate cohort")
    planned_tests = (len(models) - 1) * len(cohort_configs) * len(TEST_METRICS)
    if planned_tests > FAMILY_SIZE or (not smoke and (planned_tests != FAMILY_SIZE or len(models) != 14 or len(cohort_configs) != 2)):
        raise ValueError("formal analysis requires 26 planned model/cohort pairs (52 tests); --smoke permits fewer")
    mapping_path = resolve_path(manifest_path.parent, manifest["mapping_path"])
    mapping_doc = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    mapping = mapping_doc["cases"]
    cohorts = {c["id"]: build_cohort(mapping, c) for c in cohort_configs}
    indexed = {}
    seen_ids = set()
    for item in manifest["predictions"]:
        key = (item["cohort"], item["model"])
        if key in indexed:
            raise ValueError(f"duplicate model/cohort entry: {key}")
        if item["model"] not in model_ids or item["cohort"] not in cohorts:
            raise ValueError(f"prediction outside planned model/cohort list: {key}")
        if not item.get("id") or item["id"] in seen_ids:
            raise ValueError("every prediction entry requires a unique id")
        seen_ids.add(item["id"])
        indexed[key] = item
    output_dir.mkdir(parents=True, exist_ok=True)
    validations, metrics_rows, pair_rows, cohort_audits = [], [], [], []
    estimates, draws, availability = {}, {}, {}
    for cohort_id, cohort in cohorts.items():
        print(f"cohort={cohort_id}, records={len(cohort['ids'])}, patients={len(cohort['groups'])}, B={bootstrap}", flush=True)
        patient_weights = cluster_weights(cohort, bootstrap, seed)
        safe_cohort = "".join(c if c.isalnum() or c in "_-" else "_" for c in cohort_id)
        weights_path = output_dir / f"bootstrap_patient_weights_{safe_cohort}.npz"
        np.savez_compressed(weights_path, weights=patient_weights,
                            patient_ids=np.asarray(cohort["groups"]), patient_labels=cohort["group_y"],
                            record_ids=np.asarray(cohort["ids"]), record_group_index=cohort["record_group_index"])
        cohort_audits.append({"cohort": cohort_id, "records": len(cohort["ids"]), "patients": len(cohort["groups"]),
                             "patient_WT": int((cohort["group_y"] == 0).sum()), "patient_MT": int((cohort["group_y"] == 1).sum()),
                             "id_set_sha256": cohort["id_set_sha256"], "weights_file": weights_path.name,
                             "weights_array_sha256": sha256_bytes(patient_weights.tobytes()),
                             "weights_file_sha256": sha256_file(weights_path)})
        index_rows = [{"cohort": cohort_id, "ID": rid, "idh_truth": int(cohort["y"][i]),
                       "patient_group": cohort["record_groups"][i], "patient_index": int(cohort["record_group_index"][i]),
                       "source": cohort["sources"][i]} for i, rid in enumerate(cohort["ids"])]
        write_csv(output_dir / f"patient_index_{safe_cohort}.csv", index_rows)
        for model in models:
            model_id = model["model"]
            key = (cohort_id, model_id)
            item = indexed.get(key)
            audit = {"cohort": cohort_id, "model": model_id, "display_name": model.get("display_name", model_id),
                     "prediction_id": item.get("id") if item else None}
            computer = None
            if item is None:
                audit.update(status="missing", reason="no manifest entry for planned model/cohort")
            elif item.get("status", "selected") != "selected" or not item.get("path"):
                status = item.get("status", "missing")
                audit.update(status="missing" if status == "selected" else status, reason=item.get("reason", "prediction not selected or path missing"))
            else:
                try:
                    computer, details = load_prediction(item, cohort, manifest_path.parent)
                    audit.update(status="valid", reason="", **details)
                except (OSError, ValueError, KeyError, UnicodeError) as error:
                    audit.update(status="invalid", reason=str(error))
            validations.append(audit)
            availability[key] = audit
            if computer is not None:
                estimates[key] = computer.point()
                draws[key] = np.empty((bootstrap, len(METRICS)), dtype=float)
                for begin in range(0, bootstrap, batch_size):
                    end = min(begin + batch_size, bootstrap)
                    record_weights = patient_weights[begin:end, cohort["record_group_index"]]
                    draws[key][begin:end] = computer.evaluate(record_weights)
                print(f"  {model_id}: valid", flush=True)
            else:
                print(f"  {model_id}: {audit['status']} ({audit['reason']})", flush=True)
            for metric_index, metric in enumerate(METRICS):
                row = {"cohort": cohort_id, "model": model_id, "display_name": model.get("display_name", model_id),
                       "metric": metric, "status": audit["status"], "reason": audit["reason"],
                       "estimate": None, "ci_low": None, "ci_high": None, "valid_bootstrap": 0,
                       "bootstrap_replicates": bootstrap, "seed": seed,
                       "n_records": len(cohort["ids"]), "n_patients": len(cohort["groups"]), "warning": ""}
                if computer is not None:
                    low, high, valid_n = percentile_interval(draws[key][:, metric_index])
                    row.update(estimate=float(estimates[key][metric_index]), ci_low=low, ci_high=high, valid_bootstrap=valid_n)
                    if valid_n != bootstrap:
                        row["warning"] = f"undefined in {bootstrap-valid_n}/{bootstrap} replicates; CI uses finite replicates"
                    if not np.isfinite(estimates[key][metric_index]):
                        row["warning"] += "; point estimate undefined"
                    if valid_n and abs(high - low) < 1e-14:
                        row["warning"] += "; degenerate percentile interval"
                metrics_rows.append(row)
        for model in models:
            if model["model"] == ours:
                continue
            key_a, key_b = (cohort_id, ours), (cohort_id, model["model"])
            for metric in TEST_METRICS:
                row = {"cohort": cohort_id, "model_A": ours, "model_B": model["model"], "metric": metric,
                       "difference": None, "ci_low": None, "ci_high": None,
                       "difference_pp": None, "ci_low_pp": None, "ci_high_pp": None,
                       "p_raw": None, "p_holm": None, "holm_family_size": FAMILY_SIZE,
                       "valid_bootstrap": 0, "bootstrap_replicates": bootstrap, "seed": seed,
                       "test_status": "unavailable", "test_note": "", "significant_holm_0_05": None}
                if key_a in draws and key_b in draws:
                    metric_index = METRICS.index(metric)
                    row.update(paired_summary(estimates[key_a][metric_index], estimates[key_b][metric_index],
                                              draws[key_a][:, metric_index], draws[key_b][:, metric_index]))
                else:
                    row["test_note"] = "; ".join(f"{k[1]}: {availability[k]['status']}: {availability[k]['reason']}"
                                                 for k in (key_a, key_b) if k not in draws)
                pair_rows.append(row)
    adjusted = holm_adjust([row["p_raw"] for row in pair_rows])
    for row, corrected in zip(pair_rows, adjusted):
        row["p_holm"] = corrected
        row["significant_holm_0_05"] = corrected < 0.05 if corrected is not None else None
    write_csv(output_dir / "metrics_long.csv", metrics_rows)
    write_csv(output_dir / "paired_comparisons.csv", pair_rows)
    write_csv(output_dir / "input_validation.csv", validations)
    wide = []
    for cohort_id in cohorts:
        for model in models:
            row = {"cohort": cohort_id, "model": model["model"], "display_name": model.get("display_name", model["model"])}
            for item in metrics_rows:
                if item["cohort"] == cohort_id and item["model"] == model["model"]:
                    row["status"], row["reason"] = item["status"], item["reason"]
                    for suffix, value in (("", item["estimate"]), ("_ci_low", item["ci_low"]), ("_ci_high", item["ci_high"])):
                        row[item["metric"] + suffix] = value
            wide.append(row)
    write_csv(output_dir / "main_results.csv", wide)
    arrays = {f"model_{i:03d}": draws[key] for i, key in enumerate(draws)}
    np.savez_compressed(output_dir / "bootstrap_metrics.npz", **arrays)
    invalid = sum(row["status"] == "invalid" for row in validations)
    absent = sum(row["status"] != "valid" for row in validations)
    summary = {
        "analysis_kind": "SMOKE_TEST_NOT_FOR_PAPER" if smoke else "FORMAL_FIXED_MODEL_ANALYSIS",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path), "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path), "script_sha256": sha256_file(__file__),
        "seed": seed, "bootstrap": bootstrap, "rng": "numpy.random.default_rng / PCG64",
        "sampling": "class-stratified patient-cluster multinomial counts; sorted patient IDs; include all scans; same weights across models within each cohort; RNG reinitialized with the recorded seed for each cohort",
        "estimand": "record-weighted fixed trained-model performance; not training-seed variability",
        "positive_class": "IDH-mutant=1", "threshold": "score>0.5; exact tie classified WT",
        "ci": "pointwise 2.5%/97.5% percentile, numpy quantile method=linear; not simultaneous intervals",
        "p_value": "(1 + count(abs(delta_boot-delta_observed)>=abs(delta_observed))) / (B_valid+1); 1e-14 floating boundary tolerance; nonzero constant bootstrap differences yield NA",
        "holm": "52 prespecified hypotheses; unavailable hypotheses use p=1 internally but display NA",
        "AP": "noninterpolated average precision, not trapezoidal PR area",
        "n_planned_model_cohort_rows": len(validations), "n_valid": len(validations)-absent,
        "n_unavailable": absent, "n_invalid": invalid, "n_planned_test_rows": len(pair_rows),
        "n_testable": sum(row["p_raw"] is not None for row in pair_rows), "holm_family_size": FAMILY_SIZE,
        "all_planned_predictions_available": absent == 0,
        "cohorts": cohort_audits, "bootstrap_metric_arrays": {f"model_{i:03d}": {"cohort": key[0], "model": key[1], "columns": METRICS} for i, key in enumerate(draws)},
        "metric_warnings": [row for row in metrics_rows if row["warning"]],
        "software": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "limitations": ["patient groups are public-ID/follow-up mappings, not a completed cross-dataset identity audit", "patient bootstrap does not repair train/validation/test patient overlap", "CI and approximate p-values condition on the fixed trained models and cohort class composition", "percentile CIs and centered-bootstrap tests need not agree at finite-sample boundaries"],
    }
    write_json(output_dir / "analysis_summary.json", summary)
    write_json(output_dir / "manifest_used.json", manifest)
    for cohort_id in cohorts:
        safe_cohort = "".join(c if c.isalnum() or c in "_-" else "_" for c in cohort_id)
        title = "SMOKE TEST — NOT FOR PAPER\n\n" if smoke else ""
        lines = [title + f"# {cohort_id}: fixed-model estimates and 95% patient-cluster CI", "", "All values are percentages. NA rows retain their stated reason.", "", "| Model | ACC | AUC | Specificity | Sensitivity |", "|---|---:|---:|---:|---:|"]
        for model in models:
            row_values = []
            for metric in METRICS[:4]:
                entry = next(row for row in metrics_rows if row["cohort"] == cohort_id and row["model"] == model["model"] and row["metric"] == metric)
                if entry["estimate"] is None or not np.isfinite(entry["estimate"]):
                    row_values.append("NA")
                else:
                    row_values.append(f"{100*entry['estimate']:.2f} [{100*entry['ci_low']:.2f}, {100*entry['ci_high']:.2f}]")
            lines.append("| " + " | ".join([model.get("display_name", model["model"])] + row_values) + " |")
        lines += ["", "Unavailable inputs:"]
        for audit in validations:
            if audit["cohort"] == cohort_id and audit["status"] != "valid":
                lines.append(f"- {audit['display_name']}: {audit['status']}; {audit['reason']}")
        if not any(a["cohort"] == cohort_id and a["status"] != "valid" for a in validations):
            lines.append("- None.")
        (output_dir / f"main_table_{safe_cohort}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("analysis_kind", "n_valid", "n_unavailable", "n_invalid", "n_testable")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--smoke", action="store_true", help="mark all outputs as verification-only and permit a smaller analysis")
    args = parser.parse_args()
    summary = analyze(args.manifest, args.out, args.bootstrap, args.seed, args.batch_size, args.smoke)
    # Missing predictions are explicit partial results. Malformed supplied inputs
    # additionally signal a failed validation to the invoking process.
    return 2 if summary["n_invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
