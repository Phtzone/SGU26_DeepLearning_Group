"""Read-only chest temperature analysis for the verified preprocessing group."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy
from scipy.signal import resample_poly


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GROUP = HERE.parent
BASELINE = "common_rate64_cnn1d_grouped5fold_v1"
BASE_GROUP = "20261123T072126Z_2c81daac"
BASE_RUN = ROOT / "runs" / BASELINE / BASE_GROUP
BASE_RESULT = ROOT / "results" / BASELINE / BASE_GROUP
ROLES = ("train", "validation")
METHODS = ("baseline_z", "centered_z")
CLASSES = ("Baseline", "Stress", "Amusement")
QUANTILES = (1, 5, 25, 50, 75, 95, 99, 99.9)
QUANTILE_NAMES = ("p01", "p05", "p25", "p50", "p75", "p95", "p99", "p999")
THRESHOLDS = (3, 5, 10, 20, 50, 100)
HIST_EDGES = (0., .1, .25, .5, 1., 2., 3., 5., 10., 20., 50., 100., np.inf)
TEMPERATURE = "chest_Temp"
TARGET = "S2_84900_86820"
FS, NATIVE_FS = 64, 700
WIDTH = 30 * FS
COLORS = {"baseline_z": "#526F8A", "centered_z": "#AD431C"}
CAPTIONS = {"baseline_z": "Z-score baseline", "centered_z": "Median Centering + Z-score"}


@dataclass(frozen=True)
class Window:
    subject: str
    identifier: str
    start: int
    end: int
    original_label: int
    label: int


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(out, name, rows):
    with (out / name).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(out, name, value):
    with (out / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha256(data):
    value = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_figure(out, fig, name):
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        with (directory / f"{name}.{extension}").open("xb") as stream:
            fig.savefig(stream, format=extension, dpi=160)
    plt.close(fig)


def context():
    previous = read_json(GROUP / "preprocessing_verification.json")
    assert previous["status"] == "passed" and not previous["training_started"]
    assert previous["group_id"] == GROUP.name and previous["experiment_id"] == GROUP.parent.name
    assert previous["baseline_group_id"] == BASE_GROUP and previous["baseline_experiment_id"] == BASELINE
    source = ROOT / previous["source"]
    index = (ROOT / previous["window_manifest"]).parent
    assert index == BASE_RUN / "window_index"
    split = read_json(ROOT / previous["split_manifest"])
    assert json_sha256(split) == previous["split_manifest_checksum"]
    assert split == read_json(ROOT / previous["effective_config"]["split"]["manifest"])
    cfg = read_json(source / "config.json")
    assert cfg["target_fs"] == FS and cfg["label_source_fs"] == NATIVE_FS
    assert cfg["channels"] == previous["channels"]
    assert cfg["channels"].index(TEMPERATURE) == previous["chest_temperature_channel_index_zero_based"]
    assert cfg["subjects"] == split["original_subject_order"]
    meta = read_json(index / "common_rate_index.json")
    return previous, source, index, split, cfg["channels"], meta


def protected_hashes(source, meta):
    paths = {p for folder in (BASE_RUN, BASE_RESULT) for p in folder.rglob("*") if p.is_file()}
    # Only direct files and the original scaler and figure directories are inputs in this group.
    paths.update(p for p in GROUP.iterdir() if p.is_file())
    paths.update(p for folder in (GROUP / "seed_42", GROUP / "figures") for p in folder.rglob("*") if p.is_file())
    paths.update(source / key for key in meta["audit"]["source_file_sha256"])
    paths.update(ROOT / "data/raw/WESAD/S2" / name for name in ("S2.pkl", "S2_respiban.txt", "S2_readme.txt"))
    paths.update(ROOT / path for path in ("data/raw/WESAD/wesad_readme.pdf", "notebooks/02_preprocessing_common_rate.ipynb",
                                        "splits/grouped_5fold_seed42.json", "docs/S2_Complete_Data_Understanding.md"))
    paths.update(p for p in (ROOT / "experiments" / GROUP.parent.name).iterdir() if p.is_file())
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def load_windows(source, index, split, channel):
    per_subject, raw, residual = {}, {}, {}
    all_rows = {}
    for row in read_csv(index / "windows.csv"):
        assert row["quality_status"] == "valid" and row["window_id"] not in all_rows
        all_rows[row["window_id"]] = row
    for subject in split["original_subject_order"]:
        timeline = np.load(source / f"{subject}_X.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(source / f"{subject}_quality_mask.npy", mmap_mode="r", allow_pickle=False)
        assert timeline.dtype == np.float32 and timeline.shape == (mask.size, 14)
        assert mask.dtype == np.bool_
        windows, values = [], []
        for row in read_csv(index / subject / "windows.csv"):
            assert row == all_rows[row["window_id"]]
            start, end, label, original = (int(row[k]) for k in ("start_index", "end_index", "label", "original_label"))
            assert end - start == WIDTH and mask[start:end].all()
            assert row["subject_id"] == subject and label == original - 1 and label in range(len(CLASSES))
            windows.append(Window(subject, row["window_id"], start, end, original, label))
            values.append(timeline[start:end, channel])
        matrix = np.stack(values)
        medians = np.median(matrix.astype(np.float64), axis=1, keepdims=True)
        centered = (matrix.astype(np.float64) - medians).astype(np.float32)
        assert np.isfinite(matrix).all() and np.isfinite(centered).all()
        assert not np.median(centered, axis=1).any()
        per_subject[subject], raw[subject], residual[subject] = windows, matrix, centered
    assert len(all_rows) == sum(map(len, per_subject.values())) == 2134
    assert {w.identifier for ws in per_subject.values() for w in ws} == set(all_rows)
    return per_subject, raw, residual


def load_scalers(split, channel):
    result = {}
    for fold in split["folds"]:
        fid = fold["fold_id"]
        path = Path("seed_42") / f"fold_{fid:02d}" / "normalization.json"
        original, centered = read_json(BASE_RUN / path), read_json(GROUP / path)
        assert centered["window_centering"] == {"enabled": True, "method": "median", "channels": [TEMPERATURE]}
        assert centered["train_subjects"] == original["train_subjects"] == fold["train"]
        assert centered["ddof"] == original["ddof"] == 0
        parameters = {}
        for method, artifact in zip(METHODS, (original, centered), strict=True):
            assert len(artifact["mean"]) == len(artifact["std"]) == 14
            mu, scale = (np.float32(artifact[key][channel]) for key in ("mean", "std"))
            assert np.isfinite(mu) and np.isfinite(scale) and scale > 0
            parameters[method] = (mu, scale)
        result[fid] = parameters
    return result


def zscore(matrix, parameters):
    mu, std = parameters
    return (matrix - mu) / std


def percentiles(values, prefix="", axis=None):
    q = np.percentile(values, QUANTILES, axis=axis, method="linear")
    return {prefix + name: estimate for name, estimate in zip(QUANTILE_NAMES, q, strict=True)}


def label_name(index):
    return CLASSES[index] if index is not None else "All"


def subject_max_at_timestamps(windows, absolute):
    bounds = {}
    for w in windows:
        lo, hi = bounds.get(w.subject, (w.start, w.end))
        bounds[w.subject] = (min(lo, w.start), max(hi, w.end))
    maxima = {s: np.full(hi - lo, -1, dtype=np.float32) for s, (lo, hi) in bounds.items()}
    for window, values in zip(windows, absolute, strict=True):
        offset = bounds[window.subject][0]
        target = maxima[window.subject][window.start - offset:window.end - offset]
        np.maximum(target, values, out=target)
    return maxima


def distribution_row(fid, role, method, scope, subject, label, windows, z):
    assert z.shape == (len(windows), WIDTH) and np.isfinite(z).all()
    absolute = np.abs(z)
    abs_window_max = absolute.max(axis=1)
    unique = subject_max_at_timestamps(windows, absolute)
    n_unique = sum(int(np.count_nonzero(a >= 0)) for a in unique.values())
    result = {"fold_id": fid, "role": role, "method": method, "scope": scope, "subject_id": subject,
              "label": label_name(label), "subjects": len(unique), "windows": len(windows),
              "timestep_occurrences": int(z.size), "unique_timestamps": n_unique,
              "mean_z": float(z.mean(dtype=np.float64)), "std_z": float(z.std(dtype=np.float64)),
              "min_z": float(z.min()), "max_z": float(z.max()), "max_abs_z": float(absolute.max()),
              **{k: float(v) for k, v in (percentiles(z) | percentiles(absolute, "abs_")).items()}}
    for threshold in THRESHOLDS:
        count = int(np.count_nonzero(absolute > threshold))
        affected_windows = int(np.count_nonzero(abs_window_max > threshold))
        time_counts = [int(np.count_nonzero(values > threshold)) for values in unique.values()]
        segments = sum(int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))
                       for values in unique.values() if (mask := values > threshold).any())
        n_time = sum(time_counts)
        assert n_time <= count
        result.update({f"n_abs_gt_{threshold}": count, f"pct_abs_gt_{threshold}": 100 * count / z.size,
                       f"windows_gt_{threshold}": affected_windows,
                       f"pct_windows_gt_{threshold}": 100 * affected_windows / len(windows),
                       f"subjects_gt_{threshold}": sum(n > 0 for n in time_counts),
                       f"timestamps_gt_{threshold}": n_time,
                       f"pct_timestamps_gt_{threshold}": 100 * n_time / n_unique,
                       f"contiguous_segments_gt_{threshold}": segments})
    return result


def summarize_groups(fid, role, method, windows, z):
    partitions = defaultdict(list)
    for i, window in enumerate(windows):
        partitions["subject", window.subject, None].append(i)
        partitions["class", "All", window.label].append(i)
        partitions["subject_class", window.subject, window.label].append(i)
    rows = [distribution_row(fid, role, method, "role", "All", None, windows, z)]
    rows.extend(distribution_row(fid, role, method, scope, subject, label,
                                 [windows[i] for i in order], z[order])
                for (scope, subject, label), order in partitions.items())
    for threshold in THRESHOLDS:
        for scope in ("subject", "class", "subject_class"):
            assert sum(r[f"n_abs_gt_{threshold}"] for r in rows if r["scope"] == scope) == rows[0][f"n_abs_gt_{threshold}"]
    return rows


def summarize_windows(fid, role, method, windows, z):
    absolute = np.abs(z)
    thresholds = {cutoff: (absolute > cutoff).sum(axis=1) for cutoff in THRESHOLDS}
    quantiles = percentiles(z, axis=1) | percentiles(absolute, "abs_", axis=1)
    ids = absolute.argmax(axis=1)
    min_z, max_z = z.min(axis=1), z.max(axis=1)
    means, deviations = z.mean(axis=1, dtype=np.float64), z.std(axis=1, dtype=np.float64)
    rows = []
    for i, window in enumerate(windows):
        peak = int(ids[i])
        row = {"fold_id": fid, "role": role, "method": method, "subject_id": window.subject,
               "label": label_name(window.label), "original_label": window.original_label, "window_id": window.identifier,
               "start_index_64": window.start, "end_index_64_exclusive": window.end,
               "timestep_occurrences": int(z.shape[1]), "mean_z": float(means[i]), "std_z": float(deviations[i]),
               "min_z": float(min_z[i]), "max_z": float(max_z[i]), "max_abs_z": float(absolute[i, peak]),
               "peak_signed_z": float(z[i, peak]), "peak_source_index_64": window.start + peak,
               **{key: float(val[i]) for key, val in quantiles.items()}}
        for cutoff, counts in thresholds.items():
            n = int(counts[i])
            row[f"n_abs_gt_{cutoff}"] = n
            row[f"pct_abs_gt_{cutoff}"] = 100 * n / z.shape[1]
        rows.append(row)
    return rows


def histogram_rows(fid, role, method, z):
    counts, _ = np.histogram(np.abs(z), bins=HIST_EDGES)
    assert sum(counts) == z.size
    return [{"fold_id": fid, "role": role, "method": method, "abs_z_lower_inclusive": lo,
             "abs_z_upper_exclusive": hi if np.isfinite(hi) else "Inf", "count": int(n), "pct": 100 * int(n) / z.size}
            for lo, hi, n in zip(HIST_EDGES[:-1], HIST_EDGES[1:], counts, strict=True)]


def train_scale_row(fid, role_raw, role_residual, scales):
    raw_q25, raw_q75 = np.percentile(role_raw, (25, 75))
    q25, q75 = np.percentile(role_residual, (25, 75))
    _, old_std = scales["baseline_z"]
    _, new_std = scales["centered_z"]
    return {"fold_id": fid, "train_windows": int(len(role_raw)), "raw_scaler_std_c": float(old_std),
            "centered_scaler_std_c": float(new_std), "scale_amplification": float(old_std) / float(new_std),
            "raw_train_iqr_c": float(raw_q75 - raw_q25), "centered_train_q25_c": float(q25),
            "centered_train_q75_c": float(q75), "centered_train_iqr_c": float(q75 - q25),
            "centered_train_iqr_divided_by_std": float(q75 - q25) / float(new_std)}


def analyze_distribution(out, split, windows_by_subject, raw, residual, scalers):
    group_rows, window_rows, histograms, scale_rows, curves = [], [], [], [], {}
    val_windows = []
    validation = {name: [] for name in METHODS}
    thresholds = np.unique(np.r_[np.geomspace(.03, 160, 210), THRESHOLDS])
    for fold in split["folds"]:
        fid = fold["fold_id"]
        assert len(fold["train"]) == 9 and len(fold["validation"]) == 3
        assert set(fold["train"]).isdisjoint(fold["validation"])
        for role in ROLES:
            subjects = fold[role]
            windows = [w for subject in subjects for w in windows_by_subject[subject]]
            x = np.concatenate([raw[subject] for subject in subjects])
            r = np.concatenate([residual[subject] for subject in subjects])
            if role == "train":
                scale_rows.append(train_scale_row(fid, x, r, scalers[fid]))
            else:
                val_windows.extend(windows)
            for method, matrix in zip(METHODS, (x, r), strict=True):
                z = zscore(matrix, scalers[fid][method])
                summaries = summarize_groups(fid, role, method, windows, z)
                per_window = summarize_windows(fid, role, method, windows, z)
                for cutoff in THRESHOLDS:
                    key = f"n_abs_gt_{cutoff}"
                    assert summaries[0][key] == sum(row[key] for row in per_window)
                group_rows.extend(summaries)
                window_rows.extend(per_window)
                histograms.extend(histogram_rows(fid, role, method, z))
                distribution = np.sort(np.abs(z).reshape(-1))
                curves[fid, role, method] = 100 * (z.size - np.searchsorted(distribution, thresholds, side="right")) / z.size
                if role == "validation":
                    validation[method].append(z)
        print(f"Fold {fid}: counted Train and Validation tail occurrences and distinct timestamps", flush=True)
    assert len(val_windows) == len({w.identifier for w in val_windows}) == 2134
    for method, parts in validation.items():
        pooled = np.concatenate(parts)
        group_rows.extend(summarize_groups("all", "validation", method, val_windows, pooled))
        histograms.extend(histogram_rows("all", "validation", method, pooled))
    write_csv(out, "zscore_statistics_by_group.csv", group_rows)
    write_csv(out, "zscore_statistics_by_window.csv", window_rows)
    write_csv(out, "absolute_z_histogram.csv", histograms)
    write_csv(out, "temperature_train_scale_context.csv", scale_rows)
    plot_survival(out, thresholds, curves)
    plot_rates(out, group_rows)
    return group_rows, window_rows, histograms, scale_rows


def summarize_variation(out, windows_by_subject, residual):
    rows, window_amplitude = [], {}
    for subject, windows in windows_by_subject.items():
        values = residual[subject].astype(np.float64)
        sd = values.std(axis=1)
        range_ = np.ptp(values, axis=1)
        q05, q95 = np.percentile(values, (5, 95), axis=1)
        adjacent_p99 = np.percentile(np.abs(np.diff(values, axis=1)), 99, axis=1)
        window_amplitude[subject] = (sd, range_)
        labels = np.array([w.label for w in windows])
        for label in (None, 0, 1, 2):
            selected = np.ones(len(values), dtype=bool) if label is None else labels == label
            row = {"subject_id": subject, "label": label_name(label), "distinct_windows": int(selected.sum()),
                   "flat_windows": int(np.count_nonzero(range_[selected] == 0))}
            for field, observations in (("within_window_std", sd), ("within_window_ptp", range_),
                                        ("within_window_p95_minus_p05", q95 - q05),
                                        ("within_window_adjacent_abs_diff_p99", adjacent_p99)):
                for k, v in percentiles(observations[selected]).items():
                    row[f"{field}_{k}_c"] = float(v)
                row[f"{field}_max_c"] = float(observations[selected].max())
            rows.append(row)
    write_csv(out, "centered_window_variation_by_subject.csv", rows)
    plot_subject_variation(out, window_amplitude)
    return rows


def plot_survival(out, grid, curves):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), sharex=True, sharey=True, layout="constrained")
    for r, role in enumerate(ROLES):
        for column, fid in enumerate(range(1, 6)):
            ax = axes[r, column]
            for method in METHODS:
                values = curves[fid, role, method]
                ax.plot(grid, np.where(values > 0, values, np.nan), color=COLORS[method],
                        label=CAPTIONS[method], lw=1.65)
            ax.set(xscale="log", yscale="log", xlim=(.03, 160), ylim=(1e-5, 110),
                   title=f"Fold {fid} · {role.title()}", xticks=[.1, 1, 3, 10, 100])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.axvline(10, color=".6", lw=.8, ls=":")
            ax.grid(alpha=.2)
            if r == 1:
                ax.set_xlabel("Ngưỡng |Z|")
            if column == 0:
                ax.set_ylabel("% lượt timestep vượt ngưỡng")
    axes[0, 0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle("Đuôi phân phối chest_Temp theo fold và tập", fontsize=15)
    save_figure(out, fig, "absolute_z_survival_by_fold")


def plot_rates(out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained", sharey=True)
    role_level = [r for r in rows if r["scope"] == "role" and r["fold_id"] != "all"]
    for ax, method in zip(axes, METHODS, strict=True):
        subset = [r for r in role_level if r["method"] == method]
        rates = np.array([[r[f"pct_abs_gt_{t}"] for t in THRESHOLDS] for r in subset])
        # Fixed and shared color mapping; annotations retain percentage units.
        ax.imshow(np.log10(1 + rates * 100), aspect="auto", cmap="YlOrRd", vmin=0, vmax=4)
        for i, j in np.ndindex(rates.shape):
            value = rates[i, j]
            label = "0" if not value else f"{value:.3g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if value > 8 else "#192225")
        ax.set(title=CAPTIONS[method], xticks=range(len(THRESHOLDS)), xticklabels=THRESHOLDS,
               yticks=range(len(subset)), yticklabels=[f"Fold {r['fold_id']} · {r['role'].title()}" for r in subset],
               xlabel="Ngưỡng |Z|")
    fig.suptitle("Tỷ lệ lượt timestep vượt ngưỡng (%)", fontsize=15)
    save_figure(out, fig, "threshold_rates_by_fold")


def plot_subject_variation(out, amplitude):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for i, (ax, title) in enumerate(zip(axes, ("Độ lệch chuẩn trong cửa sổ", "Biên độ đỉnh–đỉnh trong cửa sổ"), strict=True)):
        labels = list(amplitude)
        result = ax.boxplot([amplitude[s][i] for s in labels], tick_labels=labels, patch_artist=True,
                            whis=(5, 95), flierprops={"markersize": 2.5, "alpha": .65})
        for box in result["boxes"]:
            box.set_facecolor("#81B9B6")
        ax.set(title=title, ylabel="°C", yscale="symlog")
        ax.set_yscale("symlog", linthresh=.001)
        ax.tick_params(axis="x", labelrotation=50)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Biến thiên chest_Temp sau trừ median · 2.134 cửa sổ duy nhất", fontsize=14)
    save_figure(out, fig, "centered_window_variation_by_subject")


def native_interval(window):
    return tuple((index * NATIVE_FS + FS - 1) // FS for index in (window.start, window.end))


def compare_respiban_temperature(path, start, end, native):
    # Row alignment observed in the previous S2 audit; rechecked here on every target sample.
    offset = 92017
    with path.open(encoding="utf-8") as stream:
        header = stream.readline()
        description = json.loads(stream.readline().removeprefix("# "))
        assert header.startswith("# OpenSignals")
        device, = description.values()
        assert device["sampling rate"] == NATIVE_FS and stream.readline().strip() == "# EndOfHeader"
        sensor = device["sensor"].index("TEMP")
        channel_column = device["column"].index(device["label"][sensor])
        resolution = device["resolution"][sensor]
        adc_values = []
        for i, line in enumerate(islice(stream, offset + start, offset + end), start=offset + start):
            row = line.split()
            assert int(row[0]) == i
            adc_values.append(int(row[channel_column]))
    adc_values = np.array(adc_values, dtype=np.int32)
    assert len(adc_values) == end - start
    vcc = 3.0
    voltage = adc_values * vcc / (2**resolution - 1)
    log_resistance = np.log(1e4 * voltage / (vcc - voltage))
    degrees = 1 / (1.12764514e-3 + 2.34282709e-4 * log_resistance + 8.77303013e-8 * log_resistance**3) - 273.15
    max_error = float(np.abs(degrees - native[start:end]).max())
    assert max_error < 1e-4
    return adc_values, degrees, {"channel_column": device["column"][channel_column], "adc_resolution_bits": resolution,
                                  "nseq_offset_from_pkl": offset, "first_nseq": offset + start,
                                  "last_nseq": offset + end - 1, "samples_compared": len(adc_values),
                                  "max_abs_difference_vs_pkl_c": max_error}


def window_detail(window, x, r, z, source_peak=None):
    peak = int(np.argmax(np.abs(z)))
    row = {"window_id": window.identifier, "subject_id": window.subject, "label": label_name(window.label),
           "original_label": window.original_label, "mapped_label": window.label,
           "start_index_64": window.start, "end_index_64_exclusive": window.end,
           "start_s": window.start / FS, "end_s_exclusive": window.end / FS,
           "min_c": float(x.min()), "max_c": float(x.max()), "median_c": float(np.median(x.astype(np.float64))),
           "within_window_std_c": float(x.std(dtype=np.float64)), "ptp_c": float(np.ptp(x.astype(np.float64))),
           "centered_min_c": float(r.min()), "centered_max_c": float(r.max()),
           "min_z": float(z.min()), "max_z": float(z.max()), "peak_signed_z": float(z[peak]),
           "peak_sample_64": window.start + peak, "peak_time_s": (window.start + peak) / FS}
    peak_in_window = source_peak is not None and window.start <= source_peak < window.end
    row.update({"contains_target_peak_timestamp": peak_in_window,
                "z_at_target_peak_timestamp": float(z[source_peak - window.start]) if peak_in_window else None})
    for t in THRESHOLDS:
        row[f"n_abs_gt_{t}"] = int(np.count_nonzero(np.abs(z) > t))
    return row


def plot_s2_window(out, target, x, r, z, native, peak):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    a, b = native_interval(target)
    within_time = np.arange(WIDTH) / FS
    axes[0].plot(np.arange(a, b) / NATIVE_FS - target.start / FS, native[a:b], color="#A5ACB0",
                 lw=.7, label="PKL 700 Hz")
    for ax, values, title, units in zip(
            axes, (x, r, z), ("Trước Median Centering", "Sau Median Centering", "Sau Z-score theo Train Fold 4"),
            ("°C", "Chênh lệch (°C)", "Z-score"), strict=True):
        ax.plot(within_time, values, color="#007C83", label="Common-rate 64 Hz", lw=1.45)
        ax.axvline(peak / FS, color="#AD431C", lw=1, ls="--")
        ax.scatter(peak / FS, values[peak], color="#AD431C", s=25, zorder=4)
        ax.set(title=title, ylabel=units, xlim=(0, WIDTH / FS))
        ax.grid(alpha=.22)
        if ax is not axes[0]:
            ax.axhline(0, color=".55", lw=.7)
    axes[0].legend(loc="lower right", frameon=False)
    axes[2].annotate(f"{z[peak]:.2f}", (peak / FS, z[peak]), xytext=(18, 5), textcoords="offset points", color="#97370D")
    axes[2].set_xlabel("Giây tính từ đầu cửa sổ")
    fig.suptitle(f"{target.identifier} · {label_name(target.label)} · Validation Fold 4", fontsize=14)
    save_figure(out, fig, "s2_window_three_stages")


def plot_s2_context(out, neighbors, target, source_64, native, peak_source):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), layout="constrained")
    lo, hi = neighbors[0].start, neighbors[-1].end
    axes[0].plot(np.arange(lo, hi) / FS, source_64[lo:hi], color="#007C83", lw=1.25)
    prev, _, following = neighbors[len(neighbors) // 2 - 1:len(neighbors) // 2 + 2]
    for window, color, label, alpha in [(prev, "#4F6F92", "Cửa sổ trước", .12),
                                        (following, "#B26D35", "Cửa sổ sau", .12),
                                        (target, "#009E73", "Cửa sổ đang xét", .25)]:
        axes[0].axvspan(window.start / FS, window.end / FS, color=color, alpha=alpha, label=label)
    axes[0].axvline(peak_source / FS, color="#AD431C", ls="--", lw=1)
    axes[0].set(title="Bối cảnh 64 Hz và các cửa sổ chồng lấp", xlabel="Giây từ đầu bản ghi PKL", ylabel="°C")
    axes[0].legend(loc="lower right", ncol=3, fontsize=8, frameon=False)
    peak_t = peak_source / FS
    n_start = int(np.ceil((peak_t - .5) * NATIVE_FS))
    n_next = int(np.ceil((peak_t + .5) * NATIVE_FS))
    short = np.arange(peak_source - 32, peak_source + 32)
    axes[1].plot(np.arange(n_start, n_next) / NATIVE_FS - peak_t, native[n_start:n_next],
                 color="#7E8589", lw=.7, label="PKL 700 Hz")
    axes[1].plot(short / FS - peak_t, source_64[short], color="#007C83", lw=1.4,
                 marker=".", ms=3, label="Common-rate 64 Hz")
    axes[1].scatter(0, source_64[peak_source], color="#AD431C", s=24, zorder=4)
    axes[1].axvline(0, color="#AD431C", ls="--", lw=1)
    axes[1].set(title=f"So khớp tín hiệu trong ±0,5 s quanh mẫu 64 Hz {peak_source}",
                xlabel="Giây so với mẫu có |Z| lớn nhất trong cửa sổ", ylabel="°C")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.23)
    save_figure(out, fig, "s2_native_context_and_neighbors")


def analyze_s2(out, source, split, channel_order, scalers, s2_windows, x_values, residual):
    folder = ROOT / "data/raw/WESAD/S2"
    # This is the local WESAD input already used by the repository.
    with (folder / "S2.pkl").open("rb") as stream:
        record = pickle.load(stream, encoding="latin1")
    assert record["subject"] == "S2"
    native = record["signal"]["chest"]["Temp"][:, 0]
    labels = record["label"]
    acc = record["signal"]["chest"]["ACC"]
    assert native.ndim == labels.ndim == 1 and len(native) == len(labels) == len(acc)
    timeline = np.load(source / "S2_X.npy", mmap_mode="r", allow_pickle=False)
    source_64 = timeline[:, channel_order.index(TEMPERATURE)]
    mask = np.load(source / "S2_quality_mask.npy", mmap_mode="r", allow_pickle=False)
    replay = resample_poly(native, up=16, down=175, axis=0).astype(np.float32)
    np.testing.assert_array_equal(replay, source_64)
    assert mask.all()
    n_temp_flag = np.count_nonzero(np.isclose(native, -273.15, atol=.01, rtol=0))
    n_acc_flag = np.count_nonzero(np.isclose(acc, -6.6, atol=.001, rtol=0).all(axis=1))
    assert n_temp_flag == n_acc_flag == 0
    attachment_note = next(line.strip() for line in (folder / "S2_readme.txt").read_text(encoding="utf-8").splitlines()
                           if "RespiBAN temperature sensor" in line)
    ordinal, = (i for i, w in enumerate(s2_windows) if w.identifier == TARGET)
    target = s2_windows[ordinal]
    lo, hi = native_interval(target)
    adc, degrees, respiban = compare_respiban_temperature(folder / "S2_respiban.txt", lo, hi, native)
    assert np.all(labels[lo:hi] == target.original_label)
    segments = [row for row in read_csv(source / "label_segments.csv") if row["Subject"] == "S2"]
    segment, = (row for row in segments if int(row["Start_sample_700"]) <= lo and hi <= int(row["End_sample_700"]))
    assert int(segment["Label"]) == target.original_label
    assert "S2" in split["folds"][3]["validation"]
    x, r = x_values[ordinal], residual[ordinal]
    baseline = zscore(x, scalers[4]["baseline_z"])
    current = zscore(r, scalers[4]["centered_z"])
    peak = int(np.argmax(np.abs(current)))
    peak_source = target.start + peak
    peak_700 = peak_source * NATIVE_FS / FS
    peak_floor = int(peak_700)
    short_idx = range(peak_floor - 350, peak_floor + 351)
    write_csv(out, "s2_target_trace_64hz.csv",
              [{"index_64": target.start + i, "time_s": (target.start + i) / FS,
                "within_window_time_s": i / FS, "chest_temp_c": float(value), "centered_c": float(r[i]),
                "baseline_z": float(baseline[i]), "centered_z": float(current[i]), "quality_mask": bool(mask[target.start + i]),
                "original_label": target.original_label, "label": label_name(target.label)}
               for i, value in enumerate(x)])
    write_csv(out, "s2_native_peak_zoom.csv",
              [{"pkl_index_700": i, "time_s": i / NATIVE_FS, "temp_pkl_c": float(native[i]),
                "respiban_nseq": i + respiban["nseq_offset_from_pkl"], "respiban_adc": int(adc[i - lo]) if lo <= i < hi else None,
                "respiban_temp_c": float(degrees[i - lo]) if lo <= i < hi else None, "original_label": int(labels[i])}
               for i in short_idx])
    neighbors = s2_windows[ordinal - 3:ordinal + 4]
    neighbor_rows = []
    for fold in split["folds"]:
        fid = fold["fold_id"]
        for role in ROLES:
            if target.subject not in fold[role]:
                continue
            for offset, window in enumerate(neighbors, start=ordinal - 3):
                a, b = native_interval(window)
                assert np.all(labels[a:b] == window.original_label) and mask[window.start:window.end].all()
                old_z = zscore(x_values[offset], scalers[fid]["baseline_z"])
                z = zscore(residual[offset], scalers[fid]["centered_z"])
                overlap = max(0, min(target.end, window.end) - max(target.start, window.start))
                detail = window_detail(window, x_values[offset], residual[offset], z, peak_source)
                neighbor_rows.append({"fold_id": fid, "role": role, "relative_window_offset": offset - ordinal,
                                      "overlap_with_target_samples": overlap, "quality_mask_valid_samples": WIDTH,
                                      "native_min_c": float(native[a:b].min()), "native_max_c": float(native[a:b].max()),
                                      "native_median_c": float(np.median(native[a:b].astype(np.float64))),
                                      "baseline_max_abs_z": float(np.abs(old_z).max()), **detail})
    write_csv(out, "s2_neighbor_windows.csv", neighbor_rows)
    plot_s2_window(out, target, x, r, current, native, peak)
    plot_s2_context(out, neighbors, target, source_64, native, peak_source)
    wrist = timeline[target.start:target.end, channel_order.index("wrist_TEMP")]
    acceleration = timeline[target.start:target.end, [channel_order.index("chest_ACC_" + axis) for axis in "XYZ"]]
    close_native = native[peak_floor - 175:peak_floor + 176]
    return {
        "target": window_detail(target, x, r, current, peak_source),
        "resampling": {"source_rate_hz": NATIVE_FS, "destination_rate_hz": FS, "up": 16, "down": 175,
                       "samples_replayed_equal_float32": len(source_64), "max_difference_c": 0.0},
        "respiban_file_check": respiban, "native_window_indices_700": [lo, hi], "label_segment": segment,
        "all_s2_valid_mask_samples": int(mask.sum()), "s2_invalid_mask_samples": int((~mask).sum()),
        "raw_sentinel_temp_count": int(n_temp_flag), "raw_sentinel_acc_count": int(n_acc_flag),
        "device_attachment_note": attachment_note,
        "native_min_c": float(native[lo:hi].min()), "native_max_c": float(native[lo:hi].max()),
        "native_median_c": float(np.median(native[lo:hi].astype(np.float64))),
        "native_interpolated_index_at_peak": peak_700, "native_left_of_peak_c": float(native[peak_floor]),
        "native_right_of_peak_c": float(native[peak_floor + 1]),
        "native_near_peak_min_c": float(close_native.min()), "native_near_peak_max_c": float(close_native.max()),
        "temp_64_at_peak_c": float(x[peak]), "centered_at_peak_c": float(r[peak]), "baseline_z_at_peak": float(baseline[peak]),
        "fold4_train_centered_mean_c": float(scalers[4]["centered_z"][0]),
        "fold4_train_centered_std_c": float(scalers[4]["centered_z"][1]),
        "first_5s_median_c": float(np.median(x[:5 * FS])), "last_5s_median_c": float(np.median(x[-5 * FS:])),
        "max_64_adjacent_abs_diff_c": float(np.abs(np.diff(x)).max()), "wrist_temp_64_ptp_c": float(np.ptp(wrist)),
        "chest_acc_64_ptp_by_axis": np.ptp(acceleration, axis=0).tolist(), "neighbor_rows": len(neighbor_rows),
    }


def main():
    if not __debug__:
        raise RuntimeError("Assertions are required for this analysis.")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python analyze_extremes.py [new_audit_subdirectory]")
    out = HERE
    if len(sys.argv) == 2:
        if Path(sys.argv[1]).name != sys.argv[1] or sys.argv[1] in (".", ".."):
            raise ValueError("Give a single new directory name under this group.")
        out = GROUP / sys.argv[1]
        out.mkdir(exist_ok=False)
    elif any(p.name != Path(__file__).name for p in out.iterdir()):
        raise FileExistsError("Outputs already exist. Supply a new subdirectory name to run again.")
    previous, source, index, split, channel_order, meta = context()
    original = protected_hashes(source, meta)
    assert original[previous["window_manifest"]] == previous["window_manifest_sha256"]
    for path, files in ((source, meta["audit"]["source_file_sha256"]), (index, meta["index_file_sha256"])):
        assert all(original[(path / name).relative_to(ROOT).as_posix()] == expected for name, expected in files.items())
    channel = channel_order.index(TEMPERATURE)
    scalers = load_scalers(split, channel)
    windows, raw, centered = load_windows(source, index, split, channel)
    s2 = analyze_s2(out, source, split, channel_order, scalers, windows["S2"], raw["S2"], centered["S2"])
    print("S2 raw export, native PKL, 64 Hz series, mask, label and neighbors verified", flush=True)
    groups, by_window, histograms, scale_rows = analyze_distribution(out, split, windows, raw, centered, scalers)
    variation = summarize_variation(out, windows, centered)
    for row in groups:
        if row["scope"] != "role" or not isinstance(row["fold_id"], int):
            continue
        previous_fold = previous["diagnostics"]["per_fold"][row["fold_id"] - 1]
        method_name = "baseline" if row["method"] == "baseline_z" else "centered"
        field = f"{method_name}_{row['role']}_max_abs_z"
        assert row["max_abs_z"] == previous_fold[field]
    assert sum(w["role"] == "validation" and w["method"] == "centered_z" for w in by_window) == 2134
    assert protected_hashes(source, meta) == original
    figure_paths = sorted(p.relative_to(out).as_posix() for p in (out / "figures").iterdir())
    write_json(out, "audit_verification.json", {
        "status": "passed", "experiment_id": GROUP.parent.name, "group_id": GROUP.name,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "data_roles_analyzed": list(ROLES),
        "scalers_used": "Saved Train-only scalers from the baseline and the centered group",
        "model_calls": 0, "training_steps": 0, "protection_sha256_files": len(original),
        "input_file_collection_sha256": json_sha256(original), "unique_windows": sum(map(len, windows.values())),
        "quantiles_pct": QUANTILES, "strict_abs_thresholds": THRESHOLDS,
        "counts": {"group_statistics_rows": len(groups), "window_statistics_rows": len(by_window),
                   "histogram_rows": len(histograms), "variation_rows": len(variation), "train_scale_rows": len(scale_rows)},
        "figures": figure_paths, "numpy": np.__version__, "scipy": scipy.__version__, "s2_evidence": s2,
    })
    print(f"Verified {len(original)} input file hashes and {len(by_window)} fold/role/method window rows.", flush=True)


if __name__ == "__main__":
    main()
