"""Controlled depth experiment: fixed width, data and optimizer; vary hidden layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_experiments_ab import (
    DEFAULT_SEEDS, METRIC_FIELDS, RUN_FIELDS, _aggregate, _save_run_plot, _write_rows,
)
from train import _save_prediction_plot, _save_surface_plot
from training import (
    TrainingConfig, evaluate_model, load_csv_dataset, save_history_csv, train_model,
)

PROJECT_DIR = Path(__file__).resolve().parent


def save_comparisons(summary, histories, output):
    depths = [int(row["value"]) for row in summary]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for split in ("train", "validation", "test"):
        axes[0].errorbar(
            depths, [row[f"{split}_mse_mean"] for row in summary],
            yerr=[row[f"{split}_mse_std"] for row in summary],
            marker="o", capsize=3, label=split.capitalize(),
        )
    axes[0].set(ylabel="MSE (log scale)", yscale="log", title="Train / validation / test MSE")
    axes[0].legend()
    for axis, metric, title, unit in (
        (axes[1], "test_mae", "Test MAE", "MAE"),
        (axes[2], "training_seconds", "Training time", "Seconds"),
    ):
        axis.errorbar(depths, [r[f"{metric}_mean"] for r in summary],
                      yerr=[r[f"{metric}_std"] for r in summary], marker="o", capsize=3)
        axis.set(title=title, ylabel=unit)
    for axis in axes:
        axis.set(xlabel="Hidden layers (C)", xticks=depths)
        axis.grid(True, alpha=0.3)
    figure.suptitle("Experiment C: mean +/- sample standard deviation across seeds")
    figure.tight_layout()
    figure.savefig(output / "experiment_c_errors_and_time.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    # Each seed is drawn separately: no padding or averaging after early stopping.
    for axis, depth in zip(axes.flat, depths):
        for seed, history in histories[depth]:
            epochs = [row["epoch"] for row in history]
            line, = axis.plot(epochs, [row["train_loss"] for row in history],
                              label=f"Train, seed {seed}", alpha=0.8)
            axis.plot(epochs, [row["validation_loss"] for row in history],
                      linestyle="--", color=line.get_color(), label=f"Validation, seed {seed}")
        axis.set(title=f"C = {depth}", xlabel="Epoch", ylabel="MSE loss", yscale="log")
        axis.legend(fontsize=7)
        axis.grid(True, alpha=0.3)
    for axis in list(axes.flat)[len(depths):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "experiment_c_convergence.png", dpi=180)
    plt.close(figure)


def save_analysis(summary, rows, manifest, output):
    selected = min(summary, key=lambda row: row["validation_mse_mean"])
    lines = [
        "# Thực nghiệm C – Ảnh hưởng của số tầng ẩn", "",
        "Chỉ thay đổi số tầng ẩn C; mọi tầng ẩn có cùng độ rộng. "
        "Dùng lại nguyên vẹn các CSV train/validation/test của A và B.", "",
        f"Cấu hình: seeds {manifest['seeds']}; {manifest['training_samples']} mẫu train, "
        f"{manifest['validation_samples']} validation, {manifest['test_samples']} test; "
        f"{manifest['hidden_width']} nơ-ron/tầng; activation {manifest['activation']}; "
        f"Adam; batch size {manifest['training_config']['batch_size']}; "
        f"learning rate {manifest['training_config']['learning_rate']}; "
        f"tối đa {manifest['training_config']['max_epochs']} epochs; "
        f"early stopping patience {manifest['training_config']['patience']}, "
        f"min_delta {manifest['training_config']['min_delta']}; weight decay 0. "
        "Chuẩn hóa đầu vào x1/2, x2/4; đầu ra tuyến tính; loss MSE. "
        "Checkpoint mỗi run được chọn bằng validation MSE, không dùng test.", "",
        "## Bảng kết quả", "",
        "Trung bình ± độ lệch chuẩn mẫu giữa các seeds (không phải khoảng tin cậy).", "",
        "| C | Tham số | Train MSE | Validation MSE | Test MSE | Test MAE | Test RMSE | Train (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        depth = int(row["value"])
        parameters = next(r["parameter_count"] for r in rows if r["value"] == depth)
        cells = [f"{row[m + '_mean']:.6g} ± {row[m + '_std']:.3g}" for m in
                 ("train_mse", "validation_mse", "test_mse", "test_mae", "test_rmse", "training_seconds")]
        lines.append(f"| {depth} | {parameters} | " + " | ".join(cells) + " |")
    first, last = summary[0], summary[-1]
    lines += ["", "## Nhận xét", "",
        f"- Chọn C = {selected['value']} theo validation MSE trung bình nhỏ nhất "
        f"({selected['validation_mse_mean']:.6g}); test chỉ dùng mô tả chất lượng cuối cùng.",
        f"- Từ C = {first['value']} đến C = {last['value']}, train MSE đổi từ "
        f"{first['train_mse_mean']:.6g} thành {last['train_mse_mean']:.6g}; validation MSE "
        f"từ {first['validation_mse_mean']:.6g} thành {last['validation_mse_mean']:.6g}; "
        f"test MSE từ {first['test_mse_mean']:.6g} thành {last['test_mse_mean']:.6g}.",
        "- Nhiều tầng tạo thêm các phép biến đổi phi tuyến, giúp biểu diễn đỉnh Gaussian "
        "và vùng biến thiên hẹp. Tuy nhiên độ sâu cũng tăng số tham số và độ khó tối ưu; "
        "không thể quy toàn bộ chênh lệch cho độ sâu độc lập với dung lượng mô hình.",
        f"- Thời gian train trung bình ở hai đầu dải C là "
        f"{first['training_seconds_mean']:.3f}s và {last['training_seconds_mean']:.3f}s. "
        "Đây là thời gian vòng train/validation, không gồm xuất hình hay đánh giá test. "
        "Thời gian còn phụ thuộc số epoch dừng và môi trường máy; không so trực tiếp "
        "với thời gian A/B từ một phiên chạy khác.",
    ]
    monotonic = all(b['validation_mse_mean'] <= a['validation_mse_mean']
                    for a, b in zip(summary, summary[1:]))
    lines.append("- " + (
        "Trong dải đã khảo sát, validation MSE trung bình giảm khi tăng C. "
        if monotonic else
        "Validation MSE trung bình không giảm đều khi tăng C. "
    ) + "Khởi tạo ngẫu nhiên và quỹ đạo tối ưu có thể tạo khác biệt giữa seeds; "
        "mạng sâu hơn không bảo đảm tốt hơn ở mọi lần chạy. Cần đọc cả độ lệch chuẩn "
        "và đường hội tụ thay vì chỉ một giá trị trung bình.")
    for row in summary:
        ratio = row["validation_mse_mean"] / max(row["train_mse_mean"], 1e-15)
        depth_runs = [run for run in rows if run["value"] == row["value"]]
        epochs = [run["epochs_completed"] for run in depth_runs]
        best_epochs = [run["best_epoch"] for run in depth_runs]
        lines.append(f"- C = {row['value']}: validation/train MSE = {ratio:.3f}. " + (
            "Validation cao hơn train đáng kể: cần thận trọng về khoảng cách tổng quát hóa; "
            "chỉ riêng tỷ số này chưa đủ kết luận quá khớp."
            if ratio > 1.5 else
            "Train và validation cùng bậc sai số; chưa thấy khoảng cách lớn chứng minh quá khớp."
        ))
        lines.append(f"  Số epoch thực chạy: {min(epochs)}–{max(epochs)}; "
                     f"epoch checkpoint tốt nhất: {min(best_epochs)}–{max(best_epochs)}. "
                     + ("Có run chạm trần epoch, nên chưa thể khẳng định đã hội tụ hoàn toàn."
                        if max(epochs) == manifest['training_config']['max_epochs'] else
                        "Các run dừng sớm theo mức cải thiện validation đã quy định."))
    lines += [
        "- Xem `experiment_c_convergence.png`: nét liền là train, nét đứt là validation; "
        "từng seed được vẽ riêng và dừng đúng epoch thực tế. Loss train trong log là "
        "trung bình các minibatch khi cập nhật trọng số; train MSE trong bảng được tính "
        "lại tại checkpoint tốt nhất nên có thể khác điểm cuối đường loss.",
        "- Giới hạn: 3 seeds mặc định, test chỉ 100 mẫu và một bộ dữ liệu cố định; "
        "không khẳng định ý nghĩa thống kê hay độ sâu tối ưu cho mọi bài toán. "
        "Early stopping và trần epoch cố định có thể khiến cấu hình dừng trước khi hội tụ hoàn toàn.",
        "", "## Minh chứng", "",
        "- `runs.csv`, `summary.csv`: số liệu từng run và thống kê tổng hợp.",
        "- `experiment_config.json`: thiết lập, phiên bản môi trường và SHA-256 dữ liệu.",
        "- `runs/C_*/`: checkpoint, cấu hình/metrics, loss CSV, learning curves, "
        "test predictions CSV/hình và bề mặt xấp xỉ cho từng run.",
        "- `experiment_c_errors_and_time.png`: sai số và thời gian theo C.",
        "- `experiment_c_convergence.png`: so sánh hội tụ train/validation.",
    ]
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--hidden-width", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--activation", choices=("tanh", "relu", "sigmoid"), default="tanh")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results/experiments/C")
    args = parser.parse_args()
    if not args.depths or len(set(args.depths)) != len(args.depths) or any(d not in (1, 2, 3, 4) for d in args.depths):
        parser.error("--depths must be distinct values from 1, 2, 3, 4")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("--seeds must be distinct non-negative integers")
    if args.hidden_width <= 0 or args.threads <= 0:
        parser.error("--hidden-width and --threads must be positive")
    torch.set_num_threads(args.threads)
    config = TrainingConfig(batch_size=args.batch_size, max_epochs=args.epochs,
                            learning_rate=args.learning_rate, patience=args.patience, device=args.device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_DIR / "data"
    train = load_csv_dataset(data_dir / "train.csv")
    validation = load_csv_dataset(data_dir / "validation.csv")
    manifest = {
        "experiment": "C", "depths": sorted(args.depths), "seeds": args.seeds,
        "hidden_width": args.hidden_width, "activation": args.activation,
        "optimizer": "Adam", "training_config": asdict(config),
        "training_samples": len(train), "validation_samples": len(validation),
        "test_samples": None, "torch_threads": args.threads,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "dataset_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                           for name in ("train.csv", "validation.csv", "test.csv")},
    }
    rows, histories = [], {}
    for depth in sorted(args.depths):
        histories[depth] = []
        for seed in args.seeds:
            run_dir = output / "runs" / f"C_{depth:04d}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_config = TrainingConfig(**{**asdict(config), "seed": seed})
            result = train_model(train, validation, hidden_sizes=(args.hidden_width,) * depth,
                                 activation=args.activation, config=run_config,
                                 checkpoint_path=run_dir / "best_model.pt")
            metrics = {"train": evaluate_model(result.model, train, result.device)[0],
                       "validation": evaluate_model(result.model, validation, result.device)[0]}
            # The test split never participates in optimization or checkpoint selection.
            test = load_csv_dataset(data_dir / "test.csv")
            metrics["test"], targets, predictions = evaluate_model(result.model, test, result.device)
            manifest["test_samples"] = len(test)
            save_history_csv(run_dir / "training_history.csv", result.history)
            _save_run_plot("C", depth, result.history, run_dir / "learning_curves.png")
            _save_prediction_plot(targets, predictions, run_dir / "test_predictions.png")
            _save_surface_plot(result.model, run_dir / "predicted_surface.png", result.device)
            inputs = test.tensors[0].numpy()
            np.savetxt(run_dir / "test_predictions.csv", np.column_stack((inputs, targets, predictions)),
                       delimiter=",", header="x1,x2,y,predicted_y", comments="")
            try:
                recorded_dir = run_dir.relative_to(PROJECT_DIR)
            except ValueError:
                recorded_dir = run_dir
            row = dict(experiment="C", value=depth, seed=seed, training_samples=len(train),
                       hidden_width=args.hidden_width, hidden_layers=depth, activation=args.activation,
                       batch_size=config.batch_size, learning_rate=config.learning_rate,
                       max_epochs=config.max_epochs, patience=config.patience,
                       epochs_completed=len(result.history), best_epoch=result.best_epoch,
                       best_validation_mse=result.best_validation_loss,
                       train_mse=metrics["train"]["mse"], validation_mse=metrics["validation"]["mse"],
                       test_mse=metrics["test"]["mse"], test_mae=metrics["test"]["mae"],
                       test_rmse=metrics["test"]["rmse"], training_seconds=result.training_seconds,
                       device=result.device, run_dir=str(recorded_dir),
                       parameter_count=sum(p.numel() for p in result.model.parameters()))
            details = {**row, "model": {"hidden_sizes": list(result.model.hidden_sizes),
                       "activation": args.activation, "input_scale": [2.0, 4.0]},
                       "training_config": asdict(run_config), "optimizer": "Adam",
                       "metrics": metrics}
            (run_dir / "run_config_and_metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
            rows.append(row)
            histories[depth].append((seed, result.history))
            _write_rows(output / "runs.csv", RUN_FIELDS + ("parameter_count",), rows)
            print(f"C={depth}, seed={seed}: val_mse={row['validation_mse']:.6g}, "
                  f"test_mse={row['test_mse']:.6g}, seconds={result.training_seconds:.2f}", flush=True)
    summary = _aggregate("C", rows)
    fields = ("experiment", "value", "runs") + tuple(
        f"{metric}_{stat}" for metric in METRIC_FIELDS for stat in ("mean", "std"))
    _write_rows(output / "summary.csv", fields, summary)
    (output / "experiment_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_comparisons(summary, histories, output)
    save_analysis(summary, rows, manifest, output)
    print(f"Experiment C results: {output}")


if __name__ == "__main__":
    main()
