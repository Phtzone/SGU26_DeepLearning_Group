"""Build the FER2013 report, slides and experiment workbook from a completed run."""

import argparse
import csv
import json
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
INK = "#10263a"
ACCENT = "#007276"
MUTED = "#486071"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK})


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_journal(path, results, runs, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    members = [("Họ tên", "MSSV", "Phần việc")]
    if path.exists() and path.stat().st_size:
        old = load_workbook(path, read_only=True)
        try:
            if "Phan cong" in old.sheetnames:
                members = list(old["Phan cong"].values)
        finally:
            old.close()
    workbook = Workbook()
    overview = workbook.active
    overview.title = "Thi nghiem"
    overview.append(("Cấu hình", "Lật ngang", "Dropout", "Optimizer", "Learning rate",
                     "Batch size", "Seed", "Số epoch", "Epoch tốt nhất", "Val accuracy",
                     "Val loss", "Thời gian (s)"))
    epoch_sheet = workbook.create_sheet("Ket qua tung epoch")
    epoch_sheet.append(("Cấu hình", "Epoch", "Train loss", "Train accuracy",
                        "Val loss", "Val accuracy"))
    for run in runs:
        overview.append((run["config"], run["horizontal_flip"] == "True", float(run["dropout"]),
                         run["optimizer"], float(run["learning_rate"]), int(run["batch_size"]),
                         int(run["seed"]), int(run["epochs"]), int(run["best_epoch"]),
                         float(run["val_accuracy"]), float(run["val_loss"]), float(run["seconds"])))
        for row in csv_rows(results / f"{run['config']}_epochs.csv"):
            epoch_sheet.append((run["config"], int(row["epoch"]), float(row["train_loss"]),
                                float(row["train_accuracy"]), float(row["val_loss"]),
                                float(row["val_accuracy"])))
    for sheet, columns in ((overview, (10,)), (epoch_sheet, (4, 6))):
        for column in columns:
            for cell in list(sheet.columns)[column - 1][1:]:
                cell.number_format = "0.00%"
    test = workbook.create_sheet("PrivateTest")
    test.append(("Chỉ số", "Giá trị"))
    test.append(("Cấu hình được chọn", metrics["selected_config"]))
    test.append(("Accuracy", metrics["test_accuracy"]))
    test.append(("Macro F1", metrics["test_macro_f1"]))
    test.append(("Loss", metrics["test_loss"]))
    for name, score in metrics["class_f1"].items():
        test.append((f"F1 – {name}", score))
    for index in (3, 4, *range(6, test.max_row + 1)):
        test.cell(index, 2).number_format = "0.00%"
    assignment = workbook.create_sheet("Phan cong")
    for row in members:
        assignment.append(row)
    for sheet in workbook:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="007276")
        for index, column in enumerate(sheet.columns, 1):
            width = max(len(str(cell.value or "")) for cell in column) + 4
            sheet.column_dimensions[get_column_letter(index)].width = max(14, min(width, 48))
    workbook.save(path)


def page(title, index, slide=False):
    figure = plt.figure(figsize=(13.33, 7.5) if slide else (8.27, 11.69), facecolor="#fbfcfd")
    figure.text(0.07, 0.945, title, fontsize=25 if slide else 20, weight="bold")
    figure.text(0.07, 0.035, "BTL02   /   FER2013", fontsize=9, color=MUTED)
    figure.text(0.93, 0.035, str(index), fontsize=9, color=MUTED, ha="right")
    return figure


def block(figure, title, text, top, width=75, size=11):
    figure.text(0.07, top, title, fontsize=size + 3, color=ACCENT, weight="bold", va="top")
    figure.text(0.07, top - 0.043, "\n".join(wrap(text, width)), fontsize=size, va="top",
                linespacing=1.55)


def picture(figure, path, bounds):
    axis = figure.add_axes(bounds)
    axis.imshow(plt.imread(path))
    axis.axis("off")


def write_page(pdf, figure):
    pdf.savefig(figure)
    plt.close(figure)


def comparison_table(figure, runs, bounds):
    axis = figure.add_axes(bounds)
    axis.axis("off")
    rows = [(run["config"], "Có" if run["horizontal_flip"] == "True" else "Không",
             run["dropout"], run["best_epoch"], f"{float(run['val_accuracy']):.2%}",
             f"{float(run['val_loss']):.3f}") for run in runs]
    table = axis.table(cellText=rows, colLabels=["Cấu hình", "Lật ngang", "Dropout",
                                                "Epoch", "Val acc.", "Val loss"],
                       colWidths=[0.24, 0.16, 0.15, 0.12, 0.17, 0.16],
                       cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2e8")
        cell.set_facecolor(ACCENT if row == 0 else "#f1f6f8" if row % 2 else "white")
        cell.get_text().set_color("white" if row == 0 else INK)


def make_documents(output, results, runs, metrics):
    selected = metrics["selected_config"]
    best = next(run for run in runs if run["config"] == selected)
    scores = {run["config"]: float(run["val_accuracy"]) for run in runs}
    accuracy = f"{metrics['test_accuracy']:.2%}"
    macro_f1 = f"{metrics['test_macro_f1']:.2%}"
    labels = metrics["class_order"]
    with (results / "confusion_matrix.csv").open(encoding="utf-8", newline="") as file:
        matrix = [[int(value) for value in row] for row in list(csv.reader(file))[1:]]
    worst_source, worst_target = max(
        ((source, target) for source in range(7) for target in range(7) if source != target),
        key=lambda pair: matrix[pair[0]][pair[1]])
    class_f1 = metrics["class_f1"]
    weakest = min(class_f1, key=class_f1.get)
    strongest = max(class_f1, key=class_f1.get)
    conclusion = (
        f"Cấu hình {selected} đạt {float(best['val_accuracy']):.2%} trên PublicTest ở epoch "
        f"{best['best_epoch']}. Trên PrivateTest, accuracy đạt {accuracy}, macro F1 đạt {macro_f1}."
    )
    errors = (
        f"Cặp nhầm lẫn nhiều nhất theo số ảnh là {labels[worst_source]} → {labels[worst_target]} "
        f"({matrix[worst_source][worst_target]} ảnh). F1 thấp nhất thuộc về {weakest} "
        f"({class_f1[weakest]:.2%}), cao nhất thuộc về {strongest} ({class_f1[strongest]:.2%})."
    )
    if {"baseline", "flip", "dropout", "flip_dropout"} <= scores.keys():
        flip_low = 100 * (scores["flip"] - scores["baseline"])
        flip_high = 100 * (scores["flip_dropout"] - scores["dropout"])
        drop_no_flip = 100 * (scores["dropout"] - scores["baseline"])
        drop_with_flip = 100 * (scores["flip_dropout"] - scores["flip"])
        effects = (f"Lật ngang làm accuracy validation thay đổi {flip_low:+.2f} điểm phần trăm "
                   f"ở dropout 0,25 và {flip_high:+.2f} điểm ở dropout 0,45. Tăng dropout làm "
                   f"accuracy thay đổi {drop_no_flip:+.2f} điểm khi tắt lật ngang và "
                   f"{drop_with_flip:+.2f} điểm khi bật lật ngang.")
    else:
        effects = "Bảng so sánh ghi accuracy và loss tại checkpoint tốt nhất của từng lượt chạy."
    final_epoch = csv_rows(results / f"{selected}_epochs.csv")[-1]
    train_end = float(final_epoch["train_accuracy"])
    val_end = float(final_epoch["val_accuracy"])
    gap = (f"Ở epoch cuối của {selected}, accuracy train là {train_end:.2%}, validation "
           f"là {val_end:.2%}. Chênh lệch là {100 * (train_end - val_end):+.2f} điểm phần trăm.")
    limits = ("Mỗi cấu hình dùng một seed. Cần lặp lại với nhiều seed để biết các chênh lệch "
              "nhỏ giữa các cấu hình có ổn định hay không.")
    support = sum(matrix[labels.index(weakest)])
    class_context = (f"Lớp {weakest} có {support} ảnh trong PrivateTest. Macro F1 cho mỗi lớp "
                     "trọng số bằng nhau, nên kết quả của các lớp ít ảnh vẫn ảnh hưởng đến chỉ số này.")
    filter_explanation = (
        "Hình gồm 96 bộ lọc đầu ra. Với mỗi bộ lọc, chọn lát 3×3 của kênh đầu vào có "
        "tổng bình phương trọng số lớn nhất. Màu đỏ biểu thị trọng số dương, xanh biểu thị "
        "trọng số âm theo cùng một thang màu."
    )
    architecture = ("AvgPool 2×2 đưa ảnh 48×48 về 24×24. Ba khối Conv 3×3, BatchNorm, ReLU "
                    "và MaxPool dùng 24, 48 và 96 kênh. Bộ phân loại dùng Dense 128, "
                    "ReLU và Dropout để tạo 7 logits, huấn luyện bằng CrossEntropyLoss.")
    settings = (f"AdamW, learning rate {best['learning_rate']}, batch {best['batch_size']}, "
                f"{best['epochs']} epoch, seed {best['seed']}, thiết bị {metrics['device'].upper()}. "
                "Chọn theo accuracy PublicTest cao nhất, dùng loss để phân giải khi hòa.")
    data = ("FER2013 gồm 35.887 ảnh xám 48×48. Cột Usage chia dữ liệu thành 28.709 ảnh "
            "Training, 3.589 ảnh PublicTest và 3.589 ảnh PrivateTest. Đưa pixel về [0, 1].")
    with PdfPages(output / "report.pdf", metadata={"Title": "BTL02 - Báo cáo FER2013"}) as pdf:
        figure = page("Nhận dạng biểu cảm khuôn mặt", 1)
        block(figure, "Kết quả", conclusion, 0.86)
        block(figure, "Dữ liệu", data + " Bảy lớp gồm " + ", ".join(labels) + ".", 0.71)
        block(figure, "Mô hình", architecture, 0.55)
        block(figure, "Thiết kế thực nghiệm", settings, 0.37)
        block(figure, "Mã nguồn", "code/train.py đọc CSV, tạo CNN, chạy huấn luyện và xuất "
              "checkpoint, nhật ký từng epoch, ma trận nhầm lẫn và hình bộ lọc.",
              0.20)
        write_page(pdf, figure)

        figure = page("So sánh các cấu hình", 2)
        comparison_table(figure, runs, [0.07, 0.66, 0.86, 0.20])
        picture(figure, results / "learning_curves.png", [0.07, 0.39, 0.86, 0.24])
        block(figure, "Ảnh hưởng của lật ngang và dropout", effects, 0.34)
        block(figure, "Khoảng cách train và validation", gap + " " + limits, 0.19)
        write_page(pdf, figure)

        figure = page("Đánh giá và bộ lọc cuối", 3)
        block(figure, "Ma trận nhầm lẫn trên PrivateTest", errors + " " + class_context, 0.87)
        picture(figure, results / "confusion_matrix.png", [0.20, 0.46, 0.60, 0.26])
        block(figure, "Trọng số lớp convolution cuối", filter_explanation, 0.45)
        picture(figure, results / "final_conv_filters.png", [0.12, 0.085, 0.76, 0.24])
        figure.text(0.07, 0.063, "Dữ liệu từ Goodfellow và cộng sự, Challenges in "
                    "Representation Learning, 2013.", fontsize=8, color=MUTED)
        write_page(pdf, figure)

    with PdfPages(output / "slides.pdf", metadata={"Title": "BTL02 - Trình bày FER2013"}) as pdf:
        figure = page("Nhận dạng biểu cảm khuôn mặt", 1, slide=True)
        block(figure, "Bài toán phân loại 7 biểu cảm bằng CNN", data, 0.78, width=82, size=16)
        block(figure, "Kết quả trên PrivateTest", conclusion, 0.42, width=82, size=16)
        write_page(pdf, figure)

        figure = page("Kiến trúc và cách huấn luyện", 2, slide=True)
        block(figure, "CNN PyTorch", architecture, 0.79, width=90, size=16)
        block(figure, "Chọn checkpoint trên PublicTest", settings, 0.46, width=90, size=16)
        write_page(pdf, figure)

        figure = page("Kết quả trên PublicTest", 3, slide=True)
        comparison_table(figure, runs, [0.07, 0.47, 0.86, 0.34])
        block(figure, "So sánh theo từng yếu tố", effects, 0.37, width=105, size=14)
        write_page(pdf, figure)

        figure = page("Diễn biến qua các epoch", 4, slide=True)
        picture(figure, results / "learning_curves.png", [0.07, 0.19, 0.86, 0.62])
        figure.text(0.07, 0.14, "\n".join(wrap(gap + " " + limits, 116)), fontsize=12, va="top")
        write_page(pdf, figure)

        figure = page("Đánh giá trên PrivateTest", 5, slide=True)
        picture(figure, results / "confusion_matrix.png", [0.42, 0.10, 0.54, 0.77])
        block(figure, f"Acc. {accuracy} | Macro F1 {macro_f1}", errors, 0.72, width=42, size=14)
        write_page(pdf, figure)

        figure = page("Trực quan hóa bộ lọc convolution cuối", 6, slide=True)
        picture(figure, results / "final_conv_filters.png", [0.06, 0.18, 0.88, 0.69])
        figure.text(0.07, 0.12, "\n".join(wrap(filter_explanation, 116)), fontsize=12, va="top")
        write_page(pdf, figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    runs = csv_rows(args.results / "experiments.csv")
    metrics = json.loads((args.results / "test_metrics.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    make_documents(args.output, args.results, runs, metrics)
    save_journal(args.output / "journal" / "detail.xlsx", args.results, runs, metrics)
