"""Build the combined BTL02 report and slides from the completed rounds."""

import argparse
import json
from pathlib import Path
from textwrap import wrap
from zipfile import ZIP_DEFLATED, ZipFile

from make_submission import (
    ACCENT, INK, ROOT, PdfPages, block, csv_rows, page, picture, validation_curve, write_page,
)
from matplotlib.ticker import PercentFormatter
import numpy as np


def validation_rank(run):
    return float(run["val_macro_f1"]), -float(run["val_loss"])


def pct(value):
    return f"{float(value):.2%}".replace(".", ",")


def number(value):
    return f"{float(value):.2f}".replace(".", ",")


def table(figure, headings, rows, bounds, widths, size=10):
    axis = figure.add_axes(bounds)
    axis.axis("off")
    grid = axis.table(cellText=rows, colLabels=headings, colWidths=widths,
                      cellLoc="center", bbox=(0, 0, 1, 1))
    grid.auto_set_font_size(False)
    grid.set_fontsize(size)
    for (row, _), cell in grid.get_celld().items():
        cell.set_edgecolor("#d9e2e8")
        cell.set_facecolor(ACCENT if row == 0 else "#f1f6f8" if row % 2 else "white")
        text = cell.get_text()
        text.set_color("white" if row == 0 else INK)
        text.set_weight("bold" if row == 0 else "normal")


def run_table(figure, rows, bounds):
    values = []
    for date, run in rows:
        score = run.get("val_macro_f1")
        values.append((date, run["config"], run["epochs"], run["best_epoch"],
                       pct(run["val_accuracy"]), pct(score) if score else "—"))
    table(figure, ["Ngày", "Cấu hình", "Epoch chạy", "Epoch chọn", "Val acc.", "Val macro F1"],
          values, bounds, [0.10, 0.29, 0.135, 0.135, 0.15, 0.19], size=9)


def test_table(figure, results, bounds):
    rows = [(day, data["selected_config"], pct(data["test_accuracy"]),
             pct(data["test_macro_f1"])) for day, data in results]
    table(figure, ["Ngày", "Checkpoint", "Accuracy", "Macro F1"], rows, bounds,
          [0.16, 0.34, 0.25, 0.25])


def class_plot(figure, initial, final, final_day, bounds):
    left, bottom, width, height = bounds
    axis = figure.add_axes([left + 0.07, bottom, width - 0.07, height])
    classes = final["class_order"]
    index = np.arange(len(classes))
    axis.barh(index + 0.17, [initial["class_f1"][name] for name in classes],
              height=0.3, color="#9cb4c7", label="21/11 • flip")
    axis.barh(index - 0.17, [final["class_f1"][name] for name in classes],
              height=0.3, color=ACCENT, label=f"{final_day} • {final['selected_config']}")
    axis.set(yticks=index, yticklabels=classes, xlabel="F1 trên PrivateTest", xlim=(0, 0.85))
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right", fontsize=8)


def pooling_curve(figure, rounds, bounds):
    axis = figure.add_axes(bounds)
    for folder, run in rounds:
        history = csv_rows(folder / f"{run['config']}_epochs.csv")
        epochs = [int(row["epoch"]) for row in history]
        scores = [float(row["val_macro_f1"]) for row in history]
        line, = axis.plot(epochs, scores, label=run["config"])
        best = int(run["best_epoch"])
        axis.scatter(best, scores[best - 1], color=line.get_color(), zorder=3)
    axis.set(xlabel="Epoch", ylabel="Macro F1 trên PublicTest")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, loc="lower right")


def make_documents(baseline, followup, output, third=None):
    initial_round = csv_rows(baseline / "experiments.csv")
    later = csv_rows(followup / "experiments.csv")
    rows = [("21/11", run) for run in initial_round] + [("23/11", run) for run in later]
    control, weighted = (next(run for run in later if run["config"] == config)
                         for config in ("flip", "flip_weighted"))
    initial, second_test = (json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
                            for folder in (baseline, followup))
    tests = [("21/11", initial), ("23/11", second_test)]
    selected = weighted
    final_folder, final_day, final = followup, "23/11", second_test
    late = None
    if third is not None:
        late, = csv_rows(third / "experiments.csv")
        rows.append(("26/11", late))
        if validation_rank(late) > validation_rank(weighted):
            selected, final_folder, final_day = late, third, "26/11"
            final = json.loads((third / "test_metrics.json").read_text(encoding="utf-8"))
            tests.append((final_day, final))
    control_score = float(control["val_macro_f1"])
    weighted_score = float(weighted["val_macro_f1"])
    data = ("FER2013 gồm 35.887 ảnh mặt xám 48×48, chia sẵn theo cột Usage. "
            "Dùng 28.709 ảnh Training để học, 3.589 ảnh PublicTest để chọn checkpoint và "
            "3.589 ảnh PrivateTest để đánh giá checkpoint đã chọn.")
    classes = "Angry, Disgust, Fear, Happy, Sad, Surprise và Neutral."
    architecture = ("Ở mạng cơ sở, ảnh 1×48×48 được chuẩn hóa pixel về [0, 1] và qua AvgPool 2×2. "
                    "Ba khối Conv 3×3, BatchNorm, ReLU và MaxPool lần lượt có 24, 48 và 96 kênh. "
                    "Bộ phân loại dùng Dense 128, ReLU và Dropout để tạo 7 logits.")
    settings = ("AdamW, learning rate 0,001, batch size 256 và seed 42 trên CPU. Khi bật "
                "augmentation, mỗi ảnh Training có xác suất 50% được lật ngang. "
                "Các tập đánh giá dùng ảnh gốc đã chuẩn hóa và cross-entropy chuẩn.")
    first_design = ("21/11 · Bốn cấu hình, mỗi cấu hình 12 epoch. So sánh lật ngang bật/tắt "
                    "và dropout 0,25/0,45. Chọn checkpoint theo accuracy PublicTest.")
    second_design = ("23/11 · Chạy lại flip và flip_weighted với cùng ngân sách 24 epoch. "
                     "Trọng số lớp là nghịch đảo căn bậc hai số ảnh Training rồi chuẩn hóa "
                     "về trung bình 1. Chọn checkpoint theo macro F1 PublicTest. Khi hòa, "
                     "cả hai vòng ưu tiên validation loss thấp hơn.")
    summary = (f"Checkpoint cuối là {final['selected_config']} ngày {final_day}, epoch "
               f"{selected['best_epoch']}. Trên PublicTest, macro F1 đạt "
               f"{pct(selected['val_macro_f1'])}. "
               f"Trên PrivateTest, accuracy đạt {pct(final['test_accuracy'])}, macro F1 đạt "
               f"{pct(final['test_macro_f1'])}.")
    weighting = (f"Ở vòng 23/11, thêm trọng số lớp làm macro F1 tăng từ {pct(control_score)} lên "
                 f"{pct(weighted_score)} trên PublicTest, tương đương "
                 f"{number(100 * (weighted_score - control_score))} điểm phần trăm.")
    if late is not None:
        pool_design = ("Giữ lật ngang, dropout 0,25, trọng số lớp, seed 42 và ngân sách 24 epoch. "
                       "Dời AvgPool từ đầu mạng xuống sau khối Conv đầu tiên để khối này xử lý ảnh "
                       "48×48. Từ khối thứ hai, kích thước đặc trưng giữ như mạng cơ sở.")
        pool_result = (f"Macro F1 PublicTest {pct(weighted['val_macro_f1'])} → "
                       f"{pct(late['val_macro_f1'])}. Accuracy {pct(weighted['val_accuracy'])} → "
                       f"{pct(late['val_accuracy'])}. Chọn {final['selected_config']} dựa trên "
                       "macro F1, dùng loss khi hòa.")
        if final_folder != third:
            pool_result += " Lượt 26/11 được đánh giá trên PublicTest."
    records = sum(int(run["epochs"]) for _, run in rows)
    seconds = round(sum(float(run["seconds"]) for _, run in rows))
    provenance = (f"Script tự ghi {records} dòng sau từng epoch của {len(rows)} lượt chạy "
                  f"trong {len({date for date, _ in rows})} ngày. Tổng thời gian các lượt khoảng {seconds // 60} phút "
                  f"{seconds % 60} giây theo cột seconds, chưa tính chuẩn bị dữ liệu và vẽ hình.")
    limits = ("Mỗi cấu hình dùng một seed. So sánh PrivateTest với vòng 21/11 mang tính mô tả "
              "vì tiêu chí chọn checkpoint và ngân sách epoch khác nhau. Các đối chứng cho "
              "trọng số lớp và vị trí pooling dùng PublicTest.")
    baseline_scores = {run["config"]: float(run["val_accuracy"]) for run in initial_round}
    flips = (f"Ở dropout 0,25, lật ngang tăng accuracy PublicTest từ {pct(baseline_scores['baseline'])} "
             f"lên {pct(baseline_scores['flip'])}. Ở dropout 0,45, lật ngang đưa accuracy từ "
             f"{pct(baseline_scores['dropout'])} xuống {pct(baseline_scores['flip_dropout'])}. "
             "Hiệu ứng quan sát được phụ thuộc mức dropout.")
    early_scores = []
    for run in (control, weighted):
        history = csv_rows(followup / f"{run['config']}_epochs.csv")
        score = max(float(item["val_macro_f1"]) for item in history[:12])
        early_scores.append(f"{run['config']} {pct(score)} → {pct(run['val_macro_f1'])}")
    budget = "Macro F1 tốt nhất sau 12 và 24 epoch trên PublicTest. " + ", ".join(early_scores) + "."
    matrix = np.loadtxt(final_folder / "confusion_matrix.csv", delimiter=",", skiprows=1, dtype=int)
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    actual, guessed = np.unravel_index(errors.argmax(), errors.shape)
    labels = final["class_order"]
    weakest = sorted(labels, key=final["class_f1"].get)[:2]
    weakest_f1 = " và ".join(f"{label} ({pct(final['class_f1'][label])})" for label in weakest)
    mistakes = (f"Nhầm nhiều nhất là {labels[actual]} → {labels[guessed]} "
                f"({errors[actual, guessed]} ảnh). F1 thấp nhất ở {weakest_f1}.")
    per_class = (f"F1 PrivateTest từ vòng 21/11 đến checkpoint cuối. Disgust "
                 f"{pct(initial['class_f1']['Disgust'])} → {pct(final['class_f1']['Disgust'])}, "
                 f"Fear {pct(initial['class_f1']['Fear'])} → {pct(final['class_f1']['Fear'])}. "
                 "Macro F1 lấy trung bình F1 của 7 lớp.")
    filter_note = ("Lớp Conv cuối có trọng số 96×48×3×3. Mỗi ô là lát 3×3 của kênh đầu vào có "
                   "tổng bình phương trọng số lớn nhất trong một bộ lọc đầu ra. "
                   "Đỏ là trọng số dương, xanh là trọng số âm, dùng chung thang màu.")
    extra = int(late is not None)
    output.mkdir(parents=True, exist_ok=True)

    with PdfPages(output / "report.pdf", metadata={"Title": "BTL02 - Báo cáo FER2013"}) as pdf:
        figure = page("Nhận dạng biểu cảm khuôn mặt", 1)
        block(figure, "Tóm tắt", summary, 0.86)
        block(figure, "Dữ liệu và giao thức đánh giá", data + " Các lớp gồm " + classes, 0.69)
        block(figure, "Kiến trúc CNN và tiền xử lý", architecture, 0.51)
        block(figure, "Thiết lập chung", settings, 0.34)
        block(figure, "Mã nguồn", "code/train.py đọc dữ liệu, huấn luyện CNN, chọn checkpoint "
              "và xuất log, ma trận nhầm lẫn và hình bộ lọc. code/final_report.py tạo báo cáo "
              "và slide từ các tệp đã lưu.", 0.18)
        write_page(pdf, figure)

        figure = page("Thiết kế và kết quả thực nghiệm", 2)
        block(figure, "Vòng 21/11 · Lật ngang và dropout", first_design, 0.87)
        block(figure, "Vòng 23/11 · Trọng số theo lớp", second_design, 0.74)
        run_table(figure, rows, (0.07, 0.39, 0.86, 0.22))
        block(figure, "Cách đọc nhật ký", provenance, 0.35)
        block(figure, "Kết quả vòng 21/11", flips, 0.19)
        write_page(pdf, figure)

        figure = page("Trọng số lớp và ngân sách epoch", 3)
        validation_curve(figure, followup, later, (0.07, 0.48, 0.86, 0.39), True)
        block(figure, "Ảnh hưởng của trọng số", weighting, 0.43)
        block(figure, "Mốc 12 và 24 epoch", budget, 0.27)
        block(figure, "Giới hạn", limits, 0.16)
        write_page(pdf, figure)

        if late is not None:
            figure = page("26/11 · Dời pooling sau khối đầu", 4)
            block(figure, "Thiết kế đối chứng", pool_design, 0.87)
            run_table(figure, [("23/11", weighted), ("26/11", late)], (0.07, 0.61, 0.86, 0.12))
            pooling_curve(figure, ((followup, weighted), (third, late)), (0.14, 0.28, 0.76, 0.26))
            block(figure, "Kết quả và quyết định", pool_result, 0.20)
            write_page(pdf, figure)

        figure = page("Kết quả trên PrivateTest", 4 + extra)
        test_table(figure, tests, (0.07, 0.76, 0.86, 0.12))
        class_plot(figure, initial, final, final_day, (0.09, 0.50, 0.80, 0.22))
        block(figure, "F1 theo lớp", per_class, 0.44)
        picture(figure, final_folder / "confusion_matrix.png", (0.07, 0.075, 0.55, 0.26))
        figure.text(0.64, 0.29, "\n".join(wrap(mistakes, 29)),
                    fontsize=10, va="top", linespacing=1.5)
        write_page(pdf, figure)

        figure = page("Bộ lọc convolution cuối", 5 + extra)
        block(figure, f"Trọng số của {final['selected_config']}", filter_note, 0.87)
        picture(figure, final_folder / "final_conv_filters.png", (0.06, 0.18, 0.88, 0.56))
        block(figure, "Tài liệu tham khảo", "I. Goodfellow và cộng sự, Challenges in "
              "Representation Learning: A Report on Three Machine Learning Contests, "
              "arXiv:1307.0414, 2013.", 0.15)
        write_page(pdf, figure)

    with PdfPages(output / "slides.pdf", metadata={"Title": "BTL02 - Slide FER2013"}) as pdf:
        figure = page("Nhận dạng biểu cảm khuôn mặt", 1, slide=True)
        block(figure, "CNN PyTorch trên FER2013", data, 0.80, width=86, size=16)
        block(figure, "Kết quả chính", summary, 0.49, width=86, size=16)
        write_page(pdf, figure)

        figure = page("Dữ liệu và kiến trúc", 2, slide=True)
        block(figure, "Đầu ra gồm bảy lớp", classes, 0.80, width=90, size=16)
        block(figure, "Luồng xử lý của CNN", architecture, 0.60, width=88, size=16)
        block(figure, "Thiết lập chung", settings, 0.34, width=95, size=14)
        write_page(pdf, figure)

        figure = page("Các vòng thực nghiệm", 3, slide=True)
        block(figure, "21/11 · Thí nghiệm cơ sở", first_design, 0.80, width=92, size=15)
        block(figure, "23/11 · Weighted cross-entropy", second_design, 0.58, width=92, size=15)
        block(figure, "Nhật ký chương trình", provenance, 0.27, width=100, size=14)
        write_page(pdf, figure)

        figure = page("So sánh trên PublicTest", 4, slide=True)
        run_table(figure, rows, (0.07, 0.36, 0.86, 0.47))
        block(figure, "Ảnh hưởng của lật ngang", flips, 0.27, width=109, size=14)
        write_page(pdf, figure)

        figure = page("Macro F1 qua từng epoch", 5, slide=True)
        validation_curve(figure, followup, later, (0.07, 0.23, 0.86, 0.59), True)
        block(figure, "Mốc 12 và 24 epoch", budget, 0.16, width=114, size=12)
        write_page(pdf, figure)

        if late is not None:
            figure = page("26/11 · Dời pooling sau khối đầu", 6, slide=True)
            block(figure, "Thiết kế đối chứng", pool_design, 0.83, width=116, size=12)
            run_table(figure, [("23/11", weighted), ("26/11", late)], (0.07, 0.54, 0.86, 0.14))
            pooling_curve(figure, ((followup, weighted), (third, late)), (0.135, 0.28, 0.77, 0.21))
            block(figure, "Kết quả và quyết định", pool_result, 0.18, width=120, size=12)
            write_page(pdf, figure)

        figure = page("Kết quả trên PrivateTest", 6 + extra, slide=True)
        test_table(figure, tests, (0.07, 0.67, 0.86, 0.17))
        class_plot(figure, initial, final, final_day, (0.105, 0.29, 0.78, 0.32))
        block(figure, "Thay đổi ở Disgust và Fear", per_class, 0.20, width=105, size=13)
        write_page(pdf, figure)

        figure = page("Các lỗi thường gặp", 7 + extra, slide=True)
        picture(figure, final_folder / "confusion_matrix.png", (0.43, 0.10, 0.54, 0.77))
        block(figure, "Nhầm lẫn của checkpoint cuối", mistakes, 0.73, width=42, size=14)
        block(figure, "Giới hạn thực nghiệm", limits, 0.43, width=42, size=13)
        write_page(pdf, figure)

        figure = page("Bộ lọc convolution cuối", 8 + extra, slide=True)
        picture(figure, final_folder / "final_conv_filters.png", (0.06, 0.22, 0.88, 0.64))
        block(figure, "Đọc hình", filter_note, 0.16, width=115, size=12)
        write_page(pdf, figure)


def package(destination, output, journal, baseline, followup, third=None):
    followup_name = "results/2026-11-23_weighted_loss"
    members = [
        (output / "report.pdf", "report.pdf"),
        (output / "slides.pdf", "slides.pdf"),
        (journal, "journal/detail.xlsx"),
        (ROOT / "README.md", "README.md"),
        (ROOT / "requirements.txt", "requirements.txt"),
    ]
    members.extend((ROOT / "code" / name, f"code/{name}")
                   for name in ("train.py", "make_submission.py", "final_report.py"))
    sources = [(baseline, "results"), (followup, followup_name)]
    if third is not None:
        sources.append((third, "results/2026-11-26_late_pool"))
    for directory, prefix in sources:
        members.extend((path, f"{prefix}/{path.name}") for path in sorted(directory.iterdir())
                       if path.is_file() and path.suffix in {".csv", ".json", ".png"})
    checkpoint = followup / "flip_weighted.pt"
    if checkpoint.is_file():
        members.append((checkpoint, f"{followup_name}/{checkpoint.name}"))
    if third is not None and (third / "test_metrics.json").is_file():
        checkpoint = third / "flip_weighted_late_pool.pt"
        members.append((checkpoint, f"results/2026-11-26_late_pool/{checkpoint.name}"))
    for path, _ in members:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path, name in members:
            archive.write(path, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results")
    parser.add_argument("--followup", type=Path, default=ROOT / "results" / "2026-11-23_weighted_loss")
    parser.add_argument("--third", type=Path, default=ROOT / "results" / "2026-11-26_late_pool")
    parser.add_argument("--output", type=Path, default=ROOT / "submission")
    parser.add_argument("--journal", type=Path, default=ROOT / "journal" / "1.xlsx")
    parser.add_argument("--archive", type=Path, help="Path for an optional new submission ZIP.")
    args = parser.parse_args()
    third = args.third if (args.third / "experiments.csv").is_file() else None
    make_documents(args.baseline, args.followup, args.output, third)
    if args.archive is not None:
        package(args.archive, args.output, args.journal, args.baseline, args.followup, third)
