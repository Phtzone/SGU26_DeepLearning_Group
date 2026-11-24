"""Export Vietnamese progress figures from local project evidence; no training."""
from pathlib import Path
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'presentation_assets' / '20261122'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'data/processed/common_rate_64hz'
cfg = json.loads((DATA / 'config.json').read_text(encoding='utf-8'))
checks = []
for subject in cfg['subjects']:
    x = np.load(DATA / f'{subject}_X.npy', mmap_mode='r')
    mask = np.load(DATA / f'{subject}_quality_mask.npy', mmap_mode='r')
    checks.append({'subject': subject, 'shape': list(x.shape),
                   'finite': bool(np.isfinite(x).all()),
                   'float32': x.dtype == np.float32,
                   'mask_matches': len(mask) == len(x),
                   'flagged_timepoints': int((~mask).sum())})
assert all(c['finite'] and c['float32'] and c['mask_matches'] and c['shape'][1] == 14 for c in checks)
(OUT / 'data_verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18,
                     'svg.fonttype': 'none'})
INK, MUTED, BLUE, TEAL = '#172B46', '#52647A', '#2563EB', '#087F8C'
files = []

def base(title, subtitle, n):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor='#F5F8FC')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    ax.text(.7, 8.2, title, fontsize=29, weight='bold', color=INK)
    ax.text(.7, 7.65, subtitle, fontsize=16, color=MUTED)
    ax.text(.7, .3, 'DL13 · WESAD  |  Tiến độ ngày 22/11/2026', fontsize=11, color=MUTED)
    ax.text(15.25, .3, f'{n:02}', fontsize=12, color=MUTED, ha='right')
    return fig, ax

def box(ax, x, y, w, h, title, body, color=BLUE, size=18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.15',
                              facecolor='white', edgecolor='#D8E2EF', linewidth=1.2))
    ax.text(x+.28, y+h-.46, title, fontsize=21, color=color, weight='bold', va='top')
    ax.text(x+.28, y+h-1.05, body, fontsize=size, color=INK, va='top', linespacing=1.65)

def save(fig, name):
    for ext in ('png', 'svg'):
        p = OUT / f'{name}.{ext}'; fig.savefig(p, dpi=120); files.append(p)
    plt.close(fig)

fig, ax = base('Bài toán của nhóm', 'Nhận diện căng thẳng từ tín hiệu đeo cổ tay và ngực trên WESAD', 1)
box(ax, .7, 4.7, 14.6, 2.4, '15 người · 3 trạng thái · 10 nhóm tín hiệu / 14 kênh',
    'Baseline: trạng thái nền     |     Stress: căng thẳng     |     Amusement: giải trí\nMục tiêu: phân loại đúng cả với người chưa xuất hiện trong tập huấn luyện.', TEAL, 19)
box(ax, .7, 1.3, 7.05, 2.95, 'Ngực: 8 kênh',
    'ACC (X, Y, Z), ECG, EMG, EDA,\nTemp và Resp.\nTần số gốc: 700 Hz.', BLUE, 20)
box(ax, 8.1, 1.3, 7.2, 2.95, 'Cổ tay: 6 kênh',
    'ACC (X, Y, Z): 32 Hz · BVP: 64 Hz\nEDA và TEMP: 4 Hz.\nNhãn nằm riêng, không tính là kênh vào.', TEAL, 19)
save(fig, '01_bai_toan_va_tin_hieu')

fig, ax = base('Dữ liệu 14 kênh đang ở bước nào?', 'Đã tạo đủ file cho 15 người; chưa đưa bộ dữ liệu mới vào huấn luyện.', 2)
box(ax, .7, 3.7, 7.05, 3.4, 'ĐÃ LÀM',
    '1. Đọc dữ liệu gốc và khảo sát tín hiệu.\n2. Đưa các kênh về cùng lưới 64 Hz.\n3. Lưu ma trận float32 [thời gian, 14].\n4. Lưu nhãn theo đoạn và quality mask.\n5. Kiểm tra đủ 15 người, không NaN/Inf.', TEAL)
box(ax, 8.1, 3.7, 7.2, 3.4, 'BƯỚC TIẾP THEO',
    '1. Nối dữ liệu mới vào code huấn luyện.\n2. Tạo cửa sổ và áp dụng quality mask.\n3. Chia người: Train / Validation / Test.\n4. Fit chuẩn hóa bằng Train của mỗi fold.\n5. Huấn luyện lại trên đủ 14 kênh.', BLUE)
box(ax, .7, 1.2, 14.6, 2.0, 'Cần nhớ khi trình bày',
    '64 Hz là lựa chọn thử ban đầu. Tăng tần số không tạo thêm thông tin;\ngiảm từ 700 Hz có thể làm mất chi tiết ECG/EMG nên cần đánh giá ảnh hưởng.', '#A65C00', 19)
save(fig, '02_tien_do_tien_xu_ly')

fig, ax = base('Gặp dữ liệu đáng nghi thì xử lý ra sao?', 'Ví dụ S3: đánh dấu vùng cần loại khỏi bước tạo mẫu, giữ nguyên dữ liệu gốc.', 3)
box(ax, .7, 3.85, 4.65, 3.2, '01 · PHÁT HIỆN',
    '83 mốc ở tần số gốc 700 Hz\ncó giá trị đáng nghi:\nTemp ngực ≈ −273,15 °C\nhoặc cả 3 trục ACC ≈ −6,6.\nĐộ dài đoạn: khoảng 0,119 giây.', BLUE, 17)
box(ax, 5.7, 3.85, 4.65, 3.2, '02 · ĐÁNH DẤU',
    'Thêm vùng đệm 2 giây\nmỗi phía của đoạn lỗi.\nSau quy đổi sang 64 Hz:\n265 mốc có mask = False.\nKhông xóa / tự điền dữ liệu gốc.', TEAL, 17)
box(ax, 10.7, 3.85, 4.6, 3.2, '03 · KHI TẠO MẪU',
    'Cần dùng mask để xử lý\ncửa sổ chạm vùng đáng nghi.\nGhi lại số mẫu bị loại\nvà kiểm tra phân bố lớp.\nPhần này chưa được tích hợp.', BLUE, 17)
box(ax, .7, 1.15, 14.6, 2.15, 'Giới hạn của kiểm tra hiện tại',
    'Các file đều hữu hạn không có nghĩa là tín hiệu đã sạch mọi loại nhiễu.\nVùng đệm 2 giây là giả định ban đầu; chưa kết luận đây là lựa chọn tối ưu.', '#A65C00', 19)
save(fig, '03_xu_ly_du_lieu_dang_nghi')

fig, ax = base('Đánh giá trên người mới', 'Quy trình 5 folds đang dùng: chia theo người, không trộn cửa sổ giữa các tập.', 4)
for x, w, title, body, col in [
    (.7, 4.65, 'TRAIN · 9 người', 'Học trọng số mô hình.\nFit mean/std chuẩn hóa.\nTính trọng số các lớp.', BLUE),
    (5.7, 4.65, 'VALIDATION · 3 người', 'Chọn checkpoint tốt nhất.\nDừng sớm và so cấu hình.\nKhông cập nhật trọng số.', TEAL),
    (10.7, 4.6, 'TEST · 3 người', 'Chỉ đánh giá sau khi chọn\ncheckpoint bằng Validation.\nKhông dùng để tuning.', '#A65C00')]:
    box(ax, x, 3.9, w, 3.1, title, body, col, 18)
box(ax, .7, 1.15, 14.6, 2.2, 'Lặp đủ 5 folds → mỗi người được Test đúng một lần',
    'Chỉ số chính: tính Test Macro-F1 riêng cho từng người, rồi trung bình đều 15 người.\nMỗi fold khởi tạo lại mô hình; tham số chuẩn hóa chỉ lấy từ Train của fold đó.', BLUE, 19)
save(fig, '04_cach_chia_va_danh_gia')

fig, ax = base('Kết quả thử nghiệm hiện có', 'CHỈ áp dụng cho dữ liệu cổ tay cũ: 5 kênh · 32 Hz · 3 lớp · seed 42.', 5)
chart = fig.add_axes([.08, .31, .46, .43], facecolor='#F5F8FC')
vals = [.42420334395166537, .3101951151782572]
bars = chart.bar(['CNN-1D', 'GRU'], vals, color=[BLUE, TEAL], width=.5)
chart.set_ylim(0, .6); chart.set_ylabel('Test Macro-F1\ntrung bình theo người', fontsize=15)
chart.tick_params(labelsize=16); chart.spines[['top', 'right']].set_visible(False)
chart.grid(axis='y', alpha=.15); chart.set_axisbelow(True)
for b, v in zip(bars, vals): chart.text(b.get_x()+b.get_width()/2, v+.015, f'{v:.4f}', ha='center', fontsize=22, weight='bold', color=INK)
box(ax, 9.4, 2.6, 5.9, 4.45, 'Đọc kết quả thế nào?',
    'CNN và GRU đã chạy đủ 5 folds.\nCó log, checkpoint và dự đoán.\n\nGRU có fold rất thấp.\nCần kiểm tra nguyên nhân.\n\nCNN-GRU và ablation chưa chạy.', BLUE, 17)
ax.text(.8, 1.2, 'Đây là kết quả ban đầu của bộ 5 kênh. Chưa có kết quả huấn luyện 14 kênh.',
        color='#A65C00', fontsize=20, weight='bold')
ax.text(.8, .78, 'Một seed và một số cấu hình chưa đủ để kết luận CNN luôn tốt hơn GRU.', color=MUTED, fontsize=16)
save(fig, '05_ket_qua_thu_5_kenh')

fig, ax = base('Làm tiếp theo thứ tự nào?', 'Chốt dữ liệu và cách đánh giá trước khi chạy thêm nhiều cấu hình.', 6)
box(ax, .7, 3.9, 7.05, 3.2, 'ƯU TIÊN TRƯỚC MẮT',
    '1. Nối dữ liệu 14 kênh vào pipeline.\n2. Chốt cửa sổ, quality mask và chuẩn hóa.\n3. Kiểm tra nhãn, số kênh và tách người.\n4. Chạy thử nhỏ để kiểm tra toàn luồng.', BLUE, 19)
box(ax, 8.1, 3.9, 7.2, 3.2, 'SAU KHI PIPELINE ỔN',
    '5. Chạy CNN, GRU, rồi CNN-GRU.\n6. Tuning bằng Train / Validation.\n7. Làm ablation EDA và BVP, huấn luyện lại.\n8. Tổng hợp theo người và theo seed.', TEAL, 19)
box(ax, .7, 1.15, 14.6, 2.2, 'Mỗi lượt chạy phải truy lại được',
    'Ghi cấu hình và thay đổi → lưu log / kết quả / hình → ghi nhật ký, lỗi và cách xử lý.\nChốt rõ bỏ EDA cổ tay, EDA ngực hay cả hai trước khi làm ablation.', '#A65C00', 19)
save(fig, '06_viec_can_lam_tiep')

with zipfile.ZipFile(OUT / 'WESAD_anh_trinh_bay_20261122.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in files + [OUT / 'data_verification.json']:
        archive.write(p, p.name)
print(json.dumps({'output': str(OUT), 'figures': len(files)//2, 'verified_subjects': len(checks),
                  'flagged_timepoints': sum(c['flagged_timepoints'] for c in checks)}, ensure_ascii=False))
