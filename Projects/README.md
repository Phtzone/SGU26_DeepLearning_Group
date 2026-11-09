# Projects — Không gian nghiên cứu Deep Learning của nhóm

Thư mục này tập hợp source code và bằng chứng thực nghiệm để các thành viên có thể tiếp tục phát triển, kiểm tra kết quả và mở lại dashboard mà không cần huấn luyện lại. Framework chính của dự án hiện tại là **PyTorch**; **TensorBoard** đọc logs đã lưu và không trực tiếp huấn luyện model.

## Phạm vi và cấu trúc thực tế

Hiện chỉ có một project trong `Projects`: [DL13-WESAD](DL13-WESAD/README.md), nghiên cứu phân loại baseline/stress/amusement từ tín hiệu cổ tay WESAD. Các bài thực hành khác đang ở bên ngoài thư mục `Projects` trong repository Git hiện có; không di chuyển hay đưa thêm những thay đổi ngoài phạm vi này vào các commits của WESAD.

```text
Projects/
├── README.md
├── DL13-WESAD/
│   ├── src/                         # Dữ liệu, model, training, đánh giá, logging
│   ├── scripts/                     # Điều phối, tổng hợp và hậu kiểm
│   ├── configs/, experiments/       # Cấu hình và tài liệu experiments
│   ├── splits/                      # Manifest chia người cố định
│   ├── tests/                       # Unit và regression tests
│   ├── runs/experiment_e1/           # Checkpoints, logs, TensorBoard, source snapshot
│   ├── results/experiment_e1/        # OOF, metrics, figures, bằng chứng finalization
│   └── data/                        # Chỉ có trên máy mỗi thành viên; không chia sẻ qua Git
└── *.zip                            # Các snapshots audit cục bộ; xem quy tắc bên dưới
```

Git root hiện tại là thư mục **cha của Projects**, cùng lịch sử của repository [SGU26_DeepLearning_Group](https://github.com/Phtzone/SGU26_DeepLearning_Group). Không chạy `git init` trong `Projects` hoặc `DL13-WESAD` để tránh tạo repository lồng nhau. Việc chuyển Git root cần một phương án riêng bảo toàn lịch sử; không xóa hoặc di chuyển `.git` thủ công.

## Clone, chọn main và cài môi trường

Các lệnh dùng PowerShell và đường dẫn tương đối, không phụ thuộc ổ đĩa của người tạo baseline:

```powershell
git clone https://github.com/Phtzone/SGU26_DeepLearning_Group.git
cd SGU26_DeepLearning_Group
git switch main
git pull --ff-only origin main
cd Projects/DL13-WESAD

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-tracking.txt
python -m pip check
python -c "import sys, torch, tensorboard; from torch.utils.tensorboard import SummaryWriter; print(sys.version); print(torch.__version__, torch.cuda.is_available()); print(tensorboard.__version__)"
```

Python theo khai báo project là từ 3.11. Môi trường thực sự của E1 là **Python 3.14.6, PyTorch 2.14.0 CPU, TensorBoard 2.21.0**; [metadata môi trường](DL13-WESAD/runs/experiment_e1/20261108T154303Z_474b38f9/system.json) và [danh sách phiên bản](DL13-WESAD/runs/experiment_e1/20261108T154303Z_474b38f9/requirements-runtime.txt) là bằng chứng để tái lập. Cài phiên bản mới hơn theo requirements không bảo đảm kết quả bitwise giống E1. Không cần TensorFlow hoặc W&B để chạy pipeline hiện tại.

## Lấy WESAD và đặt dữ liệu

Tải từ [trang WESAD tại UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection), theo liên kết nguồn và điều kiện sử dụng/phân phối của bên cung cấp. Không giả định cache hoặc dữ liệu phái sinh luôn có quyền tái phân phối. Repository nhóm giữ raw/interim/processed local, không commit các bản sao tín hiệu.

Giải nén để có `data/raw/WESAD/S2/S2.pkl`, …, `S17/S17.pkl`; danh sách đầy đủ là S2–S11 và S13–S17. Hoặc cấu hình thư mục raw nằm ngoài checkout:

```powershell
# Chạy từ Projects/DL13-WESAD; thay bằng vị trí dataset của chính bạn
$env:WESAD_ROOT = (Resolve-Path 'data/raw/WESAD').Path
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --dry-run
```

Dry-run chỉ kiểm tra kế hoạch/cấu hình và sự hiện diện subjects; không preprocessing, training hoặc tạo group artifacts. `.env.example` chỉ là tài liệu; project không tự đọc `.env`.

## E1 canonical baseline và kết quả

Canonical GROUP_ID: **`20261108T154303Z_474b38f9`**. [Baseline registry](DL13-WESAD/experiments/e1_cnn/baseline_registry.json) phải giữ trạng thái `finalized`. Không chỉnh registry hoặc artifacts để thay kết quả đã được chấp nhận.

E1 dùng 15 subjects, grouped subject-wise 5-fold CV, mỗi fold 9 Train / 3 Validation / 3 Test, split seed 42 và training seed 42. Đã hoàn tất **137 epochs, 5.554 mini-batches, 2.136 OOF Test windows**. Pooled OOF Accuracy **0.5066**, Macro-F1 **0.4500**, Balanced Accuracy **0.4547**; các giá trị làm tròn ở đây chỉ để hiển thị.

| Nội dung cần truy xuất | Vị trí |
|---|---|
| Tài liệu dự án và E1 | [README dự án](DL13-WESAD/README.md), [README CNN-1D](DL13-WESAD/experiments/e1_cnn/README.md) |
| Logs, checkpoints, source lúc training | [Canonical runs](DL13-WESAD/runs/experiment_e1/20261108T154303Z_474b38f9/) |
| Best/last của Fold 1 | [Checkpoints](DL13-WESAD/runs/experiment_e1/20261108T154303Z_474b38f9/seed_42/fold_01/checkpoints/); tương tự `fold_02`…`fold_05` |
| OOF, fold/subject metrics, summary | [Canonical results](DL13-WESAD/results/experiment_e1/20261108T154303Z_474b38f9/) |
| 131 PNG + 131 SVG chính thức | [Scientific-v2](DL13-WESAD/results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations/scientific-v2_44c886afdd32da86/) |
| Phiên bản hình để đối chiếu | [Scientific-v1](DL13-WESAD/results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations/scientific-v1_049cc17f3e342328/) |
| Kiểm chứng checkpoints/metrics/TensorBoard | [Baseline verification](DL13-WESAD/results/experiment_e1/20261108T154303Z_474b38f9/baseline_verification.json) |
| Bằng chứng kiểm thử | [Finalization pytest XML](DL13-WESAD/results/experiment_e1/20261108T154303Z_474b38f9/finalization_pytest.xml): 50 passed, 2 skipped, 0 failed |

Mười checkpoints và 15 TensorBoard event files thật được giữ để các thành viên clone và sử dụng lại. `source/` trong canonical runs là code/config đã chạy training; source hiện tại có thêm các sửa lifecycle metadata và nhãn hình sau training. Không thay snapshot để giả vờ rằng những sửa đổi đó đã có từ đầu.

## Mở lại TensorBoard mà không cần dataset hoặc training

Từ `Projects/DL13-WESAD` trong môi trường đã cài TensorBoard:

```powershell
python -m tensorboard.main --logdir runs/experiment_e1/20261108T154303Z_474b38f9 --port 6006
```

Mở **http://localhost:6006**. Chọn `epochs` để xem Train/Validation scalars, `batches` để xem batch loss và `evaluation` để xem Test scalars/images. Để xem mọi group E1, đổi logdir thành `runs/experiment_e1`; để xem các experiments tương lai, dùng `runs`. Đổi logdir không sửa dữ liệu. Ctrl+C dừng dashboard.

Các logs trong `results/tensorboard_upgrade/smoke_events/` chỉ là dữ liệu kiểm thử tổng hợp, không phải E1; không track thư mục này như kết quả nghiên cứu. Hai ZIP audit cũ được giữ nguyên trên máy; chúng chứa một phần bằng chứng smoke test trong snapshot. Không tự đưa ZIP lên Git khi chưa giải quyết rõ yêu cầu loại smoke logs, không sửa/xóa ZIP để làm mất snapshot lịch sử.

## Chạy E1 mới và phát triển E2/E3

Chỉ chạy khi chủ động muốn tạo một lượt thực nghiệm mới với dataset local:

```powershell
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml
```

Một lệnh chạy tuần tự đủ 5 folds, tự tạo GROUP_ID mới. Không dùng canonical GROUP_ID để thử cấu hình. Mỗi fold khởi tạo model mới; selection chỉ dựa Validation, Test đánh giá sau khi chọn best checkpoint. Cách chạy một fold và resume an toàn đã có trong [README E1](DL13-WESAD/experiments/e1_cnn/README.md).

E2 GRU, E3 CNN-GRU còn là placeholders với trạng thái `planned`; sensor ablations là kế hoạch. Khi triển khai, dùng experiment directories và GROUP_ID riêng, cùng protocol để so sánh; không sửa weights, logs, predictions hoặc figures E1 để phục vụ model mới.

## Sử dụng checkpoint cho inference

Không cần huấn luyện lại để nạp model. Dùng best checkpoint của fold cần khảo sát và `normalization.json` của chính fold đó. Với một cửa sổ mới đã được xử lý đúng 5 kênh × 960 mẫu, thực hiện như sau; đoạn mã này là hướng dẫn, không tự tạo dữ liệu đầu vào giả:

```python
from pathlib import Path
import json
import numpy as np
import torch
from scripts.verify_e1_baseline import safe_load_checkpoint
from src.models.cnn import CNN1D

fold = Path('runs/experiment_e1/20261108T154303Z_474b38f9/seed_42/fold_01')
state = safe_load_checkpoint(fold / 'checkpoints/best.pt')
model = CNN1D()
model.load_state_dict(state['model'], strict=True)
model.eval()
scaler = json.loads((fold / 'normalization.json').read_text(encoding='utf-8'))

# window phải là ndarray float32 [5, 960], cùng kênh/tần số/đơn vị như preprocessing E1.
def predict_window(window):
    mean = np.asarray(scaler['mean'], dtype=np.float32)[:, None]
    std = np.asarray(scaler['std'], dtype=np.float32)[:, None]
    x = torch.from_numpy(((window - mean) / std).astype(np.float32))[None]
    with torch.inference_mode():
        probabilities = model(x).softmax(dim=1)[0].numpy()
    names = ['baseline', 'stress', 'amusement']
    return names[int(probabilities.argmax())], probabilities
```

Chỉ nạp checkpoints đã xác minh từ repository nhóm; `safe_load_checkpoint` sử dụng `weights_only=True` và allowlist NumPy RNG types cố định. Không fit lại scaler trên Test hoặc dữ liệu đang inference.

## Làm việc trực tiếp trên main

1. Trước khi chỉnh sửa, chạy `git status` và `git pull --ff-only origin main` từ Git root. Nếu fast-forward bị chặn vì lịch sử phân kỳ, trao đổi với nhóm, không force-push.
2. Trao đổi trước khi cùng sửa một file. Không sửa artifacts của baseline đã finalized.
3. Commit theo chức năng với Conventional Commits, ví dụ `feat(training): ...`, `test: ...`, `docs: ...`, `chore(experiment): ...`. Không dồn các việc không liên quan vào một commit.
4. Stage bằng các đường dẫn cụ thể trong `Projects`, kiểm tra `git diff --cached --stat`, `git diff --cached`, `git diff --check` và `git status`.
5. Kiểm tra staged paths không có raw/cache dataset, secrets, môi trường ảo hoặc smoke logs. Kiểm tra checkpoint/event/figure cần chia sẻ không bị ignore nhầm.
6. Push trực tiếp `git push origin main` sau khi kiểm tra; không dùng `--force` hoặc `--force-with-lease`, không bypass branch protection.

Không cần Git LFS cho bộ E1 hiện tại: checkpoint lớn nhất chỉ **799.098 bytes**, ZIP lớn nhất **24.909.896 bytes**. GitHub cảnh báo trên 50 MiB và chặn trên 100 MiB cho từng tệp Git thường theo [tài liệu giới hạn](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Tệp nhị phân/checkpoints/ZIP vẫn làm lịch sử Git lớn dần qua các phiên bản; trao đổi trước khi thêm snapshot mới. Nếu sau này cần LFS, xác minh dung lượng/quyền tải, cấu hình patterns trước commit và hướng dẫn thành viên `git lfs pull`; không mua dịch vụ hoặc migrate lịch sử tự động.

## Quy tắc bảo mật và dữ liệu

- `src/data/` là source và phải được commit; `data/raw/`, `data/interim/`, `data/processed/` là dữ liệu local và không được commit.
- Không commit `.env`, API keys, tokens, private keys hoặc credentials. Chỉ dùng `.env.example` không có bí mật.
- Không commit `.venv*`, `venv`, `__pycache__`, pytest/tool caches hoặc file tạm.
- Không ignore toàn bộ `runs/`, `results/`, `.pt`, `.csv`, `.log`, TensorBoard events hay PNG/SVG; đó có thể là bằng chứng nghiên cứu thật.
- Chỉ thay đổi code/protocol mới trong phạm vi experiment được nhóm thống nhất. Các sai sót không liên quan đến nhiệm vụ hiện tại phải được ghi nhận, không tự sửa toàn bộ dự án.
