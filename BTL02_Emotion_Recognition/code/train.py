"""Train and compare CNN configurations on the original FER2013 splits."""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
SPLITS = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}
CONFIGS = {
    "baseline": (False, 0.25),
    "flip": (True, 0.25),
    "dropout": (False, 0.45),
    "flip_dropout": (True, 0.45),
    "flip_weighted": (True, 0.25),
    "flip_weighted_late_pool": (True, 0.25),
}
BASELINE_CONFIGS = ("baseline", "flip", "dropout", "flip_dropout")


def load_data(path):
    images = {name: np.empty((size, 1, 48, 48), np.uint8)
              for name, size in SPLITS.items()}
    labels = {name: np.empty(size, np.int64) for name, size in SPLITS.items()}
    counts = dict.fromkeys(SPLITS, 0)
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            split = row["Usage"].strip()
            index = counts[split]
            label = int(row["emotion"])
            if index >= SPLITS[split] or not 0 <= label < len(CLASSES):
                raise ValueError(f"Invalid label or split size in {split}, row {index + 1}")
            pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.int16)
            if pixels.size != 2304 or pixels.min() < 0 or pixels.max() > 255:
                raise ValueError(f"Invalid pixels in {split}, row {index + 1}")
            images[split][index, 0] = pixels.reshape(48, 48)
            labels[split][index] = label
            counts[split] += 1
    if counts != SPLITS:
        raise ValueError(f"Unexpected split sizes: {counts}")
    return {name: TensorDataset(torch.from_numpy(images[name]), torch.from_numpy(labels[name]))
            for name in SPLITS}


class EmotionCNN(nn.Module):
    def __init__(self, dropout, late_pool=False):
        super().__init__()
        layers = [] if late_pool else [nn.AvgPool2d(2)]
        for input_channels, output_channels in ((1, 24), (24, 48), (48, 96)):
            layers.extend([nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(output_channels), nn.ReLU(), nn.MaxPool2d(2)])
            if late_pool and input_channels == 1:
                layers.append(nn.AvgPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(96 * 3 * 3, 128),
                                        nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 7))

    def forward(self, images):
        return self.classifier(self.features(images))


def run_epoch(model, loader, device, criterion, optimizer=None, flip=False, confusion=None):
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    loss_weight = torch.zeros((), device=device) if criterion.weight is not None else None
    correct = torch.zeros((), device=device, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True,
                               memory_format=torch.channels_last) / 255
            labels = labels.to(device, non_blocking=True)
            if flip:
                mask = torch.rand((len(images), 1, 1, 1), device=device) < 0.5
                images = torch.where(mask, images.flip(-1), images)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_weight = len(labels)
            if criterion.weight is not None:
                batch_weight = criterion.weight[labels].sum()
                loss_weight += batch_weight
            total_loss += loss.detach() * batch_weight
            predictions = logits.detach().argmax(dim=1)
            correct += predictions.eq(labels).sum()
            if confusion is not None:
                indices = labels * len(CLASSES) + predictions
                confusion += torch.bincount(indices.cpu(), minlength=len(CLASSES)**2).view(7, 7)
    denominator = len(loader.dataset) if loss_weight is None else loss_weight.item()
    return total_loss.item() / denominator, correct.item() / len(loader.dataset)


def class_f1(confusion):
    matrix = confusion.numpy()
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return np.divide(2 * matrix.diagonal(), denominator,
                     out=np.zeros(len(CLASSES)), where=denominator > 0)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_learning_curves(output, histories):
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        for axis, metric in ((loss_axis, "loss"), (accuracy_axis, "accuracy")):
            line, = axis.plot(epochs, [row[f"val_{metric}"] for row in history], label=name)
            axis.plot(epochs, [row[f"train_{metric}"] for row in history],
                      linestyle=":", color=line.get_color(), alpha=0.6)
            axis.set(xlabel="Epoch", ylabel=metric.capitalize())
    accuracy_axis.legend(title="Solid: validation | Dotted: train", fontsize=8)
    figure.savefig(output / "learning_curves.png", dpi=160)
    plt.close(figure)


def save_test_figures(output, confusion, model):
    figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Images")
    axis.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted label", ylabel="True label", title="PrivateTest confusion matrix")
    axis.tick_params(axis="x", labelrotation=30)
    for actual, predicted in np.ndindex(confusion.shape):
        value = confusion[actual, predicted]
        axis.text(predicted, actual, str(value), ha="center", va="center",
                  color="white" if value > confusion.max() / 2 else "black")
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    last_conv = [layer for layer in model.features if isinstance(layer, nn.Conv2d)][-1]
    weights = last_conv.weight.detach().cpu().numpy()
    # For each output filter, show the 3×3 slice from its strongest input channel.
    strongest = (weights**2).sum(axis=(2, 3)).argmax(axis=1)
    slices = weights[np.arange(len(weights)), strongest]
    scale = abs(slices).max()
    figure, axes = plt.subplots(8, 12, figsize=(12, 8), layout="constrained")
    for index, axis in enumerate(axes.flat):
        axis.imshow(slices[index], cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set_title(f"{index + 1} / c{strongest[index] + 1}", fontsize=7)
        axis.axis("off")
    figure.suptitle("Final convolution: strongest 3×3 input-channel slice per output filter")
    figure.savefig(output / "final_conv_filters.png", dpi=160)
    plt.close(figure)


def evaluate_test(model, loader, device, output, name, selection_metric):
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.long)
    test_loss, test_acc = run_epoch(model, loader, device, nn.CrossEntropyLoss(), confusion=confusion)
    per_class_f1 = class_f1(confusion)
    results = dict(selected_config=name, test_loss=test_loss, test_accuracy=test_acc,
                   test_macro_f1=float(per_class_f1.mean()), class_f1=dict(zip(CLASSES, per_class_f1)),
                   class_order=CLASSES, splits=SPLITS, device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   pytorch=str(torch.__version__), selection_metric=selection_metric)
    (output / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savetxt(output / "confusion_matrix.csv", confusion.numpy(), fmt="%d", delimiter=",",
               header=",".join(CLASSES), comments="")
    save_test_figures(output, confusion.numpy(), model)
    print(f"PrivateTest | {name} | accuracy {test_acc:.2%} | "
          f"macro F1 {per_class_f1.mean():.2%}", flush=True)


def train(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 6))
    datasets = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"{device} | PyTorch {torch.__version__} | split sizes {SPLITS}", flush=True)
    criterion = nn.CrossEntropyLoss()
    counts = datasets["Training"].tensors[1].bincount(minlength=len(CLASSES))
    histories = {}
    summary = []
    best_model = None
    best_rank = (-1, float("-inf"))
    best_name = ""
    for name in args.configs:
        flip, dropout = CONFIGS[name]
        late_pool = name == "flip_weighted_late_pool"
        weighted = name == "flip_weighted" or late_pool
        weights = None
        if weighted:
            if torch.any(counts == 0):
                raise ValueError("Weighted loss requires a training example for every class")
            weights = counts.float().rsqrt()
            weights /= weights.mean()
        train_criterion = nn.CrossEntropyLoss(weight=weights).to(device)
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        loaders = {
            split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "Training",
                              generator=generator, pin_memory=device.type == "cuda")
            for split, dataset in datasets.items()
        }
        model = EmotionCNN(dropout, late_pool=late_pool).to(device, memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        history = []
        run_rank = (-1, float("-inf"))
        checkpoint = args.output / f"{name}.pt"
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, loaders["Training"], device, train_criterion,
                                              optimizer=optimizer, flip=flip)
            val_confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.long)
            val_loss, val_acc = run_epoch(model, loaders["PublicTest"], device, criterion,
                                         confusion=val_confusion)
            val_f1 = class_f1(val_confusion)
            val_macro_f1 = float(val_f1.mean())
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                       val_loss=val_loss, val_accuracy=val_acc, val_macro_f1=val_macro_f1)
            row.update({f"val_f1_{label}": score for label, score in zip(CLASSES, val_f1)})
            history.append(row)
            write_csv(args.output / f"{name}_epochs.csv", history)
            print(f"{name:12s} {epoch:02d}/{args.epochs}  train {train_acc:.2%}  "
                  f"val {val_acc:.2%}  val macro F1 {val_macro_f1:.2%}", flush=True)
            score = val_macro_f1 if args.selection_metric == "macro-f1" else val_acc
            if (score, -val_loss) > run_rank:
                run_rank = (score, -val_loss)
                best_row = row
                torch.save(model.state_dict(), checkpoint)
        histories[name] = history
        summary.append(dict(config=name, horizontal_flip=flip, dropout=dropout, optimizer="AdamW",
                            learning_rate=args.lr, batch_size=args.batch_size, seed=args.seed,
                            epochs=args.epochs, best_epoch=best_row["epoch"],
                            val_accuracy=best_row["val_accuracy"], val_loss=best_row["val_loss"],
                            seconds=round(time.perf_counter() - started, 1),
                            val_macro_f1=best_row["val_macro_f1"], weighted_loss=weighted,
                            selection_metric=args.selection_metric, late_pool=late_pool,
                            class_weights=json.dumps(dict(zip(CLASSES, weights.tolist())))
                            if weighted else ""))
        write_csv(args.output / "experiments.csv", summary)
        if run_rank > best_rank:
            best_rank = run_rank
            best_name = name
            best_model = model
            best_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    save_learning_curves(args.output, histories)
    if not args.validation_only:
        evaluate_test(best_model, loaders["PrivateTest"], device, args.output, best_name,
                      args.selection_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fer2013" / "fer2013.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=BASELINE_CONFIGS)
    parser.add_argument("--selection-metric", choices=["accuracy", "macro-f1"], default="accuracy")
    parser.add_argument("--validation-only", action="store_true",
                        help="Save validation results and checkpoints without evaluating PrivateTest.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    if len(set(args.configs)) != len(args.configs):
        parser.error("config names must be unique")
    train(args)
