import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

# -------------------------
# Config
# -------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)

# Pick common SpeechCommands keywords
KEYWORDS = ["yes", "no", "up", "down", "left", "right", "stop", "go"]
LABELS = KEYWORDS + ["unknown", "silence"]

LABEL_TO_ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_ID.items()}


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)


def pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: [1, T]
    T = waveform.shape[-1]
    if T > CLIP_SAMPLES:
        return waveform[..., :CLIP_SAMPLES]
    if T < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - T
        return torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def to_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # normalize
    m = wav.abs().max()
    if m > 0:
        wav = wav / m
    wav = pad_or_trim(wav)
    return wav  # [1, 16000]


class MFCCFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=TARGET_SR,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T] or [B, T]
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        x = self.mfcc(wav)  # [B, n_mfcc, frames]
        # add channel dimension for CNN: [B, 1, n_mfcc, frames]
        return x.unsqueeze(1)


class SmallKWSNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SpeechCommandsKWS(torch.utils.data.Dataset):
    def __init__(self, split: str):
        super().__init__()
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(BASE_DIR / "data"),
            download=True,
        )

        # Build file list for split
        def load_list(filename):
            path = Path(self.dataset._path) / filename
            if not path.exists():
                return set()
            return set(path.read_text().splitlines())

        val_list = load_list("validation_list.txt")
        test_list = load_list("testing_list.txt")

        all_items = []
        for wav_path, sr, label, *_ in self.dataset:
            rel = str(Path(wav_path).relative_to(self.dataset._path))
            all_items.append((wav_path, label, rel))

        if split == "train":
            self.items = [(p, l) for p, l, r in all_items if r not in val_list and r not in test_list]
        elif split == "val":
            self.items = [(p, l) for p, l, r in all_items if r in val_list]
        elif split == "test":
            self.items = [(p, l) for p, l, r in all_items if r in test_list]
        else:
            raise ValueError("split must be train/val/test")

        # Keep dataset smaller for faster training (optional)
        random.shuffle(self.items)
        max_items = 6000 if split == "train" else 1000
        self.items = self.items[:max_items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = to_mono_16k(path)

        if label in KEYWORDS:
            y = LABEL_TO_ID[label]
        else:
            y = LABEL_TO_ID["unknown"]

        return wav, y


def collate(batch):
    wavs, ys = zip(*batch)
    wavs = torch.stack([w for w in wavs], dim=0)  # [B, 1, T]
    ys = torch.tensor(ys, dtype=torch.long)
    return wavs, ys


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def main():
    set_seed()
    device = "cpu"

    featurizer = MFCCFeaturizer().to(device)
    model = SmallKWSNet(num_classes=len(LABELS)).to(device)

    train_ds = SpeechCommandsKWS("train")
    val_ds = SpeechCommandsKWS("val")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val = 0.0
    epochs = 5

    for epoch in range(1, epochs + 1):
        model.train()
        total_acc = 0.0
        for wavs, y in train_loader:
            wavs, y = wavs.to(device), y.to(device)
            feats = featurizer(wavs)
            logits = model(feats)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_acc += accuracy(logits, y)

        train_acc = total_acc / max(len(train_loader), 1)

        model.eval()
        val_acc_sum = 0.0
        with torch.no_grad():
            for wavs, y in val_loader:
                wavs, y = wavs.to(device), y.to(device)
                feats = featurizer(wavs)
                logits = model(feats)
                val_acc_sum += accuracy(logits, y)

        val_acc = val_acc_sum / max(len(val_loader), 1)
        print(f"Epoch {epoch}/{epochs} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": LABELS,
                },
                str(MODELS_DIR / "kws_model.pth"),
            )
            (MODELS_DIR / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
            print("Saved best model to models/kws_model.pth")

    print("Done. Best val acc:", best_val)


if __name__ == "__main__":
    main()
