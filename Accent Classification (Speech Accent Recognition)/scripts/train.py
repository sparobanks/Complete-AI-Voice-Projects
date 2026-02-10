from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import AutoProcessor, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import json

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST = BASE_DIR / "data" / "manifest.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
TARGET_SR = 16000
MODEL_NAME = "facebook/wav2vec2-base"
BATCH_SIZE = 4
EPOCHS = 5
LR = 2e-4
MAX_SECONDS = 8.0  # truncate long clips

def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # normalize
    m = wav.abs().max()
    if m > 0:
        wav = wav / m
    # trim
    max_len = int(TARGET_SR * MAX_SECONDS)
    if wav.shape[-1] > max_len:
        wav = wav[..., :max_len]
    return wav.squeeze(0)  # [T]

class AccentDS(Dataset):
    def __init__(self, df, label_to_id):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "label"]
        audio = load_audio(path)
        y = self.label_to_id[label]
        return audio, y

def collate_fn(batch, processor):
    audios, ys = zip(*batch)
    ys = torch.tensor(ys, dtype=torch.long)
    # Processor pads variable-length audio for wav2vec2
    inputs = processor(list(audios), sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    return inputs, ys

class Head(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def main():
    if not MANIFEST.exists():
        raise SystemExit("Missing data/manifest.csv. Run: python scripts/prepare_data.py")

    df = pd.read_csv(MANIFEST)
    labels = sorted(df["label"].unique().tolist())
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    print("Labels:", labels)
    print("Train size:", len(train_df), "Val size:", len(val_df))

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    backbone.eval()  # freeze wav2vec2

    # Determine embedding dim using a dummy forward
    with torch.no_grad():
        dummy = processor([torch.zeros(TARGET_SR)], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        out = backbone(**{k: v.to(DEVICE) for k, v in dummy.items()}).last_hidden_state
        emb_dim = out.shape[-1]

    head = Head(emb_dim, num_classes=len(labels)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_ds = AccentDS(train_df, label_to_id)
    val_ds = AccentDS(val_df, label_to_id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, processor))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor))

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for inputs, ys in pbar:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            ys = ys.to(DEVICE)

            with torch.no_grad():
                hs = backbone(**inputs).last_hidden_state  # [B, T, D]
                emb = hs.mean(dim=1)  # mean-pool -> [B, D]

            logits = head(emb)
            loss = criterion(logits, ys)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = logits.argmax(dim=1)
            correct += (preds == ys).sum().item()
            total += ys.numel()
            pbar.set_postfix(acc=correct / max(total, 1), loss=float(loss.detach().cpu()))

        # Validation
        head.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for inputs, ys in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                ys = ys.to(DEVICE)
                hs = backbone(**inputs).last_hidden_state
                emb = hs.mean(dim=1)
                logits = head(emb)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_true.extend(ys.cpu().numpy().tolist())

        acc = (np.array(all_preds) == np.array(all_true)).mean()
        print(f"\nVal accuracy: {acc:.3f}")
        print(classification_report(all_true, all_preds, target_names=labels, zero_division=0))

        if acc > best_acc:
            best_acc = acc
            # Save head weights + label mapping + model name
            torch.save({"head_state": head.state_dict()}, MODELS_DIR / "accent_head.pth")
            (MODELS_DIR / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
            (MODELS_DIR / "config.json").write_text(json.dumps({"model_name": MODEL_NAME}, indent=2), encoding="utf-8")
            print("✅ Saved best model to models/")

    print("Done. Best val acc:", best_acc)

if __name__ == "__main__":
    main()
