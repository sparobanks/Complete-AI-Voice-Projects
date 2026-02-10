from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_CSV = BASE_DIR / "data" / "manifest.csv"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}

def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Missing folder: {RAW_DIR}")

    rows = []
    for accent_dir in sorted([p for p in RAW_DIR.iterdir() if p.is_dir()]):
        label = accent_dir.name
        for f in sorted(accent_dir.rglob("*")):
            if f.suffix.lower() in AUDIO_EXTS:
                rows.append({"path": str(f.resolve()), "label": label})

    if not rows:
        raise SystemExit("No audio files found in data/raw/<accent_name>/ folders.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved manifest: {OUT_CSV}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
