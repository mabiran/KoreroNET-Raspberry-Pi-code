# -------------------- Update Successful --------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# process.py — KōreroNET + BirdNET batch processor with config, overlap, time-map merge,
#              positives backup (CSV copies for BirdNET, 5s chunks for KōreroNET with class+confidence in filename),
#              BirdNET positives also backed up as WAV chunks with class+confidence in filename,
#              global indices, and done flag for wakeup GUI.
#
# UPDATED for ResNet-34 AUDIO MELS:
#   • SR = 48_000 Hz
#   • Segment = 5.0 s
#   • n_fft = 512, hop = 128
#   • n_mels = 224, fmin = 150 Hz, fmax = 20000 Hz
#   • HTK mel, power=2.0 → dB → per-example z-score
#   • Resize slices to [1, 224, 224] (freq x time)
#
# Config keys in /home/amin/bn15/config.ini:
#   birdnet=1|0
#   koreronet=1|0
#   birdnetconf=0.70            -> BirdNET min_conf (legacy fallbacks: conf/confidence/min_conf)
#   koreronetconf=0.60          -> KōreroNET min probability threshold
#   overlap=2.5                  -> seconds overlap for windowing (both pipelines)
#   lat, lon, week               -> BirdNET geotemporal params (as before)
#   classmap=/path/to/classmap.csv
#   checkpoint=/path/to/koreronetsonicXX.pth
#   birdnetchunklen=5.0          -> BN chunk target length (seconds)
#   birdnetprepad=1.0            -> BN preferred seconds BEFORE detection window
#   birdnetpostpad=1.0           -> BN preferred seconds AFTER detection window
#
# Outputs:
#   - BirdNET per-file CSVs, combined CSV, mapped summary (bn_detections_mapped_*.csv)
#   - KORERONET per-file CSVs (*.KORERONET.results.csv), combined, mapped summary
#   - backup/birdnet/  (COPIES OF per-file BirdNET CSVs + WAV CHUNKS for each BN positive + sidecar CSVs)
#   - backup/koreronet/ (5s chunks WAV with <class>+p<prob> fused in filename + matching sidecar CSV)
#   - birdnet_detections_all.csv / koreronet_detections_all.csv (row-append indices)
#   - done.ini to signal completion

import os
import sys
import time
import shutil
import csv
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
import multiprocessing as mp
import datetime
import inspect
import re

# BirdNET-Analyzer
from birdnet_analyzer.analyze.core import analyze

# Audio & DSP
import librosa
import soundfile as sf
import numpy as np

# Torch for KōreroNET
import torch
import torch.nn.functional as F
from torchvision import models

# ── Multiprocessing context ───────────────────────────────────────────────────
def mp_ctx():
    if sys.platform != "win32":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context("spawn")

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_DIR   = Path("/media/amin/16GB DRIVE")         # source audio (search root #1)
RAM_STAGE   = Path("/dev/shm/korero_stage")          # RAM-disk (tmpfs)
OUT_DIR     = Path("/home/amin/KoreroNET/out")
BN15_DIR    = Path("/home/amin/bn15")
EXPORT_DIR  = BN15_DIR / "export"
REMAP_CSV   = BN15_DIR / "out" / "remapped_times.csv"
CONFIG_INI  = BN15_DIR / "config.ini"
DONE_FLAG   = OUT_DIR / "done.ini"

EXTRA_AUDIO_ROOT = Path("/media/amin/16GB DRIVE")    # optional extra root to search audio

# ── Pipeline knobs ───────────────────────────────────────────────────────────
BATCH_MAX_FILES     = 10
COPY_BUFSIZE        = 64 * 1024 * 1024
RAM_SAFETY_MB       = 128
DELETE_BATCH_AFTER  = False

BN_THREADS          = 4
DEFAULT_MIN_CONF    = 0.7
BATCH_SIZE          = 4
RTYPES              = {"csv"}
SKIP_EXISTING       = True

# ── KōreroNET mel/resize settings (UPDATED to match ResNet-34 training) ──────
TARGET_SR           = 48_000
TARGET_SECONDS      = 5.0
N_FFT               = 512
HOP_LENGTH          = 128
N_MELS              = 224
FMIN                = 150
FMAX                = 20_000
RESIZE_H            = 224   # keep freq axis at 224
RESIZE_W            = 224   # resize time axis to 224

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
SENTINEL   = "__DONE__"

# ── Config / INI helpers ─────────────────────────────────────────────────────
def parse_simple_ini(path: Path) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not path.exists():
        return d
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";", "[")):
            continue
        sep = "=" if "=" in line else (":" if ":" in line else None)
        if not sep:
            continue
        k, v = line.split(sep, 1)
        key = k.strip().lower().replace(" ", "")
        d[key] = v.strip()
    return d

def as_bool(val, default=False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on", "y")

def _parse_float(s: Optional[str], fallback: float) -> float:
    if s is None: return fallback
    try:
        return float(s)
    except Exception:
        return fallback

def _parse_int(s: Optional[str], fallback: int) -> int:
    if s is None: return fallback
    try:
        return int(s)
    except Exception:
        return fallback

# ── BirdNET params ───────────────────────────────────────────────────────────
def load_birdnet_params(cfg: Dict[str,str]) -> Tuple[float, float, float, int]:
    # Prefer birdnetconf; accept legacy keys as fallback
    min_conf = _parse_float(
        cfg.get("birdnetconf") or
        cfg.get("conf") or
        cfg.get("confidence") or
        cfg.get("min_conf"),
        DEFAULT_MIN_CONF
    )

    lat_raw = cfg.get("lat")
    lon_raw = cfg.get("lon") or cfg.get("long") or cfg.get("longitude")

    lat = -1.0
    lon = -1.0
    if lat_raw and lon_raw and lat_raw.strip().lower() == "nz" and lon_raw.strip().lower() == "nz":
        lat, lon = -36.8485, 174.7633
    else:
        lat = _parse_float(lat_raw, -1.0)
        lon = _parse_float(lon_raw, -1.0)

    week_raw = cfg.get("week")
    if week_raw is not None and week_raw.strip() == "1":
        week = datetime.date.today().isocalendar()[1]
    else:
        week = _parse_int(week_raw, -1)

    return (min_conf, lat, lon, week)

def load_birdnet_chunk_params(cfg: Dict[str,str]) -> Tuple[float, float, float]:
    """Read BN chunk sizing from config with sane defaults."""
    target_len = _parse_float(cfg.get("birdnetchunklen"), 5.0)
    pre_pad    = _parse_float(cfg.get("birdnetprepad"), 1.0)
    post_pad   = _parse_float(cfg.get("birdnetpostpad"), 1.0)
    # guardrails
    target_len = max(0.5, min(30.0, target_len))
    pre_pad    = max(0.0, min(10.0, pre_pad))
    post_pad   = max(0.0, min(10.0, post_pad))
    return target_len, pre_pad, post_pad

# ── Helpers ──────────────────────────────────────────────────────────────────
def format_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def iter_audio_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p

def total_audio_seconds(folder: Path) -> float:
    total = 0.0
    for f in iter_audio_files(folder):
        try:
            total += float(librosa.get_duration(path=str(f)))
        except Exception as e:
            print(f"[warn] Could not read duration of: {f} ({e})")
    return total

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def rel_to(base: Path, p: Path) -> Path:
    try:
        return p.relative_to(base)
    except Exception:
        return Path(p.name)

def disk_free_bytes(dir_path: Path) -> int:
    usage = shutil.disk_usage(dir_path)
    return int(usage.free)

def wait_for_ram_bytes(dir_path: Path, need_bytes: int, safety_mb: int = 128):
    safety = safety_mb * 1_000_000
    while True:
        if disk_free_bytes(dir_path) >= need_bytes + safety:
            return
        time.sleep(0.05)

def copy_linux_fast(src: Path, dst: Path) -> int:
    ensure_parent(dst)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        try: tmp.unlink()
        except FileNotFoundError: pass

    total = 0
    with open(src, "rb", buffering=0) as fsrc, open(tmp, "wb", buffering=0) as fdst:
        if hasattr(os, "sendfile"):
            size = os.fstat(fsrc.fileno()).st_size
            offset = 0
            while offset < size:
                sent = os.sendfile(fdst.fileno(), fsrc.fileno(), offset, size - offset)
                if sent == 0: break
                offset += sent
            total = size
        else:
            shutil.copyfileobj(fsrc, fdst, length=COPY_BUFSIZE)
            total = os.fstat(fsrc.fileno()).st_size

    os.replace(tmp, dst)
    return total

def find_all_csvs(directory: Path) -> List[Path]:
    return sorted(directory.rglob("*.csv"))

def combine_csvs(out_dir: Path, combined_path: Path, pattern_suffix: Optional[str] = None) -> int:
    files = [p for p in find_all_csvs(out_dir)
             if (p != combined_path) and (pattern_suffix is None or p.name.endswith(pattern_suffix))]
    if not files:
        return 0
    ensure_parent(combined_path)
    written = 0
    header_written = False
    with combined_path.open("w", newline="", encoding="utf-8") as fout:
        writer = None
        for f in files:
            try:
                with f.open("r", newline="", encoding="utf-8") as fin:
                    rows = list(csv.reader(fin))
                if not rows: continue
                header, *data = rows
                if writer is None:
                    writer = csv.writer(fout)
                if not header_written:
                    writer.writerow(header); header_written = True
                for row in data:
                    writer.writerow(row); written += 1
            except Exception as e:
                print(f"[warn] Could not merge {f}: {e}")
    return written

def plan_batches(files: List[Path], max_files: int) -> List[List[Path]]:
    return [files[i:i + max_files] for i in range(0, len(files), max_files)]

def batch_bytes(batch: List[Path]) -> int:
    s = 0
    for p in batch:
        try: s += p.stat().st_size
        except FileNotFoundError: pass
    return s

# ── Time-map & mapped summary ────────────────────────────────────────────────
def load_time_map(remap_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not remap_csv.exists():
        print(f"[info] No time map found at {remap_csv} — proceeding without mapping.")
        return mapping
    with remap_csv.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f); rows = list(rdr)
    if not rows: return mapping

    header = [h.strip().lower().replace(" ", "") for h in rows[0]]
    body   = rows[1:] if len(rows) > 1 else []

    filename_keys = ("filename", "file", "basename", "audiofile")
    time_keys     = ("intended_time", "intendedtime", "intended",
                     "time", "timestamp", "realtime", "actual_time_24h")

    def find_col(cands):
        for k in cands:
            if k in header: return header.index(k)
        return None

    fi = find_col(filename_keys)
    ti = find_col(time_keys)
    if fi is None: fi = 0

    for r in body:
        if not r: continue
        try:
            name = Path(r[fi]).name
            tval = r[ti] if (ti is not None and ti < len(r)) else ""
            mapping[name] = tval
        except Exception:
            pass

    print(f"[info] Loaded {len(mapping)} filename→time mappings from {remap_csv.name}.")
    return mapping

def detect_filecol(header: List[str]) -> int:
    norm = [h.strip().lower() for h in header]
    for key in ("file", "filepath", "file path", "path", "audiofile", "audio file"):
        if key in norm: return norm.index(key)
    return len(header) - 1

def write_mapped_summary(combined_csv: Path, time_map: Dict[str, str],
                         out_dir: Path, prefix: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{prefix}_detections_mapped_{ts}.csv"
    ensure_parent(out_path)

    if not combined_csv.exists():
        with out_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Start","End","Label/Name","Confidence","File","ActualTime"])
        return out_path

    with combined_csv.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f); rows = list(rdr)

    if not rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Start","End","Label/Name","Confidence","File","ActualTime"])
        return out_path

    header = rows[0]; data = rows[1:]
    fcol = detect_filecol(header)
    new_header = header + ["ActualTime"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        for r in data:
            try:
                fpath = r[fcol] if fcol < len(r) else ""
                bname = Path(fpath).name
            except Exception:
                bname = ""
            intended = time_map.get(bname, "")
            w.writerow(r + [intended])
    return out_path

# ── Copy worker (feeds both BN + KN queues) ───────────────────────────────────
def copy_batches_worker(batch_qs: List[mp.Queue], metric_q: mp.Queue,
                        src_root: Path, ram_root: Path,
                        batches: List[List[Path]]) -> None:
    bytes_done = 0
    t0 = time.perf_counter()

    for idx, batch in enumerate(batches, 1):
        if not batch: continue

        need = batch_bytes(batch)
        wait_for_ram_bytes(ram_root, need)

        batch_dir = ram_root / f"batch_{idx:05d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        t_batch = time.perf_counter()
        copied_files = 0
        for src in batch:
            rel = rel_to(src_root, src)
            dst = batch_dir / rel.name
            try:
                b = copy_linux_fast(src, dst)
                bytes_done += b; copied_files += 1
                print(f"[copy] {dst.name} ({b/1_000_000:.2f} MB)")
            except Exception as e:
                print(f"[copy] ERROR {src} → {dst}: {e}")

        dt_batch = time.perf_counter() - t_batch
        print(f"[copy] Batch {idx:05d} → {copied_files} files in {dt_batch:.3f}s | est {need/1_000_000:.2f} MB")

        for q in batch_qs:
            q.put(str(batch_dir))

    for q in batch_qs:
        q.put(SENTINEL)
    metric_q.put(("COPY_DONE", bytes_done, time.perf_counter() - t0))

# ── KōreroNET model utils ────────────────────────────────────────────────────
def load_classmap(path: Optional[Path]) -> Dict[int, str]:
    cmap: Dict[int, str] = {}
    if not path or not path.exists():
        return cmap
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
        if rows and rows[0]:
            hdr = [h.strip().lower() for h in rows[0]]
            try:
                id_idx = hdr.index("class_id") if "class_id" in hdr else hdr.index("id")
            except ValueError:
                id_idx = 0
            try:
                name_idx = hdr.index("class_name") if "class_name" in hdr else hdr.index("name")
            except ValueError:
                name_idx = 1 if len(hdr) > 1 else 0
            for r in rows[1:]:
                if not r: continue
                try:
                    idx = int(r[id_idx]); name = r[name_idx].strip()
                    cmap[idx] = name if name else f"class_{idx}"
                except Exception:
                    pass
    return cmap

def build_resnet(backbone_name: str = "resnet34", in_ch: int = 1, num_classes: int = 2):
    # Default to resnet34; if checkpoint says resnet18 we will rebuild accordingly.
    if backbone_name == "resnet34":
        m = models.resnet34(weights=None)
    else:
        m = models.resnet18(weights=None)
    w = m.conv1.weight
    if in_ch == 1 and w.shape[1] == 3:
        with torch.no_grad():
            m.conv1.weight = torch.nn.Parameter(w.mean(dim=1, keepdim=True))
    elif in_ch == 3 and w.shape[1] == 1:
        with torch.no_grad():
            m.conv1.weight = torch.nn.Parameter(w.repeat(1, 3, 1, 1))
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def load_koreronet_checkpoint(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    backbone = ckpt.get("backbone", "resnet34")  # default to 34 now
    num_classes = int(ckpt.get("num_classes", 2))
    to_rgb = bool(ckpt.get("to_rgb", False))
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    in_ch = 3 if to_rgb else 1
    m = build_resnet(backbone, in_ch=in_ch, num_classes=num_classes)
    m.load_state_dict(sd, strict=False)
    m.eval()
    torch.set_num_threads(1)
    return m, to_rgb, num_classes

# ── Mel generation matching the new training template ────────────────────────
def make_mel(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    MelSpectrogram:
      - sr=48k, n_fft=512, hop=128
      - n_mels=224, fmin=150, fmax=20000
      - HTK scale, power=2.0 -> dB -> z-score
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, htk=True, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mu = float(S_db.mean()); sd = float(S_db.std()) or 1.0
    Z = (S_db - mu) / sd
    return Z  # [224, T]

def mel_windows(mel: np.ndarray, win_frames: int, step_frames: int) -> List[Tuple[int, int, np.ndarray]]:
    n_frames = mel.shape[1]
    out = []
    i = 0
    while i + win_frames <= n_frames:
        out.append((i, i + win_frames, mel[:, i:i+win_frames]))
        i += max(1, step_frames)
    return out

def tensor_from_mel_slice(mel_slice: np.ndarray, to_rgb: bool) -> torch.Tensor:
    # mel_slice: [224, T_slice] → tensor [1,1,224,T_slice] → resize to [1,1,224,224]
    t = torch.tensor(mel_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(RESIZE_H, RESIZE_W), mode="bilinear", align_corners=False)
    if to_rgb:
        t = t.repeat(1, 3, 1, 1)
    return t

def slug(s: str) -> str:
    # safe label for filenames: letters, digits, dash/underscore only
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "unknown"

def write_chunk_audio(src_path: Path, start_s: float, end_s: float, dst_path: Path):
    ensure_parent(dst_path)
    y, sr = librosa.load(str(src_path), sr=TARGET_SR, mono=True, offset=start_s, duration=(end_s - start_s))
    sf.write(str(dst_path), y, TARGET_SR, subtype="PCM_16")

# ── KōreroNET analyzer worker ────────────────────────────────────────────────
def analyze_batches_worker_koreronet(batch_q: mp.Queue, metric_q: mp.Queue,
                                     conf_thr: float, overlap_s: float,
                                     ckpt_path: Path, classmap_path: Optional[Path]) -> None:
    try:
        model, to_rgb, num_classes = load_koreronet_checkpoint(ckpt_path)
        print(f"[koreronet] Loaded {ckpt_path.name}  backbone=auto  to_rgb={to_rgb}  classes={num_classes}")
    except Exception as e:
        print(f"[koreronet] ERROR loading checkpoint: {e}")
        metric_q.put(("KN_DONE", 0, 0, 0.0, 0.0))
        return

    classmap = load_classmap(classmap_path)
    print(f"[koreronet] Class map entries: {len(classmap)}")

    # Frames per 5 s window under (sr=48k, hop=128)
    win_frames = int(round(TARGET_SECONDS * TARGET_SR / HOP_LENGTH))
    ov_frames = int(round(max(0.0, overlap_s) * TARGET_SR / HOP_LENGTH)) if overlap_s else 0
    step_frames = max(1, win_frames - ov_frames)
    print(f"[koreronet] MEL spec: sr={TARGET_SR}, n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}, f=[{FMIN},{FMAX}]")
    print(f"[koreronet] Window: {TARGET_SECONDS}s → ~{win_frames} frames; overlap={overlap_s}s → step={step_frames} frames; resized→(224×224)")

    total_files = 0
    total_audio_s = 0.0
    t_start = time.perf_counter()

    backup_kn_dir = OUT_DIR / "backup" / "koreronet"
    backup_kn_dir.mkdir(parents=True, exist_ok=True)

    while True:
        item = batch_q.get()
        if item == SENTINEL:
            break
        batch_dir = Path(item)
        for wav in (p for p in batch_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS):
            total_files += 1
            try:
                y, sr = librosa.load(str(wav), sr=TARGET_SR, mono=True)
            except Exception as e:
                print(f"[koreronet] Read error: {wav.name} ({e})")
                continue

            dur = float(len(y)) / TARGET_SR
            total_audio_s += dur

            mel = make_mel(y, TARGET_SR)  # [224, T]
            # recompute in case overlap changed
            win_frames = int(round(TARGET_SECONDS * TARGET_SR / HOP_LENGTH))
            ov_frames = int(round(max(0.0, overlap_s) * TARGET_SR / HOP_LENGTH)) if overlap_s else 0
            step_frames = max(1, win_frames - ov_frames)
            slices = mel_windows(mel, win_frames, step_frames)

            out_csv = OUT_DIR / (wav.stem + ".KORERONET.results.csv")
            ensure_parent(out_csv)
            wrote_header = False
            appended_rows = 0

            for (a, b, melslc) in slices:
                start_s = a * (HOP_LENGTH / TARGET_SR)
                end_s   = start_s + TARGET_SECONDS

                timg = tensor_from_mel_slice(melslc, to_rgb)
                with torch.no_grad():
                    logits = model(timg)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                cls_id = int(np.argmax(probs))
                prob = float(probs[cls_id])
                if prob < conf_thr:
                    continue

                label = classmap.get(cls_id, f"class_{cls_id}")
                # Write per-file detection row
                with out_csv.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not wrote_header:
                        w.writerow(["Start","End","Label","Confidence","File"])
                        wrote_header = True
                    w.writerow([f"{start_s:.2f}", f"{end_s:.2f}", label, f"{prob:.4f}", wav.name])
                    appended_rows += 1

                # === Backup this positive 5s chunk with class + confidence fused in filename ===
                safe_label = slug(label)
                chunk_name = f"{wav.stem}__kn_{start_s:06.2f}_{end_s:06.2f}__{safe_label}__p{prob:.2f}.wav"
                chunk_path = backup_kn_dir / chunk_name
                try:
                    write_chunk_audio(wav, start_s, end_s, chunk_path)
                    # Sidecar CSV uses the SAME fused basename (.csv)
                    sidecar = chunk_path.with_suffix(".csv")
                    with sidecar.open("w", newline="", encoding="utf-8") as sc:
                        cw = csv.writer(sc)
                        cw.writerow(["file","label","prob","src","start_s","end_s"])
                        cw.writerow([chunk_name, label, f"{prob:.4f}", wav.name, f"{start_s:.2f}", f"{end_s:.2f}"])
                except Exception as e:
                    print(f"[koreronet] Backup chunk error: {e}")

            print(f"[koreronet] {wav.name}: {appended_rows} detection(s)")

    metric_q.put(("KN_DONE", total_files, 0, total_audio_s, time.perf_counter() - t_start))

# ── BirdNET analyzer worker (unchanged logic) ────────────────────────────────
def analyze_batches_worker_birdnet(batch_q: mp.Queue, metric_q: mp.Queue,
                                   bn_min_conf: float, bn_lat: float, bn_lon: float,
                                   bn_week: int, overlap_s: float) -> None:
    total_audio_s = 0.0
    total_analyze_s = 0.0
    batches_done = 0
    files_counted = 0

    kw = dict(
        output=str(OUT_DIR),
        threads=BN_THREADS,
        min_conf=bn_min_conf,
        lat=bn_lat, lon=bn_lon, week=bn_week,
        rtype=RTYPES, batch_size=BATCH_SIZE,
        combine_results=False, skip_existing_results=SKIP_EXISTING,
    )
    try:
        sig = inspect.signature(analyze)
        if "overlap" in sig.parameters and overlap_s is not None and overlap_s > 0:
            kw["overlap"] = float(overlap_s)
            print(f"[birdnet] Using overlap={overlap_s}s")
    except Exception:
        pass

    while True:
        item = batch_q.get()
        if item == SENTINEL:
            break
        batch_dir = Path(item)
        t1 = time.perf_counter()
        analyze(str(batch_dir), **kw)
        dt = time.perf_counter() - t1

        batch_sec = total_audio_seconds(batch_dir)
        total_audio_s += batch_sec
        total_analyze_s += dt
        batches_done += 1
        files_counted += len([p for p in batch_dir.iterdir() if p.is_file()])
        print(f"[birdnet] Batch {batch_dir.name}: {format_hms(batch_sec)} audio in {dt:.3f}s")

    metric_q.put(("BN_DONE", batches_done, files_counted, total_audio_s, total_analyze_s))

# ── BirdNET CSV backup (copy per-file detection CSVs) ────────────────────────
def csv_has_detections(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            if header is None: return False
            for _ in rdr: return True
        return False
    except Exception:
        return False

def parse_time_guess(s: str) -> Optional[datetime.datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt)
        except Exception:
            pass
    return None

def _prefix_with_time(basename: str, audio_basename: str, time_map: Dict[str, str]) -> str:
    intended_str = time_map.get(audio_basename, "")
    dt = parse_time_guess(intended_str) if intended_str else None
    if not dt:
        return basename
    return f"{dt.strftime('%Y%m%d_%H%M%S')}__{basename}"

def collect_birdnet_csvs_to_backup(out_dir: Path,
                                   time_map: Dict[str, str]) -> Tuple[int, int]:
    backup_dir = out_dir / "backup" / "birdnet"
    backup_dir.mkdir(parents=True, exist_ok=True)

    csvs = [p for p in out_dir.rglob("*.BirdNET.results.csv")]
    examined = 0
    copied = 0

    for csvp in csvs:
        examined += 1
        if not csv_has_detections(csvp):
            continue

        audio_basename = None
        try:
            with csvp.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows and len(rows) > 1:
                hdr = [h.strip().lower() for h in rows[0]]
                if "file" in hdr:
                    fi = hdr.index("file")
                    if fi < len(rows[1]):
                        audio_basename = Path(rows[1][fi]).name
        except Exception:
            audio_basename = None

        if not audio_basename:
            audio_basename = csvp.name[:-len(".BirdNET.results.csv")] if csvp.name.endswith(".BirdNET.results.csv") else csvp.stem

        dest_name = _prefix_with_time(
            basename=f"{Path(audio_basename).stem}.BirdNET.results.csv",
            audio_basename=Path(audio_basename).name,
            time_map=time_map
        )
        dest = backup_dir / dest_name

        try:
            shutil.copy2(csvp, dest)
            copied += 1
            print(f"[backup] (BirdNET CSV) {csvp.name} → {dest.name}")
        except Exception as e:
            print(f"[backup] ERROR copying {csvp} → {dest}: {e}")

    return copied, examined

# ── Append all detection rows into backup indices ─────────────────────────────
def append_results_to_backup_index(out_dir: Path, pattern_suffix: str, index_name: str) -> Tuple[Path, int, int]:
    backup_dir = out_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    index_path = backup_dir / index_name

    csv_files = sorted(p for p in out_dir.rglob("*.csv") if p.name.endswith(pattern_suffix))
    appended_rows = 0; scanned = 0

    index_exists = index_path.exists() and index_path.stat().st_size > 0
    index_header: Optional[List[str]] = None
    if index_exists:
        try:
            with index_path.open("r", newline="", encoding="utf-8") as fidx:
                r = csv.reader(fidx); index_header = next(r, None)
        except Exception:
            index_header = None

    with index_path.open("a", newline="", encoding="utf-8") as fidx:
        w = csv.writer(fidx)
        for csvp in csv_files:
            scanned += 1
            try:
                with csvp.open("r", newline="", encoding="utf-8") as fin:
                    rdr = csv.reader(fin)
                    header = next(rdr, None)
                    if header is None: continue

                    if not index_exists and index_header is None:
                        w.writerow(header); index_exists = True; index_header = header

                    had_row = False
                    for row in rdr:
                        if row:
                            w.writerow(row); appended_rows += 1; had_row = True
                    if not had_row: continue
            except Exception as e:
                print(f"[index] WARNING: could not read {csvp.name}: {e}")

    print(f"[index] Appended {appended_rows} row(s) from {scanned} CSV(s) → {index_path}")
    return index_path, appended_rows, scanned

# ── NEW: Audio index & BirdNET WAV backup helpers (with 5s clips) ────────────
def build_audio_index(roots: List[Path]) -> Dict[str, Path]:
    """
    Build a basename → fullpath index for original audio so we can re-slice
    after RAM batches are cleaned up.
    """
    index: Dict[str, Path] = {}
    seen_dupes: set = set()
    for root in roots:
        if not root or not root.exists():
            continue
        for p in iter_audio_files(root):
            bn = p.name
            if bn in index and bn not in seen_dupes:
                print(f"[warn] Duplicate audio basename: {bn} (keeping first: {index[bn]})")
                seen_dupes.add(bn)
                continue
            index.setdefault(bn, p)
    print(f"[index] Audio files indexed: {len(index)}")
    return index

def _norm_header(hs: List[str]) -> List[str]:
    return [re.sub(r"[()\s]", "", h.strip().lower()) for h in hs]

def _find_col(norm: List[str], keywords: List[str]) -> Optional[int]:
    for i, col in enumerate(norm):
        ok = True
        for kw in keywords:
            if kw not in col:
                ok = False; break
        if ok: return i
    return None

def parse_birdnet_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Robustly parse a BirdNET per-file CSV, returning rows with keys:
    start, end, sci, common, conf, file
    """
    out: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.reader(f)
            rows = list(rdr)
    except Exception as e:
        print(f"[backup] Read error (BN CSV): {csv_path.name} ({e})")
        return out
    if not rows or not rows[0]:
        return out

    hdr_raw = rows[0]
    hdr = _norm_header(hdr_raw)

    i_start = _find_col(hdr, ["start"])
    i_end   = _find_col(hdr, ["end"])
    i_conf  = _find_col(hdr, ["conf"])
    i_file  = _find_col(hdr, ["file"])
    i_common = _find_col(hdr, ["common"])
    i_sci    = _find_col(hdr, ["scient"])

    for r in rows[1:]:
        if not r: continue
        try:
            start = float(r[i_start]) if (i_start is not None and i_start < len(r)) else None
            end   = float(r[i_end])   if (i_end   is not None and i_end   < len(r)) else None
            conf  = float(r[i_conf])  if (i_conf  is not None and i_conf  < len(r)) else None
            filev = r[i_file] if (i_file is not None and i_file < len(r)) else ""
            common = r[i_common] if (i_common is not None and i_common < len(r)) else ""
            sci    = r[i_sci]    if (i_sci    is not None and i_sci    < len(r)) else ""
            if start is None or end is None or conf is None:
                continue
            out.append({
                "start": start, "end": end, "conf": conf,
                "file": filev, "common": common, "sci": sci
            })
        except Exception:
            continue
    return out

def _choose_5s_window(start: float, end: float, target_len: float, pre: float, post: float, audio_len: float) -> Tuple[float, float]:
    """
    Try to include a bit before/after, then adjust to exactly target_len (default 5.0s)
    centered on the detection where possible, clamped to [0, audio_len].
    """
    # initial attempt: pad before/after
    s0 = max(0.0, start - pre)
    e0 = min(audio_len, end + post)
    cur = e0 - s0

    if cur >= target_len - 1e-6:
        # shrink around detection center to exact target_len
        mid = 0.5 * (start + end)
        s = max(0.0, mid - target_len / 2.0)
        e = s + target_len
        if e > audio_len:
            e = audio_len
            s = max(0.0, e - target_len)
        return (float(s), float(e))

    # cur < target_len → expand symmetrically around detection center
    mid = 0.5 * (start + end)
    s = max(0.0, mid - target_len / 2.0)
    e = s + target_len
    if e > audio_len:
        e = audio_len
        s = max(0.0, e - target_len)
    return (float(s), float(e))

def backup_birdnet_audio_from_csvs(out_dir: Path,
                                   audio_index: Dict[str, Path],
                                   min_conf: float,
                                   target_len_s: float,
                                   pre_pad_s: float,
                                   post_pad_s: float) -> Tuple[int, int]:
    """
    For every *.BirdNET.results.csv in OUT_DIR, slice WAV chunks for each row
    (start→end) with conf >= min_conf. Export fixed-length clips (default 5.0s),
    including some context before/after. Save to backup/birdnet with label+prob in name.
    """
    backup_dir = out_dir / "backup" / "birdnet"
    backup_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(out_dir.rglob("*.BirdNET.results.csv"))
    made, considered = 0, 0

    for csvp in csvs:
        rows = parse_birdnet_csv_rows(csvp)
        if not rows:
            continue

        # Determine audio basename from first row or CSV name
        audio_bn = None
        if rows and rows[0]["file"]:
            audio_bn = Path(rows[0]["file"]).name
        if not audio_bn:
            audio_bn = csvp.name.replace(".BirdNET.results.csv", "")

        src = audio_index.get(audio_bn)
        if not src or not src.exists():
            print(f"[backup] WARN: Source audio not found for {audio_bn}; skipping WAV chunk backup.")
            continue

        try:
            audio_len = float(librosa.get_duration(path=str(src)))
        except Exception:
            audio_len = float("inf")  # let writer clamp/truncate

        for row in rows:
            considered += 1
            conf = float(row["conf"])
            if conf < min_conf:
                continue
            det_start = float(row["start"])
            det_end   = float(row["end"])
            label     = row["common"] or row["sci"] or "unknown"

            # choose 5s (or target) window with context
            start_s, end_s = _choose_5s_window(det_start, det_end, target_len_s, pre_pad_s, post_pad_s, audio_len)

            safe_label = slug(label)
            stem = Path(audio_bn).stem
            chunk_name = f"{stem}__bn_{start_s:06.2f}_{end_s:06.2f}__{safe_label}__p{conf:.2f}.wav"
            chunk_path = backup_dir / chunk_name
            if SKIP_EXISTING and chunk_path.exists():
                continue

            try:
                write_chunk_audio(src, start_s, end_s, chunk_path)
                # Sidecar CSV with metadata (matches KN sidecar schema)
                sidecar = chunk_path.with_suffix(".csv")
                with sidecar.open("w", newline="", encoding="utf-8") as sc:
                    cw = csv.writer(sc)
                    cw.writerow(["file","label","prob","src","start_s","end_s","det_start","det_end"])
                    cw.writerow([chunk_name, label, f"{conf:.4f}", src.name,
                                 f"{start_s:.2f}", f"{end_s:.2f}", f"{det_start:.2f}", f"{det_end:.2f}"])
                made += 1
            except Exception as e:
                print(f"[backup] ERROR writing BN chunk: {chunk_name} ({e})")

    print(f"[backup] (BirdNET WAV) Created {made} chunk(s) from {considered} positive row(s).")
    return made, considered

# ── Done-flag helpers ────────────────────────────────────────────────────────
def clear_done_flag():
    try:
        DONE_FLAG.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[warn] Could not clear old done flag: {e}")

def write_done_flag():
    try:
        ensure_parent(DONE_FLAG)
        tmp = DONE_FLAG.with_suffix(DONE_FLAG.suffix + ".part")
        with tmp.open("w", encoding="utf-8") as f:
            f.write("1\n")
        os.replace(tmp, DONE_FLAG)
        print(f"[flag] Wrote done flag → {DONE_FLAG}")
    except Exception as e:
        print(f"[warn] Could not write done flag: {e}")

# ── BirdNET pipeline wrapper ─────────────────────────────────────────────────
def run_birdnet_pipeline(bn_min_conf: float, bn_lat: float, bn_lon: float, bn_week: int, overlap_s: float) -> Path:
    print(f"Scanning input: {INPUT_DIR}")
    all_files = [p for p in iter_audio_files(INPUT_DIR)]
    print(f"Files to process: {len(all_files)}")
    total_sec = total_audio_seconds(INPUT_DIR)
    print(f"Total audio duration: {format_hms(total_sec)} ({total_sec:.1f} s)")
    print(f"[birdnet] min_conf={bn_min_conf:.2f}  lat={bn_lat}  lon={bn_lon}  week={bn_week}")

    RAM_STAGE.mkdir(parents=True, exist_ok=True)
    for p in RAM_STAGE.glob("batch_*"):
        shutil.rmtree(p, ignore_errors=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    batches = plan_batches(all_files, BATCH_MAX_FILES)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    birdnet_combined_name = f"birdnet_{ts}.csv"
    if not batches:
        print("[info] No input files found.")
        combined_csv = OUT_DIR / birdnet_combined_name
        ensure_parent(combined_csv)
        with combined_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Start","End","Scientific name","Common name","Confidence","File"])
        return combined_csv

    ctx = mp_ctx()
    bn_q  = ctx.Queue(maxsize=2)
    kn_q  = ctx.Queue(maxsize=2)
    metric_q = ctx.Queue()

    p_bn = ctx.Process(target=analyze_batches_worker_birdnet,
                       args=(bn_q, metric_q, bn_min_conf, bn_lat, bn_lon, bn_week, overlap_s))
    p_bn.daemon = False; p_bn.start()

    cfg = parse_simple_ini(CONFIG_INI)
    ckpt_path = Path(cfg.get("checkpoint") or "/home/amin/bn15/koreronetsonicv1.4.pth")
    classmap_path = BN15_DIR / "koreronet_class_map.csv"
    kn_thr = _parse_float(cfg.get("koreronetconf"), 0.6)
    p_kn = ctx.Process(target=analyze_batches_worker_koreronet,
                       args=(kn_q, metric_q, kn_thr, _parse_float(cfg.get("overlap"), 0.0), ckpt_path, classmap_path))
    p_kn.daemon = False; p_kn.start()

    p_cp = ctx.Process(target=copy_batches_worker,
                       args=([bn_q, kn_q], metric_q, INPUT_DIR, RAM_STAGE, batches))
    p_cp.daemon = False; p_cp.start()

    copy_bytes = 0; copy_time  = 0.0
    bn_done = False; kn_done = False; copy_done = False
    bn_stats = (0,0,0.0,0.0); kn_stats = (0,0,0.0,0.0)

    while not (copy_done and bn_done and kn_done):
        tag, *payload = metric_q.get()
        if tag == "COPY_DONE":
            copy_bytes, copy_time = payload; copy_done = True
        elif tag == "BN_DONE":
            bn_stats = payload; bn_done = True
        elif tag == "KN_DONE":
            kn_stats = payload; kn_done = True

    p_cp.join(); bn_q.put(SENTINEL); kn_q.put(SENTINEL)
    p_bn.join(); p_kn.join()

    try:
        for p in RAM_STAGE.glob("batch_*"):
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass

    combined_csv = OUT_DIR / birdnet_combined_name
    merged_rows = combine_csvs(OUT_DIR, combined_csv, pattern_suffix=".BirdNET.results.csv")

    copy_mb = copy_bytes / 1_000_000 if copy_bytes else 0.0
    copy_mbps = (copy_mb / copy_time) if copy_time > 0 else 0.0

    print("\n=== Analysis Summary ===")
    print(f"BirdNET batches:    {bn_stats[0]}  files~{bn_stats[1]}  audio={format_hms(bn_stats[2])}  t={bn_stats[3]:.2f}s")
    print(f"KōreroNET files:    {kn_stats[0]}  audio={format_hms(kn_stats[2])}  t={kn_stats[3]:.2f}s")
    print(f"Copy volume:        {copy_mb:.2f} MB in {copy_time:.2f} s → {copy_mbps:.2f} MB/s")
    print(f"RAM staging dir:    {RAM_STAGE}")
    print(f"Output folder:      {OUT_DIR}")
    if merged_rows:
        print(f"Combined CSV:       {combined_csv} ({merged_rows} BirdNET rows)")

    return combined_csv

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    clear_done_flag()

    cfg = parse_simple_ini(CONFIG_INI)
    birdnet_on   = as_bool(cfg.get("birdnet"), default=True)
    koreronet_on = as_bool(cfg.get("koreronet"), default=False)
    overlap_s    = _parse_float(cfg.get("overlap"), 0.0)
    bn_min_conf, bn_lat, bn_lon, bn_week = load_birdnet_params(cfg)
    bn_chunk_len, bn_pre_pad, bn_post_pad = load_birdnet_chunk_params(cfg)

    print(f"[config] birdnet={int(birdnet_on)}  koreronet={int(koreronet_on)}  overlap={overlap_s}")
    print(f"[config] BN chunk: len={bn_chunk_len}s  pre={bn_pre_pad}s  post={bn_post_pad}s")
    print(f"[config] KN mel spec: sr={TARGET_SR}, n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}, f=[{FMIN},{FMAX}], resize→({RESIZE_H}×{RESIZE_W})")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    combined_birdnet: Optional[Path] = None

    try:
        if birdnet_on or koreronet_on:
            combined_birdnet = run_birdnet_pipeline(bn_min_conf, bn_lat, bn_lon, bn_week, overlap_s)
        else:
            print("[info] Both analyzers disabled in config; nothing to do.")
            combined_birdnet = OUT_DIR / "combined_results.csv"
            ensure_parent(combined_birdnet)
            with combined_birdnet.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Start","End","Scientific name","Common name","Confidence","File"])

        time_map = load_time_map(REMAP_CSV)

        if combined_birdnet is None:
            combined_birdnet = OUT_DIR / "combined_results.csv"
        bn_mapped = write_mapped_summary(combined_birdnet, time_map, EXPORT_DIR, prefix="bn")
        print(f"[export] BN mapped detections → {bn_mapped}")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        koreronet_combined_name = f"koreronet_{ts}.csv"
        combined_kn = OUT_DIR / koreronet_combined_name
        kn_rows = combine_csvs(OUT_DIR, combined_kn, pattern_suffix=".KORERONET.results.csv")
        if kn_rows:
            kn_mapped = write_mapped_summary(combined_kn, time_map, EXPORT_DIR, prefix="kn")
            print(f"[export] KN mapped detections → {kn_mapped}")
        else:
            print("[export] No KōreroNET detections to export.")

        # Copy BirdNET per-file detections CSVs
        copied, examined = collect_birdnet_csvs_to_backup(OUT_DIR, time_map)
        print(f"[backup] (BirdNET CSVs) Examined: {examined}  |  Copied: {copied}  → {OUT_DIR/'backup'/'birdnet'}")

        # Back up BirdNET positive WAV chunks (5s default with context)
        audio_index = build_audio_index([INPUT_DIR, EXTRA_AUDIO_ROOT])
        bn_chunks_made, bn_rows_considered = backup_birdnet_audio_from_csvs(
            OUT_DIR, audio_index, bn_min_conf, bn_chunk_len, bn_pre_pad, bn_post_pad
        )
        print(f"[backup] (BirdNET WAVs) Made: {bn_chunks_made}  |  Rows considered: {bn_rows_considered}")

        # Append (verbatim) all detection rows into backup index CSVs
        append_results_to_backup_index(OUT_DIR, ".BirdNET.results.csv",   "birdnet_detections_all.csv")
        append_results_to_backup_index(OUT_DIR, ".KORERONET.results.csv", "koreronet_detections_all.csv")

        print("[done] process.py completed.")
    except Exception as e:
        print(f"[error] Exception in main: {e}")
    finally:
        write_done_flag()

if __name__ == "__main__":
    main()
