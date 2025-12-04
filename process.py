#!/usr/bin/env python3
#30 oct 2025
# in this version the file handling for the backup files will be totally diffrent and the front end has to adapt. 
# -*- coding: utf-8 -*-
# process.py — Lean build
# KōreroNET + BirdNET batch processor with:
#   • BN: run BirdNET-Analyzer → post-process per-file BN CSVs to make 5s clips + master CSV
#   • KN: run our ResNet-based detector directly → make 5s clips + master CSV (no per-file KN CSV)
#   • ActualStartTime = IntendedTime(from remap) + detection start offset (seconds)
#
# Master CSV schema (both BN & KN):
#   Clip,ActualStartTime,Label,Probability
#
# UPDATED for ResNet-34 AUDIO MELS:
#   SR=48k, Segment=5.0s, n_fft=1024, hop=256, n_mels=224, fmin=150, fmax=20000
#   HTK mel, power=2.0 → dB → per-example z-score, resize to [1,224,224] (freq x time)
#
# Config (/home/amin/bn15/config.ini):
#   birdnet=1|0
#   koreronet=1|0
#   birdnetconf=0.70
#   koreronetconf=0.60
#   overlap=2.5
#   lat, lon, week
#   classmap=/path/to/classmap.csv
#   checkpoint=/path/to/koreronetsonicXX.pth
#   birdnetchunklen=5.0
#   birdnetprepad=1.0
#   birdnetpostpad=1.0

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

# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing context
# ─────────────────────────────────────────────────────────────────────────────
def mp_ctx():
    if sys.platform != "win32":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context("spawn")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR   = Path("/media/amin/16GB DRIVE")         # source audio (search root #1)
RAM_STAGE   = Path("/dev/shm/korero_stage")          # RAM-disk (tmpfs)
OUT_DIR     = Path("/home/amin/KoreroNET/out")
BN15_DIR    = Path("/home/amin/bn15")
EXPORT_DIR  = BN15_DIR / "export"
REMAP_CSV   = BN15_DIR / "out" / "remapped_times.csv"
CONFIG_INI  = BN15_DIR / "config.ini"
DONE_FLAG   = OUT_DIR / "done.ini"

# Optional extra place to find originals when slicing BN chunks
EXTRA_AUDIO_ROOT = Path("/media/amin/16GB DRIVE")

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline knobs
# ─────────────────────────────────────────────────────────────────────────────
BATCH_MAX_FILES     = 10
COPY_BUFSIZE        = 64 * 1024 * 1024
RAM_SAFETY_MB       = 128

BN_THREADS          = 4
DEFAULT_MIN_CONF    = 0.7
BATCH_SIZE          = 4
RTYPES              = {"csv"}   # BirdNET outputs per-file CSVs

TARGET_SR           = 48_000
TARGET_SECONDS      = 5.0
N_FFT               = 1024
HOP_LENGTH          = 256
N_MELS              = 224
FMIN                = 150
FMAX                = 20_000
RESIZE_H            = 224
RESIZE_W            = 224

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
SENTINEL   = "__DONE__"

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    if val is None: return default
    return str(val).strip().lower() in ("1", "true", "yes", "on", "y")

def _parse_float(s: Optional[str], fallback: float) -> float:
    if s is None: return fallback
    try: return float(s)
    except Exception: return fallback

def _parse_int(s: Optional[str], fallback: int) -> int:
    if s is None: return fallback
    try: return int(s)
    except Exception: return fallback

# ─────────────────────────────────────────────────────────────────────────────
# BirdNET params
# ─────────────────────────────────────────────────────────────────────────────
def load_birdnet_params(cfg: Dict[str,str]) -> Tuple[float, float, float, int]:
    min_conf = _parse_float(
        cfg.get("birdnetconf") or cfg.get("conf") or cfg.get("confidence") or cfg.get("min_conf"),
        DEFAULT_MIN_CONF
    )

    lat_raw = cfg.get("lat")
    lon_raw = cfg.get("lon") or cfg.get("long") or cfg.get("longitude")

    lat = -1.0; lon = -1.0
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
    target_len = _parse_float(cfg.get("birdnetchunklen"), 5.0)
    pre_pad    = _parse_float(cfg.get("birdnetprepad"), 1.0)
    post_pad   = _parse_float(cfg.get("birdnetpostpad"), 1.0)
    target_len = max(0.5, min(30.0, target_len))
    pre_pad    = max(0.0, min(10.0, pre_pad))
    post_pad   = max(0.0, min(10.0, post_pad))
    return target_len, pre_pad, post_pad

# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────
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
        try: total += float(librosa.get_duration(path=str(f)))
        except Exception as e: print(f"[warn] Could not read duration of: {f} ({e})")
    return total

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def rel_to(base: Path, p: Path) -> Path:
    try: return p.relative_to(base)
    except Exception: return Path(p.name)

def disk_free_bytes(dir_path: Path) -> int:
    usage = shutil.disk_usage(dir_path)
    return int(usage.free)

def wait_for_ram_bytes(dir_path: Path, need_bytes: int, safety_mb: int = RAM_SAFETY_MB):
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

def plan_batches(files: List[Path], max_files: int) -> List[List[Path]]:
    return [files[i:i + max_files] for i in range(0, len(files), max_files)]

def batch_bytes(batch: List[Path]) -> int:
    s = 0
    for p in batch:
        try: s += p.stat().st_size
        except FileNotFoundError: pass
    return s

# ─────────────────────────────────────────────────────────────────────────────
# Time map & actual time helpers
# ─────────────────────────────────────────────────────────────────────────────
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

def parse_time_guess(s: str) -> Optional[datetime.datetime]:
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%d_%H%M%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
    )
    for fmt in fmts:
        try:
            return datetime.datetime.strptime(s.strip(), fmt)
        except Exception:
            pass
    return None

def actual_start_str(audio_basename: str, start_offset_s: float, time_map: Dict[str, str]) -> str:
    intended = time_map.get(audio_basename, "")
    if not intended:
        return ""
    dt0 = parse_time_guess(intended)
    if not dt0:
        return ""
    dt = dt0 + datetime.timedelta(seconds=float(start_offset_s))
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ─────────────────────────────────────────────────────────────────────────────
# Copy worker (feeds both BN + KN queues)
# ─────────────────────────────────────────────────────────────────────────────
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
            dst = batch_dir / rel_to(src_root, src).name
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

# ─────────────────────────────────────────────────────────────────────────────
# KōreroNET model & mel utils
# ─────────────────────────────────────────────────────────────────────────────
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
    backbone = ckpt.get("backbone", "resnet34")
    num_classes = int(ckpt.get("num_classes", 2))
    to_rgb = bool(ckpt.get("to_rgb", False))
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    in_ch = 3 if to_rgb else 1
    m = build_resnet(backbone, in_ch=in_ch, num_classes=num_classes)
    m.load_state_dict(sd, strict=False)
    m.eval()
    torch.set_num_threads(1)
    return m, to_rgb, num_classes

def make_mel(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
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
    t = torch.tensor(mel_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(RESIZE_H, RESIZE_W), mode="bilinear", align_corners=False)
    if to_rgb:
        t = t.repeat(1, 3, 1, 1)
    return t

def slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "unknown"

def write_chunk_audio(src_path: Path, start_s: float, end_s: float, dst_path: Path):
    ensure_parent(dst_path)
    y, sr = librosa.load(str(src_path), sr=TARGET_SR, mono=True, offset=start_s, duration=(end_s - start_s))
    sf.write(str(dst_path), y, TARGET_SR, subtype="PCM_16")

# ─────────────────────────────────────────────────────────────────────────────
# BirdNET CSV parsing helpers (single, non-duplicated versions)
# ─────────────────────────────────────────────────────────────────────────────
def _norm_header(hs: List[str]) -> List[str]:
    return [re.sub(r"[()\s]", "", h.strip().lower()) for h in hs]

def _find_col(norm: List[str], keywords: List[str]) -> Optional[int]:
    for i, col in enumerate(norm):
        if all(kw in col for kw in keywords):
            return i
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
        print(f"[bn] Read error (BN CSV): {csv_path.name} ({e})")
        return out
    if not rows or not rows[0]:
        return out

    hdr_raw = rows[0]
    hdr = _norm_header(hdr_raw)

    i_start  = _find_col(hdr, ["start"])
    i_end    = _find_col(hdr, ["end"])
    i_conf   = _find_col(hdr, ["conf"])
    i_file   = _find_col(hdr, ["file"])
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
            out.append({"start": start, "end": end, "conf": conf, "file": filev, "common": common, "sci": sci})
        except Exception:
            continue
    return out

# ─────────────────────────────────────────────────────────────────────────────
# KōreroNET analyzer worker → clips + master CSV (no sidecars)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_batches_worker_koreronet(batch_q: mp.Queue, metric_q: mp.Queue,
                                     conf_thr: float, overlap_s: float,
                                     ckpt_path: Path, classmap_path: Optional[Path],
                                     master_csv_path: Path, master_lock: mp.Lock,
                                     time_map: Dict[str, str]) -> None:
    try:
        model, to_rgb, num_classes = load_koreronet_checkpoint(ckpt_path)
        print(f"[kn] Loaded {ckpt_path.name}  to_rgb={to_rgb}  classes={num_classes}")
    except Exception as e:
        print(f"[kn] ERROR loading checkpoint: {e}")
        metric_q.put(("KN_DONE", 0, 0, 0.0, 0.0))
        return

    classmap = load_classmap(classmap_path)
    print(f"[kn] Class map entries: {len(classmap)}")

    win_frames = int(round(TARGET_SECONDS * TARGET_SR / HOP_LENGTH))
    ov_frames = int(round(max(0.0, overlap_s) * TARGET_SR / HOP_LENGTH)) if overlap_s else 0
    step_frames = max(1, win_frames - ov_frames)
    print(f"[kn] MEL sr={TARGET_SR}, n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}, f=[{FMIN},{FMAX}]")
    print(f"[kn] Window={TARGET_SECONDS}s (~{win_frames} frames), overlap={overlap_s}s → step={step_frames} frames")

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
                print(f"[kn] Read error: {wav.name} ({e})"); continue

            dur = float(len(y)) / TARGET_SR
            total_audio_s += dur

            mel = make_mel(y, TARGET_SR)  # [224, T]
            win_frames = int(round(TARGET_SECONDS * TARGET_SR / HOP_LENGTH))
            ov_frames  = int(round(max(0.0, overlap_s) * TARGET_SR / HOP_LENGTH)) if overlap_s else 0
            step_frames = max(1, win_frames - ov_frames)
            slices = mel_windows(mel, win_frames, step_frames)

            appended = 0
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

                # Skip detections for class ID 147 (Noise) or explicit
                # 'Noise' labels.  These represent non-biological
                # sounds and should not be included in the summary or
                # backed up.
                if cls_id == 147 or str(label).strip().lower() == "noise":
                    continue

                safe_label = slug(label)
                chunk_name = f"{wav.stem}__kn_{start_s:06.2f}_{end_s:06.2f}__{safe_label}__p{prob:.2f}.wav"
                chunk_path = backup_kn_dir / chunk_name

                try:
                    write_chunk_audio(wav, start_s, end_s, chunk_path)
                except Exception as e:
                    print(f"[kn] Chunk write error: {chunk_name} ({e})")
                    continue

                actual_str = actual_start_str(wav.name, start_s, time_map)
                # Append to KN master CSV safely
                with master_lock:
                    with master_csv_path.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([chunk_name, actual_str, label, f"{prob:.4f}"])
                appended += 1

            print(f"[kn] {wav.name}: {appended} detection(s)")

    metric_q.put(("KN_DONE", total_files, 0, total_audio_s, time.perf_counter() - t_start))

# ─────────────────────────────────────────────────────────────────────────────
# BirdNET analyzer worker (just runs BN; post-processing happens in main)
# ─────────────────────────────────────────────────────────────────────────────
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
        combine_results=False, skip_existing_results=True,
    )
    try:
        sig = inspect.signature(analyze)
        if "overlap" in sig.parameters and overlap_s is not None and overlap_s > 0:
            kw["overlap"] = float(overlap_s)
            print(f"[bn] Using overlap={overlap_s}s")
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
        print(f"[bn] Batch {batch_dir.name}: {format_hms(batch_sec)} audio in {dt:.3f}s")

    metric_q.put(("BN_DONE", batches_done, files_counted, total_audio_s, total_analyze_s))

# ─────────────────────────────────────────────────────────────────────────────
# BN post-processing → clips + BN master CSV (no sidecars)
# ─────────────────────────────────────────────────────────────────────────────
def build_audio_index(roots: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    seen_dupes: set = set()
    for root in roots:
        if not root or not root.exists():
            continue
        for p in iter_audio_files(root):
            bn = p.name
            if bn in index and bn not in seen_dupes:
                print(f"[warn] Duplicate audio basename: {bn} (keeping first: {index[bn]})")
                seen_dupes.add(bn); continue
            index.setdefault(bn, p)
    print(f"[index] Audio files indexed: {len(index)}")
    return index

def _choose_5s_window(start: float, end: float, target_len: float, pre: float, post: float, audio_len: float) -> Tuple[float, float]:
    s0 = max(0.0, start - pre)
    e0 = min(audio_len, end + post)
    cur = e0 - s0
    if cur >= target_len - 1e-6:
        mid = 0.5 * (start + end)
        s = max(0.0, mid - target_len / 2.0)
        e = s + target_len
        if e > audio_len:
            e = audio_len; s = max(0.0, e - target_len)
        return (float(s), float(e))
    mid = 0.5 * (start + end)
    s = max(0.0, mid - target_len / 2.0)
    e = s + target_len
    if e > audio_len:
        e = audio_len; s = max(0.0, e - target_len)
    return (float(s), float(e))

def birdnet_export_clips_and_master(out_dir: Path,
                                    audio_index: Dict[str, Path],
                                    min_conf: float,
                                    target_len_s: float,
                                    pre_pad_s: float,
                                    post_pad_s: float,
                                    time_map: Dict[str, str],
                                    master_csv_path: Path) -> Tuple[int, int]:
    backup_dir = out_dir / "backup" / "birdnet"
    backup_dir.mkdir(parents=True, exist_ok=True)

    csvs = [p for p in out_dir.rglob("*.BirdNET.results.csv") if (out_dir / "backup") not in p.parents]
    made, considered = 0, 0
    for csvp in csvs:
        rows = parse_birdnet_csv_rows(csvp)
        if not rows: continue

        audio_bn = None
        if rows and rows[0]["file"]:
            audio_bn = Path(rows[0]["file"]).name
        if not audio_bn:
            audio_bn = csvp.name.replace(".BirdNET.results.csv", "")

        src = audio_index.get(audio_bn)
        if not src or not src.exists():
            print(f"[bn] WARN: Source audio not found for {audio_bn}; skipping chunks.")
            continue

        try:
            audio_len = float(librosa.get_duration(path=str(src)))
        except Exception:
            audio_len = float("inf")

        for row in rows:
            considered += 1
            conf = float(row["conf"])
            if conf < min_conf:
                continue
            det_start = float(row["start"])
            det_end   = float(row["end"])
            label     = row["common"] or row["sci"] or "unknown"

            start_s, end_s = _choose_5s_window(det_start, det_end, target_len_s, pre_pad_s, post_pad_s, audio_len)
            safe_label = slug(label)
            stem = Path(audio_bn).stem
            chunk_name = f"{stem}__bn_{start_s:06.2f}_{end_s:06.2f}__{safe_label}__p{conf:.2f}.wav"
            chunk_path = backup_dir / chunk_name

            try:
                write_chunk_audio(src, start_s, end_s, chunk_path)
            except Exception as e:
                print(f"[bn] ERROR writing BN chunk: {chunk_name} ({e})")
                continue

            actual_str = actual_start_str(audio_bn, start_s, time_map)
            with master_csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([chunk_name, actual_str, label, f"{conf:.4f}"])
            made += 1

    print(f"[bn] Master rows added: {made}  |  BN positives considered: {considered}")
    return made, considered

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
def clear_done_flag():
    try: DONE_FLAG.unlink()
    except FileNotFoundError: pass
    except Exception as e: print(f"[warn] Could not clear old done flag: {e}")

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

def run_pipelines():
    cfg = parse_simple_ini(CONFIG_INI)
    birdnet_on   = as_bool(cfg.get("birdnet"), default=True)
    koreronet_on = as_bool(cfg.get("koreronet"), default=False)
    overlap_s    = _parse_float(cfg.get("overlap"), 0.0)
    bn_min_conf, bn_lat, bn_lon, bn_week = load_birdnet_params(cfg)
    bn_chunk_len, bn_pre_pad, bn_post_pad = load_birdnet_chunk_params(cfg)

    print(f"[config] birdnet={int(birdnet_on)}  koreronet={int(koreronet_on)}  overlap={overlap_s}")
    print(f"[config] BN chunk: len={bn_chunk_len}s  pre={bn_pre_pad}s  post={bn_post_pad}s")
    print(f"[config] KN mel: sr={TARGET_SR} n_fft={N_FFT} hop={HOP_LENGTH} n_mels={N_MELS} f=[{FMIN},{FMAX}] resize=({RESIZE_H}x{RESIZE_W})")

    # Prepare dirs and staging
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAM_STAGE.mkdir(parents=True, exist_ok=True)
    for p in RAM_STAGE.glob("batch_*"):
        shutil.rmtree(p, ignore_errors=True)

    # Master CSVs with header
    #
    # Derive a timestamp for this run to make master CSVs unique.  We
    # follow the same pattern used in the old process script – a
    # compact date/time string of the form YYYYMMDD_HHMMSS.  This
    # ensures that multiple runs on the same day do not clobber each
    # other and makes it easier to track which session produced a
    # given master file.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Compose file names by prefixing the timestamp.  This yields
    # e.g. 20251030_123402_birdnet_master.csv.  See README for
    # discussion of format; the user prefers a timestamp to track
    # sessions.
    bn_master = EXPORT_DIR / f"{ts}_birdnet_master.csv"
    kn_master = EXPORT_DIR / f"{ts}_koreronet_master.csv"
    # Create master CSV files with header rows.  Use ensure_parent to
    # create directories if necessary.
    for pth in (bn_master, kn_master):
        ensure_parent(pth)
        with pth.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Clip", "ActualStartTime", "Label", "Probability"])

    # Mapping for ActualStartTime
    time_map = load_time_map(REMAP_CSV)

    # Input batches
    all_files = [p for p in iter_audio_files(INPUT_DIR)]
    print(f"[scan] Files to process: {len(all_files)}  |  total audio ~ {format_hms(total_audio_seconds(INPUT_DIR))}")
    if not all_files:
        print("[info] No input files found; nothing to do.")
        return

    batches = plan_batches(all_files, BATCH_MAX_FILES)

    ctx = mp_ctx()
    bn_q  = ctx.Queue(maxsize=2)
    kn_q  = ctx.Queue(maxsize=2)
    metric_q = ctx.Queue()
    kn_lock = ctx.Lock()

    # Workers
    if birdnet_on:
        p_bn = ctx.Process(target=analyze_batches_worker_birdnet,
                           args=(bn_q, metric_q, bn_min_conf, bn_lat, bn_lon, bn_week, overlap_s))
        p_bn.daemon = False; p_bn.start()
    else:
        p_bn = None

    ckpt_path = Path(cfg.get("checkpoint") or "/home/amin/bn15/koreronetsonicv1.4.pth")
    classmap_path = Path(cfg.get("classmap") or (BN15_DIR / "koreronet_class_map.csv"))
    kn_thr = _parse_float(cfg.get("koreronetconf"), 0.6)

    if koreronet_on:
        p_kn = ctx.Process(target=analyze_batches_worker_koreronet,
                           args=(kn_q, metric_q, kn_thr, _parse_float(cfg.get("overlap"), 0.0),
                                 ckpt_path, classmap_path, kn_master, kn_lock, time_map))
        p_kn.daemon = False; p_kn.start()
    else:
        p_kn = None

    p_cp = ctx.Process(target=copy_batches_worker,
                       args=([q for q in (bn_q, kn_q) if q is not None], metric_q, INPUT_DIR, RAM_STAGE, batches))
    p_cp.daemon = False; p_cp.start()

    copy_bytes = 0; copy_time  = 0.0
    bn_done = (not birdnet_on); kn_done = (not koreronet_on); copy_done = False
    bn_stats = (0,0,0.0,0.0); kn_stats = (0,0,0.0,0.0)

    # Gather metrics
    while not (copy_done and bn_done and kn_done):
        tag, *payload = metric_q.get()
        if tag == "COPY_DONE":
            copy_bytes, copy_time = payload; copy_done = True
        elif tag == "BN_DONE":
            bn_stats = payload; bn_done = True
        elif tag == "KN_DONE":
            kn_stats = payload; kn_done = True

    # Clean queues & join
    p_cp.join()
    if p_bn: bn_q.put(SENTINEL); p_bn.join()
    if p_kn: kn_q.put(SENTINEL); p_kn.join()

    try:
        for p in RAM_STAGE.glob("batch_*"):
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass

    copy_mb = copy_bytes / 1_000_000 if copy_bytes else 0.0
    copy_mbps = (copy_mb / copy_time) if copy_time > 0 else 0.0

    print("\n=== Stage 1 Summary (analysis) ===")
    if birdnet_on:
        print(f"BirdNET batches: {bn_stats[0]}  files~{bn_stats[1]}  audio={format_hms(bn_stats[2])}  t={bn_stats[3]:.2f}s")
    if koreronet_on:
        print(f"KōreroNET files: {kn_stats[0]}  audio={format_hms(kn_stats[2])}  t={kn_stats[3]:.2f}s")
    print(f"Copy volume: {copy_mb:.2f} MB in {copy_time:.2f} s → {copy_mbps:.2f} MB/s")
    print(f"Output folder: {OUT_DIR}")

    # Stage 2: Post-process BN per-file CSVs into 5s clips + BN master CSV
    if birdnet_on:
        audio_index = build_audio_index([INPUT_DIR, EXTRA_AUDIO_ROOT])
        made, considered = birdnet_export_clips_and_master(
            OUT_DIR, audio_index, bn_min_conf, bn_chunk_len, bn_pre_pad, bn_post_pad, time_map, bn_master
        )
        print(f"[bn] Clips made: {made} / rows considered: {considered}")
    else:
        print("[bn] Skipped (disabled).")

    print(f"[done] Masters:")
    print(f"  KN → {kn_master}")
    print(f"  BN → {bn_master}")

    # ------------------------------------------------------------------
    # Copy master CSVs into backup folder
    #
    # The old process script created a pair of combined-detection CSVs in
    # the backup directory.  To maintain parity with that behaviour and
    # simplify downstream ingestion, copy the master CSVs into the
    # OUT_DIR/backup folder after processing completes.  These copies
    # intentionally omit the timestamp prefix in their names so that
    # consumers looking for `birdnet_master.csv` and
    # `koreronet_master.csv` will always find the latest session's
    # results.  If koreronet processing is disabled there will be no
    # koreronet master and we skip copying.
    try:
        backup_root = OUT_DIR / "backup"
        backup_root.mkdir(parents=True, exist_ok=True)
        # copy BirdNET master
        if bn_master and bn_master.exists():
            dest_bn = backup_root / "birdnet_master.csv"
            shutil.copy2(bn_master, dest_bn)
            print(f"[backup] Copied BN master → {dest_bn}")
        # copy Koreronet master if it was produced
        if koreronet_on and kn_master and kn_master.exists():
            dest_kn = backup_root / "koreronet_master.csv"
            shutil.copy2(kn_master, dest_kn)
            print(f"[backup] Copied KN master → {dest_kn}")
    except Exception as e:
        print(f"[backup] WARN: failed to copy master CSVs: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────
def main():
    clear_done_flag()
    try:
        run_pipelines()
        print("[done] process.py completed.")
    except Exception as e:
        print(f"[error] Exception in main: {e}")
    finally:
        write_done_flag()

if __name__ == "__main__":
    main()
