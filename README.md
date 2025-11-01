KōreroNET — Python Orchestration (wakeup + process + updater)
================================================================

This README documents the Python side of the KōreroNET node that runs on a
Raspberry Pi (or Linux desktop). It covers the GUI/launcher (`koreronet-wakeup.py`),
the batch pipeline (`process.py`), and the self-updater (`updater.py`), including
how to install them and register **wakeup.py as a systemd *user* service**.

Files
-----
- `koreronet-wakeup.py` — GUI + orchestrator. Mounts Google Drive via rclone,
  pulls `config.ini`, runs a short UART session with the STM32, builds a
  `remapped_times.csv`, optionally launches `process.py` in a terminal, monitors
  for `done.ini`, and manages the self-update workflow.  (See inline comments.)
- `process.py` — Batch pipeline for **BirdNET** and **KōreroNET (ResNet-34)**.
  Copies audio in **batches** to a RAM-disk, runs analyzers, writes per-file
  CSVs, creates mapped summaries, **backs up positives** (CSV + WAV chunks), and
  drops `done.ini` on completion.
- `updater.py` — Atomic update applier. Pulls payload from cache
  (`/home/bn15/_update_cache`), replaces files in `/home/bn15`,
  then either **reboots** the system or **relaunches** `koreronet-wakeup.py`
  if a `NO_REBOOT` flag is present in the cache.

Minimum Requirements
--------------------
- Python 3.10+ (venv recommended)
- Packages (pip): `pyserial`, `librosa`, `soundfile`, `numpy`, `torch`,
  `torchvision`, `tqdm` (optional), `birdnet-analyzer` (the Python API),
  and any UI deps for Tk (`tk` is present by default on RPi OS Desktop).
- `rclone` installed and a remote named **gdrive** configured with access to
  your Drive (service account or OAuth).
- A Google Drive folder structure with:
  - `To the node/` — incoming control/config and **update files/**
  - `From the node/` — outputs uploaded by the node
  - `From the node/Power Logs/` (and subfolder `raw/`) optionally used by wakeup
- A connected STM32 node on `/dev/ttyAMA0` (default) or another serial port.

Paths & Conventions (defaults)
------------------------------
These match the scripts’ constants so your systemd unit does not need extra env:
- **BN15 root*
- **GUI/logs**
- **Local OUT (wakeup)**`
- **Pipeline OUT (process)**
- **Export (local)**
- **Google Drive mount**
- **Drive “To the node”**
- **Drive “From the node”**
- **Update pickup** / / then applied by `updater.py`.

Configuration — `config.ini`
----------------------------
Place your `config.ini` on Drive at `To the node/config.ini`. The GUI will copy
it to `/home/bn15/config.ini` on start. Recognized keys (whitespace and
case ignored; `key = value` or `key: value` both OK):

BirdNET & KōreroNET thresholds and assets (used by `process.py`):
- `birdnet=1|0` — enable/disable BirdNET
- `koreronet=1|0` — enable/disable KōreroNET
- `birdnetconf=0.70` — minimum confidence for BirdNET (legacy fallbacks:
  `conf`, `confidence`, `min_conf` accepted)
- `koreronetconf=0.60` — minimum probability for KōreroNET
- `overlap=2.5` — seconds of overlap for 5 s windows (both pipelines)
- `lat`, `lon`, `week` — BirdNET geo/temporal params; `week=1` picks current ISO week
- `checkpoint=/path/to/koreronetsonicXX.pth` — KōreroNET checkpoint
- `classmap=/path/to/classmap.csv` — id→label mapping for KōreroNET
- `birdnetchunklen=5.0`, `birdnetprepad=1.0`, `birdnetpostpad=1.0` — BN chunk sizing

Timetable (optional; parsed by the GUI if provided):
- `timetable={S,U,0,P,...(24 entries)},{rec_seconds,sleep_seconds}`

`koreronet-wakeup.py` — What it does
------------------------------------
1. **Ensures folders** exist locally; **mounts Drive** (`rclone mount gdrive:`
   → `/mnt/gdrive`) and copies `To the node/config.ini` locally if available.
2. **Self-update**: if Drive folder `To the node/update files/` contains any
   files (except `done.ini`), copies them to `_update_cache/`, purges the Drive
   folder, writes a completion marker `done.ini` there, and **spawns
   `updater.py`**, exiting immediately so files can be replaced safely.
3. **UART session**: opens `/dev/ttyAMA0` @ 115200-8-N-1 and sends a few short
   commands to the STM32 (power stats, history, timetable request). It logs the
   responses to `bn15/logs/session_*.log` and writes each command’s output as a
   text file in `bn15/out/`.
4. **Timemap build**: constructs `remapped_times.csv` in `bn15/out/` by scanning
   the removable drive (`/media/16GB DRIVE`) and aligning files onto the
   intended timetable windows (or uses the timetable in `config.ini` if present).
5. **Optional run**: when `birdnet=1`, launches `process.py` in a new terminal,
   otherwise skips analysis.
6. **done.ini monitor**: watches `/home/KoreroNET/out/done.ini`. When the
   pipeline drops it, the GUI performs a short wrap-up and removes stale flags.

`process.py` — What it does
---------------------------
- Reads `config.ini`, computes BirdNET & KōreroNET parameters.
- **Copies audio in batches** to a RAM-disk (`/dev/shm/korero_stage`) using
  big-buffer or `sendfile()`-based copy for speed.
- **BirdNET**: calls the Python API (`birdnet_analyzer.analyze.core.analyze`)
  with threads, min_conf, geo/temporal args, and optional overlap. Produces
  per-file CSVs under `/home/KoreroNET/out`.
- **KōreroNET** (ResNet-34 pipeline matching your training): 
  - Audio → mono 48 kHz → mel-spectrogram (*HTK scale*, `n_fft=512`, `hop=128`,
    `n_mels=224`, `fmin=150`, `fmax=20000`, **power=2.0 → dB → z-score**),
    then **resize to 224×224**; to-RGB if the checkpoint expects it.
  - Sliding 5 s windows (with `overlap` in seconds), softmax, thresholding.
  - Writes per-file CSVs and **backs up positive 5 s chunks** into
    `out/backup/koreronet/` with filename format:
    `<audio>__kn_<start>_<end>__<label>__p<prob>.wav` (+ a CSV sidecar).
- **Backups & indices**:
  - Copies per-file BirdNET CSVs that *do* contain detections into
    `out/backup/birdnet/`, prefixing the file with the **intended timestamp**
    from `remapped_times.csv` when available.
  - Appends rows from per-file CSVs into running index CSVs in `out/backup/`.
- **Mapped summaries** (BN & KN): creates `*_detections_mapped_YYYYMMDD_HHMMSS.csv`
  with an extra `ActualTime` column derived from `remapped_times.csv`.
- On completion, writes **`/home/KoreroNET/out/done.ini`** for the GUI.

`updater.py` — What it does
---------------------------
- Uses **`.updater.lock`** to avoid concurrent runs.
- Waits briefly for the GUI to exit, **atomically replaces** files in BN15
  from `_update_cache/` (preserves permissions), cleans the cache, `sync`s FS.
- If `NO_REBOOT` exists in the cache: skips reboot and **relaunches the GUI**.
  Otherwise it tries several reboot commands; if all fail, it relaunches the GUI.

Install (recommended venv)
-------------------------
```bash
# As user 'amin' on the Pi:
python3 -m venv /home/bn15
/home/bn15/bin/pip install --upgrade pip
/home/bn15/bin/pip install pyserial librosa soundfile numpy torch torchvision birdnet-analyzer tqdm
# (Optional extras you use: matplotlib, pandas, etc.)
```

Rclone setup
------------
```bash
# Configure a remote called 'gdrive'
rclone config
# Test the remote
rclone lsd gdrive:
# (Optional) Manual mount test:
mkdir -p /mnt/gdrive
rclone mount gdrive: /mnt/gdrive --vfs-cache-mode=writes --allow-other --daemon
```

Register **koreronet-wakeup.py** as a *user* service (systemd)
--------------------------------------------------------------
Create the unit file at: `~/.config/systemd/user/koreronet-wakeup.service`

```
[Unit]
Description=KōreroNET Wakeup GUI (user)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
# Use the project venv if present; fall back to system Python otherwise
ExecStart=/home/bn15/bin/python3 -u /home/bn15/koreronet-wakeup.py
WorkingDirectory=/home/bn15
Restart=always
RestartSec=3
# If running on a desktop session with GUI, ensure DISPLAY is available:
Environment=PYTHONUNBUFFERED=1
# Uncomment if you need to force the display/shell
# Environment=DISPLAY=:0
# Environment=XDG_RUNTIME_DIR=/run/user/%U

[Install]
WantedBy=default.target
```

Enable and start it:
```bash
systemctl --user daemon-reload
systemctl --user enable koreronet-wakeup.service
systemctl --user start  koreronet-wakeup.service
# View logs
journalctl --user -u koreronet-wakeup.service -f
```

If running **headless** (no desktop), you can still keep the service;
the GUI will log to `bn15/logs/` and run the UART + pipeline orchestration.
If you require a virtual display for any reason, consider `xvfb-run` or use
a terminal-only mode (future enhancement).

Manual launch (for debugging)
-----------------------------
```bash
/home/bn15/bin/python3 -u /home/bn15/koreronet-wakeup.py
```

Typical daily flow
------------------
1. User (or automation) drops new audio on the removable drive and updates
   `To the node/config.ini` as needed.
2. Service starts `koreronet-wakeup.py` at login/boot; it mounts Drive, pulls
   **config**, checks **update files**, and spawns `updater.py` if needed.
3. GUI runs a short **UART** session with STM32, logs snapshots & history, and
   builds **remapped_times.csv**.
4. If `birdnet=1`, it launches the **process.py** pipeline; else you can start
   it manually later. When the pipeline finishes, it writes **done.ini**.
5. GUI sees `done.ini`, cleans up stale flags, and you’ll find artifacts under
   `/home/KoreroNET/out/` and `From the node/` on Drive.

Key Outputs & Where to Find Them
--------------------------------
- Per-file CSVs: `/home/KoreroNET/out/*.csv`
- **Mapped summaries**: `/home/KoreroNET/out/*_detections_mapped_*.csv`
- **Backups**:
  - `.../out/backup/birdnet/*.BirdNET.results.csv` (only those with detections, time-prefixed when mapped)
  - `.../out/backup/koreronet/*.wav` (+ CSV sidecars) for positive chunks
- **Indices**: `.../out/backup/birdnet_detections_all.csv` and/or `koreronet_detections_all.csv`
- Completion flag: `/home/KoreroNET/out/done.ini`
- GUI logs: `/home/bn15/logs/`
- Update cache: `/home/bn15/_update_cache/`

Troubleshooting
---------------
- **Service doesn’t start**: `journalctl --user -u koreronet-wakeup.service -e`.
  Check Python paths, permissions, and rclone availability on PATH.
- **Drive not mounted**: run `rclone version`; verify the `gdrive` remote;
  try a manual `rclone mount` (see above).
- **UART port not found**: adjust `CANDIDATE_PORTS` in the script to your device
  (e.g., `/dev/ttyUSB0`). Ensure user is in the `dialout` group.
- **No BirdNET outputs**: ensure `birdnet=1` and threshold is not too high.
  Confirm the `birdnet-analyzer` package and its model files are accessible.
- **KōreroNET checkpoint missing**: set `checkpoint=` in `config.ini` and
  ensure the path is readable by the service user. If the model expects RGB,
  the script will expand 1→3 channels automatically.
- **Updates not applying**: confirm files were dropped to
  `To the node/update files/` and that `_update_cache/` fills on the Pi.
  Look at `bn15/logs/updater_*.log`.

License
-------
Code is licensed under the **Polyform Noncommercial License 1.0.0** (non‑commercial
use, attribution required, no warranty). Documentation (this README) may be used
under **CC BY‑NC 4.0**. If you prefer a single license, keep everything under
Polyform Noncommercial.

© 2025 KōreroNET Project 
