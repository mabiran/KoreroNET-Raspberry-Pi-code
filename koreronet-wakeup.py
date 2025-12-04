#!/usr/bin/env python3
# -------------------- Koreronet Wakeup (revised) --------------------
import os, sys, time, datetime, threading, queue, subprocess, glob, re, csv, shutil, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# -------------------- Paths & Constants --------------------
LOG_DIR = Path("/home/amin/bn15/logs")
OUT_DIR = Path("/home/amin/bn15/out")                     # local OUT for wakeup
PROCESS_OUT_DIR = Path("/home/amin/KoreroNET/out")        # OUT used by process.py
DONE_FLAG = PROCESS_OUT_DIR / "done.ini"
EXPORT_LOCAL = Path("/home/amin/bn15/export")             # export made by process.py
BACKUP_LOCAL = PROCESS_OUT_DIR / "backup"

# Prefer stable alias on RPi; fallbacks included (USB CDC, etc.)
CANDIDATE_PORTS = ["/dev/ttyAMA0"]
BAUD = 115200
EOL = b"\r\n"
CMDS = ["hi", "nucleo power stats", "nucleo give me time table", "nucleo tell me time", "nucleo power history"]

# Flash drive & audio location
CHECK_PATH = "/media/amin/16GB DRIVE"
DRIVE_DIR = Path(CHECK_PATH)

# ---- rclone / Google Drive (mounted) ----
RCLONE = "rclone"
RCLONE_REMOTE = "gdrive"  # remote name
MOUNT_POINT = Path("/mnt/gdrive")
IMPORT_DIR = MOUNT_POINT / "To the node 1"
EXPORT_DIR = MOUNT_POINT / "From the node 1"
CONFIG_REMOTE = IMPORT_DIR / "config.ini"
CONFIG_LOCAL = Path("/home/amin/bn15/config.ini")

# ---- Self-update paths ----
BN15_DIR = Path("/home/amin/bn15")
UPDATE_CACHE_DIR = BN15_DIR / "_update_cache"
UPDATER_SCRIPT = BN15_DIR / "updater.py"
REMOTE_UPDATE_DIR = IMPORT_DIR / "update files"   # exact path requested
UPDATER_LOCK = BN15_DIR / ".updater_spawned.lock" # prevent double spawns

# Terminal candidates to show process.py live
TERMINALS = [
    ("x-terminal-emulator", ["-e"]),     # Debian alternatives
    ("lxterminal", ["-e"]),
    ("gnome-terminal", ["--", "bash", "-lc"]),  # we append full "python -u ..." after this
    ("xfce4-terminal", ["-e"]),
    ("konsole", ["-e"]),
    ("xterm", ["-e"]),
]

# ---- Power log publish locations on Drive ----
POWERLOG_REMOTE_BASE = "From the node 1/Power Logs"
RAW_SUBDIR = POWERLOG_REMOTE_BASE + "/raw"

# ----------------------------------------------------------
# ---- Debug override (put near other constants) ----
DEBUG_FORCE_TT = False
FORCED_TT = "{U,U,U,0,0,S,S,S,S,S,0,P,0,0,0,0,0,0,S,S,S,S,S,S},{60,60}"
# ---------------------------------------------------

def now(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_LOCAL.mkdir(parents=True, exist_ok=True)
    BACKUP_LOCAL.mkdir(parents=True, exist_ok=True)

# -------------------- Rclone Mount Helpers --------------------
def _have_rclone() -> bool:
    try:
        subprocess.run([RCLONE, "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def _is_mounted(p: Path) -> bool:
    return os.path.ismount(str(p))

def ensure_mount(timeout_s: float = 12.0) -> bool:
    if _is_mounted(MOUNT_POINT):
        return True
    if not _have_rclone():
        print(f"[{now()}] ❌ rclone not found on PATH; cannot mount.")
        return False
    MOUNT_POINT.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [RCLONE, "mount", f"{RCLONE_REMOTE}:", str(MOUNT_POINT),
             "--vfs-cache-mode=writes", "--allow-other", "--daemon"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"[{now()}] ❌ rclone mount invocation failed: {e}")
        return False
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if _is_mounted(MOUNT_POINT):
            return True
        time.sleep(0.25)
    return _is_mounted(MOUNT_POINT)

def fetch_remote_config() -> bool:
    # 1) Prefer direct rclone → bypass FUSE
    if _have_rclone():
        try:
            remote_path = f"{RCLONE_REMOTE}:To the node 1/config.ini"
            res = subprocess.run(
                [RCLONE, "copyto", remote_path, str(CONFIG_LOCAL)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if res.returncode == 0:
                print(f"[{now()}] DEBUG: rclone copyto config.ini succeeded.")
                return True
            else:
                print(f"[{now()}] DEBUG: rclone copyto failed rc={res.returncode}")
        except Exception as e:
            print(f"[{now()}] DEBUG: rclone copyto raised {e!r}")

    # 2) Fallback: try the mounted path (best-effort)
    if not ensure_mount():
        print(f"[{now()}] DEBUG: ensure_mount() failed inside fetch_remote_config")
        return False

    print(f"[{now()}] DEBUG: checking CONFIG_REMOTE={CONFIG_REMOTE}")
    try:
        if CONFIG_REMOTE.exists():
            CONFIG_LOCAL.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CONFIG_REMOTE, CONFIG_LOCAL)
            print(f"[{now()}] DEBUG: copy2 via mount succeeded.")
            return True
        else:
            print(f"[{now()}] DEBUG: CONFIG_REMOTE does not exist at init time.")
    except Exception as e:
        print(f"[{now()}] ⚠️ Failed to copy config via mount: {e}")
    return False


# -------------------------------------------------------------

# -------------------- Config / INI ---------------------------
def parse_simple_ini(path: Path):
    d = {}
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
        d[k.strip().lower().replace(" ", "")] = v.strip()
    return d

def as_bool(s, default=False):
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "on", "y")
# -------------------------------------------------------------



# ---------------- Timetable parsing + remap ------------------
def _parse_tt_block(text: str):
    m = re.search(r"\{([USP0 ,]+)\}\s*,\s*\{(\d+)\s*,\s*(\d+)\}", text)
    if not m:
        raise ValueError("Timetable not found")
    hours = [h.strip() for h in m.group(1).split(",")]
    if len(hours) != 24:
        raise ValueError("Expected 24 hour marks")
    rec_s, gap_s = int(m.group(2)), int(m.group(3))
    return hours, rec_s, gap_s

def _load_timetable_from_out():
    tt_file = OUT_DIR / "nucleo_give_me_time_table.txt"
    return _parse_tt_block(tt_file.read_text(encoding="utf-8"))

def _find_window(hours):
    nowdt = datetime.datetime.now()
    cur_hr = nowdt.hour
    if 'P' not in hours:
        anchor = cur_hr
        prevP = (cur_hr - 1) % 24
        start_date = (nowdt - datetime.timedelta(days=1)).date()
    else:
        anchor = cur_hr if hours[cur_hr] == 'P' else max((i for i, h in enumerate(hours) if h == 'P'), default=cur_hr)
        prev_candidates = [i for i, h in enumerate(hours) if h == 'P' and i < anchor]
        if prev_candidates:
            prevP = prev_candidates[-1]
            start_date = nowdt.date()
        else:
            prevP = max((i for i, h in enumerate(hours) if h == 'P'), default=anchor)
            start_date = (nowdt - datetime.timedelta(days=1)).date()
    seq, day, h = [], start_date, (prevP + 1) % 24
    while True:
        if h == anchor:
            break
        seq.append((day, h, hours[h]))
        h = (h + 1) % 24
        if h == 0:
            day = day + datetime.timedelta(days=1)
    return anchor, seq

def _expected_slots(seq, rec_s, gap_s):
    slots, period = [], max(1, rec_s + gap_s)
    for day, hour, mark in seq:
        if mark in ('S', 'U'):
            t0 = datetime.datetime.combine(day, datetime.time(hour, 0, 0))
            k = 0
            while k * period < 3600:
                slots.append(t0 + datetime.timedelta(seconds=k * period))
                k += 1
    return slots

def remap_and_write_csv(cfg):
    if DEBUG_FORCE_TT:
        hours, rec_s, gap_s = _parse_tt_block(FORCED_TT)
    elif cfg and 'timetable' in cfg:
        hours, rec_s, gap_s = _parse_tt_block(cfg['timetable'])
    else:
        hours, rec_s, gap_s = _load_timetable_from_out()

    _, seq = _find_window(hours)
    slots = _expected_slots(seq, rec_s, gap_s)

    files = sorted(
        [p for ext in ("*.WAV", "*.wav", "*.wav.gz", "*.flac") for p in DRIVE_DIR.rglob(ext)],
        key=lambda p: (p.stat().st_mtime, p.name)
    )

    csv_out = OUT_DIR / "remapped_times.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "actual_time_24h"])
        if not slots or not files:
            w.writerow(["-", datetime.datetime.now().strftime("%d/%m/%Y")])
            return csv_out
        n = min(len(files), len(slots))
        for i in range(n):
            w.writerow([str(files[i].name), slots[i].strftime("%Y-%m-%d %H:%M:%S")])
        if len(files) < len(slots):
            w.writerow(["MISSING_RECORDINGS", datetime.datetime.now().strftime("%d/%m/%Y")])
    return csv_out
# -------------------------------------------------------------

# ----------------------------- UI ----------------------   ------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Koreronet Wakeup")
        self.geometry("900x600")
        self.msgq = queue.Queue()
        self.worker = None
        self.script_proc = None
        self.last_session_log: Path | None = None
        self._done_monitor_thread = None
        self._closing = False
    

        # UI
        top = ttk.Frame(self); top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Other script:").pack(side="left")
        self.combo = ttk.Combobox(top, width=60, values=self._scan_scripts(), state="readonly")
        self.combo.pack(side="left", padx=6)
        ttk.Button(top, text="Run", command=self.run_selected).pack(side="left")
        ttk.Button(top, text="Stop", command=self.stop_selected).pack(side="left", padx=6)
        ttk.Button(top, text="Refresh List", command=self.refresh_scripts).pack(side="left")
        ttk.Button(top, text="Start UART Job", command=self.start_worker).pack(side="left", padx=10)

        self.log = scrolledtext.ScrolledText(self, wrap="word", height=28)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)
        self.after(100, self._drain)

        ensure_dirs()
        # Record session start and clear any leftover done flags BEFORE monitoring begins
        self._session_started_at = time.time()
        self._clear_stale_done_flags()

        self._init_config()           # mounts, pulls config, ensures folders, checks updates
        self.start_worker()
        self._start_done_monitor()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _upload_export_pair(self) -> bool:
        """
        Upload contents of /home/amin/bn15/export into 'From the node 1'
        on the mounted gdrive. Returns True on success.
        """
        if not EXPORT_LOCAL.exists() or not any(EXPORT_LOCAL.iterdir()):
            self.logln(f"[{now()}] export/ is empty; nothing to upload.")
            return True  # nothing to do is not a hard failure

        # Log what we are about to send
        try:
            files = [p.name for p in EXPORT_LOCAL.iterdir() if p.is_file()]
            self.logln(f"[{now()}] export/ contains: {', '.join(files) or '(no files)'}")
        except Exception as e:
            self.logln(f"[{now()}] Could not list export/: {e}")

        # Use the mounted filesystem explicitly
        if not ensure_mount():
            self.logln(f"[{now()}] ❌ Cannot mount gdrive; export upload aborted.")
            return False

        dst = MOUNT_POINT / "From the node 1"
        try:
            dst.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logln(f"[{now()}] ❌ Cannot create remote folder {dst}: {e}")
            return False

        try:
            copied = self._mount_copy_dir(EXPORT_LOCAL, dst)
            self.logln(f"[{now()}] Uploaded {copied} file(s) from export/ to {dst}")
            return copied > 0
        except Exception as e:
            self.logln(f"[{now()}] ❌ export upload exception: {e}")
            return False
   
    # ----- Clear leftover done flags on startup -----
    def _clear_stale_done_flags(self):
        """
        Remove stale done.ini files from previous runs so we don't immediately
        enter the completion/shutdown path.
        """
        candidates = {
            DONE_FLAG,                           # /home/amin/KoreroNET/out/done.ini
            BN15_DIR / "done.ini",              # safety: if ever dropped at BN15 root
            OUT_DIR / "done.ini",               # safety: if ever dropped under BN15/out
        }
        removed = 0
        for p in candidates:
            try:
                if p.exists():
                    # Treat as stale if clearly older than this session start
                    if p.stat().st_mtime < self._session_started_at - 1:
                        p.unlink()
                        removed += 1
                        self.logln(f"[{now()}] 🧹 Removed stale done flag: {p}")
            except Exception as e:
                self.logln(f"[{now()}] ⚠️ Could not remove {p}: {e}")
        if removed == 0:
            self.logln(f"[{now()}] (no stale done flags found)")

    # ----- Config init (ensures Drive is mounted & pulls config.ini) -----
    def _init_config(self):
        mounted = ensure_mount()
        if mounted:
            self.logln(f"[{now()}] ☁️  gdrive mounted at {MOUNT_POINT}")
        else:
            self.logln(f"[{now()}] ❌ Unable to mount gdrive at {MOUNT_POINT} (check rclone/service)")

        ok = fetch_remote_config()
        if ok:
            self.logln(f"[{now()}] ☁️  Pulled config.ini → {CONFIG_LOCAL}")
        else:
            self.logln(f"[{now()}] ⚠️  No remote config at {CONFIG_REMOTE}; using local if present.")

        self.cfg = parse_simple_ini(CONFIG_LOCAL)
        self.logln(f"[{now()}] Config: " + (", ".join(f"{k}={v}" for k, v in self.cfg.items()) or "EMPTY"))

        try:
            if ensure_mount():
                IMPORT_DIR.mkdir(parents=True, exist_ok=True)
                EXPORT_DIR.mkdir(parents=True, exist_ok=True)
                # check for self-update payload right after we know Drive is available
                self._check_updates_and_trigger()
        except Exception as e:
            self.logln(f"[{now()}] ⚠️ Could not ensure cloud subfolders / update check: {e}")

    # ---------- Self-update helpers ----------
    def _list_files_except_done(self, root: Path):
        try:
            return [p for p in root.rglob("*") if p.is_file() and p.name.lower() != "done.ini"]
        except Exception:
            return []

    def _clear_directory_contents(self, root: Path) -> int:
        """Delete all children of 'root' (leave root directory). Returns removed count."""
        removed = 0
        if not root.exists():
            return 0
        for child in list(root.iterdir()):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    shutil.rmtree(child, ignore_errors=True)
                removed += 1
            except Exception as e:
                self.logln(f"[{now()}] Purge error at {child}: {e}")
        return removed

    def _mount_copy_dir(self, src: Path, dst: Path) -> int:
        """Copy directory tree src->dst (preserve structure). Returns number of files copied."""
        if not ensure_mount():
            self.logln(f"[{now()}] No mount; skipping mount copy.")
            return 0
        count = 0
        try:
            dst.mkdir(parents=True, exist_ok=True)
            for p in src.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(src)
                    outp = dst / rel
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, outp)
                    count += 1
            return count
        except Exception as e:
            self.logln(f"[{now()}] Mount copy error: {e}")
            return count

    def _spawn_updater(self, updater_path: Path):
        """Spawn updater and mark a lock to avoid duplicate spawns, then hard-exit."""
        # Prevent duplicate spawns
        try:
            if UPDATER_LOCK.exists():
                age = time.time() - UPDATER_LOCK.stat().st_mtime
                if age < 180:
                    self.logln(f"[{now()}] Updater already spawned recently; skipping re-run.")
                    return
            UPDATER_LOCK.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass

        python_bin = "/home/amin/bn15/bin/python3"
        py = python_bin if Path(python_bin).exists() else sys.executable

        log_file = LOG_DIR / "updater_live.log"
        self.logln(f"[{now()}] ▶ Starting updater: {py} -u {updater_path}")
        try:
            with log_file.open("a", encoding="utf-8") as uf:
                uf.write(f"[{now()}] spawn: {py} -u {updater_path}\n")
                subprocess.Popen([py, "-u", str(updater_path)], stdout=uf, stderr=uf)
        except Exception as e:
            self.logln(f"[{now()}] ❌ Failed to spawn updater: {e}")
            return

        # Exit immediately so updater can replace files safely
        self.logln(f"[{now()}] Exiting wakeup to let updater run …")
        try:
            self._closing = True
            self.destroy()
        finally:
            os._exit(0)

    def _check_updates_and_trigger(self):
        """
        If 'To the node 1/update files' contains files:
          - copy to local cache (/home/amin/bn15/_update_cache),
          - purge that remote folder and drop done.ini there (ack of pickup),
          - launch **the cached updater.py if present**; otherwise BN15/updater.py,
          - terminate this app so updater can safely replace files.
        """
        if not ensure_mount():
            self.logln(f"[{now()}] Skipping update check: Drive not mounted.")
            return

        if not REMOTE_UPDATE_DIR.exists() or not REMOTE_UPDATE_DIR.is_dir():
            self.logln(f"[{now()}] No update folder at {REMOTE_UPDATE_DIR}")
            return

        payload = self._list_files_except_done(REMOTE_UPDATE_DIR)
        if not payload:
            self.logln(f"[{now()}] No update payload found in {REMOTE_UPDATE_DIR}")
            return

        self.logln(f"[{now()}] 🔁 Update payload detected: {len(payload)} file(s) at {REMOTE_UPDATE_DIR}")

        # Prepare local cache
        try:
            if UPDATE_CACHE_DIR.exists():
                _ = self._clear_directory_contents(UPDATE_CACHE_DIR)
            UPDATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            copied = self._mount_copy_dir(REMOTE_UPDATE_DIR, UPDATE_CACHE_DIR)
            self.logln(f"[{now()}] Cached {copied} file(s) → {UPDATE_CACHE_DIR}")
        except Exception as e:
            self.logln(f"[{now()}] ❌ Failed to cache update payload: {e}")
            return

        # Purge remote and leave done.ini
        try:
            removed = self._clear_directory_contents(REMOTE_UPDATE_DIR)
            self.logln(f"[{now()}] 🧹 Purged {removed} item(s) from {REMOTE_UPDATE_DIR}")
            done_path = REMOTE_UPDATE_DIR / "done.ini"
            with done_path.open("w", encoding="utf-8") as f:
                f.write(f"updated_at={now()}\n")
            self.logln(f"[{now()}] Dropped {done_path}")
        except Exception as e:
            self.logln(f"[{now()}] ⚠️ Could not finalize remote purge/done.ini: {e}")

        # Decide which updater to run: prefer the cached updater if present
        cached_updater = UPDATE_CACHE_DIR / "updater.py"
        updater_to_run = cached_updater if cached_updater.exists() else UPDATER_SCRIPT
        if not updater_to_run.exists():
            self.logln(f"[{now()}] ❌ No updater.py found (cache or BN15). Aborting update.")
            return

        self._spawn_updater(updater_to_run)

    def _scan_scripts(self):
        all_py = sorted(Path("/home/amin/bn15").glob("*.py"))
        return [str(p) for p in all_py if p.name != "koreronet-wakeup.py"]

    def refresh_scripts(self):
        self.combo["values"] = self._scan_scripts()
        if not self.combo.get() and self.combo["values"]:
            self.combo.set(self.combo["values"][0])

    def logln(self, msg):
        sys.stdout.write(msg + "\n"); sys.stdout.flush()
        self.msgq.put(msg)

    def _drain(self):
        try:
            while True:
                msg = self.msgq.get_nowait()
                self.log.insert("end", msg + "\n"); self.log.see("end")
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _on_close(self):
        self._closing = True
        self.destroy()

    # ---------------- UART worker ----------------
    def start_worker(self):
        if self.worker and self.worker.is_alive():
            self.logln(f"[{now()}] Worker already running."); return
        self.worker = threading.Thread(target=self._uart_job, daemon=True)
        self.worker.start()

    def _open_serial(self):
        try:
            import serial
        except Exception as e:
            self.logln(f"[{now()}] pyserial not installed: {e}")
            return None, None
        s, port = None, None
        for port_candidate in CANDIDATE_PORTS:
            try:
                s = serial.Serial(port_candidate, BAUD, timeout=1.0, write_timeout=2.0)
                port = port_candidate; break
            except Exception as e:
                self.logln(f"[{now()}] Port {port_candidate} failed: {e}")
        return s, port

    def _uart_job(self):
        log_path = LOG_DIR / f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
        self.last_session_log = log_path
        self.logln(f"[{now()}] Session log: {log_path}")
        try:
            s, port = self._open_serial()
            if not s:
                self.logln(f"[{now()}] ERROR opening serial. UART job aborted.")
                with log_path.open("a", encoding="utf-8") as log: log.write(f"[{now()}] ERROR: no UART.\n")
                return

            try:
                import serial
                with log_path.open("a", encoding="utf-8") as log:
                    with s:
                        self.logln(f"[{now()}] Opened UART: {port} @ {BAUD}")
                        for cmd in CMDS:
                            self.logln(f"[{now()}] ▶ {cmd}")
                            s.reset_input_buffer(); s.reset_output_buffer()
                            s.write(cmd.encode() + EOL); s.flush(); time.sleep(1.0)
                            deadline = time.monotonic() + 2.5
                            lines, buf = [], b""
                            while time.monotonic() < deadline:
                                chunk = s.read(1024)
                                if chunk:
                                    buf += chunk
                                    while b"\n" in buf:
                                        line, buf = buf.split(b"\n", 1)
                                        line = line.rstrip(b"\r").decode(errors="replace")
                                        lines.append(line); self.logln(f"→ {line}")
                                else:
                                    time.sleep(0.02)
                            if buf:
                                line = buf.rstrip(b"\r").decode(errors="replace")
                                lines.append(line); self.logln(f"→ {line}")
                            out_file = OUT_DIR / (cmd.replace(" ", "_") + ".txt")
                            with out_file.open("w", encoding="utf-8") as f:
                                for ln in lines: f.write(ln + "\n")
                            log.write(f"\n[{now()}] ---- {cmd} ----\n")
                            log.writelines([ln + "\n" for ln in lines]); log.flush()
            except Exception as e:
                self.logln(f"[{now()}] UART session error: {e}")

            self.logln(f"[{now()}] ⏳ Waiting 5 s before checking USB drive …")
            time.sleep(5)
            exists = os.path.ismount(CHECK_PATH) or os.path.isdir(CHECK_PATH)
            msg = f'[{now()}] Drive check: {"FOUND ✅" if exists else "NOT FOUND ❌"} → {CHECK_PATH}'
            self.logln(msg)
            try:
                with log_path.open("a", encoding="utf-8") as log: log.write(msg + "\n")
            except Exception:
                pass

            self.logln(f"[{now()}] ✅ UART session done. Logs: {log_path}")

            # ---------- Remap using timetable ----------
            try:
                csv_path = remap_and_write_csv(self.cfg)
                self.logln(f"[{now()}] 🗂 Remap CSV: {csv_path}")
            except Exception as e:
                self.logln(f"[{now()}] Remap failed: {e}")

            # ---------- Optional: trigger BirdNET ----------
            birdnet_on = as_bool(self.cfg.get("birdnet"))
            if birdnet_on:
                self.logln(f"[{now()}] 🐦 BirdNET enabled → starting process.py in a terminal")
                self._launch_process_with_terminal("/home/amin/bn15/process.py")
            else:
                self.logln(f"[{now()}] 🐦 BirdNET disabled by config.ini; skipping.")

        except Exception as e:
            self.logln(f"[{now()}] FATAL: {e}")

    # ---------------- Terminal launcher for process.py ---------------
    def _which(self, exe: str) -> bool:
        return shutil.which(exe) is not None

    def _launch_process_with_terminal(self, script_path: str):
        python_bin = "/home/amin/bn15/bin/python3"
        if not Path(python_bin).exists():
            python_bin = sys.executable
        cmd_line = f'{python_bin} -u "{script_path}" ; read -p "Press Enter to close…"'

        for term, flags in TERMINALS:
            if not self._which(term):
                continue
            try:
                if term in ("gnome-terminal",):
                    full = [term] + flags + [cmd_line]     # gnome-terminal -- bash -lc "<cmd>"
                else:
                    full = [term] + flags + ["bash", "-lc", cmd_line]  # others: -e bash -lc "<cmd>"
                subprocess.Popen(full)
                self.logln(f"[{now()}] Opened terminal: {term} → running process.py")
                return
            except Exception as e:
                self.logln(f"[{now()}] {term} failed: {e}")

        # Fallback: run inline and pipe output into UI
        self.logln(f"[{now()}] ⚠️ No terminal found. Running process.py inline.")
        env = os.environ.copy(); env.setdefault("PYTHONUNBUFFERED", "1")
        self.script_proc = subprocess.Popen(
            [python_bin, "-u", script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
        )
        threading.Thread(target=self._pump_proc_output, daemon=True).start()

    # ---------------- done.ini monitor & wrap-up ----------------
    def _start_done_monitor(self):
        if self._done_monitor_thread and self._done_monitor_thread.is_alive():
            return
        self._done_monitor_thread = threading.Thread(target=self._done_monitor_loop, daemon=True)
        self._done_monitor_thread.start()

    def _done_monitor_loop(self):
        self.logln(f"[{now()}] ⏳ Monitoring for done flag at {DONE_FLAG}")
        while not self._closing:
            try:
                if DONE_FLAG.exists():
                    # Ignore stale flags that predate this session
                    try:
                        mtime = DONE_FLAG.stat().st_mtime
                    except Exception:
                        mtime = time.time()
                    if mtime < getattr(self, "_session_started_at", time.time()) - 1:
                        try:
                            DONE_FLAG.unlink()
                            self.logln(f"[{now()}] 🧹 Ignored & removed stale done.ini")
                        except Exception as e:
                            self.logln(f"[{now()}] ⚠️ Could not remove stale done.ini: {e}")
                        time.sleep(2)
                        continue

                    self.logln(f"[{now()}] ✅ done.ini detected. Waiting 2 seconds…")
                    time.sleep(2)
                    try:
                        self._handle_completion()
                    finally:
                        # Always remove the flag if still present
                        try:
                            if DONE_FLAG.exists():
                                DONE_FLAG.unlink()
                                self.logln(f"[{now()}] done.ini cleared.")
                        except Exception as e:
                            self.logln(f"[{now()}] Could not delete done.ini: {e}")
                    break
            except Exception as e:
                self.logln(f"[{now()}] done-flag monitor error: {e}")
            time.sleep(3)

    # ---- Cloud helpers ----
    def _rclone_copy(self, src: Path, remote_subpath: str) -> bool:
        if not _have_rclone():
            self.logln(f"[{now()}] rclone not available.")
            return False
        remote = f'{RCLONE_REMOTE}:{remote_subpath}'
        try:
            res = subprocess.run(
                [RCLONE, "copy", str(src), remote, "--ignore-existing", "--create-empty-src-dirs",
                 "--transfers", "4", "--checkers", "8"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            self.logln(f"[{now()}] rclone copy {src} → {remote} rc={res.returncode}")
            if res.stdout:
                self.logln(res.stdout.strip())
            return res.returncode == 0
        except Exception as e:
            self.logln(f"[{now()}] rclone copy exception: {e}")
            return False

    def _mount_copy_dir_legacy(self, src: Path, dst: Path) -> bool:
        copied = self._mount_copy_dir(src, dst)
        return copied > 0

    def _copy_to_cloud(self, src: Path, remote_subdir: str) -> bool:
        ok = self._rclone_copy(src, remote_subdir)
        if ok:
            return True
        dst = MOUNT_POINT / remote_subdir
        return self._mount_copy_dir_legacy(src, dst)

    def _safe_clear_contents(self, path: Path):
        try:
            if not path.exists():
                return
            for child in path.iterdir():
                try:
                    if child.is_file():
                        child.unlink()
                    else:
                        shutil.rmtree(child, ignore_errors=True)
                except Exception as e:
                    self.logln(f"[{now()}] Cleanup error in {path}: {e}")
            self.logln(f"[{now()}] Cleared: {path}")
        except Exception as e:
            self.logln(f"[{now()}] Cleanup exception in {path}: {e}")

    # ---- Purge USB drive ----
    def _purge_drive_dir(self):
        """
        Recursively delete all contents of the USB drive folder (CHECK_PATH).
        Leaves the top-level folder in place. Safe no-op if missing.
        """
        base = DRIVE_DIR
        try:
            if not base.exists():
                self.logln(f"[{now()}] USB path not found; nothing to purge: {base}")
                return
            if str(base) != CHECK_PATH:
                self.logln(f"[{now()}] Refusing to purge unexpected path: {base}")
                return
            purged = 0
            for child in base.iterdir():
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    else:
                        shutil.rmtree(child, ignore_errors=True)
                    purged += 1
                except Exception as e:
                    self.logln(f"[{now()}] Purge error at {child}: {e}")
            self.logln(f"[{now()}] 🧹 Purged {purged} item(s) from {base}")
        except Exception as e:
            self.logln(f"[{now()}] Purge exception for {base}: {e}")

    # ---- Nucleo helpers ----
    def _send_nucleo(self, cmd: str, wait_s: float = 2.0):
        s, port = self._open_serial()
        if not s:
            self.logln(f"[{now()}] Cannot open UART to send: {cmd}")
            return
        try:
            with s:
                self.logln(f"[{now()}] ▶ {cmd}")
                s.reset_input_buffer(); s.reset_output_buffer()
                s.write(cmd.encode() + EOL); s.flush()
                deadline = time.monotonic() + wait_s
                buf = b""
                while time.monotonic() < deadline:
                    chunk = s.read(256)
                    if not chunk:
                        time.sleep(0.05); continue
                    buf += chunk
                if buf:
                    for ln in buf.decode(errors="replace").splitlines():
                        self.logln(f"→ {ln}")
        except Exception as e:
            self.logln(f"[{now()}] Send failed: {e}")

    def _internet_online(self) -> bool:
        try:
            res = subprocess.run(["ping", "-c", "1", "-W", "2", "8.8.8.8"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0
        except Exception:
            return False

    def _update_system_time_from_internet(self):
        try:
            subprocess.run(["sudo", "timedatectl", "set-ntp", "true"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
        except Exception as e:
            self.logln(f"[{now()}] timedatectl error: {e}")

    # --------- Power log helpers ----------
    def _find_latest_session_log(self) -> Path | None:
        try:
            candidates = [p for p in LOG_DIR.glob("session_*.log") if p.is_file()]
            if not candidates:
                return self.last_session_log if (self.last_session_log and self.last_session_log.exists()) else None
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            return self.last_session_log if (self.last_session_log and self.last_session_log.exists()) else None

    def _find_gui_autostart_log(self) -> Path | None:
        for root in (LOG_DIR, Path("/home/amin/bn15")):
            try:
                matches = list(root.glob("*gui_autostart*.log"))
                if matches:
                    return max(matches, key=lambda p: p.stat().st_mtime)
            except Exception:
                pass
        return None

    def _extract_power_history_block(self, text: str) -> tuple[str | None, list[str], list[str]]:
        lines = text.splitlines()
        snapshot = []
        i = 0
        while i < len(lines):
            if "---- nucleo power stats ----" in lines[i]:
                j = i + 1
                while j < len(lines) and not lines[j].startswith("["):
                    ln = lines[j].strip()
                    if ln and not ln.startswith("ACK:"):
                        snapshot.append(ln)
                    j += 1
                break
            i += 1

        ts_str = None
        ph_lines = []
        i = 0
        while i < len(lines):
            if "---- nucleo power history ----" in lines[i]:
                m = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", lines[i])
                if m:
                    ts_str = m.group(1)
                j = i + 1
                while j < len(lines) and not lines[j].startswith("["):
                    if lines[j].strip().startswith("PH_"):
                        ph_lines.append(lines[j].strip())
                    j += 1
                break
            i += 1
        return ts_str, snapshot, ph_lines

    def _build_and_upload_power_history(self, session_log: Path):
        if not (session_log and session_log.exists()):
            self.logln(f"[{now()}] No session log found; skipping power history upload.")
            return

        try:
            txt = session_log.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.logln(f"[{now()}] Could not read session log: {e}")
            return

        ts_str, snapshot, ph_lines = self._extract_power_history_block(txt)
        if not ts_str:
            ts_str = datetime.datetime.fromtimestamp(session_log.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        tmpdir = Path(tempfile.mkdtemp(prefix="_powerlog_"))
        ts_for_name = ts_str.replace(":", "").replace("-", "").replace(" ", "_")
        out_path = tmpdir / f"power_history_{ts_for_name}.log"

        try:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(ts_str + "\n")
                for ln in snapshot:
                    f.write(ln + "\n")
                for ln in ph_lines:
                    f.write(ln + "\n")

            self.logln(f"[{now()}] Uploading power history → {POWERLOG_REMOTE_BASE}")
            self._copy_to_cloud(tmpdir, POWERLOG_REMOTE_BASE)
        except Exception as e:
            self.logln(f"[{now()}] Failed to build/upload power history: {e}")
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

        try:
            rawtmp = Path(tempfile.mkdtemp(prefix="_rawlogs_"))
            sl_ts = datetime.datetime.fromtimestamp(session_log.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            sl_copy = rawtmp / f"{sl_ts}__{session_log.name}"
            shutil.copy2(session_log, sl_copy)

            ga = self._find_gui_autostart_log()
            if ga and ga.exists():
                ga_ts = datetime.datetime.fromtimestamp(ga.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
                ga_copy = rawtmp / f"{ga_ts}__{ga.name}"
                shutil.copy2(ga, ga_copy)

            self.logln(f"[{now()}] Uploading raw logs → {RAW_SUBDIR}")
            self._copy_to_cloud(rawtmp, RAW_SUBDIR)
        except Exception as e:
            self.logln(f"[{now()}] Raw log upload failed: {e}")
        finally:
            try:
                shutil.rmtree(rawtmp, ignore_errors=True)
            except Exception:
                pass

    # ---- Completion handler ----
    def _handle_completion(self):
        self.logln(f"[{now()}] 🚩 Handling completion sequence…")

        # 0) Publish power logs
        latest_log = self._find_latest_session_log()
        if latest_log:
            self._build_and_upload_power_history(latest_log)
        else:
            self.logln(f"[{now()}] No session log available for power history.")

        # 1) Upload EXPORT_LOCAL → From the node/
        remap_file = OUT_DIR / "remapped_times.csv"
        remap_uploaded = False
        if remap_file.exists():
            remap_subdir = "From the node 1/Remap"
            self.logln(f"[{now()}] Uploading remapped_times.csv to cloud subfolder {remap_subdir}…")
            remap_uploaded = self._copy_to_cloud(remap_file, remap_subdir)
            
        # Upload export/ (BN + KN master CSVs) into From the node 1
        self.logln(f"[{now()}] Uploading export/ (BN/KN masters) to From the node 1 …")
        if remap_uploaded:
            # keep remap only in its own Remap/ folder
            for f in EXPORT_LOCAL.glob("remapped_times.csv"):
                try:
                    f.unlink()
                except Exception as e:
                    self.logln(f"[{now()}] Could not remove remapped_times.csv from export/: {e}")
        ok_export = self._upload_export_pair()


        # 2) If sendbackup flag, upload backup + session log to From the node/Backup/<ts>/
        cfg = parse_simple_ini(CONFIG_LOCAL)
        send_backup = as_bool(cfg.get("sendbackup")) or as_bool(cfg.get("backup")) or as_bool(cfg.get("uploadbackup"))
        ok_backup = True
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_remote_subdir = f"From the node 1/Backup/{ts}"
        if send_backup:
            if BACKUP_LOCAL.exists() and any(BACKUP_LOCAL.iterdir()):
                self.logln(f"[{now()}] Uploading backup/ to cloud at {backup_remote_subdir} …")
                ok_backup = self._copy_to_cloud(BACKUP_LOCAL, backup_remote_subdir)
            else:
                self.logln(f"[{now()}] backup/ is empty; skipping audio backup.")
                ok_backup = True

            if self.last_session_log and self.last_session_log.exists():
                try:
                    temp_dir = Path("/tmp/_nucleo_logs")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_copy = temp_dir / self.last_session_log.name
                    shutil.copy2(self.last_session_log, temp_copy)
                    self._copy_to_cloud(temp_dir, backup_remote_subdir + "/logs")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    self.logln(f"[{now()}] Could not upload session log to backup area: {e}")

                try:
                    if EXPORT_LOCAL.exists() and any(EXPORT_LOCAL.iterdir()):
                        self.logln(f"[{now()}] Adding BN/KN master CSVs to backup folder {backup_remote_subdir} …")
                        if ensure_mount():
                            dst = MOUNT_POINT / backup_remote_subdir
                            dst.mkdir(parents=True, exist_ok=True)
                            copied = self._mount_copy_dir(EXPORT_LOCAL, dst)
                            self.logln(f"[{now()}] Copied {copied} export file(s) into backup folder.")
                        else:
                            self.logln(f"[{now()}] ❌ Could not mount Drive; export files NOT copied to backup.")
                    else:
                        self.logln(f"[{now()}] export/ empty → no BN/KN files to include in backup.")
                except Exception as e:
                    self.logln(f"[{now()}] Exception while copying export files to backup: {e}")
        # 3) On successful cloud copies, clear local export, out, backup

        if ok_export and ok_backup:
            self._safe_clear_contents(EXPORT_LOCAL)
            self._safe_clear_contents(PROCESS_OUT_DIR)  # includes backup inside
        else:
            self.logln(f"[{now()}] ⚠️ Upload not fully successful; local files retained.")

        # 3.5) Purge the USB drive folder regardless
        self._purge_drive_dir()

        # 4) Refresh config.ini from Drive and apply timetable if provided
        if fetch_remote_config():
            self.logln(f"[{now()}] ✅ Refreshed config from Drive.")
        else:
            self.logln(f"[{now()}] ⚠️ Could not refresh config from Drive (using local).")
        cfg = parse_simple_ini(CONFIG_LOCAL)
        timetable = cfg.get("timetable")
        if timetable:
            tt_line = " ".join(timetable.split())
            self._send_nucleo(f"nucleo timetable {tt_line}", wait_s=3.0)
        else:
            self.logln(f"[{now()}] No timetable in config.ini; skipping timetable set.")

        # 5) If online → sync time and set Nucleo time
        if self._internet_online():
            self.logln(f"[{now()}] 🌐 Internet detected → updating system time …")
            self._update_system_time_from_internet()
            dt = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            self._send_nucleo(f"nucleo time is {dt}", wait_s=2.0)
        else:
            self.logln(f"[{now()}] 🌐 No internet; skipping time sync and Nucleo time set.")

        # 6) Tell Nucleo we're done, then poweroff
        self._send_nucleo("nucleo processing completed", wait_s=1.5)
        self.logln(f"[{now()}] 📴 Shutting down now …")
        try:
            subprocess.Popen(["sudo", "poweroff"])
        except Exception as e:
            self.logln(f"[{now()}] Could not power off: {e}")

    # ---------------- External script runner (manual) ---------------
    def run_selected(self):
        path = self.combo.get()
        if not path:
            messagebox.showwarning("No script", "Select a script first."); return
        if self.script_proc and self.script_proc.poll() is None:
            messagebox.showinfo("Running", "Another script is already running."); return
        self.logln(f"[{now()}] ▶ Running: {path}")
        env = os.environ.copy(); env.setdefault("PYTHONUNBUFFERED", "1")
        py = "/home/amin/bn15/bin/python3" if Path("/home/amin/bn15/bin/python3").exists() else sys.executable
        self.script_proc = subprocess.Popen(
            [py, "-u", path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
        )
        threading.Thread(target=self._pump_proc_output, daemon=True).start()

    def _pump_proc_output(self):
        try:
            assert self.script_proc and self.script_proc.stdout
            for line in self.script_proc.stdout:
                self.logln(line.rstrip("\n"))
            code = self.script_proc.wait()
            self.logln(f"[{now()}] ▶ Script exited with code {code}")
        except Exception as e:
            self.logln(f"[{now()}] ▶ Script error: {e}")

    def stop_selected(self):
        if self.script_proc and self.script_proc.poll() is None:
            self.logln(f"[{now()}] ⛔ Terminating script…"); self.script_proc.terminate()
        else:
            self.logln(f"[{now()}] No running script.")

# ----------------------------- Run ---------------------------
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ensure_dirs()
    App().mainloop()
