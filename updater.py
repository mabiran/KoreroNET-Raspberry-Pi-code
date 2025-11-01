#update success
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
updater.py — apply payload from BN15/_update_cache → BN15/, then reboot (or relaunch GUI if NO_REBOOT present)

Improvements over previous version:
• Single-instance guard via lock file (.updater.lock) to prevent concurrent runs.
• Best-effort wait for koreronet-wakeup.py to exit before replacing files.
• Atomic copy/replace with permission preservation.
• Honors NO_REBOOT file in the cache: skip reboot and relaunch koreronet-wakeup.py.
• If reboot attempts fail, fallback to relaunching koreronet-wakeup.py so the node keeps running.
• More defensive logging and safety checks.
"""

import os, sys, time, shutil, subprocess, traceback
from pathlib import Path

BN15        = Path("/home/amin/bn15")
CACHE       = BN15 / "_update_cache"
WAKEUP      = BN15 / "koreronet-wakeup.py"
PY_VENV     = BN15 / "bin/python3"
LOG_DIR     = BN15 / "logs"
LOCK_FILE   = BN15 / ".updater.lock"

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"updater_{time.strftime('%Y%m%d_%H%M%S')}.log"

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass
    print(f"[{ts}] {msg}", flush=True)

def single_instance_guard() -> bool:
    """Avoid concurrent updater runs. Treat a very recent lock as 'active'."""
    try:
        if LOCK_FILE.exists():
            age = time.time() - LOCK_FILE.stat().st_mtime
            if age < 600:  # 10 minutes
                log("Another updater instance appears to be running; exiting.")
                return False
        LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except Exception as e:
        log(f"Lock create error: {e}")
        return False

def wait_wakeup_exit(timeout: float = 8.0) -> None:
    """Best-effort: wait until koreronet-wakeup.py is gone before replacing files."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            out = subprocess.run(["pgrep", "-f", "koreronet-wakeup.py"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if out.returncode != 0 or not out.stdout.strip():
                return
        except Exception:
            return
        time.sleep(0.25)

def copy_replace_tree(src_root: Path, dst_root: Path) -> int:
    """
    Recursively copy all files from src_root into dst_root, preserving
    relative paths and atomically replacing existing files.
    Returns number of files replaced.
    """
    count = 0
    for p in sorted(src_root.rglob("*")):
        if p.is_dir():
            continue
        if p.name.lower() == "done.ini":
            continue
        rel = p.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Atomic replace: copy to .new then os.replace (preserves atomicity)
        tmp = dst.with_suffix(dst.suffix + ".new")
        shutil.copy2(p, tmp)
        # Ensure mode is preserved even if umask interferes
        try:
            st = p.stat()
            os.chmod(tmp, st.st_mode & 0o777)
        except Exception:
            pass
        os.replace(tmp, dst)
        count += 1
    return count

def clean_dir_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in list(path.iterdir()):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child, ignore_errors=True)
        except Exception as e:
            log(f"cleanup warning for {child}: {e}")

def _try_run(cmd: list[str]) -> bool:
    try:
        log(f"exec: {' '.join(cmd)}")
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        log(f"exec failed: {e}")
        return False

def reboot_system() -> bool:
    """
    Try several reboot methods (with and without sudo). Return True if any issued.
    """
    candidates = [
        ["sudo", "-n", "systemctl", "reboot"],
        ["sudo", "-n", "reboot"],
        ["sudo", "-n", "shutdown", "-r", "now"],
        ["systemctl", "reboot"],
        ["reboot"],
        ["shutdown", "-r", "now"],
        ["/sbin/reboot"],
        ["/sbin/shutdown", "-r", "now"],
    ]
    for cmd in candidates:
        if _try_run(cmd):
            return True
    return False

def relaunch_wakeup() -> None:
    """Relaunch the GUI if we're not rebooting."""
    py = str(PY_VENV) if PY_VENV.exists() else sys.executable
    try:
        log(f"Relaunching GUI: {py} -u {WAKEUP}")
        subprocess.Popen([py, "-u", str(WAKEUP)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"Failed to relaunch wakeup: {e}")

def main() -> int:
    try:
        log("Updater starting.")
        if not single_instance_guard():
            return 0

        # Safety checks
        if not BN15.exists():
            log(f"Target folder missing: {BN15}")
            return 1
        if not CACHE.exists():
            log("No _update_cache folder; nothing to do.")
            return 0

        payload_files = [p for p in CACHE.rglob("*") if p.is_file() and p.name.lower() != "done.ini"]
        if not payload_files:
            log("Cache is empty; nothing to update.")
            return 0

        # Prefer that the wakeup GUI is gone
        wait_wakeup_exit(timeout=8.0)

        log(f"Applying {len(payload_files)} file(s) from {CACHE} → {BN15}")
        replaced = copy_replace_tree(CACHE, BN15)
        log(f"Replaced {replaced} file(s).")

        # Clean cache afterwards
        clean_dir_contents(CACHE)
        log("Cache cleaned.")

        # Flush to disk before reboot/relaunch
        try:
            log("Syncing filesystems …")
            os.sync()
        except Exception:
            pass
        time.sleep(0.5)

        # Decide reboot vs relaunch
        no_reboot = (CACHE / "NO_REBOOT").exists()
        if no_reboot:
            log("NO_REBOOT flag present — skipping reboot.")
            relaunch_wakeup()
            return 0

        log("Requesting system reboot …")
        if reboot_system():
            log("Reboot command issued. Exiting updater.")
            return 0
        else:
            log("Failed to issue reboot via all methods. Falling back to relaunch GUI.")
            relaunch_wakeup()
            return 2

    except Exception as e:
        log(f"Updater error: {e}")
        log(traceback.format_exc())
        return 1
    finally:
        # Best-effort: remove stale lock after we've decided what to do
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(main())
