#!/usr/bin/env bash
set -euo pipefail
APPDIR="$HOME/bn15"
mkdir -p "$APPDIR" "$HOME/.config/systemd/user" "$APPDIR/logs" "$APPDIR/out"

# 1) Python deps (pip-managed)
cat > "$APPDIR/requirements.txt" <<'EOF'
pyserial
EOF

# 2) User service unit (GUI-friendly; starts after login)
cat > "$HOME/.config/systemd/user/koreronet-wakeup.service" <<'EOF'
[Unit]
Description=Koreronet Wakeup GUI (user)
Wants=graphical-session.target
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u %h/bn15/koreronet-wakeup.py
Restart=on-failure

[Install]
WantedBy=graphical-session.target
EOF

# 3) Optional XDG autostart fallback (launcher waits for GUI env)
cat > "$APPDIR/koreronet-wakeup-launch.sh" <<'EOF'
#!/usr/bin/env bash
LOG="$HOME/bn15/logs/gui_autostart.log"
exec >>"$LOG" 2>&1
for i in {1..60}; do
  [[ -n "$DISPLAY" && -n "$DBUS_SESSION_BUS_ADDRESS" && -n "$XDG_RUNTIME_DIR" ]] && break
  sleep 1
done
exec /usr/bin/python3 -u "$HOME/bn15/koreronet-wakeup.py"
EOF
chmod +x "$APPDIR/koreronet-wakeup-launch.sh"

cat > "$HOME/.config/autostart/koreronet-wakeup.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Koreronet Wakeup
Exec=/home/amin/bn15/koreronet-wakeup-launch.sh
X-GNOME-Autostart-enabled=true
Terminal=false
EOF

# 4) Installer to set everything up on any new machine
cat > "$APPDIR/install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APPDIR="$HOME/bn15"

echo "TIP: To ensure DISPLAY/DBUS reach systemd-user each login:"
grep -q 'systemctl --user import-environment' ~/.profile || \
echo 'systemctl --user import-environment DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS' >> ~/.profile
echo "Done. Reboot & log in: the GUI should auto-start. For belt-and-braces, XDG autostart also exists."
  # System packages (Tk for GUI, libsndfile1 for soundfile, rclone for cloud, pip for Python deps)
  sudo apt-get update -y
  sudo apt-get install -y python3-tk xwayland libsndfile1 rclone python3-pip
  sudo usermod -aG dialout "$USER" || true

  # Python deps (user scope)
  pip3 install --user -r "$APPDIR/requirements.txt"

  # Register user service
  systemctl --user daemon-reload
  systemctl --user enable --now koreronet-wakeup.service

  echo "TIP: To ensure DISPLAY/DBUS reach systemd-user each login:"
  grep -q 'systemctl --user import-environment' ~/.profile || \
  echo 'systemctl --user import-environment DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS' >> ~/.profile

  echo "Done. Reboot & log in: the GUI should auto-start. For belt-and-braces, XDG autostart also exists."
EOF
chmod +x "$APPDIR/install.sh"

# 5) Tiny README
cat > "$APPDIR/README.txt" <<'EOF'
Install on this or another Pi:
  1) Copy bn15/ to the target user home.
  2) Run:  ~/bn15/install.sh
Check status:
  systemctl --user status koreronet-wakeup.service --no-pager
Logs:
  journalctl --user -u koreronet-wakeup.service -f -o cat
EOF

echo "✅ Files ready in $APPDIR:"
echo " - requirements.txt"
echo " - install.sh"
echo " - (installed) ~/.config/systemd/user/koreronet-wakeup.service"
echo " - koreronet-wakeup-launch.sh  +  ~/.config/autostart/koreronet-wakeup.desktop (fallback)"
echo "Run:  $APPDIR/install.sh"
