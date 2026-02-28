# X11 Forwarding Setup (Windows → Remote Linux Server)

This guide explains how to run GUI applications (e.g. `pygame`) on a remote Linux server and display them on your local Windows machine using X11 forwarding.

> **Note:** X11 forwarding works well for simple GUI apps (e.g. `xclock`, text editors). For real-time graphics like `pygame`, consider using [VNC](#alternative-vnc-for-better-performance) or [NoMachine](#alternative-nomachine-fastest) instead, as X11 forwarding will be very laggy for frame-by-frame rendering.

---

## Prerequisites

- **Windows PC** (local machine)
- **Remote Linux server** with SSH access

---

## Step 1: Install an X Server on Windows

You need an X server running locally to receive and display the GUI output.

1. Download and install **VcXsrv** from [sourceforge.net/projects/vcxsrv](https://sourceforge.net/projects/vcxsrv/).
2. Launch **XLaunch** (included with VcXsrv) and configure:
   - Display settings: **Multiple windows**
   - Display number: `0`
   - Client startup: **Start no client**
   - Extra settings: ✅ **Disable access control**
3. Click **Finish**. VcXsrv will run in your system tray — keep it running whenever you need X11 forwarding.

---

## Step 2: Install and Configure PuTTY

> **Important:** The built-in Windows OpenSSH client (`ssh` in PowerShell) does **not** support X11 forwarding. You must use PuTTY.

1. Download and install **PuTTY** from [putty.org](https://www.putty.org/).
2. Open PuTTY and configure:
   - **Session**:
     - Host Name: `172.22.34.208`
     - Port: `22`
     - Connection type: SSH
   - **Connection → SSH → X11**:
     - ✅ Check **"Enable X11 forwarding"**
     - X display location: `localhost:0`
   - *(Optional)* **Connection → SSH**:
     - ✅ Check **"Enable compression"** (reduces lag slightly)
3. *(Optional)* Go back to **Session**, type a name under "Saved Sessions", and click **Save** so you don't have to reconfigure each time.
4. Click **Open** to connect.
5. When prompted with a security alert about the host key, click **Accept**.
6. Log in with your username and password.

---

## Step 3: Configure the Remote Server (One-Time Setup)

Ensure the SSH server has X11 forwarding enabled.

1. Check the current configuration:

   ```bash
   grep -i x11 /etc/ssh/sshd_config
   ```

2. You should see:

   ```
   X11Forwarding yes
   ```

   If it says `no` or is commented out, edit the file:

   ```bash
   sudo nano /etc/ssh/sshd_config
   ```

   Ensure these lines are present and uncommented:

   ```
   X11Forwarding yes
   X11DisplayOffset 10
   X11UseLocalhost yes
   ```

   Then restart the SSH daemon:

   ```bash
   sudo systemctl restart sshd
   ```

3. Make sure `xauth` is installed:

   ```bash
   which xauth
   ```

   If not found:

   ```bash
   sudo apt install xauth   # Debian/Ubuntu
   sudo yum install xauth   # RHEL/CentOS
   ```

---

## Step 4: Verify X11 Forwarding

After connecting via PuTTY, run:

```bash
echo $DISPLAY
```

You should see something like `localhost:10.0`. If it is empty, X11 forwarding is not working — revisit the steps above.

Then test with a simple GUI app:

```bash
xclock
```

A clock window should appear on your Windows desktop.

---

## Step 5: Run Pygame

Once `xclock` works, you can run pygame scripts:

```bash
SDL_VIDEODRIVER=x11 uv run scripts/play_level.py --run-dir <your-run-dir>
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Can't open display:` / `$DISPLAY` is empty | Make sure you're using PuTTY with X11 forwarding enabled, not Windows OpenSSH |
| `Can't open display: localhost:10.0` | Make sure VcXsrv is running in your system tray |
| `xclock` works but pygame is very laggy | X11 forwarding is not designed for real-time graphics — use VNC or NoMachine instead (see below) |
| Connection refused | Check that the server's SSH config has `X11Forwarding yes` and that `xauth` is installed |
| Windows Firewall blocking | Allow VcXsrv through Windows Firewall (Windows Security → Firewall → Allow an app) |

---

## Alternative: VNC (For Better Performance)

X11 forwarding sends individual drawing commands over the network, which is very slow for real-time applications like pygame. VNC sends compressed screen images instead and is much faster.

**Server setup:**

```bash
sudo apt install tigervnc-standalone-server
vncserver :1 -geometry 1920x1080
```

**Windows client:**

1. Install [TigerVNC Viewer](https://tigervnc.org/) or [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/).
2. Connect to `172.22.34.208:5901`.

---

## Alternative: NoMachine (Fastest)

[NoMachine](https://www.nomachine.com/) uses a highly optimized protocol and provides near-native performance, even for graphical applications.

1. Install NoMachine on both the remote server and your Windows machine.
2. Connect using the NoMachine client.
