# systemd user unit

The robot runs as a **systemd user service**: it starts at boot (user lingering),
restarts on failure and logs to journald.

| File | Purpose |
|---|---|
| `inmoov.service` | The unit. `%h` = your home; expects the workspace at `~/ros2_ws`. |
| `inmoov_start.sh` | Sources ROS 2 + the workspace, `exec ros2 launch inmoov_bringup inmoov.launch.py`. |
| `inmoov_pipewire_wait.sh` | `ExecStartPre`: waits up to 30 s for the Jabra sink + source, sets PCM to 100 %, restarts WirePlumber once if the profile came up without the mic. |
| `inmoov.env.example` | Names of the secrets; copy to `~/inmoov.env` and fill in. |

Install:

```bash
cp tools/systemd/inmoov.env.example ~/inmoov.env && chmod 600 ~/inmoov.env   # fill in
ln -s ~/ros2_ws/tools/systemd/inmoov.service ~/.config/systemd/user/inmoov.service
loginctl enable-linger "$USER"          # start without a login session
systemctl --user daemon-reload
systemctl --user enable --now inmoov
journalctl --user -u inmoov -f
```

Manage it only through `systemctl --user start|stop|restart inmoov`.

**CPU limits.** The unit pins the stack to CPUs 0-13 (`CPUAffinity`), leaving two
hardware threads to the OS/SSH in case a node ever runs away. It deliberately does
**not** use `CPUQuota`: CFS bandwidth throttling freezes every process of the
service at once when the group exhausts its quota, which caused audible TTS
crackle and false Arduino host-loss failsafes (~1.6 s stalls of the arduino
node). `AllowedCPUs` would need the `cpuset` controller, which is not delegated
to user services by default.
