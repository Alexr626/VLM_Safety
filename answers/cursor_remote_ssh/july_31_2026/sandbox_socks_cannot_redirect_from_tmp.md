# Can Cursor sandbox socks be forced under $HOME instead of /tmp?

Date: 2026-07-31

**Short answer:** No supported setting does that today.

- `sandbox-proxy-http-*.sock` / `sandbox-proxy-socks-*.sock` are created under
  `/tmp` by Cursor’s agent sandbox proxy.
- Official forum note (Cursor staff): sandbox temp dirs are ephemeral under
  `/tmp`; the sandbox does not set `$TMPDIR` and cleanup targets fixed `/tmp`
  paths.
- `sandbox.json` can set `disableTmpWrite`, `enableSharedBuildCache`, extra
  paths, network policy — **not** a custom directory for proxy sockets.
- Setting `TMPDIR=~/tmp` in `.bashrc` / remote env may redirect *your* shell
  temp files, but does not relocate Cursor’s sandbox-proxy sockets.

Workarounds: periodic `rm` of stale `sandbox-proxy-*.sock` owned by you;
cron under home; report as multi-user `/tmp` clutter to Cursor.
