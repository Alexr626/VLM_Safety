# lambdab2 /tmp: sandbox-proxy socks + what to delete

Date: 2026-07-31

## The “SOC/sock” flood

~847 × `sandbox-proxy-http-*.sock` + ~847 × `sandbox-proxy-socks-*.sock`,
all owned by **romanus**, mostly dated from Cursor agent sandbox sessions
(from ~Jun 15 onward). Unix domain sockets left behind when sandbox proxy
processes exit. **0 bytes on disk**, but they clutter `/tmp` (and use inodes).

Safe for romanus to delete stale ones; live Cursor sessions may recreate a few.

## Real disk (romanus)

| Path | Size | Notes |
|---|---|---|
| `/tmp/cursor-sandbox-cache/` | ~2.7G | Cursor sandbox cache — largest romanus item |
| `/tmp/mmhal_test/` | ~166M | old MMHal test zip |
| `/tmp/runai_steering_llava_2026-07-30/` | ~136M | today’s RunAI pack (keep until uploaded) |
| `/tmp/vti_repo.tar.gz` | ~103M | older git archive for RunAI |

Do **not** delete other users’ `/tmp` (e.g. gerardb UUID sockets,
`torchinductor_gerardb`, root `CoreFxPipe_*`).
