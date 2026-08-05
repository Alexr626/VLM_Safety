# Where finished RunAI job logs live on NFS

Date: 2026-07-31

Pod path: `/home/datalake/romanus/logs/runai/`
WinSCP / SFTP path: `/airl-datalake/romanus/logs/runai/`

Latest sync (this morning) wrote:
`sync-steering-llava_20260731T140021Z.log`
and ended with `VERIFY_SYNC_OK` / `SYNC_OK`.

Also always available from lambdab2 without WinSCP:
`runai training logs <job-name> -p nlm-mh`
