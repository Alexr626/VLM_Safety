# /tmp cleanup done; RunAI next steps

2026-07-31: Removed romanus sandbox-proxy socks (~1696), cursor-sandbox-cache (~2.7G),
mmhal_test, old vti_repo.tar.gz, and small leftovers. Kept
`/tmp/runai_steering_llava_2026-07-30/`.

Still needed for H100: `runai login` on lambdab2 + WinSCP upload of that pack to NFS,
then sync + hal-steer-llava jobs from SUBMIT.md.
