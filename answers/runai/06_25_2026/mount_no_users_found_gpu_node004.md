# Why NFS mount fails: `no users found` on `smoke-pope`

**Date:** 2026-06-25

## What the error means

```
CreateContainerError: mount callback failed ... no users found
```

This happens **before your command runs**. RunAI/kubelet is trying to mount:

```
/volume1/airl-datalake @ gpustorage-1 → /home/datalake in the pod
```

The mount plugin must map **your RunAI/Kubernetes user identity** to a POSIX user the NFS server accepts. **"no users found"** = that lookup failed on the **GPU node**, not in your repo or on the Synology SFTP side.

**Not caused by:** tarball, `vti_repo.tar.gz`, COCO layout, micromamba, or bash command.

---

## Strongest clue: node-specific failure

| Job | Node | NFS mount |
|-----|------|-----------|
| `probe-mm` | **gpu-node005** | Succeeded |
| `setup-vlm4-0-1` | **gpu-node005** | Succeeded |
| `probe-nfs` | **gpu-node004** | Failed (`no users found`) |
| `smoke-pope` | **gpu-node004** | Failed (`no users found`) |

Same flags every time:

```
--nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite
--gpu-devices-request 1 --node-pools h100-pool
```

**Conclusion:** Likely **gpu-node004** has a broken or misconfigured NFS/LDAP user-mapping mount agent. **gpu-node005** worked for you on 2026-06-18.

NFS **data** on gpustorage is fine (setup completed; WinSCP works). The failure is **in-cluster mount on a specific node**.

---

## Other causes (ruled out for your case)

| Cause | Evidence |
|-------|----------|
| 0 GPUs / wrong pool | You request 1 GPU + `h100-pool`; GPU allocated |
| Wrong NFS path | Same path as successful `setup-vlm4` |
| Corrupt NFS content | `setup-vlm4` wrote env + COCO + weights |
| Your script | Container never started |

---

## Why jobs keep hitting `gpu-node004`

Scheduler assigns pods to any free GPU in `h100-pool`. If `gpu-node004` is free but its mount agent is broken, you get repeated mount failures. Job may sit **Initializing**, retry the same node, and **waste GPU quota**.

---

## What to do

1. **Delete stuck `smoke-pope`** to free quota.
2. **Resubmit with a new job name** — may land on `gpu-node005` if free.
3. **Escalate to Chun-Nam / cluster admin** with:
   - Error: `mount callback failed: no users found`
   - Node: `gpu-node004`
   - Contrast: `probe-mm` / `setup-vlm4` succeeded on `gpu-node005`
   - Ask: drain or fix NFS mount on `gpu-node004`
4. **Do not re-run setup** — NFS bench is already built.

### Message template for admin

> Training jobs with NFS mount `path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake` fail on **gpu-node004** with `CreateContainerError: mount callback failed: no users found`. Same submit on **gpu-node005** succeeded (`setup-vlm4`, `probe-mm`). Please check RunAI NFS user mapping / mount agent on gpu-node004.

---

## If resubmit still fails

- Check which node: `runai training describe <job> -p nlm-mh` → Pods → Node
- If always `gpu-node004` → admin fix required
- If `gpu-node005` but another error → different issue (paste logs)
