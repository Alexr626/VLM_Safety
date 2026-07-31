# Checking H100 GPU specs on RunAI (nlm-mh)

**Date:** 2026-06-26

## Probe result (`probe-gpu` on `h100-pool`, `llm_image14:0.1`)

```
GPU 0: NVIDIA H100 80GB HBM3
memory.total: 81559 MiB (~79.6 GiB usable)
driver_version: 595.71.05
compute_cap: 9.0 (Hopper)
pci.bus_id: 00000000:E4:00.0
Node: gpu-node004
```

**Interpretation:** **H100 80GB HBM3** (Hopper). One full GPU per job with `--gpu-devices-request 1`.

### PCIe vs SXM (generation / bandwidth)

`nvidia-smi` product string is often **identical** for PCIe and SXM H100 80GB HBM3. Bandwidth differs:

| Variant | Typical HBM bandwidth | How to tell |
|---------|----------------------|-------------|
| **H100 SXM5** 80GB | ~3.35 TB/s | DGX/HGX tray; no PCIe bus id like a slot card |
| **H100 PCIe** 80GB | ~2.0 TB/s | Standard `pci.bus_id` (e.g. `00000000:E4:00.0`) |

Your probe shows a **PCIe-style bus ID** → likely **H100 PCIe 80GB**, not SXM5 — but confirm with admin if job placement matters.

For **7B VLMs** (LLaVA, Qwen2.5-VL): **80 GB is plenty** on one GPU; you are not memory-bandwidth-limited at POPE/CHAIR batch sizes — throughput is usually **generation latency**, not fitting the model.

---

## Commands you can run

### 1. One-shot probe job (best — runs on cluster hardware)

```bash
runai training submit probe-gpu -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap,pci.bus_id,driver_version --format=csv; nvidia-smi -q | head -80'

runai training logs probe-gpu -p nlm-mh
```

### 2. From inside any running training pod

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,compute_cap,pci.bus_id --format=csv
nvidia-smi -q -d MEMORY,CLOCK,PERFORMANCE
```

### 3. Via micromamba env on NFS (PyTorch view)

Add to a job command after NFS mount:

```bash
BASE=/home/datalake/romanus
"$BASE/bin/micromamba" run -p "$BASE/envs/vlm_hal" python -c "
import torch
p = torch.cuda.get_device_properties(0)
print('name:', p.name)
print('total_memory_GiB:', p.total_memory / 2**30)
print('sm_count:', p.multi_processor_count)
print('capability:', p.major, p.minor)
"
```

PyTorch reports **memory** and **SM count**, not HBM bandwidth.

### 4. CLI cluster inventory (usually admin-only)

```bash
runai node list        # insufficient permissions for researcher role
runai nodepool list    # same
```

Use probe jobs or ask Chun-Nam for node pool specs.

---

## Job setup implications (your stack)

| Resource | Your cluster (probed) | Implication |
|----------|----------------------|-------------|
| VRAM | ~81 GB | 7B VLM + activations: **single GPU, no offload** |
| Quota | 2 GPUs / project | One model job at a time unless quota free |
| Pool | `h100-pool` | Always `--gpu-devices-request 1 --node-pools h100-pool` |
| In pod | `CUDA_VISIBLE_DEVICES=0` | RunAI exposes allocated GPU as device 0 |

You do **not** need multi-GPU or CPU offload for 7B eval. Optimize **job count / hyperparameter sweeps** across sequential jobs, not sharding one 7B model.

Larger models (30B+, 70B) need a different plan (not your current smoke path).

---

## See which node a job landed on

```bash
runai training describe <job> -p nlm-mh
# Pods table → Node column (e.g. gpu-node004)
```

Specs are usually **per pool** (all `h100-pool` GPUs same SKU); verify with two probes if unsure.
