# S2 dirs failure: `GLIBCXX_3.4.29` / PIL `libLerc`

## What failed
Not direction extract logic. After train2014 + LLaVA weights were fine, importing the model stack died:

```
from PIL import Image
→ ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
  (required by …/envs/vlm_hal/…/libLerc.so.4)
```

Pillow’s `libLerc` (in the NFS micromamba env) needs a **newer** `libstdc++` than the one on the RunAI image (`llm_image14`). The loader used the **host** `/lib/x86_64-linux-gnu/libstdc++.so.6`, which is too old. Crash loop = same import error every retry.

## What already worked
- NFS space (~9.6T free)
- `demos_v2.jsonl` present
- COCO train2014 (82783 images)
- LLaVA HF snapshot cached

## Fix (prefer conda’s libstdc++)
In RunAI helpers, before `micromamba run` / python:

```bash
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Optionally verify in a probe:
```bash
"$MM" run -p "$ENV_PREFIX" python -c "from PIL import Image; print('PIL_OK', Image.__version__)"
```

Alternative: rebuild/pin Pillow in `envs/vlm_hal` so it doesn’t pull a too-new `libLerc`, or install `libstdcxx-ng` into that env (then still set `LD_LIBRARY_PATH`).

Fast path for S3 only: WinSCP the **already-extracted** LLaVA nd200 dir from lambdab2 onto NFS (tiny) and skip on-cluster extract until the env lib path is fixed.
