# A5 nonempty-only parser failure rates

Of the 300 sampled responses:
- **38 empty** (12.7%) — Qwen2.5 OOM artifacts; excluded here
- **262 nonempty**

**Parser fail rate on nonempty:** **11/262 = 4.20%**

By cell class (nonempty only):

| cell_class | n | fail | fail_rate |
|---|---:|---:|---:|
| additive_layer | 50 | 3 | 6.0% |
| additive_mlp | 52 | 2 | 3.85% |
| rotation_mlp | 160 | 6 | 3.75% |

Overall nonempty rate is under the ~5% gate; `additive_layer` alone is slightly over (6%). Failures are long-form Qwen answers without a clear leading yes/no. Written into `artifact_checks/a5_parser_audit.json`.
