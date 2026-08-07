# Is VTI `pca.mean_` the same operation as 2026-07-30 raw mean-difference?

Date: 2026-08-06

**Short answer: yes.** In the 06-19 / VTI textual path, `pca.mean_` is the mean over demos of the flattened `(value − h_value)` last-token stacks. Reshaped to `(num_layers+1, hidden_dim)`, that is the same aggregation the 07-30 `raw_mean_difference` path computes. The 06-19 **steering vector** is that mean **plus** PC1, not the mean alone.

---

## 06-19 / legacy VTI (`obtain_textual_vti`)

Per demo: last-token stacks for `h_value` and `value`, then

```text
diff_i = flatten(value_i − h_value_i)   # length = (L+1) * H
```

Stack to `X` with shape `(n_demos, flat)`. PCA fit:

```33:34:evaluation/interventions/vti/pca.py
        self.register_buffer("mean_", X.mean(1, keepdim=True))
        Z = X - self.mean_
```

(`X` is unsqueezed to `(1, n, d)`; `mean(1)` averages over demos.)

Steering reconstruction:

```108:113:evaluation/interventions/vti/directions.py
    direction = (
        pca.components_.sum(dim=1, keepdim=True) + pca.mean_
    ).mean(0).view(
        hidden_states[-1][0].size(0),
        hidden_states[-1][0].size(1),
    )
```

With `rank=1`, `components_.sum(dim=1)` is just PC1. So:

```text
direction = reshape(PC1 + mean_of_diffs)
```

`svd_flip` changes the sign of PC1 only; it does not change `mean_`.

## 07-30 meandiff (`obtain_textual_meandiff_from_stacks`)

```107:110:evaluation/interventions/vti/directions_meandiff.py
        per_layer = v_act - h_act
        diffs.append(per_layer)
        ...
    full = np.mean(np.stack(diffs, axis=0), axis=0).astype(np.float32)
```

Same polarity (`value − h_value`), same average over demos, kept in `(L+1, H)` layout instead of flattening first. Because mean is elementwise, flatten→mean→reshape equals mean of the unflattened stacks.

## What is *not* the same

| Quantity | Relation |
|----------|----------|
| `pca.mean_` (reshaped) | = raw mean-difference **as an operation** |
| 06-19 steering direction | = that mean **+ PC1** |
| 07-30 steering direction | = that mean alone |
| Numerical mean vectors across the two dated runs | Differ whenever demos / captions / images / `n` differ — same formula, different inputs |

So: the “mean” in the 06-19 PC1+mean row is defined the same way as the 07-30 raw mean-difference; the 06-19 deployed vector is that mean with PC1 added on top.
