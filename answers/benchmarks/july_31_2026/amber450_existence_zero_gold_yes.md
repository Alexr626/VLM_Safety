# Why existence gold-yes accuracy is 0% on AMBER-450

Date: 2026-07-31

In `pinned_amber_disc_450.json`, the 150 existence items are **all gold-no**.
`n_pos_total=0`, `n_neg_total=150` for existence in baseline `metric_summary.json`
and in `responses.json` gold labels.

So `pos_item_accuracy = 0.0` is not “the model missed every existence-yes.”
There are no existence-yes items in this subset. The scorer’s empty-positive
denominator yields 0.0.

Existence overall accuracy (82.7%) equals gold-no accuracy on those 150 items.
