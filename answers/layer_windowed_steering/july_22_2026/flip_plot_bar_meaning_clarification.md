# What flip-plot bars actually mean (leading-clause panes)

**Date:** 2026-07-22

## Short answers

**Yes** — on a leading-clause pane, a colored bar’s height is the number of items that flipped from:

- baseline = **leading-clause prompt, no steering**
- to steered = **same leading-clause prompt, with the intervention**

in the plot’s directed sense (yes→no or no→yes).

**Grey hatched** on those panes is a *different* contrast:

- baseline = **neutral prompt, no steering**
- to = **leading-clause prompt, no steering**

same directed sense.

So grey = prompt effect alone; colored = steering effect on top of the leading-clause prompt. Same y-axis units (item counts), **not** the same experimental contrast — comparing bar heights as “same kind of effect” is not valid.

## Extra subtlety: different item pools

Steering flips only count items whose **leading-clause baseline** was still `flip_from`. Items the leading clause already flipped away from `flip_from` cannot appear in the colored bars for that direction. Example: AMBER relation gold=yes, leading toward no — grey ≈ 11 yes→no from neutral→assertive; colored yes→no flips are only among the remaining items that still said yes under the leading-clause baseline.
EOF
