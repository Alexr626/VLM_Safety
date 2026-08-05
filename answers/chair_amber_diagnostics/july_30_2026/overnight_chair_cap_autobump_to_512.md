# Overnight CHAIR cap policy (Alex, 2026-07-30)

If the pre-grid CHAIR caption-length probe finds any caption hitting the 256-token
cap on either model, **do not halt and do not skip CHAIR**. Raise
`--chair_max_new_tokens` to **512** for the overnight grid (both models) and
continue AMBER → CHAIR → POPE as planned.

Rationale: overnight must not stop for a fix that is a one-line cap change.
