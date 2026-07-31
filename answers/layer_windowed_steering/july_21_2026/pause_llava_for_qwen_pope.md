# Pause LLaVA AMBER for Qwen POPE (2026-07-21)

- LLaVA AMBER SIGTERM'd; GPU 0 left to Qwen POPE-30 only.
- Checkpoint: 14 complete cells + partial `additive_mlp_0.9_layers_0_9` at 184/200 (manifest==acts, no corrupt JSON).
- Watcher `resume_llava_amber_after_qwen_pope.sh` resumes LLaVA AMBER when Qwen reaches 54×60 POPE cells.
