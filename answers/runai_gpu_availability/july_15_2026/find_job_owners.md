# How to find who owns running RunAI jobs

**Date:** 2026-07-15  
**Context:** `nlm-mh` quota full — running jobs `hal44` (training, 2 GPU) and `llama3-70b8` (inference, 2 GPU).

## Fastest: describe each running job

Owner / submitter shows up in the **General** section (typically **Created by** / user email).

```bash
# Training job (hal44)
runai training describe hal44 -p nlm-mh

# Inference job (llama3-70b8)
runai inference describe llama3-70b8 -p nlm-mh
```

Unified form:

```bash
runai workload describe hal44 -p nlm-mh --type training
runai workload describe llama3-70b8 -p nlm-mh --type inference
```

JSON (easier to grep for email / user fields):

```bash
runai training describe hal44 -p nlm-mh -o json | less
runai inference describe llama3-70b8 -p nlm-mh -o json | less

# if jq is available, try common keys:
runai training describe hal44 -p nlm-mh -o json | jq '.. | objects | with_entries(select(.key|test("user|created|submit|owner";"i"))) | select(length>0)'
```

## List view with a user column

Try adding a user column (exact column name varies by CLI version):

```bash
runai workload list -p nlm-mh --status Running --columns name,type,status,user,project
# or:
runai workload list -p nlm-mh --status Running --json | less
```

## Optional: kubectl (after `runai login` / valid kubeconfig)

Namespaces are often `runai-<project>`:

```bash
kubectl get pods -n runai-nlm-mh -o wide
kubectl describe pod <pod-name> -n runai-nlm-mh | rg -i 'user|created|run\.ai|annotation'
```

## What to ask them

Once you have the submitter email from **Created by**:

- How long they expect `hal44` / `llama3-70b8` to keep running
- Whether either job can be stopped or downsized (each holds **2 GPUs**; a 1-GPU H100 training job needs only 1 free quota GPU)
- Preferred handoff time so you can submit into `h100-pool`
