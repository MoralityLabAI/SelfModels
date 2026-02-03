# Workflow DSL

This document defines the deterministic Workflow DSL and its receipt format.

## WorkflowPlan (JSON)

PASTE-IN A: Deterministic Workflow DSL (JSON) + Receipt Format
- DSL is a JSON “WorkflowPlan” with:
  - plan_id, mode, budgets, inputs, variables, steps[], outputs
  - steps contain: id, op, args, save_as (optional)
- Supported ops (minimal set):
  1) route_expert
  2) tool_call
  3) verify
  4) transform
  5) branch
  6) retry
  7) ask_human
  8) emit
- Receipt format (hashable):
  {
    "plan_id": "...",
    "step_id": "...",
    "op": "...",
    "ts": <int>,
    "inputs_hash": "sha256:...",
    "output_ref": "var:...",
    "output_hash": "sha256:...",
    "metrics": {"tokens_in": <int>, "tokens_out": <int>, "wall_ms": <int>}
  }

## Opcode semantics

- route_expert: invoke a registered expert by ID with a prompt or structured input; save the expert output.
- tool_call: invoke a tool under capability constraints; save the tool result.
- verify: run a checker over a prior output and return an object like {"ok": true/false, ...}.
- transform: deterministic local transform over inputs/refs; commonly used to assemble prompts.
- branch: conditional jump to another step ID based on a boolean or a referenced value.
- retry: re-run a step by ID up to a bounded retry count; abort on exhaustion.
- ask_human: emit a structured request and halt execution.
- emit: finalize the workflow with a result reference and audit references.

## Receipts

Each step emits a receipt for auditability. Receipts are hashable and include hashes of inputs and outputs. The executor should emit deterministic receipts and fail closed on invalid schemas.

## Example WorkflowPlan

PASTE-IN B: Example WorkflowPlan (outage-aware patch plan)

```json
{
  "plan_id": "fix_bug_v1",
  "mode": "degraded",
  "budgets": {"max_steps": 18, "max_tokens": 1600, "max_wall_ms": 45000, "max_tool_spend_usd": 0},
  "inputs": {"task": "Patch TypeError in futures component", "context_ref": "ctx:repo_diff", "snapshot_ref": "snap:t381"},
  "variables": {},
  "steps": [
    {"id":"s1","op":"transform","args":{"fn":"assemble_prompt","refs":["ctx:repo_diff","snap:t381"]},"save_as":"prompt"},
    {"id":"s2","op":"route_expert","args":{"expert_id":"slm_code_v1","prompt_ref":"var:prompt","max_new_tokens":220,"temperature":0.0},"save_as":"patch"},
    {"id":"s3","op":"verify","args":{"checker_id":"diff_applies_cleanly","input_ref":"var:patch"},"save_as":"v1"},
    {"id":"s4","op":"branch","args":{"cond":{"ok":"var:v1.ok"},"then":"s7","else":"s5"}},
    {"id":"s5","op":"route_expert","args":{"expert_id":"slm_code_v2","prompt_ref":"var:prompt","max_new_tokens":220,"temperature":0.0},"save_as":"patch2"},
    {"id":"s6","op":"verify","args":{"checker_id":"diff_applies_cleanly","input_ref":"var:patch2"},"save_as":"v2"},
    {"id":"s7","op":"branch","args":{"cond":{"ok":"var:v1.ok"},"then":"s9","else":"s8"}},
    {"id":"s8","op":"branch","args":{"cond":{"ok":"var:v2.ok"},"then":"s10","else":"s12"}},
    {"id":"s9","op":"emit","args":{"result_ref":"var:patch","status":"ok","audit_refs":["var:v1"]}},
    {"id":"s10","op":"emit","args":{"result_ref":"var:patch2","status":"ok","audit_refs":["var:v2"]}},
    {
      "id":"s12","op":"ask_human",
      "args":{"request":{"kind":"needs_context","message":"Both patch attempts failed to apply cleanly. Provide file paths or error output."}}
    }
  ],
  "outputs": {"result_ref":"var:patch", "audit_refs":["var:v1","var:v2"]}
}
```
