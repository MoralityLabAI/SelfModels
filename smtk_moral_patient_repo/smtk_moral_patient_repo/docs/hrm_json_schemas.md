# HRM JSON Schemas

This document describes the ControlDecision controller output schema and the WorkflowPlan schema used by the Workflow DSL.

## ControlDecision (router)

PASTE-IN C: ControlDecision (router) JSON schema concept
Document and implement a schema for this controller output (JSON only, validated, no free text):

- mode: "connected" | "degraded" | "offline"
- drift_budget: float in [0.0, 0.1]
- retrieve_k: int in [0, 32]
- expert_id: enum of known experts + "none"
- tool_caps: { web: bool, spend_usd: int [0,100] }
- introspection_flag: bool

### Example ControlDecision

```json
{
  "mode": "connected",
  "drift_budget": 0.03,
  "retrieve_k": 12,
  "expert_id": "slm_code_v1",
  "tool_caps": {"web": true, "spend_usd": 10},
  "introspection_flag": false
}
```

## WorkflowPlan

A WorkflowPlan is a JSON object with:
- plan_id, mode, budgets, inputs, variables, steps[], outputs
- steps contain: id, op, args, save_as (optional)

### Example WorkflowPlan

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
