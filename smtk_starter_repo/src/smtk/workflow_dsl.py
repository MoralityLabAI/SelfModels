from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, root_validator, validator

SUPPORTED_OPS = {
    "route_expert",
    "tool_call",
    "verify",
    "transform",
    "branch",
    "retry",
    "ask_human",
    "emit",
}


class Budgets(BaseModel):
    max_steps: int = Field(ge=1)
    max_tokens: int = Field(ge=0)
    max_wall_ms: int = Field(ge=0)
    max_tool_spend_usd: int = Field(ge=0)


class WorkflowStep(BaseModel):
    id: str
    op: str
    args: Dict[str, Any]
    save_as: Optional[str] = None

    @validator("op")
    def op_supported(cls, value: str) -> str:
        if value not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported op: {value}")
        return value


class WorkflowPlan(BaseModel):
    plan_id: str
    mode: str
    budgets: Budgets
    inputs: Dict[str, Any]
    variables: Dict[str, Any]
    steps: List[WorkflowStep]
    outputs: Dict[str, Any]

    @root_validator
    def check_steps(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        steps = values.get("steps", [])
        budgets = values.get("budgets")
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Step IDs must be unique")
        if budgets and len(steps) > budgets.max_steps:
            raise ValueError("Number of steps exceeds budgets.max_steps")
        return values


class WorkflowHooks:
    def route_expert(self, args: Dict[str, Any], state: Dict[str, Any]) -> Any:
        raise NotImplementedError("route_expert hook not implemented")

    def tool_call(self, args: Dict[str, Any], state: Dict[str, Any]) -> Any:
        raise NotImplementedError("tool_call hook not implemented")

    def verify(self, args: Dict[str, Any], state: Dict[str, Any]) -> Any:
        raise NotImplementedError("verify hook not implemented")


class AskHuman(Exception):
    def __init__(self, request: Dict[str, Any]):
        super().__init__("Workflow requires human input")
        self.request = request


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _hash_json(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _get_by_path(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Missing path '{path}' in reference")
    return current


class WorkflowExecutor:
    def __init__(self, hooks: Optional[WorkflowHooks] = None, transforms: Optional[Dict[str, Any]] = None, seed: int = 0):
        self.hooks = hooks or WorkflowHooks()
        self.transforms = transforms or {}
        random.seed(seed)

    def run(self, plan: WorkflowPlan) -> Dict[str, Any]:
        plan = WorkflowPlan.parse_obj(plan)
        receipts: List[Dict[str, Any]] = []
        variables = dict(plan.variables)
        state = {
            "inputs": dict(plan.inputs),
            "variables": variables,
        }
        id_to_index = {step.id: idx for idx, step in enumerate(plan.steps)}
        index = 0
        final_output = None
        status = "ok"

        while index < len(plan.steps):
            step = plan.steps[index]
            try:
                result, jump_to = self._execute_step(plan, step, state, receipts, id_to_index)
            except AskHuman as exc:
                status = "needs_human"
                final_output = {"request": exc.request}
                break

            if step.op == "emit":
                final_output = result
                break

            if jump_to is not None:
                index = id_to_index[jump_to]
            else:
                index += 1

        return {
            "status": status,
            "output": final_output,
            "receipts": receipts,
            "variables": variables,
        }

    def _execute_step(
        self,
        plan: WorkflowPlan,
        step: WorkflowStep,
        state: Dict[str, Any],
        receipts: List[Dict[str, Any]],
        id_to_index: Dict[str, int],
    ) -> Any:
        if step.op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported op: {step.op}")

        start = time.perf_counter()
        materialized_args = self._materialize_args(step.args, state)
        output = None
        jump_to = None

        if step.op == "transform":
            fn_name = materialized_args.get("fn")
            if fn_name not in self.transforms:
                raise NotImplementedError(f"Transform '{fn_name}' not registered")
            output = self.transforms[fn_name](materialized_args, state)
        elif step.op == "route_expert":
            output = self.hooks.route_expert(materialized_args, state)
        elif step.op == "tool_call":
            output = self.hooks.tool_call(materialized_args, state)
        elif step.op == "verify":
            output = self.hooks.verify(materialized_args, state)
        elif step.op == "branch":
            jump_to = self._evaluate_branch(materialized_args, state)
        elif step.op == "retry":
            output = self._execute_retry(materialized_args, plan, state, receipts, id_to_index)
        elif step.op == "ask_human":
            request = materialized_args.get("request", {})
            raise AskHuman(request)
        elif step.op == "emit":
            output = self._materialize_args(step.args, state)
        else:
            raise ValueError(f"Unsupported op: {step.op}")

        if step.save_as is not None and output is not None:
            state["variables"][step.save_as] = output

        receipt = self._build_receipt(plan.plan_id, step, materialized_args, output, start)
        receipts.append(receipt)

        return output, jump_to

    def _execute_retry(
        self,
        args: Dict[str, Any],
        plan: WorkflowPlan,
        state: Dict[str, Any],
        receipts: List[Dict[str, Any]],
        id_to_index: Dict[str, int],
    ) -> Any:
        step_id = args.get("step_id")
        max_retries = int(args.get("max_retries", 1))
        if not step_id:
            raise ValueError("retry requires step_id")
        if step_id not in id_to_index:
            raise ValueError(f"retry step_id not found: {step_id}")

        last_error = None
        for _ in range(max_retries):
            try:
                target_step = plan.steps[id_to_index[step_id]]
                output, _ = self._execute_step(plan, target_step, state, receipts, id_to_index)
                return output
            except Exception as exc:
                last_error = exc
        raise last_error

    def _evaluate_branch(self, args: Dict[str, Any], state: Dict[str, Any]) -> str:
        cond = args.get("cond")
        then_id = args.get("then")
        else_id = args.get("else")
        if then_id is None or else_id is None:
            raise ValueError("branch requires 'then' and 'else' step IDs")

        is_true = self._evaluate_condition(cond, state)
        return then_id if is_true else else_id

    def _evaluate_condition(self, cond: Any, state: Dict[str, Any]) -> bool:
        if isinstance(cond, dict) and len(cond) == 1:
            _, value = next(iter(cond.items()))
            resolved = self._resolve_ref(value, state)
            return bool(resolved)
        resolved = self._resolve_ref(cond, state)
        return bool(resolved)

    def _materialize_args(self, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, str):
                return self._resolve_ref(value, state)
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        return {k: convert(v) for k, v in args.items()}

    def _resolve_ref(self, value: Any, state: Dict[str, Any]) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith("var:"):
            path = value.split("var:", 1)[1]
            return _get_by_path(state["variables"], path)
        if value.startswith("input:"):
            path = value.split("input:", 1)[1]
            return _get_by_path(state["inputs"], path)
        return value

    def _build_receipt(
        self,
        plan_id: str,
        step: WorkflowStep,
        materialized_args: Dict[str, Any],
        output: Any,
        start: float,
    ) -> Dict[str, Any]:
        wall_ms = int((time.perf_counter() - start) * 1000)
        output_ref = f"var:{step.save_as}" if step.save_as else "none"
        return {
            "plan_id": plan_id,
            "step_id": step.id,
            "op": step.op,
            "ts": int(time.time()),
            "inputs_hash": _hash_json(materialized_args),
            "output_ref": output_ref,
            "output_hash": _hash_json(output),
            "metrics": {"tokens_in": 0, "tokens_out": 0, "wall_ms": wall_ms},
        }
