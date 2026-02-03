import json
from pathlib import Path

from smtk.workflow_dsl import WorkflowExecutor, WorkflowHooks, WorkflowPlan


class StubHooks(WorkflowHooks):
    def route_expert(self, args, state):
        prompt_ref = args.get("prompt_ref", "")
        return f"// patch based on {prompt_ref}"

    def tool_call(self, args, state):
        return {"tool": args.get("tool"), "ok": True}

    def verify(self, args, state):
        return {"ok": True, "checker_id": args.get("checker_id")}


def assemble_prompt(args, state):
    refs = args.get("refs", [])
    return "\n".join(str(ref) for ref in refs)


def main() -> None:
    plan_path = Path("examples/plans/outage_patch_plan.json")
    plan = WorkflowPlan.parse_file(plan_path)

    executor = WorkflowExecutor(hooks=StubHooks(), transforms={"assemble_prompt": assemble_prompt}, seed=0)
    result = executor.run(plan)

    print(json.dumps(result["receipts"], indent=2))


if __name__ == "__main__":
    main()
