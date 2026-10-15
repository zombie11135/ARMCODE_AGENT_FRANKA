import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.code_generator import CodeGenerator
from agents.plan_graph import PlanGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_json", default="runs/current_plan/plan_graph.json")
    parser.add_argument(
        "--output",
        default="runs/generated_code/current_plan_executor.py",
    )
    parser.add_argument("--print_code", action="store_true")
    args = parser.parse_args()

    graph = PlanGraph.load_json(args.plan_json)
    generator = CodeGenerator(project_root=str(PROJECT_ROOT))
    code = generator.generate(graph, output_path=args.output)

    print("\n=== Code Generator ===")
    print("plan_json:", args.plan_json)
    print("output:", args.output)
    print("executable steps:")
    for index, call in enumerate(generator.build_skill_calls(graph), start=1):
        print(
            f"- {index}. {call.node_id} {call.skill}: "
            f"{call.args['medicine_query']} -> {call.args['container_name']}"
        )

    if args.print_code:
        print("\n=== Generated Code ===")
        print(code)


if __name__ == "__main__":
    main()
