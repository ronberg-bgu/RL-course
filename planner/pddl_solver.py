import os
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

def solve_pddl(domain_path, problem_path):
    """
    Parses PDDL files using Unified-Planning and solves them with pyperplan or another installed backend.
    """
    reader = PDDLReader()
    problem = reader.parse_problem(domain_path, problem_path)
    
    # Use Fast-Downward engine which is much more scalable for grid logic
    with OneshotPlanner(name="fast-downward") as planner:
        result = planner.solve(problem)
        if result.status in [
            up.engines.results.PlanGenerationResultStatus.SOLVED_SATISFICING,
            up.engines.results.PlanGenerationResultStatus.SOLVED_OPTIMALLY
        ]:
            print(f"Plan found with {len(result.plan.actions)} steps!")
            for action in result.plan.actions:
                print(action)
            return result.plan
        else:
            print("No plan found.")
            return None

if __name__ == "__main__":
    import sys
    # Provide domain and problem via CLI
    if len(sys.argv) == 3:
        solve_pddl(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python pddl_solver.py <domain.pddl> <problem.pddl>")
