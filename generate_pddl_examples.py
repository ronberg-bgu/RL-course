import os
from environment.multi_agent_env import MultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env

def generate_examples():
    """
    Generates two explicit PDDL examples (Domain & Problem) for students to study.
    Outputs to the /pddl/ folder.
    """
    os.makedirs("pddl", exist_ok=True)
    
    print("Generating Example 1 (Small Box Map)...")
    map1 = [
        "WWWWW",
        "W A W",
        "W B W",
        "W G W",
        "WWWWW"
    ]
    env1 = MultiAgentBoxPushEnv(ascii_map=map1)
    env1.reset()
    # rename files so they don't overwrite each other
    from environment.pddl_extractor import generate_domain, generate_problem
    generate_domain("pddl/ex1_domain.pddl")
    generate_problem(env1, "pddl/ex1_problem.pddl")

    print("Generating Example 2 (Big Box Co-op Map)...")
    map2 = [
        "WWWWWW",
        "W AA W",
        "W CC W",
        "W G  W",
        "WWWWWW"
    ]
    env2 = MultiAgentBoxPushEnv(ascii_map=map2)
    env2.reset()
    generate_domain("pddl/ex2_domain.pddl")
    generate_problem(env2, "pddl/ex2_problem.pddl")
    
    print("Done! Check the `pddl/` folder for ex1 and ex2 PDDL files.")

if __name__ == "__main__":
    generate_examples()
