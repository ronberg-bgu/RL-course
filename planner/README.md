# PDDL Planner 

We use the state-of-the-art **Unified-Planning** Python library as the backend for our logical planning module. This allows you to generate logical plans for the box-pushing grid levels natively via Python without struggling with external solvers or legacy C++ binaries.

## 📦 Installation

To use the planner, you need to install the Unified-Planning library and a Python-based classical solver (`pyperplan`) via PIP:

```bash
pip install unified-planning pyperplan
```

## 🚀 Usage

Whenever you initialize a level, the environment comes equipped with a `generate_pddl_for_env` script that translates the specific grid locations, agents, and goal requirements into precise PDDL logic representation.

It outputs two files into the `pddl/` folder:
1. `domain.pddl` (The physics rules of the game)
2. `problem.pddl` (The specific objects and level layout)

### Quick Start Example

You can run the classical solver right from this folder in your terminal!

```bash
python3 pddl_solver.py ../pddl/domain.pddl ../pddl/problem.pddl
```

Or you can use the wrapper programmatically inside your Python code:

```python
from environment.multi_agent_env import MultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl

# 1. Initialize custom grid
ascii_map = [
    "WWWWW",
    "W A W",
    "W B W",
    "W G W",
    "WWWWW"
]

env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
env.reset()

# 2. Extract PDDL representation
domain_file, problem_file = generate_pddl_for_env(env, pddl_folder="../pddl")

# 3. Solve it natively using Unified-Planning!
plan = solve_pddl(domain_file, problem_file)

if plan:
    print("Found plan!")
    for action in plan.actions:
        print(action)
```

## 💡 Customizing Solvers
The `pddl_solver.py` script defaults to using the `pyperplan` engine (`OneshotPlanner(name="pyperplan")`) because it is written entirely in Python. If your research or exercises demand cutting-edge speed for large 50x50 grids, you can easily install `fast-downward` into your Python environment and replace `"pyperplan"` with `"fast-downward"`. Unified-Planning handles the translation flawlessly.
