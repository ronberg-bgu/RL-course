import argparse
import subprocess
from pathlib import Path
from textwrap import dedent


PROMPT_TEXT = dedent("""
You are generating PDDL for a deterministic box-pushing world.

Requirements:
- Use domain name: box-push
- Use types: agent, location, box, heavybox
- Use predicates:
  - (agent-at ?a - agent ?loc - location)
  - (box-at ?b - box ?loc - location)
  - (heavybox-at ?h - heavybox ?loc - location)
  - (clear ?loc - location)
  - (adj ?l1 - location ?l2 - location)
  - (adj-left ?l1 - location ?l2 - location)
  - (adj-right ?l1 - location ?l2 - location)
  - (adj-up ?l1 - location ?l2 - location)
  - (adj-down ?l1 - location ?l2 - location)

Actions:
- move
- push-small
- push-heavy

Heavy box clarification:
- The heavy box is 1x1.
- It uses type heavybox.
- Its predicate is heavybox-at.
- Its object name is hbx_0.
- A heavy box push requires both agents to stand on the same location adjacent to the heavy box
  and push in the same direction simultaneously.

Map:
WWWWWWWW
W  AA  W
W   C  W
W W  B W
W W  B W
W WGG  W
W   G  W
WWWWWWWW

Legend:
W = wall
A = agent
B = small box
C = heavy box
G = goal
space = empty cell

Need:
1. A valid general domain.pddl
2. A valid problem.pddl derived from the map
3. The goal should refer directly to box / heavybox locations
4. Compatibility with the course simulator/repository conventions
""")


DOMAIN_PDDL = dedent("""
(define (domain box-push)
  (:requirements :strips :typing :equality :disjunctive-preconditions)

  (:types
    agent location box heavybox
  )

  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)

    (adj ?l1 - location ?l2 - location)
    (adj-left ?l1 - location ?l2 - location)
    (adj-right ?l1 - location ?l2 - location)
    (adj-up ?l1 - location ?l2 - location)
    (adj-down ?l1 - location ?l2 - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and
      (agent-at ?a ?from)
      (adj ?from ?to)
      (clear ?to)
    )
    :effect (and
      (not (agent-at ?a ?from))
      (agent-at ?a ?to)
    )
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (box-at ?b ?boxloc)
      (clear ?toloc)
      (or
        (and (adj-up ?from ?boxloc)    (adj-up ?boxloc ?toloc))
        (and (adj-down ?from ?boxloc)  (adj-down ?boxloc ?toloc))
        (and (adj-left ?from ?boxloc)  (adj-left ?boxloc ?toloc))
        (and (adj-right ?from ?boxloc) (adj-right ?boxloc ?toloc))
      )
    )
    :effect (and
      (not (agent-at ?a ?from))
      (agent-at ?a ?boxloc)

      (not (box-at ?b ?boxloc))
      (box-at ?b ?toloc)

      (clear ?boxloc)
      (not (clear ?toloc))
    )
  )

  (:action push-heavy
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (heavybox-at ?h ?boxloc)
      (clear ?toloc)
      (or
        (and (adj-up ?from ?boxloc)    (adj-up ?boxloc ?toloc))
        (and (adj-down ?from ?boxloc)  (adj-down ?boxloc ?toloc))
        (and (adj-left ?from ?boxloc)  (adj-left ?boxloc ?toloc))
        (and (adj-right ?from ?boxloc) (adj-right ?boxloc ?toloc))
      )
    )
    :effect (and
      (not (agent-at ?a1 ?from))
      (not (agent-at ?a2 ?from))
      (agent-at ?a1 ?boxloc)
      (agent-at ?a2 ?boxloc)

      (not (heavybox-at ?h ?boxloc))
      (heavybox-at ?h ?toloc)

      (clear ?boxloc)
      (not (clear ?toloc))
    )
  )
)
""").strip() + "\n"


PROBLEM_PDDL = dedent("""
(define (problem bp-map)
  (:domain box-push)

  (:objects
    loc_1_1 loc_2_1 loc_3_1 loc_4_1 loc_5_1 loc_6_1
    loc_1_2 loc_2_2 loc_3_2 loc_4_2 loc_5_2 loc_6_2
    loc_1_3 loc_3_3 loc_4_3 loc_5_3 loc_6_3
    loc_1_4 loc_3_4 loc_4_4 loc_5_4 loc_6_4
    loc_1_5 loc_3_5 loc_4_5 loc_5_5 loc_6_5
    loc_1_6 loc_2_6 loc_3_6 loc_4_6 loc_5_6 loc_6_6 - location

    agent_0 agent_1 - agent
    box_0 box_1 - box
    hbx_0 - heavybox
  )

  (:init
    (clear loc_1_1)
    (clear loc_2_1)
    (clear loc_3_1)
    (clear loc_4_1)
    (clear loc_5_1)
    (clear loc_6_1)

    (clear loc_1_2)
    (clear loc_2_2)
    (clear loc_3_2)
    (clear loc_5_2)
    (clear loc_6_2)

    (clear loc_1_3)
    (clear loc_3_3)
    (clear loc_4_3)
    (clear loc_6_3)

    (clear loc_1_4)
    (clear loc_3_4)
    (clear loc_4_4)
    (clear loc_6_4)

    (clear loc_1_5)
    (clear loc_3_5)
    (clear loc_4_5)
    (clear loc_5_5)
    (clear loc_6_5)

    (clear loc_1_6)
    (clear loc_2_6)
    (clear loc_3_6)
    (clear loc_4_6)
    (clear loc_5_6)
    (clear loc_6_6)

    (agent-at agent_0 loc_3_1)
    (agent-at agent_1 loc_4_1)

    (heavybox-at hbx_0 loc_4_2)
    (box-at box_0 loc_5_3)
    (box-at box_1 loc_5_4)

    (adj loc_1_1 loc_2_1) (adj loc_2_1 loc_1_1)
    (adj loc_2_1 loc_3_1) (adj loc_3_1 loc_2_1)
    (adj loc_3_1 loc_4_1) (adj loc_4_1 loc_3_1)
    (adj loc_4_1 loc_5_1) (adj loc_5_1 loc_4_1)
    (adj loc_5_1 loc_6_1) (adj loc_6_1 loc_5_1)

    (adj loc_1_2 loc_2_2) (adj loc_2_2 loc_1_2)
    (adj loc_2_2 loc_3_2) (adj loc_3_2 loc_2_2)
    (adj loc_3_2 loc_4_2) (adj loc_4_2 loc_3_2)
    (adj loc_4_2 loc_5_2) (adj loc_5_2 loc_4_2)
    (adj loc_5_2 loc_6_2) (adj loc_6_2 loc_5_2)

    (adj loc_1_3 loc_1_4) (adj loc_1_4 loc_1_3)
    (adj loc_3_3 loc_4_3) (adj loc_4_3 loc_3_3)
    (adj loc_4_3 loc_5_3) (adj loc_5_3 loc_4_3)
    (adj loc_5_3 loc_6_3) (adj loc_6_3 loc_5_3)

    (adj loc_1_4 loc_1_5) (adj loc_1_5 loc_1_4)
    (adj loc_3_4 loc_4_4) (adj loc_4_4 loc_3_4)
    (adj loc_4_4 loc_5_4) (adj loc_5_4 loc_4_4)
    (adj loc_5_4 loc_6_4) (adj loc_6_4 loc_5_4)

    (adj loc_1_5 loc_1_6) (adj loc_1_6 loc_1_5)
    (adj loc_3_5 loc_4_5) (adj loc_4_5 loc_3_5)
    (adj loc_4_5 loc_5_5) (adj loc_5_5 loc_4_5)
    (adj loc_5_5 loc_6_5) (adj loc_6_5 loc_5_5)

    (adj loc_1_6 loc_2_6) (adj loc_2_6 loc_1_6)
    (adj loc_2_6 loc_3_6) (adj loc_3_6 loc_2_6)
    (adj loc_3_6 loc_4_6) (adj loc_4_6 loc_3_6)
    (adj loc_4_6 loc_5_6) (adj loc_5_6 loc_4_6)
    (adj loc_5_6 loc_6_6) (adj loc_6_6 loc_5_6)

    (adj loc_1_1 loc_1_2) (adj loc_1_2 loc_1_1)
    (adj loc_2_1 loc_2_2) (adj loc_2_2 loc_2_1)
    (adj loc_3_1 loc_3_2) (adj loc_3_2 loc_3_1)
    (adj loc_4_1 loc_4_2) (adj loc_4_2 loc_4_1)
    (adj loc_5_1 loc_5_2) (adj loc_5_2 loc_5_1)
    (adj loc_6_1 loc_6_2) (adj loc_6_2 loc_6_1)

    (adj loc_1_2 loc_1_3) (adj loc_1_3 loc_1_2)
    (adj loc_3_2 loc_3_3) (adj loc_3_3 loc_3_2)
    (adj loc_4_2 loc_4_3) (adj loc_4_3 loc_4_2)
    (adj loc_5_2 loc_5_3) (adj loc_5_3 loc_5_2)
    (adj loc_6_2 loc_6_3) (adj loc_6_3 loc_6_2)

    (adj loc_3_3 loc_3_4) (adj loc_3_4 loc_3_3)
    (adj loc_4_3 loc_4_4) (adj loc_4_4 loc_4_3)
    (adj loc_5_3 loc_5_4) (adj loc_5_4 loc_5_3)
    (adj loc_6_3 loc_6_4) (adj loc_6_4 loc_6_3)

    (adj loc_3_4 loc_3_5) (adj loc_3_5 loc_3_4)
    (adj loc_4_4 loc_4_5) (adj loc_4_5 loc_4_4)
    (adj loc_5_4 loc_5_5) (adj loc_5_5 loc_5_4)
    (adj loc_6_4 loc_6_5) (adj loc_6_5 loc_6_4)

    (adj loc_3_5 loc_3_6) (adj loc_3_6 loc_3_5)
    (adj loc_4_5 loc_4_6) (adj loc_4_6 loc_4_5)
    (adj loc_5_5 loc_5_6) (adj loc_5_6 loc_5_5)
    (adj loc_6_5 loc_6_6) (adj loc_6_6 loc_6_5)

    (adj-right loc_1_1 loc_2_1) (adj-left loc_2_1 loc_1_1)
    (adj-right loc_2_1 loc_3_1) (adj-left loc_3_1 loc_2_1)
    (adj-right loc_3_1 loc_4_1) (adj-left loc_4_1 loc_3_1)
    (adj-right loc_4_1 loc_5_1) (adj-left loc_5_1 loc_4_1)
    (adj-right loc_5_1 loc_6_1) (adj-left loc_6_1 loc_5_1)

    (adj-right loc_1_2 loc_2_2) (adj-left loc_2_2 loc_1_2)
    (adj-right loc_2_2 loc_3_2) (adj-left loc_3_2 loc_2_2)
    (adj-right loc_3_2 loc_4_2) (adj-left loc_4_2 loc_3_2)
    (adj-right loc_4_2 loc_5_2) (adj-left loc_5_2 loc_4_2)
    (adj-right loc_5_2 loc_6_2) (adj-left loc_6_2 loc_5_2)

    (adj-right loc_3_3 loc_4_3) (adj-left loc_4_3 loc_3_3)
    (adj-right loc_4_3 loc_5_3) (adj-left loc_5_3 loc_4_3)
    (adj-right loc_5_3 loc_6_3) (adj-left loc_6_3 loc_5_3)

    (adj-right loc_3_4 loc_4_4) (adj-left loc_4_4 loc_3_4)
    (adj-right loc_4_4 loc_5_4) (adj-left loc_5_4 loc_4_4)
    (adj-right loc_5_4 loc_6_4) (adj-left loc_6_4 loc_5_4)

    (adj-right loc_3_5 loc_4_5) (adj-left loc_4_5 loc_3_5)
    (adj-right loc_4_5 loc_5_5) (adj-left loc_5_5 loc_4_5)
    (adj-right loc_5_5 loc_6_5) (adj-left loc_6_5 loc_5_5)

    (adj-right loc_1_6 loc_2_6) (adj-left loc_2_6 loc_1_6)
    (adj-right loc_2_6 loc_3_6) (adj-left loc_3_6 loc_2_6)
    (adj-right loc_3_6 loc_4_6) (adj-left loc_4_6 loc_3_6)
    (adj-right loc_4_6 loc_5_6) (adj-left loc_5_6 loc_4_6)
    (adj-right loc_5_6 loc_6_6) (adj-left loc_6_6 loc_5_6)

    (adj-down loc_1_1 loc_1_2) (adj-up loc_1_2 loc_1_1)
    (adj-down loc_2_1 loc_2_2) (adj-up loc_2_2 loc_2_1)
    (adj-down loc_3_1 loc_3_2) (adj-up loc_3_2 loc_3_1)
    (adj-down loc_4_1 loc_4_2) (adj-up loc_4_2 loc_4_1)
    (adj-down loc_5_1 loc_5_2) (adj-up loc_5_2 loc_5_1)
    (adj-down loc_6_1 loc_6_2) (adj-up loc_6_2 loc_6_1)

    (adj-down loc_1_2 loc_1_3) (adj-up loc_1_3 loc_1_2)
    (adj-down loc_3_2 loc_3_3) (adj-up loc_3_3 loc_3_2)
    (adj-down loc_4_2 loc_4_3) (adj-up loc_4_3 loc_4_2)
    (adj-down loc_5_2 loc_5_3) (adj-up loc_5_3 loc_5_2)
    (adj-down loc_6_2 loc_6_3) (adj-up loc_6_3 loc_6_2)

    (adj-down loc_1_3 loc_1_4) (adj-up loc_1_4 loc_1_3)
    (adj-down loc_3_3 loc_3_4) (adj-up loc_3_4 loc_3_3)
    (adj-down loc_4_3 loc_4_4) (adj-up loc_4_4 loc_4_3)
    (adj-down loc_5_3 loc_5_4) (adj-up loc_5_4 loc_5_3)
    (adj-down loc_6_3 loc_6_4) (adj-up loc_6_4 loc_6_3)

    (adj-down loc_1_4 loc_1_5) (adj-up loc_1_5 loc_1_4)
    (adj-down loc_3_4 loc_3_5) (adj-up loc_3_5 loc_3_4)
    (adj-down loc_4_4 loc_4_5) (adj-up loc_4_5 loc_4_4)
    (adj-down loc_5_4 loc_5_5) (adj-up loc_5_5 loc_5_4)
    (adj-down loc_6_4 loc_6_5) (adj-up loc_6_5 loc_6_4)

    (adj-down loc_1_5 loc_1_6) (adj-up loc_1_6 loc_1_5)
    (adj-down loc_3_5 loc_3_6) (adj-up loc_3_6 loc_3_5)
    (adj-down loc_4_5 loc_4_6) (adj-up loc_4_6 loc_4_5)
    (adj-down loc_5_5 loc_5_6) (adj-up loc_5_6 loc_5_5)
    (adj-down loc_6_5 loc_6_6) (adj-up loc_6_6 loc_6_5)
  )

  (:goal
    (and
      (box-at box_0 loc_3_5)
      (box-at box_1 loc_4_5)
      (heavybox-at hbx_0 loc_4_6)
    )
  )
)
""").strip() + "\n"


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_planner(repo_root: Path) -> subprocess.CompletedProcess[str]:
    solver = repo_root / "planner" / "pddl_solver.py"
    domain = repo_root / "pddl" / "domain.pddl"
    problem = repo_root / "pddl" / "problem.pddl"

    if not solver.exists():
        raise FileNotFoundError(f"Could not find planner script at: {solver}")

    cmd = ["py", str(solver), str(domain), str(problem)]
    return subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write PDDL files and run the planner.")
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the prompt text used for the LLM workflow.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    pddl_dir = repo_root / "pddl"

    write_text_file(repo_root / "llm_prompt.txt", PROMPT_TEXT)
    write_text_file(pddl_dir / "domain.pddl", DOMAIN_PDDL)
    write_text_file(pddl_dir / "problem.pddl", PROBLEM_PDDL)

    if args.print_prompt:
        print("=== LLM PROMPT ===")
        print(PROMPT_TEXT)
        print("==================\n")

    print("Wrote:")
    print(f"  {pddl_dir / 'domain.pddl'}")
    print(f"  {pddl_dir / 'problem.pddl'}")
    print(f"  {repo_root / 'llm_prompt.txt'}")
    print()

    try:
        result = run_planner(repo_root)
    except Exception as exc:
        error_text = f"Planner execution failed before running:\n{exc}\n"
        write_text_file(repo_root / "planner_output.txt", error_text)
        print(error_text)
        return 1

    combined_output = ""
    if result.stdout:
        combined_output += result.stdout
    if result.stderr:
        if combined_output and not combined_output.endswith("\n"):
            combined_output += "\n"
        combined_output += result.stderr

    write_text_file(repo_root / "planner_output.txt", combined_output)

    print("=== Planner output ===")
    print(combined_output, end="" if combined_output.endswith("\n") else "\n")
    print("======================")
    print(f"Saved full planner log to: {repo_root / 'planner_output.txt'}")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())