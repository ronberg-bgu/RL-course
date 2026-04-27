from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import re
import sys
import traceback

from pddl_to_map import parse_pddl_to_map
from planner.pddl_solver import solve_pddl
from visualize_plan import visualize_pddl_plan


PROMPT = """Generate two PDDL files for the RL-course box-pushing assignment: `domain.pddl` and `problem.pddl`.

Follow these requirements exactly.

Use these exact types:
- agent
- location
- box
- heavybox

Use these exact predicates:
- (agent-at ?a - agent ?loc - location)
- (box-at ?b - box ?loc - location)
- (heavybox-at ?h - heavybox ?loc - location)
- (clear ?loc - location)
- (adj ?l1 - location ?l2 - location)
- (left-of ?l1 - location ?l2 - location)
- (right-of ?l1 - location ?l2 - location)
- (up-of ?l1 - location ?l2 - location)
- (down-of ?l1 - location ?l2 - location)

Use these exact actions and parameter orders:
- move(?a, ?from, ?to)
- push-small(?a, ?from, ?boxloc, ?toloc, ?b)
- push-heavy(?a1, ?a2, ?from, ?boxloc, ?toloc, ?h)

Semantics:
- `clear` means the location contains no small box and no heavy box.
- Agents may share the same location.
- Therefore, moving agents must not change `clear`.
- Only box movement should change `clear`.
- Use directional predicates to preserve push direction. Do not rely only on `adj`.
- `move`: one agent moves to an adjacent clear location.
- `push-small`: one agent stands adjacent to a small box, pushes it one step forward in the same direction, and then occupies the box's previous location.
- `push-heavy`: the heavy box is 1x1. Two different agents must stand on the same location adjacent to the heavy box and push in the same direction simultaneously. After the push, both agents occupy the heavy box's previous location.

Compatibility requirements:
- Use valid PDDL syntax only.
- Do not invent extra predicates or extra actions.
- Domain name: `box-push`
- Problem name: `bp-map`
- It is acceptable to use `:disjunctive-preconditions`.
- Do not model `clear` as "no agent is in the cell".

Naming:
- Locations: `loc_X_Y`
- Agents: `agent_0`, `agent_1`
- Small boxes: `box_0`, `box_1`
- Heavy box: `hbx_0`

World rules:
- Grid size is 5x5.
- Coordinates are `(x,y)` where `0 <= x < 5` and `0 <= y < 5`.
- Adjacency is only 4-neighbor: up, down, left, right.
- Wall cells are blocked and not traversable.
- Do not create location objects for wall cells.
- Do not create adjacency relations through walls.
- The goal should mention only final box positions, not agent positions.
- Choose a valid layout whose resulting plan is simple for the provided simulator and visualizer to execute.

World instance:
- `agent_0` starts at `(3,0)`
- `agent_1` starts at `(0,1)`
- `box_0` starts at `(0,2)`
- `box_1` starts at `(1,2)`
- `hbx_0` starts at `(3,1)`
- Goals: `box_0` at `(0,4)`, `box_1` at `(1,4)`, `hbx_0` at `(3,4)`
- Wall cells: `(2,0)` and `(2,4)`

Output format:
- Output only the complete contents of `domain.pddl`
- Then output only the complete contents of `problem.pddl`
- Do not include explanations
- Do not include markdown code fences
- Separate the two files exactly like this:
DOMAIN.PDDL
<content>

PROBLEM.PDDL
<content>
"""

DOMAIN_PDDL = """(define (domain box-push)
  (:requirements :typing :equality :disjunctive-preconditions)
  (:types
    agent location box heavybox
  )

  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (left-of ?l1 - location ?l2 - location)
    (right-of ?l1 - location ?l2 - location)
    (up-of ?l1 - location ?l2 - location)
    (down-of ?l1 - location ?l2 - location)
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
        (and (left-of ?from ?boxloc) (left-of ?boxloc ?toloc))
        (and (right-of ?from ?boxloc) (right-of ?boxloc ?toloc))
        (and (up-of ?from ?boxloc) (up-of ?boxloc ?toloc))
        (and (down-of ?from ?boxloc) (down-of ?boxloc ?toloc))
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
        (and (left-of ?from ?boxloc) (left-of ?boxloc ?toloc))
        (and (right-of ?from ?boxloc) (right-of ?boxloc ?toloc))
        (and (up-of ?from ?boxloc) (up-of ?boxloc ?toloc))
        (and (down-of ?from ?boxloc) (down-of ?boxloc ?toloc))
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
)"""

PROBLEM_PDDL = """(define (problem bp-map)
  (:domain box-push)
  (:objects
    agent_0 agent_1 - agent
    box_0 box_1 - box
    hbx_0 - heavybox
    loc_0_0 loc_1_0 loc_3_0 loc_4_0 loc_0_1 loc_1_1 loc_2_1 loc_3_1 loc_4_1 loc_0_2 loc_1_2 loc_2_2 loc_3_2 loc_4_2 loc_0_3 loc_1_3 loc_2_3 loc_3_3 loc_4_3 loc_0_4 loc_1_4 loc_3_4 loc_4_4 - location
  )
  (:init
    (agent-at agent_0 loc_3_0)
    (agent-at agent_1 loc_0_1)
    (box-at box_0 loc_0_2)
    (box-at box_1 loc_1_2)
    (heavybox-at hbx_0 loc_3_1)
    (clear loc_0_0)
    (clear loc_1_0)
    (clear loc_3_0)
    (clear loc_4_0)
    (clear loc_0_1)
    (clear loc_1_1)
    (clear loc_2_1)
    (clear loc_4_1)
    (clear loc_2_2)
    (clear loc_3_2)
    (clear loc_4_2)
    (clear loc_0_3)
    (clear loc_1_3)
    (clear loc_2_3)
    (clear loc_3_3)
    (clear loc_4_3)
    (clear loc_0_4)
    (clear loc_1_4)
    (clear loc_3_4)
    (clear loc_4_4)
    (adj loc_0_0 loc_1_0)
    (adj loc_1_0 loc_0_0)
    (adj loc_0_0 loc_0_1)
    (adj loc_0_1 loc_0_0)
    (adj loc_1_0 loc_1_1)
    (adj loc_1_1 loc_1_0)
    (adj loc_3_0 loc_4_0)
    (adj loc_4_0 loc_3_0)
    (adj loc_3_0 loc_3_1)
    (adj loc_3_1 loc_3_0)
    (adj loc_4_0 loc_4_1)
    (adj loc_4_1 loc_4_0)
    (adj loc_0_1 loc_1_1)
    (adj loc_1_1 loc_0_1)
    (adj loc_0_1 loc_0_2)
    (adj loc_0_2 loc_0_1)
    (adj loc_1_1 loc_2_1)
    (adj loc_2_1 loc_1_1)
    (adj loc_1_1 loc_1_2)
    (adj loc_1_2 loc_1_1)
    (adj loc_2_1 loc_3_1)
    (adj loc_3_1 loc_2_1)
    (adj loc_2_1 loc_2_2)
    (adj loc_2_2 loc_2_1)
    (adj loc_3_1 loc_4_1)
    (adj loc_4_1 loc_3_1)
    (adj loc_3_1 loc_3_2)
    (adj loc_3_2 loc_3_1)
    (adj loc_4_1 loc_4_2)
    (adj loc_4_2 loc_4_1)
    (adj loc_0_2 loc_1_2)
    (adj loc_1_2 loc_0_2)
    (adj loc_0_2 loc_0_3)
    (adj loc_0_3 loc_0_2)
    (adj loc_1_2 loc_2_2)
    (adj loc_2_2 loc_1_2)
    (adj loc_1_2 loc_1_3)
    (adj loc_1_3 loc_1_2)
    (adj loc_2_2 loc_3_2)
    (adj loc_3_2 loc_2_2)
    (adj loc_2_2 loc_2_3)
    (adj loc_2_3 loc_2_2)
    (adj loc_3_2 loc_4_2)
    (adj loc_4_2 loc_3_2)
    (adj loc_3_2 loc_3_3)
    (adj loc_3_3 loc_3_2)
    (adj loc_4_2 loc_4_3)
    (adj loc_4_3 loc_4_2)
    (adj loc_0_3 loc_1_3)
    (adj loc_1_3 loc_0_3)
    (adj loc_0_3 loc_0_4)
    (adj loc_0_4 loc_0_3)
    (adj loc_1_3 loc_2_3)
    (adj loc_2_3 loc_1_3)
    (adj loc_1_3 loc_1_4)
    (adj loc_1_4 loc_1_3)
    (adj loc_2_3 loc_3_3)
    (adj loc_3_3 loc_2_3)
    (adj loc_3_3 loc_4_3)
    (adj loc_4_3 loc_3_3)
    (adj loc_3_3 loc_3_4)
    (adj loc_3_4 loc_3_3)
    (adj loc_4_3 loc_4_4)
    (adj loc_4_4 loc_4_3)
    (adj loc_0_4 loc_1_4)
    (adj loc_1_4 loc_0_4)
    (adj loc_3_4 loc_4_4)
    (adj loc_4_4 loc_3_4)
    (left-of loc_0_0 loc_1_0)
    (left-of loc_3_0 loc_4_0)
    (left-of loc_0_1 loc_1_1)
    (left-of loc_1_1 loc_2_1)
    (left-of loc_2_1 loc_3_1)
    (left-of loc_3_1 loc_4_1)
    (left-of loc_0_2 loc_1_2)
    (left-of loc_1_2 loc_2_2)
    (left-of loc_2_2 loc_3_2)
    (left-of loc_3_2 loc_4_2)
    (left-of loc_0_3 loc_1_3)
    (left-of loc_1_3 loc_2_3)
    (left-of loc_2_3 loc_3_3)
    (left-of loc_3_3 loc_4_3)
    (left-of loc_0_4 loc_1_4)
    (left-of loc_3_4 loc_4_4)
    (right-of loc_1_0 loc_0_0)
    (right-of loc_4_0 loc_3_0)
    (right-of loc_1_1 loc_0_1)
    (right-of loc_2_1 loc_1_1)
    (right-of loc_3_1 loc_2_1)
    (right-of loc_4_1 loc_3_1)
    (right-of loc_1_2 loc_0_2)
    (right-of loc_2_2 loc_1_2)
    (right-of loc_3_2 loc_2_2)
    (right-of loc_4_2 loc_3_2)
    (right-of loc_1_3 loc_0_3)
    (right-of loc_2_3 loc_1_3)
    (right-of loc_3_3 loc_2_3)
    (right-of loc_4_3 loc_3_3)
    (right-of loc_1_4 loc_0_4)
    (right-of loc_4_4 loc_3_4)
    (up-of loc_0_1 loc_0_0)
    (up-of loc_1_1 loc_1_0)
    (up-of loc_3_1 loc_3_0)
    (up-of loc_4_1 loc_4_0)
    (up-of loc_0_2 loc_0_1)
    (up-of loc_1_2 loc_1_1)
    (up-of loc_2_2 loc_2_1)
    (up-of loc_3_2 loc_3_1)
    (up-of loc_4_2 loc_4_1)
    (up-of loc_0_3 loc_0_2)
    (up-of loc_1_3 loc_1_2)
    (up-of loc_2_3 loc_2_2)
    (up-of loc_3_3 loc_3_2)
    (up-of loc_4_3 loc_4_2)
    (up-of loc_0_4 loc_0_3)
    (up-of loc_1_4 loc_1_3)
    (up-of loc_3_4 loc_3_3)
    (up-of loc_4_4 loc_4_3)
    (down-of loc_0_0 loc_0_1)
    (down-of loc_1_0 loc_1_1)
    (down-of loc_3_0 loc_3_1)
    (down-of loc_4_0 loc_4_1)
    (down-of loc_0_1 loc_0_2)
    (down-of loc_1_1 loc_1_2)
    (down-of loc_2_1 loc_2_2)
    (down-of loc_3_1 loc_3_2)
    (down-of loc_4_1 loc_4_2)
    (down-of loc_0_2 loc_0_3)
    (down-of loc_1_2 loc_1_3)
    (down-of loc_2_2 loc_2_3)
    (down-of loc_3_2 loc_3_3)
    (down-of loc_4_2 loc_4_3)
    (down-of loc_0_3 loc_0_4)
    (down-of loc_1_3 loc_1_4)
    (down-of loc_3_3 loc_3_4)
    (down-of loc_4_3 loc_4_4)
  )
  (:goal
    (and
      (box-at box_0 loc_0_4)
      (box-at box_1 loc_1_4)
      (heavybox-at hbx_0 loc_3_4)
    )
  )
)"""

def write_generated_pddl(repo_root: Path) -> tuple[Path, Path]:
    pddl_dir = repo_root / "pddl"
    pddl_dir.mkdir(exist_ok=True)

    domain_path = pddl_dir / "domain.pddl"
    problem_path = pddl_dir / "problem.pddl"

    domain_path.write_text(DOMAIN_PDDL, encoding="utf-8")
    problem_path.write_text(PROBLEM_PDDL, encoding="utf-8")

    return domain_path, problem_path


def run_planner(domain_path: Path, problem_path: Path):
    log_buffer = StringIO()

    try:
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            plan = solve_pddl(str(domain_path), str(problem_path))
    except Exception:
        traceback.print_exc(file=log_buffer)
        print(log_buffer.getvalue(), end="")
        raise

    raw_log = log_buffer.getvalue()
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    cleaned_log = ansi_re.sub("", raw_log)

    # Re-emit planner output to stdout so `tee` captures it in planner_output.txt.
    print(cleaned_log, end="")
    return plan


def _force_utf8_console() -> None:
    """Avoid Windows cp1252 crashes when downstream code prints emoji."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_console()
    repo_root = Path(__file__).resolve().parent
    domain_path, problem_path = write_generated_pddl(repo_root)

    print("Prompt used to obtain the embedded PDDL:")
    print(PROMPT)
    print()
    print(f"Wrote {domain_path}")
    print(f"Wrote {problem_path}")

    ascii_map = parse_pddl_to_map(domain_path, problem_path)
    print("Reconstructed ASCII map:")
    for row in ascii_map:
        print(row)

    run_planner(domain_path, problem_path)
    visualize_pddl_plan(ascii_map, str(domain_path), str(problem_path))


if __name__ == "__main__":
    main()
