import os
import re

from dotenv import load_dotenv
from google import genai
load_dotenv()

# ==========================================
# 1. THE MASTER PROMPT
# ==========================================
PROMPT = """
You are an expert AI classical planner. I need you to write a `domain.pddl` and `problem.pddl` for a Box Pushing grid world.

**Types:**
The types must be exactly: `agent location box heavybox`. 
(Both `box` and `heavybox` are objects to be pushed, but they behave differently).

**The Rules of the World:**
1. It is a 2D grid. Agents can move to adjacent clear locations. Agents CAN share the same cell.
2. There is a `move` action for a single agent to move to an adjacent clear location.
3. Regular boxes: A single agent can push a regular box using `push-small`. The agent moves into the box's old cell, and the box moves to an adjacent clear cell.
4. Heavy box: There is ONE heavy box. It occupies exactly one cell (`?boxloc`). To push it using `push-heavy`, TWO agents must be standing in the EXACT SAME CELL (`?from`) adjacent to the heavy box. They push it into a clear cell (`?to`). After the push, both agents end up at `?boxloc` (the heavy box's old cell).
5. **CRITICAL PDDL PREDICATE RULE:** DO NOT use union types (like `or`) and DO NOT use generic/untyped parameters. You MUST define three separate, strictly-typed positional predicates: `(agent-at ?ag - agent ?loc - location)`, `(box-at ?b - box ?loc - location)`, and `(heavybox-at ?hb - heavybox ?loc - location)`.
6. **Clear Predicate:** Define a `(clear ?loc - location)` predicate. A location is clear if it has NO box and NO heavybox on it (agents do not block locations). 
7. **CRITICAL PARAMETER ORDER:** The python visualizer is hard-coded to parse parameters in a specific order. You MUST define your actions with exactly this parameter order:
   - `move(?ag - agent, ?from - location, ?to - location)`
   - `push-small(?ag - agent, ?from - location, ?to - location, ?b - box, ?new_box_loc - location)`
   - `push-heavy(?ag1 - agent, ?ag2 - agent, ?from - location, ?to - location, ?hb - heavybox, ?new_box_loc - location)`
   *(Note: In all cases, `?to` is the cell the agent steps into).*
8. **STRAIGHT-LINE PUSHING (CRITICAL):** Agents can only push boxes in a straight line. You MUST define a `(straight ?loc1 - location ?loc2 - location ?loc3 - location)` predicate. In the initial state, you must explicitly list all valid 3-cell straight lines (horizontal and vertical) for the 5x5 grid. 
   **BOUNDARY RULE:** The grid is EXACTLY 5x5. You are strictly forbidden from generating any coordinates with a 0 or a 6 (e.g., `loc_1_0`, `loc_0_2`, or `loc_6_1` are ILLEGAL). Only use numbers 1 through 5. In your `push-small` and `push-heavy` actions, use `(straight ?from ?to ?new_box_loc)` instead of `adjacent` to ensure physical realism!

**The Problem (Initial State):**
* Map size: 5x5 grid.
* Agents: `agent_0` is at `loc_5_1`. `agent_1` is at `loc_1_2`.
* Regular Boxes: `box_0` is at `loc_2_2`. `box_1` is at `loc_3_3`.
* Heavy Box: `heavy_0` is at `loc_2_3`.
* Adjacency & Straight Lines: You must define all valid `adjacent` and `straight` logic for a 5x5 grid.
* The goal condition must ONLY refer to box locations (not agents). 
* Goal: `(box-at box_0 loc_4_4)`, `(box-at box_1 loc_4_5)`, and `(heavybox-at heavy_0 loc_1_5)`.

Please output the `domain.pddl` first, enclosed in ```lisp tags, followed by the `problem.pddl`, also enclosed in ```lisp tags. Do not write any other text.
"""


# ==========================================
# 2. THE API CALL
# ==========================================
def generate_pddl_with_gemini(prompt_text):
    print("🧠 Contacting Gemini API to generate PDDL...")

    # Initialize the client. It automatically looks for the GEMINI_API_KEY environment variable.
    client = genai.Client()

    # We use gemini-2.5-flash as it is lightning fast and excellent at coding/logic
    # https://aistudio.google.com/api-keys - to generate more keys
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt_text,
    )

    return response.text


# ==========================================
# 3. EXTRACT AND SAVE
# ==========================================
def extract_and_save_pddl(llm_response):
    print("✂️ Extracting PDDL from response...")

    os.makedirs("exercises/ex1/pddl", exist_ok=True)

    # Workaround for copy-paste bugs: matching 3 backticks safely
    pattern = r'[`]{3}(?:lisp|pddl)\n(.*?)\n[`]{3}'
    pddl_blocks = re.findall(pattern, llm_response, re.DOTALL)

    if len(pddl_blocks) >= 2:
        domain_code = pddl_blocks[0]
        problem_code = pddl_blocks[1]

        with open("exercises/ex1/pddl/domain.pddl", "w", encoding="utf-8") as f:
            f.write(domain_code.strip())
        print("✅ Successfully saved exercises/ex1/pddl/domain.pddl")

        with open("exercises/ex1/pddl/problem.pddl", "w", encoding="utf-8") as f:
            f.write(problem_code.strip())
        print("✅ Successfully saved exercises/ex1/pddl/problem.pddl")
    else:
        print("❌ Error: Could not cleanly extract two PDDL blocks from the LLM response.")
        print("Raw response was:")
        print(llm_response)