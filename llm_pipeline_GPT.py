from openai import OpenAI
#ran out of tokens, so this won't work anyway
client = OpenAI()

# ---------- CONFIG ----------
MODEL = "gpt-5.4-mini"   # or whatever model you're using
DOMAIN_FILE = "pddl/domain.pddl"
PROBLEM_FILE = "pddl/problem.pddl"


# ---------- STEP 1: GENERATE DOMAIN ----------
def generate_domain(prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=f"""
You are an expert in PDDL.

Generate ONLY a valid PDDL domain file.
No explanations. No comments outside PDDL.

Task:
{prompt}

Return ONLY the domain.pddl content.
"""
    )

    return response.output[0].content[0].text


# ---------- STEP 2: GENERATE PROBLEM ----------
def generate_problem(prompt: str, domain_pddl: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=f"""
You are an expert in PDDL.

Given this domain:
{domain_pddl}

Generate a matching PDDL problem file.

Task:
{prompt}

Return ONLY the problem.pddl content.
"""
    )

    return response.output[0].content[0].text


# ---------- PIPELINE ----------
def run_pipeline(task_prompt: str):
    # Generate domain
    domain = generate_domain(task_prompt)

    with open(DOMAIN_FILE, "w") as f:
        f.write(domain)

    print("Domain saved to domain.pddl")

    # Generate problem
    problem = generate_problem(problem_prompt, domain)

    with open(PROBLEM_FILE, "w") as f:
        f.write(problem)

    print("Problem saved to problem.pddl")


problem_prompt = """now generate a problem PDDL for this map:
   large_map = 
        "WWWWWWWW",
        "W  AA  W",
        "W B C  W",
        "W      W",
        "W   B  W",
        "W G G GW",
        "WWWWWWWW"
    
A = agent, B = box, C = heavybox, G = goal, rightmost goal is heavybox goal"""
# ---------- RUN ----------
if __name__ == "__main__":
    prompt = """
    The goal is to generate a PDDL for a box pushing problem:
      There are 2 actors and three boxes - one of them is a big box that requires two actors to push it 
     Types: agent location  box heavybox;

    Predicates: agent-at box-at heavybox-at
    example syntax: agent_0, agent_1 loc_x_y box_0, box_1 hbox_0 
     Operations and parameters: 
     move : ?a - agent ?from - location ?to - location 
     push-small: ?a - agent ?from - location  ?boxloc - location  ?toloc - location  ?b - box 
    push-heavy-up: ?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location  ?b - heavybox  
     push-heavy-down :same as above  
     push-heavy-left : same as above  
     push-heavy-right : same as above 
    the two coordinates in push-big-x are used for box location and agent location, since two agents need to push from the exact same point adjacent to the heavy box; 
    additional predicates that can be used for better effect:
    north south east west - for heavy box pushing;
    adj, clear - for marking cells;
     Example Goal: (:goal (and     (box-at box_0 loc_3_5)     (box-at box_1 loc_4_5) ))
     Big box is 1x1 in size and requires two agents to push at the same time 

    multiple agents can occupy the same space;
    make sure to avoid diagonal pushing for heavy box;
    make sure to avoid pushing into boxes (or agents);

    Please give a COMPLETE domain PDDL, problem PDDL is not necessary for now
    """

    run_pipeline(prompt)