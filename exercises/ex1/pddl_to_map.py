import re

def parse_pddl_to_map(domain_file, problem_file):
    with open(problem_file, 'r') as f:
        problem_text = f.read()

    locations = re.findall(r'loc_(\d+)_(\d+)', problem_text)
    if not locations:
        raise ValueError("No locations found in the problem file.")
        
    # Visualizer expects loc_X_Y where X=Col, Y=Row
    max_x = max([int(x) for x, y in locations])
    max_y = max([int(y) for x, y in locations])

    # Arrays are accessed as grid[row][column] -> grid[y][x]
    grid = [[' ' for _ in range(max_x)] for _ in range(max_y)]

    # Grab everything between :init and :goal
    init_match = re.search(r'\(:init(.*?)\(:goal', problem_text, re.DOTALL | re.IGNORECASE)
    
    # FIX: Grab EVERYTHING from :goal to the end of the file
    goal_match = re.search(r'\(:goal(.*)', problem_text, re.DOTALL | re.IGNORECASE)
    
    init_text = init_match.group(1) if init_match else problem_text
    goal_text = goal_match.group(1) if goal_match else ""

    # Parse X (col) and Y (row) and place them correctly in the matrix
    for x, y in re.findall(r'loc_(\d+)_(\d+)', goal_text):
        grid[int(y)-1][int(x)-1] = 'G'
        
    for x, y in re.findall(r'agent-at\s+\S+\s+loc_(\d+)_(\d+)', init_text):
        grid[int(y)-1][int(x)-1] = 'A'
    for x, y in re.findall(r'box-at\s+\S+\s+loc_(\d+)_(\d+)', init_text):
        grid[int(y)-1][int(x)-1] = 'B'
    for x, y in re.findall(r'heavybox-at\s+\S+\s+loc_(\d+)_(\d+)', init_text):
        grid[int(y)-1][int(x)-1] = 'C'

    ascii_map = []
    wall_row = "W" * (max_x + 2)
    ascii_map.append(wall_row)
    for row in grid:
        ascii_map.append("W" + "".join(row) + "W")
    ascii_map.append(wall_row)

    return ascii_map
# Quick self-test block (runs only if you run this file directly)
if __name__ == "__main__":
    try:
        # Assumes you've already run llm_pipeline.py to generate these
        test_map = parse_pddl_to_map("pddl/domain.pddl", "pddl/problem.pddl")
        for line in test_map:
            print(line)
    except FileNotFoundError:
        print("Run llm_pipeline.py first to generate the PDDL files.")

