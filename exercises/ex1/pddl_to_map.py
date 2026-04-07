import re

def parse_pddl_to_map(problem_file="exercises/ex1/pddl/problem.pddl"):
    try:
        with open(problem_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Could not find {problem_file}")
        return []

    # Create empty 7x7 grid (5x5 playable space + 1 layer of walls)
    grid = [[' ' for _ in range(7)] for _ in range(7)]

    # Add Walls ('W') around the perimeter
    for i in range(7):
        grid[0][i] = 'W' # Top
        grid[6][i] = 'W' # Bottom
        grid[i][0] = 'W' # Left
        grid[i][6] = 'W' # Right

    # Extract the Init and Goal blocks from the PDDL
    init_block = re.search(r'\(:init(.*?)\(:goal', content, re.DOTALL | re.IGNORECASE)

    # FIX: Grab everything after :goal so we don't accidentally stop at the first parenthesis!
    goal_block = re.search(r'\(:goal(.*)', content, re.DOTALL | re.IGNORECASE)

    if not init_block or not goal_block:
        print("❌ Error: Could not find :init or :goal blocks in PDDL.")
        return []

    init_text = init_block.group(1)
    goal_text = goal_block.group(1)

    def place_objects(text, pattern, symbol):
        # Finds strings like 'agent_0 loc_1_2' and extracts the X and Y
        matches = re.finditer(rf'{pattern}.*?loc_(\d+)_(\d+)', text, re.IGNORECASE)
        for match in matches:
            x, y = int(match.group(1)), int(match.group(2))
            if 1 <= x <= 5 and 1 <= y <= 5:
                grid[y][x] = symbol # y is row, x is column

    # Place the Goals ('G') first - Split into two lines to avoid regex capture group errors!
    place_objects(goal_text, r'box_\d+', 'G')
    place_objects(goal_text, r'heavy_\d+', 'G')

    # Place the Agents ('A'), Regular Boxes ('B'), and Heavy Box ('C')
    place_objects(init_text, r'agent_\d+', 'A')
    place_objects(init_text, r'box_\d+', 'B')
    place_objects(init_text, r'heavy_\d+', 'C')

    # Convert the 2D array back into a list of strings
    ascii_map = ["".join(row) for row in grid]
    return ascii_map

if __name__ == "__main__":
    # Quick test to make sure it works if run by itself
    generated_map = parse_pddl_to_map("exercises/ex1/pddl/problem.pddl")
    for row in generated_map:
        print(row)