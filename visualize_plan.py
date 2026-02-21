import pygame
import time
import re
from environment.multi_agent_env import MultiAgentBoxPushEnv
from planner.pddl_solver import solve_pddl
from minigrid.core.constants import DIR_TO_VEC

def extract_target_pos(pddl_action):
    """
    Parses a string like 'move(agent_0, loc_2_1, loc_2_2)'
    or 'push-small(agent_0, loc_2_2, loc_2_3, loc_2_4, box_0)'
    to find the agent's target coordinate.
    
    For our domain, the FIRST parameter is always the agent,
    the SECOND is 'from', and the THIRD is the 'target' position for the agent.
    """
    action_str = str(pddl_action)
    # Extracts all words inside the parentheses
    # Example: move(agent_0, loc_2_1, loc_2_2) -> ['agent_0', 'loc_2_1', 'loc_2_2']
    params = re.findall(r'[\w_]+', action_str[action_str.find("(")+1:action_str.find(")")])
    
    if len(params) >= 3:
        agent_name = params[0]
        target_loc = params[2] # e.g., 'loc_2_2'
        
        parts = target_loc.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            tx, ty = int(parts[1]), int(parts[2])
            return agent_name, (tx, ty)
            
    return None, None

def get_required_actions(env, agent, target_pos):
    """
    Returns a list of PettingZoo actions (0=left, 1=right, 2=forward) 
    needed to move the agent into the target adjacent position.
    """
    current_pos = env.agent_positions[agent]
    current_dir = env.agent_dirs[agent]
    
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    # Define which direction (0: right, 1: down, 2: left, 3: up) corresponds to dx, dy
    target_dir = None
    for d, vec in enumerate(DIR_TO_VEC):
        if vec[0] == dx and vec[1] == dy:
            target_dir = d
            break
            
    if target_dir is None:
        raise ValueError(f"Target {target_pos} is not adjacent to {current_pos}")
        
    actions = []
    
    # Rotate until facing the target direction
    while current_dir != target_dir:
        # Simplistic rotation: just turn right until we hit it (could be optimized with left turns)
        actions.append(1) # 1 = turn right
        current_dir = (current_dir + 1) % 4
        
    # Move forward
    actions.append(2)
    
    return actions

def visualize_pddl_plan(ascii_map, domain_file, problem_file):
    print("🧠 Solving PDDL problem...")
    plan = solve_pddl(domain_file, problem_file)
    
    if not plan:
        print("❌ No plan could be found. Cannot visualize.")
        return
        
    print(f"✅ Plan found with {len(plan.actions)} steps. Booting visualizer...")
    
    # Initialize Pygame environment
    env = MultiAgentBoxPushEnv(ascii_map=ascii_map, render_mode="human")
    env.reset()
    env.core_env.render()
    
    time.sleep(1) # Pause before starting so the user can look at the screen
    
    for pddl_action in plan.actions:
        print(f"Executing: {pddl_action}")
        
        agent, target_pos = extract_target_pos(pddl_action)
        
        if agent and target_pos:
            pz_actions = get_required_actions(env, agent, target_pos)
            
            for act in pz_actions:
                # Step the environment
                env.step({agent: act})
                env.core_env.render()
                
                # Check for PyGame quit events so the window doesn't freeze
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                        
                time.sleep(0.4) # Control animation speed
                
    print("🎉 Plan execution complete! Closing in 3 seconds.")
    time.sleep(3)
    pygame.quit()

if __name__ == "__main__":
    # We test visualizer on the Example 1 map we generated earlier
    ex1_map = [
        "WWWWW",
        "W A W",
        "W B W",
        "W G W",
        "WWWWW"
    ]
    visualize_pddl_plan(ex1_map, "pddl/ex1_domain.pddl", "pddl/ex1_problem.pddl")
