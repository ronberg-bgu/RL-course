import pygame
import time
from environment.multi_agent_env import MultiAgentBoxPushEnv

def run_simulation(ascii_map):
    """
    Runs a visual simulation of the PettingZoo environment with random agent actions.
    This demonstrates the rendering built on top of MiniGrid's UI.
    """
    # Initialize the environment with human rendering enabled
    env = MultiAgentBoxPushEnv(ascii_map=ascii_map, render_mode="human")
    obs, info = env.reset()
    
    # Needs to render the first frame immediately
    env.core_env.render()
    
    print("Starting visual simulation! Watch the PyGame window.")
    print("Agents will take random actions to demonstrate the physics engine.")
    
    running = True
    while running and env.agents:
        # MiniGrid/Pygame event loop to prevent the window from freezing or crashing on Mac
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
        if not running:
            break

        # Generate random actions for surviving agents
        # 0: left, 1: right, 2: forward
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render the updated state
        env.core_env.render()
        
        # Pause slightly so human eyes can see the fast grid
        time.sleep(0.3)
        
        # Check if they reached the goal naturally
        if any(terminations.values()):
            print("Goal Reached!")
            break

    print("Simulation Complete.")
    pygame.quit()

if __name__ == "__main__":
    # A fun example map with a Big Box (CC) and a Small Box (B)
    example_map = [
        "WWWWWWWW",
        "W  A B W",
        "W CC   W",
        "W  A   W",
        "W      W",
        "W G    W",
        "WWWWWWWW"
    ]
    
    # To run this, just do `python3 simulate.py` from the root directory!
    run_simulation(example_map)
