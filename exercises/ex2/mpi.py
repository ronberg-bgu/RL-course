import copy
import itertools

import sys
import os
from unittest import result

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np 
UP,RIGHT,DOWN,LEFT = 0,1,2,3
STAY = 4

class my_stoc_env():
    def __init__(self, ascii_map, move_success_prob=0.8, push_success_prob=0.8):
        self.ascii_map = ascii_map
        self.move_success_prob = move_success_prob
        self.push_success_prob = push_success_prob
        self.wall_positions, self.agent_positions, self.small_boxes_positions, self.heavy_box_positions, self.goal_positions = self.read_ascii_map(ascii_map)
    def read_ascii_map(self,ascii_map: list[str]):
        wall_positions = []
        agent_positions = []
        small_boxes_positions = []
        heavy_box_positions = []
        goal_positions = []
        for y, row in enumerate(ascii_map):
            for x, char in enumerate(row):
                if char == "W":
                    wall_positions.append((x, y))
                elif char == "A":
                    agent_positions.append((x, y))
                elif char == "B":
                    small_boxes_positions.append((x, y))
                elif char == "C":
                    heavy_box_positions.append((x, y))
                elif char == "G":
                    goal_positions.append((x, y))
        return wall_positions, agent_positions, small_boxes_positions, heavy_box_positions, goal_positions
    def generate_ascii_map(self):
        max_x = max(pos[0] for pos in self.wall_positions + self.agent_positions + self.small_boxes_positions + self.heavy_box_positions + self.goal_positions) + 1
        max_y = max(pos[1] for pos in self.wall_positions + self.agent_positions + self.small_boxes_positions + self.heavy_box_positions + self.goal_positions) + 1
        ascii_map = [[" " for _ in range(max_x)] for _ in range(max_y)]
        for x, y in self.goal_positions:
            ascii_map[y][x] = "G"
        for x, y in self.wall_positions:
            ascii_map[y][x] = "W"
        for x, y in self.agent_positions:
            ascii_map[y][x] = "A"
        for x, y in self.small_boxes_positions:
            ascii_map[y][x] = "B"
        for x, y in self.heavy_box_positions:
            ascii_map[y][x] = "C"

        return ["".join(row) for row in ascii_map]
    
    def duplicate(self):
        new_env = my_stoc_env(self.generate_ascii_map(), self.move_success_prob, self.push_success_prob)
        new_env.wall_positions = copy.deepcopy(self.wall_positions)
        new_env.agent_positions = copy.deepcopy(self.agent_positions)
        new_env.small_boxes_positions = copy.deepcopy(self.small_boxes_positions)
        new_env.heavy_box_positions = copy.deepcopy(self.heavy_box_positions)
        new_env.goal_positions = copy.deepcopy(self.goal_positions)
        return new_env
    def can_move(self, agent_index, direction):
        x, y = self.agent_positions[agent_index]
        if direction == UP:
            new_pos = (x, y - 1)
        elif direction == RIGHT:
            new_pos = (x + 1, y)
        elif direction == DOWN:
            new_pos = (x, y + 1)
        elif direction == LEFT:
            new_pos = (x - 1, y)
        else: # STAY
            return True
        if new_pos in self.wall_positions:
            return False
        if new_pos in self.small_boxes_positions or new_pos in self.heavy_box_positions:
            return False
        return True
    
    def can_push_small(self, agent_index, direction):
        x, y = self.agent_positions[agent_index]
        if direction == UP:
            box_pos = (x, y - 1)
            new_box_pos = (x, y - 2)
        elif direction == RIGHT:
            box_pos = (x + 1, y)
            new_box_pos = (x + 2, y)
        elif direction == DOWN:
            box_pos = (x, y + 1)
            new_box_pos = (x, y + 2)
        elif direction == LEFT:
            box_pos = (x - 1, y)
            new_box_pos = (x - 2, y)
        else: # STAY
            return False
        if box_pos not in self.small_boxes_positions:
            return False
        if new_box_pos in self.wall_positions or new_box_pos in self.small_boxes_positions or new_box_pos in self.heavy_box_positions or new_box_pos in self.agent_positions:
            return False
        return True

    def can_push_heavy(self, agent_index1, agent_index2, direction, direction2):
        if direction != direction2:
            return False
        if agent_index1 == agent_index2:
            return False

        x1, y1 = self.agent_positions[agent_index1]
        x2, y2 = self.agent_positions[agent_index2]
        if x1 != x2 or y1 != y2:
            return False
        if direction == UP:
            box_pos = (x1, y1 - 1)
            new_box_pos = (x1, y1 - 2)
        elif direction == RIGHT:
            box_pos = (x1 + 1, y1)
            new_box_pos = (x1 + 2, y1)
        elif direction == DOWN:
            box_pos = (x1, y1 + 1)
            new_box_pos = (x1, y1 + 2)
        elif direction == LEFT:
            box_pos = (x1 - 1, y1)
            new_box_pos = (x1 - 2, y1)
        else: # STAY
            return False
        if box_pos not in self.heavy_box_positions:
            return False
        if new_box_pos in self.wall_positions or new_box_pos in self.small_boxes_positions or new_box_pos in self.heavy_box_positions or new_box_pos in self.agent_positions:
            return False
        return True
    
    def step(self,actions:list[int]):
        for agent_index, action in enumerate(actions):
            if action not in [UP, RIGHT, DOWN, LEFT]:
                continue
            if self.can_move(agent_index, action):
                x, y = self.agent_positions[agent_index]
                if action == UP:
                    new_pos = (x, y - 1)
                elif action == RIGHT:
                    new_pos = (x + 1, y)
                elif action == DOWN:
                    new_pos = (x, y + 1)
                elif action == LEFT:
                    new_pos = (x - 1, y)
                self.agent_positions[agent_index] = new_pos
            elif self.can_push_small(agent_index, action):
                x, y = self.agent_positions[agent_index]
                if action == UP:
                    box_pos = (x, y - 1)
                    new_box_pos = (x, y - 2)
                elif action == RIGHT:
                    box_pos = (x + 1, y)
                    new_box_pos = (x + 2, y)
                elif action == DOWN:
                    box_pos = (x, y + 1)
                    new_box_pos = (x, y + 2)
                elif action == LEFT:
                    box_pos = (x - 1, y)
                    new_box_pos = (x - 2, y)
                self.agent_positions[agent_index] = box_pos
                self.small_boxes_positions.remove(box_pos)
                self.small_boxes_positions.append(new_box_pos)
            elif agent_index+1<len(self.agent_positions) and self.can_push_heavy(agent_index, agent_index+1, action, actions[agent_index+1]):
                x, y = self.agent_positions[agent_index]
                if action == UP:
                    box_pos = (x, y - 1)
                    new_box_pos = (x, y - 2)
                elif action == RIGHT:
                    box_pos = (x + 1, y)
                    new_box_pos = (x + 2, y)
                elif action == DOWN:
                    box_pos = (x, y + 1)
                    new_box_pos = (x, y + 2)
                elif action == LEFT:
                    box_pos = (x - 1, y)
                    new_box_pos = (x - 2, y)
                self.agent_positions[agent_index] = box_pos
                self.agent_positions[agent_index+1] = box_pos
                self.heavy_box_positions.remove(box_pos)
                self.heavy_box_positions.append(new_box_pos)
                
    
    
    def get_state(self):
        return (self.agent_positions[0], self.agent_positions[1], self.small_boxes_positions[0],self.small_boxes_positions[1], self.heavy_box_positions[0]), self.wall_positions
    
    def from_state(self, state,wall_positions):
        agent_pos1, agent_pos2, small_box_pos1, small_box_pos2, heavy_box_pos = state
        self.agent_positions = [agent_pos1, agent_pos2]
        self.small_boxes_positions = [small_box_pos1, small_box_pos2]
        self.heavy_box_positions = [heavy_box_pos]
        self.wall_positions = wall_positions
        self.ascii_map = self.generate_ascii_map()
    
    def is_goal_state(self):
        if self.small_boxes_positions[0] in self.goal_positions and self.small_boxes_positions[1] in self.goal_positions and self.heavy_box_positions[0] in self.goal_positions:
            return 1
        return 0
    
    def get_transitions(self, action_1, action_2):
        if action_1 in [UP, RIGHT, DOWN, LEFT]:
            if action_2 in [UP, RIGHT, DOWN, LEFT]:
                if self.can_push_heavy(0, 1, action_1, action_2):
                    return [(self.push_success_prob, self.get_state()[0], self.is_goal_state()), (1-self.push_success_prob, self.get_state()[0], self.is_goal_state())]
                elif self.can_push_small(0, action_1):
                    if self.can_push_small(1, action_2):
                        ss_env = self.duplicate()
                        ss_env.step([action_1, action_2])
                        sf_env = self.duplicate()
                        sf_env.step([action_1, STAY])
                        fs_env = self.duplicate()
                        fs_env.step([STAY, action_2])
                        ff_env = self.duplicate()
                        ff_env.step([STAY, STAY])
                        return [(self.push_success_prob*self.push_success_prob, ss_env.get_state()[0], ss_env.is_goal_state()), (self.push_success_prob*(1-self.push_success_prob), sf_env.get_state()[0], sf_env.is_goal_state()), ((1-self.push_success_prob)*self.push_success_prob, fs_env.get_state()[0], fs_env.is_goal_state()), ((1-self.push_success_prob)*(1-self.push_success_prob), ff_env.get_state()[0], ff_env.is_goal_state())]
                    elif self.can_move(1, action_2):
                        ss_env = self.duplicate()
                        ss_env.step([action_1, action_2])
                        sl_env = self.duplicate()
                        sl_env.step([action_1, (action_2-1)%4])
                        sr_env = self.duplicate()
                        sr_env.step([action_1, (action_2+1)%4])
                        fs_env = self.duplicate()
                        fs_env.step([STAY, action_2])
                        fl_env = self.duplicate()
                        fl_env.step([STAY, (action_2-1)%4])
                        fr_env = self.duplicate()
                        fr_env.step([STAY, (action_2+1)%4])
                        return [(self.push_success_prob*self.move_success_prob, ss_env.get_state()[0], ss_env.is_goal_state()), (self.push_success_prob*((1-self.move_success_prob)/2), sl_env.get_state()[0], sl_env.is_goal_state()), (self.push_success_prob*((1-self.move_success_prob)/2), sr_env.get_state()[0], sr_env.is_goal_state()), ((1-self.push_success_prob)*self.move_success_prob, fs_env.get_state()[0], fs_env.is_goal_state()), ((1-self.push_success_prob)*((1-self.move_success_prob)/2), fl_env.get_state()[0], fl_env.is_goal_state()), ((1-self.push_success_prob)*((1-self.move_success_prob)/2), fr_env.get_state()[0], fr_env.is_goal_state())]
                    else:
                        return self.get_transitions(action_1, STAY)
                elif self.can_move(0, action_1):
                    if self.can_push_small(1, action_2):
                        ss_env = self.duplicate()
                        ss_env.step([action_1, action_2])
                        ls_env = self.duplicate()
                        ls_env.step([(action_1-1)%4, action_2])
                        rs_env = self.duplicate()
                        rs_env.step([(action_1+1)%4, action_2])
                        sf_env = self.duplicate()
                        sf_env.step([action_1, STAY])
                        lf_env = self.duplicate()
                        lf_env.step([(action_1-1)%4, STAY])
                        rf_env = self.duplicate()
                        rf_env.step([(action_1+1)%4, STAY])
                        return [(self.move_success_prob*self.push_success_prob, ss_env.get_state()[0], ss_env.is_goal_state()), (self.move_success_prob*((1-self.push_success_prob)/2), ls_env.get_state()[0], ls_env.is_goal_state()), (self.move_success_prob*((1-self.push_success_prob)/2), rs_env.get_state()[0], rs_env.is_goal_state()), ((1-self.move_success_prob)*self.push_success_prob, sf_env.get_state()[0], sf_env.is_goal_state()), ((1-self.move_success_prob)*((1-self.push_success_prob)/2), lf_env.get_state()[0], lf_env.is_goal_state()), ((1-self.move_success_prob)*((1-self.push_success_prob)/2), rf_env.get_state()[0], rf_env.is_goal_state())]
                    elif self.can_move(1, action_2):
                        ss_env = self.duplicate()
                        ss_env.step([action_1, action_2])
                        sl_env = self.duplicate()
                        sl_env.step([action_1, (action_2-1)%4])
                        sr_env = self.duplicate()
                        sr_env.step([action_1, (action_2+1)%4])
                        ls_env = self.duplicate()
                        ls_env.step([(action_1-1)%4, action_2])
                        ll_env = self.duplicate()
                        ll_env.step([(action_1-1)%4, (action_2-1)%4])
                        lr_env = self.duplicate()
                        lr_env.step([(action_1-1)%4, (action_2+1)%4])
                        rs_env = self.duplicate()
                        rs_env.step([(action_1+1)%4, action_2])
                        rl_env = self.duplicate()
                        rl_env.step([(action_1+1)%4, (action_2-1)%4])
                        rr_env = self.duplicate()
                        rr_env.step([(action_1+1)%4, (action_2+1)%4])
                        return [(self.move_success_prob*self.move_success_prob, ss_env.get_state()[0], ss_env.is_goal_state()), (self.move_success_prob*((1-self.move_success_prob)/2), sl_env.get_state()[0], sl_env.is_goal_state()), (self.move_success_prob*((1-self.move_success_prob)/2), sr_env.get_state()[0], sr_env.is_goal_state()), ((1-self.move_success_prob)*self.move_success_prob, ls_env.get_state()[0], ls_env.is_goal_state()), ((1-self.move_success_prob)*((1-self.move_success_prob)/2), ll_env.get_state()[0], ll_env.is_goal_state()), ((1-self.move_success_prob)*((1-self.move_success_prob)/2), lr_env.get_state()[0], lr_env.is_goal_state()), ((1-self.move_success_prob)*self.move_success_prob, rs_env.get_state()[0], rs_env.is_goal_state()), ((1-self.move_success_prob)*((1-self.move_success_prob)/2), rl_env.get_state()[0], rl_env.is_goal_state()), ((1-self.move_success_prob)*((1-self.move_success_prob)/2), rr_env.get_state()[0], rr_env.is_goal_state())]
                    else:
                        return self.get_transitions(action_1, STAY)   
                else:
                    return self.get_transitions(STAY, action_2)
            else:
                if self.can_push_small(0, action_1):
                    succ_env = self.duplicate()
                    succ_env.step([action_1, STAY])
                    return [(self.push_success_prob, succ_env.get_state()[0], succ_env.is_goal_state()), (1-self.push_success_prob, self.get_state()[0], self.is_goal_state())]
                elif self.can_move(0, action_1):
                    succ_env = self.duplicate()
                    succ_env.step([action_1, STAY])
                    left_env = self.duplicate()
                    left_env.step([(action_1-1)%4, STAY])
                    right_env = self.duplicate()
                    right_env.step([(action_1+1)%4, STAY])
                    return [(self.move_success_prob, succ_env.get_state()[0], succ_env.is_goal_state()), ((1-self.move_success_prob)/2, left_env.get_state()[0], left_env.is_goal_state()), ((1-self.move_success_prob)/2, right_env.get_state()[0], right_env.is_goal_state())]
                else:
                    return [(1.0, self.get_state()[0], self.is_goal_state())]
        else:
            if action_2 in [UP, RIGHT, DOWN, LEFT]:
                if self.can_push_small(1, action_2):
                    succ_env = self.duplicate()
                    succ_env.step([action_1, action_2])
                    return [(self.push_success_prob, succ_env.get_state()[0], succ_env.is_goal_state()), (1-self.push_success_prob, self.get_state()[0], self.is_goal_state())]
                elif self.can_move(1, action_2):
                    succ_env = self.duplicate()
                    succ_env.step([action_1, action_2])
                    left_env = self.duplicate()
                    left_env.step([action_1, (action_2-1)%4])
                    right_env = self.duplicate()
                    right_env.step([action_1, (action_2+1)%4])
                    return [(self.move_success_prob, succ_env.get_state()[0], succ_env.is_goal_state()), ((1-self.move_success_prob)/2, left_env.get_state()[0], left_env.is_goal_state()), ((1-self.move_success_prob)/2, right_env.get_state()[0], right_env.is_goal_state())]
                else:
                    return [(1.0, self.get_state()[0], self.is_goal_state())]
            else:
                return [(1.0, self.get_state()[0], self.is_goal_state())]
    

    def get_transitions_full(self, action_1, action_2):
        
        result = {}
        items = self.get_transitions(action_1, action_2)
        for p, s, r in items:
            if s not in result:
                result[s] = [p, r]  # store sum of p and one r
            else:
                result[s][0] += p   # add to existing p sum
    
        # convert back to list of tuples (p_sum, s, r)
        return [(p_sum, s, r) for s, (p_sum, r) in result.items()]
    
    def num_on_goal(self):
        count = 0
        for box_pos in self.small_boxes_positions + self.heavy_box_positions:
            if box_pos in self.goal_positions:
                count += 1
        return count
            
            
    def print_map(self):
        ascii_map = self.generate_ascii_map()
        for row in ascii_map:
            print(row)
            
            
    def stochastic_step(self, actions:list[int]):
        if self.can_push_heavy(0, 1, actions[0], actions[1]):
            if np.random.rand() < self.push_success_prob:
                self.step(actions)
        else:
            new_actions = []
            for agent_index, action in enumerate(actions):
                if action not in [UP, RIGHT, DOWN, LEFT]:
                    new_actions.append(STAY)
                    continue
                elif self.can_move(agent_index, action):
                    randd = np.random.rand()
                    if randd < self.move_success_prob:
                        new_actions.append(action)
                    elif randd < self.move_success_prob + (1-self.move_success_prob)/2:
                        new_actions.append((action-1)%4)
                    else:
                        new_actions.append((action+1)%4)
                elif self.can_push_small(agent_index, action):
                    if np.random.rand() < self.push_success_prob:
                        new_actions.append(action)
                    else:
                        new_actions.append(STAY)
                else:
                    new_actions.append(STAY)
            self.step(new_actions)