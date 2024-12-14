from policy import Policy
import numpy as np
import random
import sys

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        if policy_id == 1:
            self.sorted_prods = []
            self.list_prods = []
            self.current_index_filled = 0  # Stores the current filled stock
        elif policy_id == 2:
            self.q_table = {}  # State-action value table
            self.alpha = 0.1   # Learning rate
            self.gamma = 0.9   # Discount factor
            self.epsilon = 0.1 # Exploration rate
            self.current_index_filled = 0  # Stores the current filled stock
    def get_action(self, observation, info):
        if self.policy_id == 1:
            # Policy 1: Sort products and place them
            if not self.sorted_prods:
                self.list_prods = observation["products"]
                for prod in self.list_prods:
                    prod_size = prod["size"]
                    prod_area = prod_size[0] * prod_size[1]
                    quantity = prod["quantity"]
                    self.sorted_prods.append([prod_area, prod, quantity])  # Include quantity
                
                # Sort products by area initially
                self.sorted_prods.sort(key=lambda x: x[0], reverse=True)
            if all(prod[2] == 0 for prod in self.sorted_prods):
                sys.exit("All products have quantity 0. Stopping the program.")

            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                if i < self.current_index_filled:
                    continue
                self.current_index_filled = i
                for prod in self.sorted_prods:
                    prod_width, prod_height = prod[1]["size"]
                    x, y = 0, 0  # Initialize x and y
                    if prod[1]["quantity"] > 0:
                        for (w, h) in [(prod_width, prod_height), (prod_height, prod_width)]:
                            for x in range(stock_w - w + 1):
                                for y in range(stock_h - h + 1):
                                    if self._can_place_(stock, (x, y), (w, h)):
                                        return {"stock_idx": i, "size": (w, h), "position": (x, y)}
                self.current_index_filled += 1
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
        elif self.policy_id == 2:
            # Policy 2: Reinforcement Learning Q-Learning
            stock_idx = self.current_index_filled
            if stock_idx >= len(observation["stocks"]):
                return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
                
            current_stock = observation["stocks"][stock_idx]
            state = self._get_state(current_stock, observation["products"])
            possible_actions = self._get_possible_actions(state, current_stock, observation["products"])
            
            if not possible_actions:
                self.current_index_filled += 1
                return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                # Explore: random action
                action = random.choice(possible_actions)
            else:
                # Exploit: best known action
                if state not in self.q_table:
                    self.q_table[state] = {str(a): 0 for a in possible_actions}
                    
                action = max(possible_actions, 
                           key=lambda a: self.q_table[state].get(str(a), 0))

            # Q-learning update
            reward = self._get_reward(action, current_stock)
            next_state = self._get_state(current_stock, observation["products"])
            
            if state not in self.q_table:
                self.q_table[state] = {}
            if str(action) not in self.q_table[state]:
                self.q_table[state][str(action)] = 0
                
            # Q-value update
            old_value = self.q_table[state][str(action)]
            next_max = max(self.q_table.get(next_state, {0: 0}).values())
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[state][str(action)] = new_value

            return {
                "stock_idx": stock_idx,
                "size": action["size"],
                "position": action["position"]
            }
        
    def _get_state(self, stock, products):
        """Convert current stock and products to state representation"""
        # Convert stock dimensions to integers/tuples
        stock_w, stock_h = map(int, self._get_stock_size_(stock))
        
        # Convert remaining products to hashable format
        remaining_products = tuple(
            (tuple(map(int, p["size"])), p["quantity"]) 
            for p in products 
            if p["quantity"] > 0
        )
    
        # Return hashable state representation
        return (stock_w, stock_h, remaining_products)
    
    def _get_possible_actions(self, state, stock, products):
        """Get all valid placements for current state"""
        stock_w, stock_h = state[0], state[1]
        actions = []
        
        for p_idx, product in enumerate(products):
            if product["quantity"] <= 0:
                continue
                
            prod_w, prod_h = product["size"]
            # Try both orientations
            for w, h in [(prod_w, prod_h), (prod_h, prod_w)]:
                for x in range(stock_w - w + 1):
                    for y in range(stock_h - h + 1):
                        if self._can_place_(stock, (x, y), (w, h)):
                            actions.append({
                                "product_idx": p_idx,
                                "size": (w, h),
                                "position": (x, y)
                            })
        return actions

    def _get_reward(self, action, stock):
        """Calculate reward for an action"""
        if action is None:
            return -10  # Penalty for invalid action
            
        w, h = action["size"]
        used_area = w * h
        stock_w, stock_h = self._get_stock_size_(stock)
        total_area = stock_w * stock_h
        return used_area / total_area  # Reward based on area utilization