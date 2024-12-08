import pickle
import json
import random
import math
from collections import defaultdict
from FOON_class import Object
from TreeNode_class import *

def import_pkl(filepath='FOON.pkl'):
    """Imports the knowledge base containing task planning information"""
    with open(filepath, 'rb') as knowledge_file:
        kb_data = pickle.load(knowledge_file)
    return kb_data["functional_units"], kb_data["object_nodes"], kb_data["object_to_FU_map"]

def parse_action_success_rates(filepath='motion.txt'):
    """Parses action success probabilities from configuration file"""
    success_rates = {}
    with open(filepath, 'r') as config:
        for entry in config:
            entry = entry.strip()
            if entry and '\t' in entry:
                try:
                    action, probability = entry.split('\t')
                    success_rates[action.strip()] = float(probability.strip())
                except ValueError:
                    print(f"Invalid configuration entry: {entry}")
    return success_rates

def verify_resource_availability(available_resources, required_resource):
    """Verifies if a required resource exists in available resources"""
    for resource in available_resources:
        if (resource["label"] == required_resource.label and
                sorted(resource["states"]) == sorted(required_resource.states) and
                sorted(resource["ingredients"]) == sorted(required_resource.ingredients) and
                resource["container"] == required_resource.container):
            return True
    return False

class TaskPlanner:
    """Implements Monte Carlo Tree Search for task planning"""
    def __init__(self, action_units, resources, resource_mapping, inventory, action_probabilities):
        self.action_units = action_units
        self.resources = resources
        self.resource_mapping = resource_mapping
        self.inventory = inventory
        self.action_probabilities = action_probabilities
        self.iteration_count = 0

    def choose_next_node(self, current_node):
        """Selects the most promising node for exploration"""
        while current_node.child_nodes:
            current_node = max(current_node.child_nodes, 
                             key=lambda n: n.calculate_priority_score(self.iteration_count))
            if not current_node.available_actions:
                break
        return current_node

    def grow_tree(self, node):
        """Expands the search tree by adding a new node"""
        if not node.available_actions:
            return node

        action_idx = node.available_actions.pop()
        action_unit = self.action_units[action_idx]

        for required_resource in action_unit.input_nodes:
            if not verify_resource_availability(self.inventory, required_resource):
                new_node = node.create_child(required_resource.id)
                new_node.available_actions = self.resource_mapping[required_resource.id].copy()
                return new_node

        return node

    def run_simulation(self, start_node):
        """Executes a simulation from the given node"""
        current = start_node
        action_sequence = []
        simulation_success = True

        while current.available_actions:
            chosen_action = random.choice(current.available_actions)
            action_unit = self.action_units[chosen_action]
            action_sequence.append(chosen_action)

            success_probability = self.action_probabilities.get(action_unit.motion_node, 0.5)
            if random.random() > success_probability:
                simulation_success = False
                break

            for required_resource in action_unit.input_nodes:
                if not verify_resource_availability(self.inventory, required_resource):
                    current = TreeNode(required_resource.id)
                    current.available_actions = self.resource_mapping[required_resource.id].copy()

            if not current.available_actions:
                break

        return simulation_success, action_sequence

    def update_tree(self, node, outcome):
        """Updates the tree with simulation results"""
        while node:
            node.record_result(outcome)
            node = node.parent

    def plan_task(self, target_node, iterations=1000):
        """Plans a sequence of actions to achieve the target state"""
        root = TreeNode(target_node.id)
        root.available_actions = self.resource_mapping[target_node.id].copy()

        for _ in range(iterations):
            self.iteration_count += 1
            
            selected = self.choose_next_node(root)
            expanded = self.grow_tree(selected)
            success, path = self.run_simulation(expanded)
            self.update_tree(expanded, 1 if success else 0)

        optimal_sequence = []
        current = root
        while current.child_nodes:
            current = max(current.child_nodes, 
                         key=lambda n: (n.successes / n.explored_count if n.explored_count > 0 else 0))
            if current.available_actions:
                best_action = max(current.available_actions,
                                key=lambda a: self.action_probabilities.get(
                                    self.action_units[a].motion_node, 0.5))
                optimal_sequence.append(best_action)

        return [self.action_units[i] for i in optimal_sequence]

def export_task_sequence(sequence, filepath):
    """Exports the planned task sequence to a file"""
    print('Exporting task sequence to:', filepath)
    with open(filepath, 'w') as output:
        output.write('//\n')
        for action in sequence:
            output.write(action.get_FU_as_text() + "\n")

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <target_state>")
        print("  <target_state>: Desired end state (use underscores for spaces)")
        sys.exit(1)

    target_state = sys.argv[1].replace('_', ' ')

    # Load system knowledge base
    action_units, resources, resource_mapping = import_pkl()

    # Initialize system configuration
    with open('utensils.txt', 'r') as f:
        available_tools = [line.rstrip() for line in f]
    inventory = json.load(open('kitchen.json'))
    target_states = json.load(open("goal_nodes.json"))
    action_probabilities = parse_action_success_rates('motion.txt')

    # Locate target state configuration
    target_config = next((state for state in target_states if state["label"] == target_state), None)
    if target_config is None:
        print(f"Target state '{target_state}' not found in configuration")
        sys.exit(1)

    # Initialize target object
    target = Object(target_config["label"])
    target.states = target_config["states"]
    target.ingredients = target_config["ingredients"]
    target.container = target_config["container"]

    # Execute planning algorithm
    for resource in resources:
        if resource.check_object_equal(target):
            planner = TaskPlanner(action_units, resources, 
                                resource_mapping, inventory, 
                                action_probabilities)
            sequence = planner.plan_task(resource, iterations=1000)

            output_path = f'mcts_{target_state.replace(" ", "_")}.txt'
            export_task_sequence(sequence, output_path)
            print(f"Task planning completed. Plan exported to {output_path}")
            break
    else:
        print(f"No matching resource found for target state '{target_state}'")