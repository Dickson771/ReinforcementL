import math
class TreeNode:
    """Represents a node in the search tree for task planning"""
    def __init__(self, node_id, parent=None):
        self.node_id = node_id
        self.parent = parent
        self.child_nodes = []
        self.successes = 0
        self.explored_count = 0
        self.available_actions = []

    def create_child(self, node_id):
        """Creates a new child node and links it to the current node"""
        new_node = TreeNode(node_id, self)
        self.child_nodes.append(new_node)
        return new_node

    def record_result(self, outcome):
        """Records the outcome of a simulation for this node"""
        self.explored_count += 1
        self.successes += outcome

    def calculate_priority_score(self, total_iterations, exploration_constant=1.414):
        """Calculates node priority using Upper Confidence Bound formula"""
        if self.explored_count == 0:
            return float('inf')
        success_rate = self.successes / self.explored_count
        exploration_term = exploration_constant * math.sqrt(math.log(total_iterations) / self.explored_count)
        return success_rate + exploration_term