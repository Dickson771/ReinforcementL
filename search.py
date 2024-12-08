import pickle
import json
from FOON_class import Object

# -----------------------------------------------------------------------------------------------------------------------------#

# Checks an ingredient exists in kitchen


def check_if_exist_in_kitchen(kitchen_items, ingredient):
    """
        parameters: a list of all kitchen items,
                    an ingredient to be searched in the kitchen
        returns: True if ingredient exists in the kitchen
    """

    for item in kitchen_items:
        if item["label"] == ingredient.label \
                and sorted(item["states"]) == sorted(ingredient.states) \
                and sorted(item["ingredients"]) == sorted(ingredient.ingredients) \
                and item["container"] == ingredient.container:
            return True

    return False


# -----------------------------------------------------------------------------------------------------------------------------#


def search_BFS(kitchen_items=[], goal_node=None):
    # list of indices of functional units
    reference_task_tree = []

    # list of object indices that need to be searched
    items_to_search = []

    # find the index of the goal node in object node list
    items_to_search.append(goal_node.id)

    # list of item already explored
    items_already_searched = []

    while len(items_to_search) > 0:
        current_item_index = items_to_search.pop(0)  # pop the first element
        if current_item_index in items_already_searched:
            continue

        else:
            items_already_searched.append(current_item_index)

        current_item = foon_object_nodes[current_item_index]

        if not check_if_exist_in_kitchen(kitchen_items, current_item):

            candidate_units = foon_object_to_FU_map[current_item_index]

            # selecting the first path
            # this is the part where you should use heuristic for Greedy Best-First search
            selected_candidate_idx = candidate_units[0]

            # if an fu is already taken, do not process it again
            if selected_candidate_idx in reference_task_tree:
                continue

            reference_task_tree.append(selected_candidate_idx)

            # all input of the selected FU need to be explored
            for node in foon_functional_units[
                    selected_candidate_idx].input_nodes:
                node_idx = node.id
                if node_idx not in items_to_search:

                    # if in the input nodes, we have bowl contains {onion} and onion, chopped, in [bowl]
                    # explore only onion, chopped, in bowl
                    flag = True
                    if node.label in utensils and len(node.ingredients) == 1:
                        for node2 in foon_functional_units[
                                selected_candidate_idx].input_nodes:
                            if node2.label == node.ingredients[
                                    0] and node2.container == node.label:

                                flag = False
                                break
                    if flag:
                        items_to_search.append(node_idx)

    # reverse the task tree
    reference_task_tree.reverse()

    # create a list of functional unit from the indices of reference_task_tree
    task_tree_units = []
    for i in reference_task_tree:
        task_tree_units.append(foon_functional_units[i])

    return task_tree_units


def save_paths_to_file(task_tree, path):

    print('writing generated task tree to ', path)
    _file = open(path, 'w')

    _file.write('//\n')
    for FU in task_tree:
        _file.write(FU.get_FU_as_text() + "\n")
    _file.close()


# -----------------------------------------------------------------------------------------------------------------------------#

# creates the graph using adjacency list
# each object has a list of functional list where it is an output


def read_universal_foon(filepath='FOON.pkl'):
    """
        parameters: path of universal foon (pickle file)
        returns: a map. key = object, value = list of functional units
    """
    pickle_data = pickle.load(open(filepath, 'rb'))
    functional_units = pickle_data["functional_units"]
    object_nodes = pickle_data["object_nodes"]
    object_to_FU_map = pickle_data["object_to_FU_map"]

    return functional_units, object_nodes, object_to_FU_map


# -----------------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    foon_functional_units, foon_object_nodes, foon_object_to_FU_map = read_universal_foon(
    )

    utensils = []
    with open('utensils.txt', 'r') as f:
        for line in f:
            utensils.append(line.rstrip())

    kitchen_items = json.load(open('kitchen.json'))

    goal_nodes = json.load(open("goal_nodes.json"))

    for node in goal_nodes:
        node_object = Object(node["label"])
        node_object.states = node["states"]
        node_object.ingredients = node["ingredients"]
        node_object.container = node["container"]

        for object in foon_object_nodes:
            if object.check_object_equal(node_object):
                output_task_tree = search_BFS(kitchen_items, object)
                save_paths_to_file(output_task_tree,
                                   'output_BFS_{}.txt'.format(node["label"]))
                break
