### ContinuationPaths.py
### MIT LICENSE 2025 Marcio Gameiro

import DSGRN_utils
from collections import defaultdict
from tqdm import tqdm
import time

def is_morse_set(par_node, morse_set, parameter_graph, level):
    """Check if morse_set is a Morse set for the Morse graph at parameter par_node"""
    # Compute Morse graph
    parameter = parameter_graph.parameter(par_node)
    morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter, level=level)
    # Get list of stable Morse nodes
    stable_nodes = [v for v in morse_graph.vertices() if not morse_graph.adjacencies(v)]
    for vert in stable_nodes:
        # Get Morse node (vertex index)
        morse_node = morse_graph.vertex_label(vert)[0]
        # Get set of cells in the Morse set corresponding to the Morse node morse_node
        vertices, edges = DSGRN_utils.MorseComponent(morse_graph, stg, graded_complex, morse_node=morse_node)
        # Check if Morse sets are equal
        if {v for v in vertices} == morse_set:
            return True
    return False

def continuation_path(path, parameter_graph, level=4):
    """Check if a path is a valid continuation path, i. e., check
       if the Morse set in the initial node of the path is a Morse
       set for all nodes in the path.
    """
    # Compute Morse graph for initial parameter in the path
    parameter = parameter_graph.parameter(path[0])
    morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter, level=level)
    # Check if Morse graph contains a single node
    if len(morse_graph.vertices()) != 1:
        return False
    # Get set of cells in the Morse set corresponding to the Morse node
    vertices, edges = DSGRN_utils.MorseComponent(morse_graph, stg, graded_complex, morse_node=0)
    # # Check if the Morse set consists of a single cell
    # if len(vertices) != 1:
    #     return False
    # Get set of cells in Morse set at initial parameter
    initial_morse_set = {v for v in vertices}
    for par_node in path[1:]:
        # Check if initial_morse_set is a Morse set at par_node
        if not is_morse_set(par_node, initial_morse_set, parameter_graph, level):
            return False
    return True

def non_trivial_path(path, parameter_graph, level=4):
    """Check if a path is non-trivial (non-constant)"""
    for par_index in path:
        parameter = parameter_graph.parameter(par_index)
        morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter, level=level)
        if len(morse_graph.vertices()) > 1:
            return True
    return False

def nontrivial_continuation_paths(cont_paths, parameter_graph, level=4, verbose=True):
    """Find all non-trivial continuation paths"""
    t_start = time.perf_counter()
    nontrivial_cont_paths = {}
    for start_node in cont_paths:
        non_trivial = lambda path: non_trivial_path(path, parameter_graph, level=level)
        nontrivial_cont_paths[start_node] = [path for path in cont_paths[start_node] if non_trivial(path)]
    t_end = time.perf_counter()
    t_elapsed = t_end - t_start
    num_nontrivial_cont_paths = sum(len(nontrivial_cont_paths[node]) for node in nontrivial_cont_paths)
    if verbose:
        print('Time to compute non-trivial continuation paths:', t_elapsed / 60, 'minutes')
        print('Number of nontrivial continuation paths:', num_nontrivial_cont_paths)
    return nontrivial_cont_paths

def compute_continuation_paths(parameter_graph, level=4, verbose=True):
    """Compute continuation paths starting at nodes with no active edges"""
    # Get a dictionary of parameters active edges
    t_start = time.perf_counter()
    param_active_edges = defaultdict(list)
    for par_index in range(parameter_graph.size()):
        num_active = DSGRN_utils.number_active_edges(par_index, parameter_graph)
        param_active_edges[num_active].append(par_index)
    t_end = time.perf_counter()
    t_elapsed = t_end - t_start
    if verbose:
        print('Time to compute dictionary of active edges:', t_elapsed, 'seconds')
    # Compute all paths starting at nodes with no active edges
    t_start = time.perf_counter()
    starting_nodes = param_active_edges[0]
    all_paths = DSGRN_utils.get_all_paths(starting_nodes, parameter_graph)
    t_end = time.perf_counter()
    t_elapsed = t_end - t_start
    if verbose:
        print('Time to compute all paths:', t_elapsed, 'seconds')
    # Get all continuation paths (i.e., Morse set is maintained)
    t_start = time.perf_counter()
    # Check if valid continuation path, i. e., a single Morse set at
    # the initial node of the path and that Morse set is a Morse set
    # for all parameter nodes along the path
    valid = lambda path: DSGRN_utils.continuation_path(path, parameter_graph, level=level)
    continuation_paths = {}
    for start_node in all_paths:
        continuation_paths[start_node] = [path for path in all_paths[start_node] if valid(path)]
    t_end = time.perf_counter()
    t_elapsed = t_end - t_start
    if verbose:
        print('Time to compute continuation paths:', t_elapsed / 60, 'minutes')
    # Get number of paths
    num_paths = sum(len(all_paths[node]) for node in all_paths)
    # Get number of continuation paths
    num_cont_paths = sum(len(continuation_paths[node]) for node in continuation_paths)
    if verbose:
        print('Total number of paths:', num_paths)
        print('Number of continuation paths:', num_cont_paths)
    return continuation_paths, all_paths, param_active_edges

def param_stability(param_node, parameter_graph, level=4):
    """Return number of stable Morse graph nodes at parameter node param_node"""
    parameter = parameter_graph.parameter(param_node)
    morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter, level=level)
    stable_nodes = [v for v in morse_graph.vertices() if not morse_graph.adjacencies(v)]
    return len(stable_nodes)

def cont_paths_from_node(start_node, parameter_graph, max_path_len, max_stability, level=4, max_weight=None):
    """Get all continuation paths with nondecreasing node weights starting at
       start_node, where the weight of a node is the number of active edges.
    """
    dim = parameter_graph.network().size()
    # Set max weight if not given
    if max_weight == None:
        max_weight = sum(max(len(parameter_graph.network().inputs(k)), 1) for k in range(dim))
    # Compute Morse graph for starting parameter node
    parameter = parameter_graph.parameter(start_node)
    morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter, level=level)
    # Check if Morse graph contains a single node
    if len(morse_graph.vertices()) != 1:
        return []
    # Get set of cells in the Morse set corresponding to the Morse node
    vertices, edges = DSGRN_utils.MorseComponent(morse_graph, stg, graded_complex, morse_node=0)
    # Get set of cells in Morse set at starting parameter
    start_morse_set = {v for v in vertices}
    # Check if increasing weight, i. e., if weight > current weight
    increasing_w = lambda node, curr_weight: DSGRN_utils.number_active_edges(node, parameter_graph) > curr_weight
    # Check if non-decreasing stability, i. e., stability >= current stability
    increasing_s = lambda node, curr_stability: param_stability(node, parameter_graph, level=level) >= curr_stability
    # Check if valid target node and multi-stable
    target_multi = lambda node: target(node) and param_stability(node, parameter_graph, level=level) > 1
    # Check if taget_multi and increasing_s
    target_multi_inc = lambda node, curr_stability: target_multi(node) and increasing_s(node, curr_stability)
    # Check if allowed next node, i. e., if increasing or target_multi_inc
    allowed = lambda node, curr_weight, curr_stability: increasing_w(node, current_weight) or target_multi_inc(node, curr_stability)
    # # Check if allowed next node, i. e., if increasing or target_multi
    # allowed = lambda node, current_weight: increasing(node, current_weight) or target_multi(node)
    # # Check if allowed next node, i. e., if weight >= current weight
    # allowed = lambda node, current_weight: DSGRN_utils.number_active_edges(node, parameter_graph) >= current_weight
    # Check if valid target node, i. e., if weight >= max weight
    target = lambda node: DSGRN_utils.number_active_edges(node, parameter_graph) >= max_weight
    # Stack with tuples (current_node, current_path)
    paths_stack = [(start_node, [start_node])]
    continuation_paths = []
    while paths_stack:
        current_node, current_path = paths_stack.pop()
        if len(current_path) == max_path_len:
            continuation_paths.append(current_path)
            # Print info if path reaches max_path_lenght
            # if param_stability(current_node, parameter_graph, level=level) > 2:
            #     print(len(current_path), param_stability(current_node, parameter_graph, level=level), current_path)
            continue
        current_stability = param_stability(current_node, parameter_graph, level=level)
        if current_stability == max_stability:
            continuation_paths.append(current_path)
            # Print info if reaches max_stability
            print(len(current_path), current_stability, current_path)
            continue
        current_weight = DSGRN_utils.number_active_edges(current_node, parameter_graph)
        adjacencies = parameter_graph.adjacencies(current_node, type='codim1')
        for node in [node for node in adjacencies if allowed(node, current_weight, current_stability)]:
            # Check if initial_morse_set is a valid a Morse set at this parameter node
            valid_morse_set = DSGRN_utils.is_morse_set(node, start_morse_set, parameter_graph, level)
            # If not a valid Morse set or previlously visited node
            if not valid_morse_set or node in current_path:
                # If valid target
                if target(current_node):
                    continuation_paths.append(current_path)
                    # if current_stability > 2:
                    #     print(len(current_path), current_stability, current_path)
                continue
            # Extend path and add to stack
            paths_stack.append((node, current_path + [node]))
    return continuation_paths

def get_all_cont_paths(starting_nodes, parameter_graph, max_path_len, max_stability, level=4, max_weight=None):
    """Get all paths starting at a node in starting_nodes"""
    cont_paths = {}
    for start_node in tqdm(starting_nodes):
        cont_paths[start_node] = cont_paths_from_node(start_node, parameter_graph, max_path_len, max_stability, level, max_weight=max_weight)
    return cont_paths
