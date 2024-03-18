from nltk.corpus import wordnet as wn
import networkx as nx
import numpy as np
import warnings


def find_hypernym_paths(hypo_source, hyper_target, with_repeats=False):
    open_list = hypo_source.hypernyms()
    open_depth = [0 for x in open_list]
    closed_list = []
    current_path = []
    all_paths = []
    current_node = None

    while open_list:
        # Read first from open list and add to current path
        current_node = open_list.pop()
        closed_list.append(current_node)
        current_depth = open_depth.pop()
        current_path = current_path[0:current_depth] + [current_node]
        if current_node == hyper_target:
            # If reached the target add the current path to the found ones
            all_paths.append([hypo_source] + current_path)
        else:
            # Else expand node
            current_hyper = current_node.hypernyms()
            for hyper in current_hyper:
                if (hyper not in closed_list and hyper not in open_list) or with_repeats:
                    open_list.append(hyper)
                    open_depth.append(len(current_path))

    return all_paths


def dict_filter_synset_path(path: list, dict: list):
    """
    Given a synsets path (as list) and a dictionary returns a new path which is a subset of the original path and such
    that each synset in the path has at least one lemma contained in the dictionary.

    Arguments:
    path -- the original synset path
    dict -- the dictionary
    """
    def lemma_in_dict(synset):
        for l in synset.lemmas():
            if l.name() in dict:
                return True
        return False

    return [syn for syn in path if lemma_in_dict(syn)]


def load_txt_dict(txt_path):
    with open(txt_path, 'r') as f:
        words = f.readlines()
    words = [x.strip() for x in words]
    return words


def create_graph_to(dict, target_syn=None, filter=True, with_repeats=False, filter_nonequal_lemmas=False):
    """
    dict : Union[str, list]. If str, is a path to a dictionary file, if a list, is a dictionary
    """
    if target_syn is None:
        target_syn = wn.synsets('object', pos=wn.NOUN)[0]
    hierarchy = nx.DiGraph()
    if isinstance(dict, str):
        dict_all = load_txt_dict(dict)
    else:
        dict_all = dict
    node_set = set()
    paths = []

    # Find all paths from all synsets of dict_all words to 'object'
    for word in dict_all:
        word_synsets = wn.synsets(word, pos=wn.NOUN)
        for syn in word_synsets:
            # Don't proceed if filter_nonequal_lemmas and the syn name root is not equal to the word
            syn_name_root = syn.name().split('.')[0]
            if filter_nonequal_lemmas and syn_name_root != word:
                continue
            paths += find_hypernym_paths(syn, target_syn, with_repeats)
    if filter:
        filtered_paths = [dict_filter_synset_path(x, dict_all) for x in paths]
    else:
        filtered_paths = paths
    # Create nodes in the graph for each synset found that is in relation with a word in the dict
    for path in filtered_paths:
        for syn in path:
            node_set.update([syn])
    # Create edges based on filtered paths
    for path in filtered_paths:
        for idx in range(1, len(path)):
            hierarchy.add_edge(path[idx], path[idx - 1])
    # Remove isolated edges (even though there should be none)
    hierarchy.remove_nodes_from(list(nx.isolates(hierarchy)))
    return hierarchy


def is_word_in_wordnet(word):
    word_wn = wn.synsets(word, pos=wn.NOUN)
    if not word_wn:
        return False
    else:
        return True


def has_path_to(word, target_synset):
    word_wn = wn.synsets(word, pos=wn.NOUN)
    for syn in word_wn:
        if find_hypernym_paths(syn, target_synset):
            return True
    return False


def hierarchy_path_to_root(node, hierarchy):
    """
    Given a hierarchy tree, returns all paths from node to the tree root
    :param node:
    :param hierarchy:
    :return:
    """
    open_list = [node]
    open_depth = [0 for x in open_list]
    closed_list = []
    current_path = []
    all_paths = []
    current_node = None

    while open_list:
        # Read first from open list and add to current path
        current_node = open_list.pop()
        closed_list.append(current_node)
        current_depth = open_depth.pop()
        current_path = current_path[0:current_depth] + [current_node]

        try:
            # This is needed if you have an unconnected node to root due to a hypernym not leading to object.n.01
            current_hyper = [a for a,b in hierarchy.in_edges(current_node)]
        except nx.NetworkXError:
            continue
        if not current_hyper:
            all_paths.append(current_path)
        else:
            for hyper in current_hyper:
                open_list.append(hyper)
                open_depth.append(len(current_path))

    return all_paths


def hierarchy_root_to_lemma_tree(lemma: str, hierarchy, filter_nonequal_lemmas=False):
    tree = nx.DiGraph()
    word_wn = wn.synsets(lemma, pos=wn.NOUN)
    for syn in word_wn:
        # Don't proceed if filter_nonequal_lemmas and the syn name root is not equal to the word
        syn_name_root = syn.name().split('.')[0]
        if filter_nonequal_lemmas and syn_name_root != lemma:
            continue
        paths = hierarchy_path_to_root(syn, hierarchy)
        for path in paths:
            for node in path:
                if not tree.has_node(node):
                    tree.add_node(node)
            for i in range(1,len(path)):
                tree.add_edge(path[i], path[i-1])
    return tree


def similarity_tree_visit_count_decorator(tree: nx.DiGraph):
    for node in tree.nodes():
        nx.set_node_attributes(tree, {node: {"visit_count": 0}})
    return tree


def similarity_tree_update_counts(similarity_tree: nx.DiGraph, new_tree: nx.DiGraph, concept):
    for node in new_tree.nodes():
        if similarity_tree.has_node(node):
            current_count = similarity_tree.nodes[node]['visit_count']
            nx.set_node_attributes(similarity_tree, {node: {"visit_count": current_count+1}})
    return similarity_tree


def get_subtree_set(tree: nx.DiGraph, node):
    """
    Returns the set of nodes from the subtree starting from node
    :param tree:
    :param node:
    :return:
    """
    # Compute subtree with associated depth
    open_set = [node]
    closed_set = []
    score_list = []
    while open_set:
        current_node = open_set.pop(0)
        if current_node in closed_set:
            continue
        open_set = open_set + [b for a,b in tree.edges(current_node)]
        closed_set = closed_set + [current_node]
    closed_set = set(closed_set)
    return closed_set


def similarity_tree_subtree_score_decorator(tree: nx.DiGraph, aggregator=None, attribute="concept_similarity"):
    """
    Adds to each node a list which is the similarity for all nodes in the subtree starting from that node, max-aggregated by depth
    :param tree:
    :param aggregator: if None leaves it as a list, if callable
    :return:
    """
    for node in tree.nodes():
        subtree = get_subtree_set(tree, node)
        # TODO is this the best way to remove duplicates? Should I check whether there is a chance for them to have same similarity with different words?
        if aggregator is None:
            scores = list(set([tree.nodes[x][attribute] for x in list(subtree)]))
        else:
            scores = aggregator([tree.nodes[x][attribute] for x in list(subtree)])
        # scores = get_subtree_depthwise_scores(tree, node)
        nx.set_node_attributes(tree, {node: {"subtree_score": scores}})
    return tree


def score_list_best_mean(scores):
    means = [np.mean(x) for x in scores]
    return np.argmax(means)


def find_tree_leaves(tree, root):
    open_list = [root]
    closed_list = []
    leaves = []
    while open_list:
        current_node = open_list.pop(0)
        if current_node in closed_list:
            continue
        closed_list.append(current_node)
        expanded = [b for a,b in tree.out_edges(current_node)]
        if len(expanded) == 0:
            leaves.append(current_node)
        open_list = open_list + expanded
    return leaves


def exist_close_elements(array, epsilon):
    x = np.reshape(array, (len(array), 1))
    diffs = np.abs(x - x.transpose())
    np.fill_diagonal(diffs, epsilon+1.0)
    if np.min(diffs) < epsilon:
        return True
    return False


def score_list_ranking(scores, epsilon=0.002):
    """
    Given a list of lists representing the path scores for all children nodes, returns the index of the child that has the highest non-common maximum
    :param scores:
    :return:
    """

    counts = len(scores)
    if counts == 1:
        return 0

    lengths = [len(x) for x in scores]
    max_length = max(lengths)
    padded_lists = [scores[i] + [0]*(max_length-lengths[i]) for i in range(counts)]
    array_scores = np.array(padded_lists)
    array_scores = np.sort(array_scores, axis=1)[:,::-1]

    unique = np.unique(array_scores[:,0]).size
    duplicate_maxima = exist_close_elements(array_scores[:,0], epsilon)
    while duplicate_maxima:  # unique != counts:
        # print(f"removing non-unique maximum: {np.max(array_scores[:,0])}")
        array_scores = np.delete(array_scores, 0, 1)
        try:
            # unique = np.unique(array_scores[:,0]).size
            duplicate_maxima = exist_close_elements(array_scores[:,0], epsilon)
        except IndexError:
            # All the maxima are the same. Return the first and warn
            warnings.warn("Identical maxima found, returning first")
            return 0

    return np.argmax(array_scores[:,0])


def similarity_tree_find_best_leaf(tree: nx.DiGraph, source_synset, aggregator="ranking", similarity_attribute="concept_similarity"):

    if aggregator == "max_leaf":
        leaves = find_tree_leaves(tree, source_synset)
        similarities = [tree.nodes[x]['concept_similarity'] for x in leaves]
        return leaves[np.argmax(similarities)]

    max_node = source_synset
    out_nodes = [b for a,b in tree.edges(max_node)]
    # As long as there are children
    while out_nodes:
        # Get lists from node attributes
        path_score_lists = []
        children_similarities = []
        for node in out_nodes:
            try:
                path_score_lists.append(tree.nodes[node]["subtree_score"])
            except KeyError:
                pass
            children_similarities.append(tree.nodes[node][similarity_attribute])
        # Determine winner
        if aggregator == "ranking":
            max_idx = score_list_ranking(path_score_lists)
        elif aggregator == "mean":
            max_idx = score_list_best_mean(path_score_lists)
        elif aggregator == "depthwise_max":
            max_idx = np.argmax(children_similarities)
        else:
            raise ValueError("Invalid aggregator")
        max_node = out_nodes[max_idx]
        out_nodes = [b for a,b in tree.edges(max_node)]

    return max_node