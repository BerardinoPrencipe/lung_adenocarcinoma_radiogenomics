import numpy as np
from utils.geometric import getEuclidanDistance

class TreeNode(object):
    """
    Tree node Class
    """

    branches = []

    def __init__(self, name='root', children=None, information_struct=None, label='root'):
        self.name = name
        self.information_struct = information_struct
        self.children = []
        self.label = label
        self.depth = 0
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def get_string(self):
        repr_string = "{:10s} : {}\n".format('Node', self.name)
        for x in self.information_struct:
            repr_string += "{:10s} : {}\n".format(x, self.information_struct[x])
        return repr_string

    def add_child(self, node):
        assert isinstance(node, TreeNode)
        self.children.append(node)

    def __str__(self, level=0, limit=None):
        ret = "  "*level+self.name+"\n"
        for child in self.children:
            if limit is None or level+1 <= limit:
                ret += child.__str__(level+1, limit=limit)
        return ret

    def get_string_label(self, level=0, limit=None):
        ret = "  " * level + self.label + "\n"
        for child in self.children:
            if limit is None or level + 1 <= limit:
                ret += child.get_string_label(level + 1, limit=limit)
        return ret

    def get_n_children(self):
        return len(self.children)

    def get_recursively_n_children(self):
        n_children = self.get_n_children()
        for child in self.children:
            n_children += child.get_recursively_n_children()
        return n_children

    def set_recursively_depth(self):
        for child in self.children:
            child.depth = self.depth+1
            child.set_recursively_depth()

    def get_max_depth(self):
        max_depth = self.depth
        for child in self.children:
            new_depth = child.get_max_depth()
            if new_depth > max_depth:
                max_depth = new_depth
        return max_depth

    def label_tree_from_branches(self):
        for child in self.children:
            child.recursively_assign_label(child.name)

    def recursively_assign_label(self, label):
        self.label = label
        for child in self.children:
            child.recursively_assign_label(label)

    # TODO
    def recursively_find_branches_tree(self, threshold_new_branch=6, do_init=False):
        if do_init:
            self.init_recursive_label()

        if self.label != 'main_branch':
            print('It is not main branch! {}'.format(self.label))
            return

        if len(self.children) >= 2:
            depth_max_childrens = []
            for child in self.children:
                depth_max_children = child.get_max_depth()
                depth_max_childrens.append(depth_max_children)

            amax_depth = np.argmax(depth_max_childrens)

            for idx, child in enumerate(self.children):
                n_tot_children = child.get_recursively_n_children()
                depth_children = child.get_max_depth()
                if idx == amax_depth:
                    # MAIN BRANCH
                    child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
                else:
                    if n_tot_children > threshold_new_branch:
                        print('Branch {} created!'.format(child.name))
                        print('Depth Children = {:3d}\tN Tot Children = {}'.format(depth_children, n_tot_children))
                        TreeNode.branches.append(child)
                        child.recursively_assign_label(child.name)
                        child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
                    else:
                        child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
        elif len(self.children) == 1:
            child = self.children[0]
            child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
        else:
            pass
        return TreeNode.branches

    ''' 
    def recursively_find_branches_tree(self, threshold_new_branch=6, do_init=False):
        if do_init:
            self.init_recursive_label()

        if len(self.children) >= 2:
            depth_max_childrens = []
            for child in self.children:
                depth_max_children = child.get_max_depth()
                depth_max_childrens.append(depth_max_children)

            amax_depth = np.argmax(depth_max_childrens)

            for idx, child in enumerate(self.children):
                n_tot_children = child.get_recursively_n_children()
                depth_children = child.get_max_depth()
                if idx != amax_depth and n_tot_children > threshold_new_branch:
                    print('Branch {} created!'.format(child.name))
                    print('Depth Children = {:3d}\tN Tot Children = {}'.format(depth_children, n_tot_children))
                    TreeNode.branches.append(child)
                    child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
                else:
                    child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
        elif len(self.children) == 1:
            child = self.children[0]
            child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
        else:
            pass
        return TreeNode.branches
    '''

    def init_recursive_label(self):
        TreeNode.branches = []
        TreeNode.branches.append(self)
        # self.recursively_assign_label(self.name)
        self.recursively_assign_label('main_branch')

class TreeNodesContainer(object):
    def __init__(self, nodes=None):
        self.nodes = []
        if nodes is not None:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node):
        assert isinstance(node, TreeNode)
        self.nodes.append(node)

    def get_nodes_of_layer(self, layer_to_cmp):
        nodes_of_layer = list()
        if layer_to_cmp not in self.get_layers():
            return None
        for node in self.nodes:
            _, layer, _ = node.name.split('_')
            if layer == layer_to_cmp:
                nodes_of_layer.append(node)
        return nodes_of_layer

    def get_layers(self):
        layers = list()
        for node in self.nodes:
            _, layer, _ = node.name.split('_')
            if layer not in layers:
                layers.append(layer)
        return layers

    def get_centroids_for_nodes_of_layer(self, layer_to_cmp):
        return self.get_key_for_nodes_of_layer(layer_to_cmp, 'centroid')

    def get_key_for_nodes_of_layer(self, layer_to_cmp, key):
        nodes_of_layer = self.get_nodes_of_layer(layer_to_cmp)
        values = list()
        for node in nodes_of_layer:
            values.append(node.information_struct[key])
        return values

    def build_distance_matrix_between_layers(self, layer1, layer2):
        centroids_layer_1 = self.get_centroids_for_nodes_of_layer(layer1)
        centroids_layer_2 = self.get_centroids_for_nodes_of_layer(layer2)

        distance_matrix = np.zeros((len(centroids_layer_1), len(centroids_layer_2)))

        for idx_ls, centroid_ls in enumerate(centroids_layer_1):
            for idx_sls, centroid_sls in enumerate(centroids_layer_2):
                distance_matrix[idx_ls, idx_sls] = getEuclidanDistance(centroids_layer_1[idx_ls],
                                                                       centroids_layer_2[idx_sls])

        return distance_matrix

    def get_node_from_name(self, node_name):
        for node in self.nodes:
            if node.name == node_name:
                return node
        return None

    def link_two_nodes(self, parent_node_name, child_node_name):
        parent_node = self.get_node_from_name(parent_node_name)
        child_node  = self.get_node_from_name(child_node_name)
        if parent_node is not None and child_node is not None:
            parent_node.add_child(child_node)


def label_tree_from_branches(branches):
    for branch in branches:
        branch.recursively_assign_label(branch.name)


''' 
 # TODO: give a label to each node of the tree
    def recursively_find_branches_tree_OLD(self, threshold_new_branch=6, do_init=False):
        if do_init:
            self.init_recursive_label()

        # TreeNode.branches.append(self)
        if len(self.children) >= 2:
            n_tot_childrens = []
            depth_max_childrens = []
            for child in self.children:
                n_tot_children = child.get_recursively_n_children()
                n_tot_childrens.append(n_tot_children)

                depth_max_children = child.get_max_depth()
                depth_max_childrens.append(depth_max_children)

            amax = np.argmax(n_tot_childrens)
            max_children = np.max(n_tot_childrens)

            amax_depth = np.argmax(depth_max_childrens)
            max_children_depth = np.max(depth_max_childrens)

            for child in self.children:
                depth_children = child.get_max_depth()
                n_tot_children = child.get_recursively_n_children()
                # if n_tot_children > threshold_new_branch and n_tot_children != max_children:
                # if depth_children > threshold_new_branch and depth_children != max_children_depth:
                if n_tot_children > threshold_new_branch and depth_children != max_children_depth:
                    print('Branch {} created!'.format(child.name))
                    print('Depth Children = {:3d}\tN Tot Children = {}'.format(depth_children, n_tot_children))
                    # child.recursively_assign_label(child.name)
                    TreeNode.branches.append(child)
                    child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
                else:
                    child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
        elif len(self.children) == 1:
            child = self.children[0]
            child.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch)
        else:
            pass
            # print('No children for {} node'.format(self.name))
        return TreeNode.branches
'''