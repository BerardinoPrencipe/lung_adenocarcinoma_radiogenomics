import numpy as np
from util.geometric import getEuclidanDistance

class TreeNode(object):
    "Tree node"
    def __init__(self, name='root', children=None, information_struct=None, label='root'):
        self.name = name
        self.information_struct = information_struct
        self.children = []
        self.label = label
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

    def __str__(self, level=0):
        ret = "  "*level+self.name+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def get_n_children(self):
        return len(self.children)

    def get_recursively_n_children(self):
        n_children = self.get_n_children()
        for child in self.children:
            n_children += child.get_recursively_n_children()
        return n_children

    def recursively_assign_label(self, label):
        self.label = label
        for child in self.children:
            child.label = label
            child.recursively_assign_label(label)

    # TODO: give a label to each node of the tree
    def recursively_label_tree(self, threshold_new_branch=6):
        branches = list()
        branches.append(self)
        for child in self.children:
            if len(self.children) > 2:
                n_tot_children = child.get_recursively_n_children()
                if n_tot_children > threshold_new_branch:
                    child.recursively_assign_label(child.name)
                    print('Branch {} created!'.format(child.name))
                    branches.append(child)
            child.recursively_label_tree(threshold_new_branch=threshold_new_branch)
        return branches

class TreeVesselContainer(object):
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