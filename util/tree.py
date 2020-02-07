class Tree(object):
    "Tree node"
    def __init__(self, name='root', children=None, information_struct=None):
        self.name = name
        self.information_struct = information_struct
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        repr_string = "{:10s} : {}\n".format('Node', self.name)
        for x in self.information_struct:
            repr_string += "{:10s} : {}\n".format(x, self.information_struct[x])
        return repr_string

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)