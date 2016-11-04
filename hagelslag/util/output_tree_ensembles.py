#!/usr/bin/env python
"""
Read a scikit-learn tree ensemble object and output the object into a human-readable text format.
"""
import pickle
import argparse


__author__ = "David John Gagne <djgagne@ou.edu>"
__copyright__ = "Copyright 2015, David John Gagne"
__email__ = "djgagne@ou.edu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input scikit-learn tree ensemble model pickle file.")
    parser.add_argument("-o", "--output", required=True, help="Output file start.")
    parser.add_argument("-a", "--attr", default="", help="Attribute list file.")
    args = parser.parse_args()
    tree_ensemble_obj = load_tree_object(args.input)
    if args.attr != "":
        with open(args.attr) as attr_obj:
            attribute_names = attr_obj.readlines()
    else:
        attribute_names = None
    output_tree_ensemble(tree_ensemble_obj, args.output, attribute_names)
    return


def load_tree_object(filename):
    """
    Load scikit-learn decision tree ensemble object from file.
    
    Parameters
    ----------
    filename : str
        Name of the pickle file containing the tree object.
    
    Returns
    -------
    tree ensemble object
    """
    with open(filename) as file_obj:
        tree_ensemble_obj = pickle.load(file_obj)
    return tree_ensemble_obj


def output_tree_ensemble(tree_ensemble_obj, output_filename, attribute_names=None):
    """
    Write each decision tree in an ensemble to a file.

    Parameters
    ----------
    tree_ensemble_obj : sklearn.ensemble object
        Random Forest or Gradient Boosted Regression object
    output_filename : str
        File where trees are written
    attribute_names : list
        List of attribute names to be used in place of indices if available.
    """
    for t, tree in enumerate(tree_ensemble_obj.estimators_):
        print("Writing Tree {0:d}".format(t))
        out_file = open(output_filename + ".{0:d}.tree", "w")
        #out_file.write("Tree {0:d}\n".format(t))
        tree_str = print_tree_recursive(tree.tree_, 0, attribute_names)
        out_file.write(tree_str)
        #out_file.write("\n")
        out_file.close()
    return


def print_tree_recursive(tree_obj, node_index, attribute_names=None):
    """
    Recursively writes a string representation of a decision tree object.

    Parameters
    ----------
    tree_obj : sklearn.tree._tree.Tree object
        A base decision tree object
    node_index : int
        Index of the node being printed
    attribute_names : list
        List of attribute names
    
    Returns
    -------
    tree_str : str
        String representation of decision tree in the same format as the parf library.
    """
    tree_str = ""
    if node_index == 0:
        tree_str += "{0:d}\n".format(tree_obj.node_count)
    if tree_obj.feature[node_index] >= 0:
        if attribute_names is None:
            attr_val = "{0:d}".format(tree_obj.feature[node_index])
        else:
            attr_val = attribute_names[tree_obj.feature[node_index]]
        tree_str += "b {0:d} {1} {2:0.4f} {3:d} {4:1.5e}\n".format(node_index,
                                                                   attr_val,
                                                                   tree_obj.weighted_n_node_samples[node_index],
                                                                   tree_obj.n_node_samples[node_index],
                                                                   tree_obj.threshold[node_index])
    else:
        if tree_obj.max_n_classes > 1:
            leaf_value = "{0:d}".format(tree_obj.value[node_index].argmax())
        else:
            leaf_value = "{0}".format(tree_obj.value[node_index][0][0])
        tree_str += "l {0:d} {1} {2:0.4f} {3:d}\n".format(node_index,
                                                          leaf_value,
                                                          tree_obj.weighted_n_node_samples[node_index],
                                                          tree_obj.n_node_samples[node_index])
    if tree_obj.children_left[node_index] > 0:
        tree_str += print_tree_recursive(tree_obj, tree_obj.children_left[node_index], attribute_names)
    if tree_obj.children_right[node_index] > 0:
        tree_str += print_tree_recursive(tree_obj, tree_obj.children_right[node_index], attribute_names)
    return tree_str

if __name__ == "__main__":
    main()
