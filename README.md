# CS 365 Lab C: DecisionTree
# Huong Phan

1. Display tree in a text file and get number of nodes in the tree
python main.py display_tree <file name>
e.g. python main.py display_tree tennis.txt
The tree is printed into a text file, whose name is 'decision_tree_' + file name (e.g. decision_tree_tennis.txt)

2. Get training set accuracy
python main.py training_set_accuracy <file name>
e.g. python main.py training_set_accuracy pets.txt

3. Get test set accuracy
python main.py accuracy_testing <file name>
e.g. python main.py accuracy_testing tennis.txt

Note: for training set accuracy and test set accuracy, please don't run it with titanic.txt as it takes very long to run.