from DecisionTree import *
from sys import argv

def main(function, filename):
    try:
        if argv[1] == "display_tree":
            function = display_tree
        elif argv[1] == "training_set_accuracy":
            function = training_set_accuracy
        elif argv[1] == "accuracy_testing":
            function = accuracy_testing
    except TypeError:
        print("Please enter the available function names: display_tree, training_set_accuracy, accuracy_testing\n") 
    try:
        if argv[2] == "tennis.txt":
            filename = argv[2]
        elif argv[2] == "titanic2.txt":
            filename = argv[2]
        elif argv[2] == "pets.txt":
            filename = argv[2]
    except TypeError:
        print("Please enter the available file names: tennis.txt, titanic2.txt, pets.txt\n")
    print(function(filename))

if __name__ == '__main__':
    main(argv[1], argv[2])
