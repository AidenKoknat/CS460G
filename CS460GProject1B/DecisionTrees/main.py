# TO DO:
    # Edit tree data so each branch shows 0 or 1
    # Presentation

import csv
import math
import numpy as np
from anytree import Node, RenderTree
import matplotlib.pyplot as plt


def file_to_data(file_name):
    data1 = []
    file = file_name
    with open(file, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            point = []
            for value in row:
                tuple = [value, 0]
                point.append(tuple)
            data1.append(point)
        return data1


def findFloorAndCeiling(data, feature):
    minimum = 0
    maximum = 0
    for point in range(len(data)):
        if float(data[point][feature][0]) < minimum:
            minimum = float(data[point][feature][0])
        if float(data[point][feature][0]) > maximum:
            maximum = float(data[point][feature][0])
    if (math.ceil(maximum - minimum)) % 2 == 1:
        maximum = maximum + 1
    CategoryFloor = int(math.floor(minimum))
    CategoryCeiling = int(math.ceil(maximum))
    return CategoryFloor, CategoryCeiling


def discretize_data(Data):
    AmountOfFeatures = len(Data[0]) - 1

    for feature in range(0, AmountOfFeatures, 1):
        CategoryFloor, CategoryCeiling = findFloorAndCeiling(Data, feature)
        for point in range(len(Data)):
            CategoryScore = 0
            for categoryRange in range(CategoryFloor, CategoryCeiling, 2):
                if float(Data[point][feature][0]) > categoryRange and float(Data[point][feature][0]) < categoryRange + 2:
                    Data[point][feature][1] = CategoryScore
                else:
                    CategoryScore = CategoryScore + 1
    return Data


def discretize_point(Data, point):
    AmountOfFeatures = len(Data[0]) - 1

    for feature in range(0, AmountOfFeatures, 1):
        CategoryFloor, CategoryCeiling = findFloorAndCeiling(Data, feature)
        CategoryScore = 0
        for categoryRange in range(CategoryFloor, CategoryCeiling, 2):
            if float(point[feature][0]) > categoryRange and float(point[feature][0]) < categoryRange + 2:
                point[feature][1] = CategoryScore
            else:
                CategoryScore = CategoryScore + 1
    return point


# For use after data is already discretized
def find_discretized_category_amount(data, feature):
    maximum = 0
    for point in range(len(data)):
        if data[point][feature][1] > maximum:
            maximum = int(data[point][feature][1])
    maximum = maximum + 1
    return maximum


def calculate_total_entropy(data, discretized_feature):
    TotalEntropy = 0
    Floor, Ceiling = findFloorAndCeiling(data, discretized_feature)
    number_of_categories = Ceiling

    for category in range(int(number_of_categories)):
        AmountOne = 0
        AmountZero = 0
        for point in range(len(data)):
            if int(data[point][discretized_feature][1]) == category:
                if int(data[point][-1][0]) == 1:
                    AmountOne = AmountOne + 1
                if int(data[point][-1][0]) == 0:
                    AmountZero = AmountZero + 1
        TotalPoints = AmountOne + AmountZero

        if TotalPoints == 0:
            # print("No points in group: " + str(category))
            Entropy = 0
        else:
            if AmountOne == TotalPoints or AmountZero == TotalPoints:
                Entropy = 0
            else:
                Entropy = - (AmountOne / TotalPoints) * math.log((AmountOne / TotalPoints), 2) - (AmountZero / TotalPoints) * math.log((AmountZero / TotalPoints), 2)
        # print("Entropy of category " + str(category) + ": " + str(Entropy))
        TotalEntropy = TotalEntropy + (Entropy * (TotalPoints / len(data)))
    return TotalEntropy


def calculate_information_gain(data, usedFeature = None) -> int:
    OverallOnes = 0
    OverallZeros = 0
    OverallEntropy = 0
    InformationGain = []
    for point in range(len(data)):
        if int(data[point][-1][0]) == 1:
            OverallOnes = OverallOnes + 1
        if int(data[point][-1][0]) == 0:
            OverallZeros = OverallZeros + 1
    OverallTotal = OverallOnes + OverallZeros
    if OverallTotal == 0:
        print("error in grouping")
    else:
        if OverallOnes == OverallTotal or OverallZeros == OverallTotal:
            OverallEntropy = 0
        else:
            OverallEntropy = - (OverallOnes / OverallTotal) * math.log((OverallOnes / OverallTotal), 2) - (OverallZeros / OverallTotal) * math.log((OverallZeros / OverallTotal), 2)
    for feature in range(len(data[0]) - 1):
        Entropy = calculate_total_entropy(data, feature)
        InformationGain.append(OverallEntropy - Entropy)
        # print("Information Gain for feature " + str(feature) + ": " + str(InformationGain[feature]))

    maxInformationGain = float(0)
    featureToBeUsed = 0
    for feature in range(len(InformationGain)):
        # print(feature)
        if feature != usedFeature:
            if InformationGain[feature] >= maxInformationGain:
                featureToBeUsed = feature
                maxInformationGain = InformationGain[feature]
    # print("The feature that will yield the most Information Gain is: feature " + str(featureToBeUsed))
    return featureToBeUsed


def split_by_feature(data, feature):
    subsetHolder = []
    if not data:
        #print(subsetHolder[feature])
        return
    else:
        categoryAmount = find_discretized_category_amount(data, feature)
        for category in range(categoryAmount):
            subset = []  # Check if resets to zero
            for point in range(len(data)):
                if data[point][feature][1] == category:
                    subset.append(data[point])
            if subset:
                subsetHolder.append(subset)
        return subsetHolder


def get_splits_data(discretized_data):
    firstInformationGain = calculate_information_gain(discretized_data)
    firstSplit = split_by_feature(discretized_data, firstInformationGain)
    secondSplit = []
    # print(str(len(firstSplit)))
    for category in range(len(firstSplit)):
        temporarySplit = []
        if len(firstSplit[category]) == 0:
            temporarySplit = []
        else:
            # Takes data from firstSplit[subCategory]
            secondInformationGain = calculate_information_gain(firstSplit[category], firstInformationGain)
            temporarySplit = split_by_feature(firstSplit[category], secondInformationGain)
            secondSplit.append(temporarySplit)
    return firstSplit, firstInformationGain, secondSplit, secondInformationGain


# Tree Generation
def create_tree(discretized_data):

    firstSplit, firstInformationGain, secondSplit, secondInformationGain = get_splits_data(discretized_data)

    root = Node("Root", parent=None, depth=0)
    root.attribute = firstInformationGain # feature it was split on
    for category in range(len(firstSplit)):
        child = Node("category: " + str(category), parent=root, depth=1)
        child.featureValue = category
        child.attribute = secondInformationGain

        for subcategory in range(len(secondSplit[category])):
            #if len(secondSplit[subcategory]) == 0 or len(secondSplit[subcategory][-1]) == 0:
            if secondSplit[category][subcategory]: # Average the 1s/0s to determine class attribute
                classValue = 0
                sum = float(0)
                for points in range(len(secondSplit[category][subcategory])):
                    # add up the points to find average
                    sum = sum + float(secondSplit[category][subcategory][points][-1][0])
                if (sum / len(secondSplit[subcategory])) > .5:
                    classValue = 1
                grandchild = Node(subcategory, parent=child, depth=2)
                grandchild.featureValue = subcategory
                grandchild.classLabel = classValue
            else:
                grandchild = Node(subcategory, parent=child, depth=2)
                grandchild.classLabel = 0
                grandchild.featureValue = subcategory
    # print(RenderTree(root))  # Comment/Uncomment to show tree
    return root


def predict_value(tree, point, data):
    point = discretize_point(data, point)  # discretizes the point
    attribute = tree.attribute  # finds attribute tree was initially split on
    pointFeatureValue = point[attribute][1]  # gets attribute of point
    for child in tree.children:
        if child.featureValue == pointFeatureValue:
            secondFeature = child.attribute
            for grandchild in child.children:
                if grandchild.featureValue == point[secondFeature][1]:
                    # print("predicted value is: " + str(grandchild.classLabel))
                    return grandchild.classLabel
    # print("prediction failed!")
    return -1


def get_accuracy(tree, data):
    accuracySum = float(0)
    accuracyTotal = float(len(data))
    for point in range(len(data)):
        if predict_value(tree, data[point], data) == int(data[point][-1][0]):
            accuracySum = accuracySum + 1
    accuracy = accuracySum / accuracyTotal * float(100)
    print("Accuracy is " + str(int(accuracySum)) + "/" + str(int(accuracyTotal)) + ": " + str(accuracy) + "%")


# Visualizing the data and predicting
def visualize_data(data, secondSplit):
    zeroArray = []
    oneArray = []

    for point in data:
        if point[-1][0] == 0:
            zeroArray.append(point)
        else:
            oneArray.append(point)
    x0 = [int(point[0][1]) for point in zeroArray]
    y0 = [int(point[1][1]) for point in zeroArray]
    x1 = [int(point[0][1]) for point in oneArray]
    y1 = [int(point[1][1]) for point in oneArray]

    figure, axis = plt.subplots()
    figure.suptitle(0)
    axis.set_xlabel("Feature 0")
    axis.set_ylabel("Feature 1")
    plt.xticks(np.arange(0,10,1.0))
    plt.yticks(np.arange(0,10,1.0))
    a = []
    b = []
    category0Amount = find_discretized_category_amount(data, 0)
    category1Amount = find_discretized_category_amount(data, 1)
    for category in range(category0Amount):
        a.append(category)
    for category in range(category1Amount):
        b.append(category)
    z = np.zeros((len(b), len(a)))

    for indexA, ai in enumerate(a):
        if ai < len(secondSplit):
            for indexB, bi in enumerate(b):

                if bi < len(secondSplit[indexA]):
                    try:
                        x = secondSplit[ai][bi][-1][0]
                        while not (secondSplit[ai][bi][-1][0] == '0' or secondSplit[ai][bi][-1][0] == '1'):
                            secondSplit[ai][bi][-1] = secondSplit[ai][bi][-1][-1]
                        z[indexB][indexA] = secondSplit[ai][bi][-1][0]
                    except TypeError:
                        z[indexB][indexA] = secondSplit[ai][bi][-1]
                else:
                    z[indexB][indexA] = '0'  # int(secondSplit[bi][ai][0][0][2])

    axis.pcolormesh(a, b, z, cmap='cool', shading = 'auto')
    axis.scatter(x0, y0, c=[[1,1,1]], marker='$0$')
    axis.scatter(x1, y1, c=[[0,0,0]], marker='$1$')
    plt.show()

# synthetic-1.csv:
data1 = file_to_data("data\synthetic-1.csv")

discretizedData1 = discretize_data(data1)
tree = create_tree(discretizedData1)
firstSplit, firstInformationGain, secondSplit, secondInformationGain = get_splits_data(discretizedData1)
visualize_data(data1, secondSplit)
# End synthetic-1.csv

# synthetic-2.csv:
data2 = file_to_data("data\synthetic-2.csv")

discretizedData2 = discretize_data(data2)
tree2 = create_tree(discretizedData2)
firstSplit2, firstInformationGain2, secondSplit2, secondInformationGain2 = get_splits_data(discretizedData2)
visualize_data(data2, secondSplit2)
# End synthetic-2.csv

# synthetic-3.csv:
data3 = file_to_data("data\synthetic-3.csv")

discretizedData3 = discretize_data(data3)
tree3 = create_tree(discretizedData3)
firstSplit3, firstInformationGain3, secondSplit3, secondInformationGain3 = get_splits_data(discretizedData3)
visualize_data(data3, secondSplit3)
# End synthetic-3.csv

# synthetic-4.csv:
data4 = file_to_data("data\synthetic-4.csv")

discretizedData4 = discretize_data(data4)
tree4 = create_tree(discretizedData4)
firstSplit4, firstInformationGain4, secondSplit4, secondInformationGain4 = get_splits_data(discretizedData4)
visualize_data(data4, secondSplit4)
# End synthetic-4.csv

# pokemon data
pokemonData = []
statFile = "data\pokemonStats.csv"
with open(statFile, 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        point = []
        for value in row:
            tuple = [value, 0]
            point.append(tuple)
        pokemonData.append(point)
legendaryFile = "data\pokemonLegendary.csv"
with open(statFile, 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for index, row in enumerate(csvreader):
        if row == "True":
            tuple = [1, 0]
            point.append(tuple)
            pokemonData[index].append(tuple)
        else:
            tuple = [0, 0]
            point.append(tuple)
            pokemonData[index].append(tuple)
discretizedPokemon = discretize_data(pokemonData)
pokemonTree = create_tree(discretizedPokemon)
firstPokemonSplit, firstPokemonInformationGain, secondPokemonSplit, secondPokemonInformationGain = get_splits_data(discretizedPokemon)
visualize_data(pokemonData, secondPokemonSplit)

get_accuracy(tree, discretizedData1)
get_accuracy(tree2, discretizedData2)
get_accuracy(tree3, discretizedData3)
get_accuracy(tree4, discretizedData4)
get_accuracy(pokemonTree, discretizedPokemon)
