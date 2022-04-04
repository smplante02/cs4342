import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def fPC(y, yhat):
    return np.mean(y == yhat)


def measureAccuracyOfPredictors(predictors, X, y):
    # found online sources for vectoring based on first part of the assignment
    ensembleVotes = np.zeros(y.shape)

    # for averaging later on
    count = 0

    for set in predictors:
        # getting pixel locations from set of predictors and comparing them
        r1, c1, r2, c2 = set
        pixelCompare = X[:, r1, c1] - X[:, r2, c2]

        # deciding what to do with the comparison information
        # if brighter, add 1 in corresponding spot in ensemblePrediction (vectorized)
        pixelCompare[pixelCompare > 0] = 1
        pixelCompare[pixelCompare < 0] = 0
        ensembleVotes += pixelCompare
        count += 1

    # getting final ensemble prediction - will get final guess of machine (vectorized)
    # if greater than 0.5, counts as a vote in favor, else vote against smile
    ensembleMean = np.divide(ensembleVotes, count)
    ensembleMean[ensembleMean > 0.5] = 1
    ensembleMean[ensembleMean <= 0.5] = 0

    return fPC(y, ensembleMean)


def stepwiseRegression(faces, labels):
    predictors = []
    for predictorSet in range(6):
        bestFeatureSoFar = None
        bestScoreSoFar = 0
        for r1 in range(24):
            for c1 in range(24):
                # print("R1C1: (", r1, ", ", c1, ")")
                for r2 in range(24):
                    for c2 in range(24):
                        # vectorization
                        testingSet = (r1, c1, r2, c2)
                        # adding new testing set to predictors with list append method, then comparing
                        testingScore = measureAccuracyOfPredictors(predictors + list((testingSet,)), faces, labels)
                        if testingScore > bestScoreSoFar:
                            bestScoreSoFar = testingScore
                            bestFeatureSoFar = testingSet
        predictors.append(bestFeatureSoFar)
    return predictors


def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


def testingMachine(trainingFaces, trainingLabels):
    # WARNING: takes about 30-45 minutes to run fully
    numExamples = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    print("n\ttrainingAccuracy\ttestingAccuracy")

    for n in numExamples:
        predictorList = stepwiseRegression(trainingFaces[:n], trainingLabels[:n])

        trainingAccuracyMeasure = measureAccuracyOfPredictors(predictorList, trainingFaces, trainingLabels)
        testingAccuracyMeasure = measureAccuracyOfPredictors(predictorList, testingFaces, testingLabels)
        print(n, "\t", trainingAccuracyMeasure, "\t", testingAccuracyMeasure)

        # only visualizing the n = 2000 for the assignment
        if n == 2000:
            show = True
            # Show an arbitrary test image in grayscale
            im = testingFaces[0, :, :]
            fig, ax = plt.subplots(1)
            ax.imshow(im, cmap='gray')
            # going through each set in predictor list
            for set in predictorList:
                print(set)
                r1, c1, r2, c2 = set
                # Show r1,c1
                rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2,
                                         edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Show r2,c2
                rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2,
                                         edgecolor='b', facecolor='none')
                ax.add_patch(rect)
            # Display the merged result
            plt.show()


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    # final testing function
    testingMachine(trainingFaces, trainingLabels)
