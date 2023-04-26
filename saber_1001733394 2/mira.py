# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            # this is the data-structure you should use
            self.weights[label] = util.Counter()

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        # this could be useful for your code later...
        self.features = trainingData[0].keys()

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        local_weights = dict(self.weights)




        for c in Cgrid:
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):

                    actual_data_ = trainingData[i]
                    actual_label = trainingLabels[i]
                    max_label_score=None
                    best_label=None               
                    for label in self.legalLabels:
                        local_label_score = 0
                        local_label_score += self.weights[label] *trainingData[i]  #calculate the prediction of the data

                        
                        
                        #save the output with largest value most likely to correct 
                        if max_label_score is None or local_label_score>max_label_score:
                            max_label_score = local_label_score
                            best_label = label
                            
                    if best_label == actual_label: 
                        None          #correct instance do nothing
                   
                    if best_label != actual_label:    #different than actual  so trying to adjust weights
                        ACT_square =( actual_data_ * actual_data_ )
                        # maximum possible of tau by a constant positive c
                        tau = min(c, ((local_weights[best_label] - local_weights[actual_label]) *
                                            actual_data_ + 1.0) / (2.0 * (ACT_square)))

                        # update the weight vectors of these labels
                        for feature in self.features:
                            local_weights[best_label][feature] = local_weights[best_label][feature]-( tau *  actual_data_[feature])
                            local_weights[actual_label][feature] = local_weights[actual_label][feature]+(tau*  actual_data_[feature])

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
