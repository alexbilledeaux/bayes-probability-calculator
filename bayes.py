# Converts a text file into an list of dictionaries,
# each of which represents an instance (row) in the data set.
def importDataset(fileName: str):
    file = open('assets/' + fileName, 'r')
    content = file.read()
    file.close()
    content = content.encode('ascii', 'ignore')
    content = content.decode()
    # Use column headers to structure dictionary
    headers = []
    headerString = content.partition("\n")[0]
    for x in range(headerString.count(' ') + 1):
        headers.append(headerString.partition(" ")[0])
        headerString = headerString.partition(" ")[2]
    # Remove column headers
    content = content.partition("\n")[2]
    # Get a list of each line in the data file
    lines = content.splitlines()
    # Convert each line into a dictionary that represents an instance (row)
    instances = []
    for line in lines:
        instance = {}
        for i, header in enumerate(headers):
            instance[header] = line.partition(" ")[0]
            line = line.partition(" ")[2]
        instances.append(instance)
    return instances

# Prints a formatted header with the given text to the console
def printHeader(header: str):
    print("\n--------------------\n" + header + "\n--------------------")

def learnAndClassify(instance, trainingData):
    # Learn
    print("Training instances = " + str(len(trainingData)))
    hypothesisClasses = getHypothesisClasses(trainingData)
    printHeader("Prior Probabilities")
    for hypothesisClass in hypothesisClasses:
        print("P("+ hypothesisClass + ") = " + str(round(getPriorProbability(hypothesisClass, trainingData), 2)))

    printHeader("New Instance")
    newInstance = ""
    for attributeKey in list(instance.keys()):
        newInstance += (instance[attributeKey] + " ")
    print(newInstance)

    printHeader("Conditional Probabilities")
    for hypothesisClass in hypothesisClasses:
        for attributeKey in list(instance.keys()):
            print("P(" + instance[attributeKey] + "|" + hypothesisClass + ") = " + str(round(getLikelihood(attributeKey, instance[attributeKey], hypothesisClass, trainingData), 3)))

    # Classify
    printHeader("Class Probabilities")
    classProbabilities = {}
    for hypothesisClass in hypothesisClasses:
        classProbability = getPriorProbability(hypothesisClass, trainingData)
        for attributeKey in list(instance.keys()):
            classProbability *= getLikelihood(attributeKey, instance[attributeKey], hypothesisClass, trainingData)
        classProbabilities[hypothesisClass] = classProbability
        print(hypothesisClass + ": " + str(round(classProbability, 3)))

    printHeader("Classification")
    classification = ""
    highestValue = 0
    for hypothesisClass in getHypothesisClasses(trainingData):
        if classProbabilities[hypothesisClass] > highestValue:
            classification = hypothesisClass
            highestValue = classProbabilities[hypothesisClass]
    print(classification)

# Returns a list of every hypothesis class in a given dataset.
def getHypothesisClasses(dataSet):
    hypothesisClasses = []
    for instance in dataSet:
        if instance["Oracle"] not in hypothesisClasses:
            hypothesisClasses.append(instance["Oracle"])
    return hypothesisClasses

# Return the probability of a given hypothesis class occuring in the data set.
def getPriorProbability(hypothesisClass: str, dataSet) -> float:
    priorProbability = 0
    for instance in dataSet:
        if (instance["Oracle"] == hypothesisClass):
            priorProbability += 1
    priorProbability = priorProbability / len(dataSet)
    return priorProbability

# Return the probability of a given attribute value, after seeing a given hypothesis class.
def getLikelihood(attributeKey, attributeValue, hypothesisClass: str, dataSet) -> float:
    likelihood = 0
    countOfHypothesisClass = 0
    for instance in dataSet:
        if instance["Oracle"] == hypothesisClass:
            if instance[attributeKey] == attributeValue:
                likelihood += 1
            countOfHypothesisClass += 1
    likelihood = likelihood / countOfHypothesisClass
    return likelihood

# Return the probability of a given attribute value occuring in the data set.
def getEvidence(attributeKey, attributeValue, dataSet) -> float:
    evidence = 0
    for instance in dataSet:
        if (instance[attributeKey] == attributeValue):
            evidence += 1
    evidence = evidence / len(dataSet)
    return evidence

# Return the probability of a given hypothesis class, after seeing a given attribute value.
def getPosteriorProbability(hypothesisClass: str, attributeKey, attributeValue, dataSet) -> float:
    posteriorProbability = 0
    posteriorProbability = (getLikelihood(attributeKey, attributeValue, hypothesisClass, dataSet) * getPriorProbability(hypothesisClass, dataSet)) / getEvidence(attributeKey, attributeValue, dataSet)
    return posteriorProbability