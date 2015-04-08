
from sobol_lib_NoNumpy import *

def assignCreatureProps(Creature,newValues):
    for neurInd in range(len(Creature.neuronList)):
        Creature.neuronList[neurInd] = assignNeurSynProps(Creature.neuronList[neurInd],newValues[0])

    for synInd in range(len(Creature.synapseList)):
        Creature.synapseList[synInd] = assignNeurSynProps(Creature.synapseList[synInd],newValues[1])

    return Creature

def assignNeurSynProps(Object,newValues):
    for propInd in range(len(Object.propertyList)):
        Object.propertyList[propInd] = newValues[propInd]
    return Object

def generatePopulationSobolPoints(PopSize,NeurCount,SynCount,NeurPropCount,SynPropCount,minVal=-1E3,maxVal=1E3,resolution=10,startSeed=0):
    numNeurDims = NeurPropCount*NeurCount
    numSynDims = SynPropCount*SynCount
    numDims = numNeurDims+numSynDims
    mins=[]
    maxs=[]
    for d in range(numDims):
        mins.append(minVal)
        maxs.append(maxVal)

    rawPoints, endSeed = generateSobolCharacterizationPoints(numDims,PopSize,mins,maxs,resolution,startSeed)

    processedPoints = []
    for c in range(PopSize):
        processedPoints.append([[],[]])
        for n in range(numNeurDims):
            processedPoints[-1][0].append(rawPoints[c][n])
        for s in range(numSynDims):
            processedPoints[-1][1].append(rawPoints[c][numNeurDims+s])

    return processedPoints, endSeed