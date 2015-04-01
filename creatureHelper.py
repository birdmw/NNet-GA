def checkConvergence(outputLists):
    '''
    Takes the given list of output lists, finds the min and max value for each output
    For all output sets:
        If the difference between the min and max value is greater than 0.01 % of the max value, return False

    Return true if all outputs pass

    outputLists = [[cycleA_output0,cycleA_output1,...],[cycleB_output0,cycleB_output1,...],...]
    '''

    percentageDifferenceToBeConverged = 0.01 #0.0001
    for outInd in range(len(outputLists[0])):
        reorderedOutputs = []
        for pt in outputLists:
            reorderedOutputs.append(pt[outInd])

        minOut = min(reorderedOutputs)
        maxOut = max(reorderedOutputs)
        diff = abs(maxOut)-abs(minOut)
        if diff > abs(maxOut *percentageDifferenceToBeConverged):
            return False

    return True


def save_creature(creature,fileName):
    fCreature = open(fileName,'wb')
    dump(creature,fCreature)
    fCreature.close()
    return fileName


def load_creature(fileName):
    fCreature = open(fileName,'r')
    creat = load(fCreature)
    fCreature.close()
    for n in range ( creat.neuronCount ):
        if n < creat.inputCount:
            creat.input[n] = creat.neuronList[n]
        if (creat.neuronCount - n-1) < creat.outputCount:
            creat.output[creat.neuronCount -n-1] = creat.neuronList[n]

    return creat

def testCreatureRepeatability(creature,inputSets,runs):
    print 'Creature fitness = ',creature.fitness
    for inputSet in inputSets:
        print 'Inputs: ',inputSet
        for i in range(len(inputSet)):
            creature.input[0].inbox = [inputSet[i]]
        for r in range(runs):
            creature.run()
            outputs = []
            for outp in creature.output:
                outputs.append(outp.outbox)

            print '  Run',r,' Outputs: ',outputs


def myGauss(self, mu,sig,x):
    '''
    Uses mu and sig to create a gaussian, then uses x as an input to the gaussian, returning the probability that x would be seen in the gaussian
    '''
    if sig == 0.0:
        if x==mu:
            return 1.0
        else:
            return 0.0
    p1 = -np.power(x-mu,2.)
    p2 = 2*np.power(sig,2.)
    g = np.exp(p1/p2)
    return g
