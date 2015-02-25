import numpy as np

def printPopulation ( population ):
    print "==SIGMA CREATURE:"
    printCreature ( population.sigmaCreature )
    print "==AVERAGE WINNING CREATURE:"
    printCreature ( population.avgWinningCreature )
    print "==AVERAGE LOSING CREATURE:"
    printCreature ( population.avgLosingCreature )
    print "==TRAINING CREATURE:"
    printCreature ( population.trainingCreature )
    for creature in population.creatureList:
        print "=Population ", len ( population.creatureList )
        print "  ",population.creatureCount," creatureCount, ", population.neuronCount, " neuronCount, ",population.inputCount," inputCount, ", population.outputCount, " outputCount, ",population.synapseCount," synapseCount"

def printCreature ( creature ):
        print "  -Creature", creature
        print "  --",creature.neuronCount," neurons, ",creature.inputCount," inputs, ",creature.outputCount," outputs, ", len(creature.synapseList)," synapses."
        print "  --",creature.ELO.mu," mu, ",creature.ELO.sigma," sigma "

def printSynapse ( synapse ):
        print "    ~Synapse"
        print "    ~~ a = ",synapse.a,", b = ",synapse.b,", c = ",synapse.c,", d = ",synapse.d

def printNeuron ( neuron ):
        print "    *Neuron"
        print "    ** inbox = ",neuron.inbox,", value = ", neuron.value, ", outbox = ", neuron.outbox, ", threshold = ",neuron.threshold,", prevOutbox = ", neuron.prevOutbox

def printPopOuts ( population ):
        print "::::::Training Outputs::::::::::::::"
        c = population.trainingCreature
        for o in c.output:
            printNeuron ( o )
        print "::::::Population Outputs::::::::::::"
        for c in population.creatureList:
            for o in c.output:
                printNeuron ( o )

def myGauss(mu,sig,x):
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

def printFinalOuts( population ):
    print "training outs:"
    for o in range (len(population.trainingCreature.output)):
        print population.trainingCreature.output[o].outbox
    print "best creature outs:"
    for c in range (len(population.creatureList[0].output)):
        print population.creatureList[0].output[c].outbox
    print

