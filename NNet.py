from math import *
from random import *
from copy import *
import numpy as np
import matplotlib.pyplot as plt

class Population:
    def __init__(self, CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT ):
        self.creatureList = []
        self.creatureCount = CREATURE_COUNT
        self.neuronCount = NEURON_COUNT
        self.inputCount = INPUT_COUNT
        self.outputCount = OUTPUT_COUNT
        self.sigmaCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )
        self.avgWinningCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
        self.avgLosingCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
        self.trainingCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
        self.deltaCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )

        for out in self.deltaCreature.output:
            out.outbox = random()
        for out in self.trainingCreature.output:
            out.outbox = random()

        self.synapseCount = len ( self.sigmaCreature.synapseList )
        self.populate()


    def populate( self ):
         while (len(self.creatureList) < 2): #If there is only one left, create a random new creature.
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
         while (len(self.creatureList) < self.creatureCount): #Breed until full population
              mother = choice( self.creatureList )
              father = choice( self.creatureList )
              if not (mother == father):
                child = mate( mother , father )
                self.creatureList.append( child )
         self.mutate()

    def mutate( self ):
        for creature in self.creatureList:
            for n in range( len ( creature.neuronList ) ):
                creature.neuronList[n].threshold = gauss( creature.neuronList[n].threshold , self.sigmaCreature.neuronList[n].threshold)
                #creature.neuronList[n].threshold = gauss( creature.neuronList[n].threshold , .5)
            for s in range ( len ( creature.synapseList ) ):
                creature.synapseList[s].a = gauss( creature.synapseList[s].a , self.sigmaCreature.synapseList[s].a )
                creature.synapseList[s].b = gauss( creature.synapseList[s].b , self.sigmaCreature.synapseList[s].b )
                creature.synapseList[s].c = gauss( creature.synapseList[s].c , self.sigmaCreature.synapseList[s].c )
                creature.synapseList[s].d = gauss( creature.synapseList[s].d , self.sigmaCreature.synapseList[s].d )
                #creature.synapseList[s].a = gauss( creature.synapseList[s].a , .01 )
                #creature.synapseList[s].b = gauss( creature.synapseList[s].b , .01 )
                #creature.synapseList[s].c = gauss( creature.synapseList[s].c , .01 )
                #creature.synapseList[s].d = gauss( creature.synapseList[s].d , .01 )


    def setTrainingCreature( self ):
        '''
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))##random bool

        self.trainingCreature.output[0].outbox = float(bool(self.trainingCreature.input[0].inbox)^bool(self.trainingCreature.input[1].inbox))##<---xor for inputs 0 and 1
        self.trainingCreature.output[1].outbox = float(not(bool(self.trainingCreature.input[0].inbox) and  bool(self.trainingCreature.input[1].inbox)))##<---xor for inputs 0 and 1
        '''
        '''
        count=0.0
        for i in self.trainingCreature.input:
            i.inbox = 0.5
        for o in self.trainingCreature.output:
            o.outbox = count
            count += 1

        '''
         # 6 output, 4 input
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))##random bool

##        self.trainingCreature.input[-2].inbox = randint(-5,5)
##        self.trainingCreature.input[-1].inbox = random()
        #xor(I0,I1)
        self.trainingCreature.output[0].outbox = float(bool(self.trainingCreature.input[0].inbox)^bool(self.trainingCreature.input[1].inbox))
        #and(I0,I1)
        self.trainingCreature.output[1].outbox = float(bool(self.trainingCreature.input[0].inbox)&bool(self.trainingCreature.input[1].inbox))
        #or(I0,I1)
        self.trainingCreature.output[2].outbox = float(bool(self.trainingCreature.input[0].inbox) or bool(self.trainingCreature.input[1].inbox))
        #0.5*I0 + 0.5*I1
        self.trainingCreature.output[3].outbox = 0.5*float(self.trainingCreature.input[0].inbox) + 0.5*float(self.trainingCreature.input[1].inbox)
        #
        '''
        self.trainingCreature.output[3].outbox = float(~(bool(self.trainingCreature.input[0].inbox) & bool(self.trainingCreature.input[1].inbox)))
        #2 * 3
        self.trainingCreature.output[4].outbox = float(bool(self.trainingCreature.input[0].inbox) or bool(self.trainingCreature.input[1].inbox)) * self.trainingCreature.input[3].inbox
        # 2 / 3
        self.trainingCreature.output[5].outbox =  float(self.trainingCreature.input[2].inbox)**self.trainingCreature.input[-2].inbox
        '''
    def compete( self, CYCLES_PER_RUN ):
        for creature in self.creatureList:
            creature.test(self, LESSONS_PER_TEST,CYCLES_PER_RUN)

    def resolve( self ):

        #Calculate fitness

        sumFitness=0.0
        for creature in self.creatureList:
            '''
            fitSum = 0.0
            for out in range(self.outputCount):

                mu = self.trainingCreature.output[out].outbox

                #sigma = self.sigmaCreature.output[out].outbox
                sigma = self.deltaCreature.output[out].outbox
                fitSum += myGauss(mu,sigma,creature.output[out].outbox)# Bug here between sigma and creature.output
                #fitSum += abs(creature.output[out].outbox - self.trainingCreature.output[out].outbox)

            creature.fitness = fitSum/self.outputCount
            '''
            tempFit = fitness(creature,self.trainingCreature,self.deltaCreature)
            if tempFit == -1:
                self.creatureList.remove(creature)
            else:
                creature.fitness = tempFit
                sumFitness += creature.fitness




        averageFitness = sumFitness / self.creatureCount

        winningCreatures = []
        losingCreatures = []

        #divide winners and losers

        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

        for i in range ( len(self.creatureList) ):
            if (i<len(self.creatureList)/2):
                winningCreatures.append(self.creatureList[i])
            else:
                losingCreatures.append(creature)


        #build sigma and delta creature

        self.avgWinningCreature.fitness = sum( w.fitness for w in winningCreatures) / len(winningCreatures)
        for i in range ( self.neuronCount ):
            self.avgWinningCreature.neuronList[i].threshold = sum( w.neuronList[i].threshold for w in winningCreatures) / len(winningCreatures)
            self.avgWinningCreature.neuronList[i].outbox = sum( w.neuronList[i].outbox for w in winningCreatures) / len(winningCreatures)
            self.avgLosingCreature.neuronList[i].threshold = sum( l.neuronList[i].threshold for l in losingCreatures ) / len(losingCreatures)
            self.avgLosingCreature.neuronList[i].outbox = sum( l.neuronList[i].threshold for l in losingCreatures ) / len(losingCreatures)
            self.sigmaCreature.neuronList[i].threshold = (self.avgWinningCreature.neuronList[i].threshold - self.avgLosingCreature.neuronList[i].threshold)/MUT_DIVISOR
            self.deltaCreature.neuronList[i].outbox = (self.trainingCreature.neuronList[i].outbox - self.avgWinningCreature.neuronList[i].outbox)*1
        for i in range ( self.synapseCount ):
            self.avgWinningCreature.synapseList[i].a = sum( w.synapseList[i].a for w in winningCreatures ) / len(winningCreatures)
            self.avgWinningCreature.synapseList[i].b = sum( w.synapseList[i].b for w in winningCreatures ) / len(winningCreatures)
            self.avgWinningCreature.synapseList[i].c = sum( w.synapseList[i].c for w in winningCreatures ) / len(winningCreatures)
            self.avgWinningCreature.synapseList[i].d = sum( w.synapseList[i].d for w in winningCreatures ) / len(winningCreatures)

            self.avgLosingCreature.synapseList[i].a = sum( l.synapseList[i].a for l in losingCreatures ) / len(losingCreatures)
            self.avgLosingCreature.synapseList[i].b = sum( l.synapseList[i].b for l in losingCreatures ) / len(losingCreatures)
            self.avgLosingCreature.synapseList[i].c = sum( l.synapseList[i].c for l in losingCreatures ) / len(losingCreatures)
            self.avgLosingCreature.synapseList[i].d = sum( l.synapseList[i].d for l in losingCreatures ) / len(losingCreatures)

            self.sigmaCreature.synapseList[i].a = (self.avgWinningCreature.synapseList[i].a - self.avgLosingCreature.synapseList[i].a)/MUT_DIVISOR
            self.sigmaCreature.synapseList[i].b = (self.avgWinningCreature.synapseList[i].b - self.avgLosingCreature.synapseList[i].b)/MUT_DIVISOR
            self.sigmaCreature.synapseList[i].c = (self.avgWinningCreature.synapseList[i].c - self.avgLosingCreature.synapseList[i].c)/MUT_DIVISOR
            self.sigmaCreature.synapseList[i].d = (self.avgWinningCreature.synapseList[i].d - self.avgLosingCreature.synapseList[i].d)/MUT_DIVISOR

        for creature in self.creatureList:
            if (creature.fitness <= 0.000001):
                self.creatureList.remove(creature)
            elif creature in losingCreatures:
                self.creatureList.remove(creature)
        if len(self.creatureList)==0:
            print '======== WARNING: ALL CREATURES DIED ========'

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount):
        self.neuronCount = neuronCount
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.neuronList  = []
        self.synapseList = []
        self.fitness = random()
        self.input = []
        self.output = []

        self.selfWorth = 0
        self.bestNeuronList = []
        self.bestSynapseList = []
        self.propertyCount = 0

        for n in range(self.neuronCount):
            self.neuronList.append(Neuron(random()))

        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])

        for o in range (self.outputCount):
            index = self.neuronCount - self.outputCount + o
            self.output.append(self.neuronList[index])
            self.propertyCount+=1

        for n1 in self.neuronList:
            for n2 in self.neuronList:
                self.synapseList.append(Synapse(n1, n2, random() / self.neuronCount, random() / (2*pi) / self.neuronCount, random() * pi / self.neuronCount, random() / self.neuronCount))
                self.propertyCount+=4
        self.updateSelf()




    def selfEvaluation(self,population):
        tempFit = 0
        fit =fitness(self,population.trainingCreature,population.deltaCreature)
##        for out in range(self.outputCount):
##            mu = population.trainingCreature.output[out].outbox
##            sigma = population.deltaCreature.output[out].outbox
##            tempFit += myGauss(mu,sigma,self.output[out].outbox)
##
##        fit =  tempFit/self.outputCount

        if self.selfWorth > fit:
            self.revertSelf()
            self.selfMutation(SELF_MUTATION_PERC)
        else:
            self.updateSelf()
            self.selfMutation(SELF_MUTATION_PERC)


    def revertSelf(self):
        self.neuronList = deepcopy(self.bestNeuronList)
        self.synapseList = deepcopy(self.bestSynapseList)

    def updateSelf(self):
        self.bestNeuronList = deepcopy(self.neuronList)
        self.bestSynapseList = deepcopy(self.synapseList)

    def selfMutation(self,percentageToMutate):
        mutationPerc = 0
        mutationCount = 0
        mutationOptions = ['threshold','a','b','c','d']
        while mutationPerc <= percentageToMutate:
            selectObjectToMutate = {'threshold':choice(self.neuronList),
                                'a':choice(self.synapseList),
                                'b':choice(self.synapseList),
                                'c':choice(self.synapseList),
                                'd':choice(self.synapseList)}

            propertyToMutate = choice(mutationOptions)

            objectToMutate = selectObjectToMutate.get(propertyToMutate)
            mu = getattr(objectToMutate,propertyToMutate)



            #TODO: Find a better sigma value
            if mu == 0:
                sigma = 0.1
            else:
                sigma = 0.1*mu


            newVal = gauss(mu,sigma)
            #print "gauss(",mu,",",sigma,")=",newVal
            setattr(objectToMutate,propertyToMutate,newVal)
            mutationCount+=1
            mutationPerc = mutationCount/self.propertyCount



    def test(self,population,lessons,cycles):
        for l in range(lessons):
            self.run(population,cycles)
            self.selfEvaluation(population)

    def run( self, population, cycles ):
        for r in range( cycles ):
            for i in range ( self.inputCount ):
                self.input[i].inbox += population.trainingCreature.input[i].inbox
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

class Neuron:
    def __init__(self, threshold):
        self.threshold = threshold
        self.inbox = random()
        self.value = random()
        self.outbox = random()
        self.prevOutbox = random()

    def run(self):
        self.prevOutbox = self.outbox
        #self.outbox = 0.0
        self.value += self.inbox
        if (self.value >= self.threshold):
            self.outbox = self.value
        self.value = 0.0
        self.inbox = 0.0

class Synapse:
    def __init__(self, n1, n2, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n1 = n1
        self.n2 = n2

    def run(self):
        try:
            self.n2.inbox += self.a * sin(self.b * self.n1.outbox + self.c) + self.d * (self.n1.prevOutbox - self.n1.outbox)
        except:
            pass
        #    self.n2.inbox = 0.0


def fitness(creature,trainingCreature,deltaCreature):
    fitSum = 0.0
    outputCount =trainingCreature.outputCount
    for out in range(outputCount):
        mu = trainingCreature.output[out].outbox
        sigma = deltaCreature.output[out].outbox
        x = creature.output[out].outbox
        if abs(x) > MAX_VALUE:
            return -1

        fitSum += myGauss(mu,sigma,x)

    return fitSum/outputCount


def mate (mother, father):
     child = deepcopy( mother )
     for i in range(len(child.neuronList)):
          if getrandbits(1):
              child.neuronList[i].threshold = father.neuronList[i].threshold
     for i in range(len(child.synapseList)):
          if getrandbits(1):
              child.synapseList[i].a = father.synapseList[i].a
          if getrandbits(1):
              child.synapseList[i].b = father.synapseList[i].b
          if getrandbits(1):
              child.synapseList[i].c = father.synapseList[i].c
          if getrandbits(1):
              child.synapseList[i].d = father.synapseList[i].d
     return child

def myGauss(mu,sig,x):
    if sig == 0.0:
        if x==mu:
            return 1.0
        else:
            return 0.0


    if (abs(mu) > MAX_VALUE):
        return 0
    if (abs(sig)>MAX_VALUE):
        return 0
    if (abs(x)>MAX_VALUE):
        return 0

    p1 = -np.power(x-mu,2.)
    p2 = 2*np.power(sig,2.)

    g = np.exp(p1/p2)
    return g

    #return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

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
        print "  -Creature"
        print "  --",creature.neuronCount," neurons, ",creature.inputCount," inputs, ",creature.outputCount," outputs, ", len(creature.synapseList)," synapses."
        print "  --",creature.fitness," fitness "

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

if __name__ == "__main__":
    CREATURE_COUNT = 500
    NEURON_COUNT= 8
    INPUT_COUNT = 2
    OUTPUT_COUNT = 4
    CYCLES_PER_RUN = (NEURON_COUNT+1)*2
    LESSONS_PER_TEST = 15
    GENERATIONS = 10


    MAX_VALUE = 10
    SELF_MUTATION_PERC = 0.05
    MUT_DIVISOR = 7




    WinnersFits=[]
    bestOutputs=[]
    trainOutputs=[]

    for i in range(1):
        WinnersFits.append([])
        bestOutputs.append([])
        trainOutputs.append([])
        population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )

        for G in range (GENERATIONS):
            print "|||||||||||||||||||||||| GENERATION: ",G,"||||||||||||||||||||||||"

            #printPopulation (population)
            #printCreature(population.creatureList[0])
            #printSynapse(population.creatureList[0].synapseList[0])
            WinnersFits[-1].append(population.avgWinningCreature.fitness)
            population.populate()
            population.setTrainingCreature()
            population.compete( CYCLES_PER_RUN )
            population.resolve()

            bestOutputs[-1].append([])
            trainOutputs[-1].append([])
            for c in range (len(population.creatureList[0].output)):
                bestOutputs[-1][-1].append(population.creatureList[0].output[c].outbox)
                trainOutputs[-1][-1].append(population.trainingCreature.output[c].outbox)


            print "best creature outs, deltas:"
            for c in range (len(population.creatureList[0].output)):
                out = population.creatureList[0].output[c].outbox
                delta = population.trainingCreature.output[c].outbox - out
                print out,'     ,     ',delta
            print "best creature fit:"
            print population.creatureList[0].fitness

    for i in range(len(WinnersFits)):
##        name = 'Round: '+str(i)
##        plt.figure(name, figsize=(8,8))
##        plt.plot(WinnersFits[i])
##        plt.axis([0, GENERATIONS, 0, 2])
        trainPlotter = []
        bestPlotter = []
        diffPlotter = []

        for o in range(OUTPUT_COUNT):
            trainPlotter.append([])
            bestPlotter.append([])
            diffPlotter.append([])
            for j in range(len(trainOutputs[i])):
                trainPlotter[-1].append(trainOutputs[i][j][o])
                bestPlotter[-1].append(bestOutputs[i][j][o])
                diffPlotter[-1].append(trainPlotter[-1][-1]-bestPlotter[-1][-1])

        for o in range(OUTPUT_COUNT):
            name = 'Output '+str(o)
            plt.figure(name, figsize=(8,8))
            plt.plot(trainPlotter[o])
            plt.plot(bestPlotter[o])
##            plt.plot(diffPlotter[o])




    print "training outs:"
    for o in range (len(population.trainingCreature.output)):
        print population.trainingCreature.output[o].outbox
    print "best creature outs:"
    for c in range (len(population.creatureList[0].output)):
        print population.creatureList[0].output[c].outbox
    print


    plt.show()

