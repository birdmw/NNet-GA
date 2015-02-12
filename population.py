from creature import *
from math import *
from random import *
import numpy as np

class Population:
    def __init__(self, CreatureCount, NeuronCount, InputCount, OutputCount,Cycles, Lessons = 1, LessonMutationDivider=1,GenerationMutationDivider=1):
        self.creatureList = []
        self.creatureCount = CreatureCount
        self.inputCount = InputCount
        self.outputCount = OutputCount
        self.cycles = Cycles
        self.lessons = Lessons
        self.lessonMutDiv = LessonMutationDivider
        self.genMutDiv = GenerationMutationDivider

        #Create pseudo-creature data structures
        self.sigmaCreature = Creature( NeuronCount, InputCount, OutputCount )
        self.avgTopCreature = Creature( NeuronCount, InputCount, OutputCount  )
        self.avgBottomCreature = Creature( NeuronCount, InputCount, OutputCount  )
        self.trainingCreature = Creature( NeuronCount, InputCount, OutputCount  )
        self.deltaCreature = Creature( NeuronCount, InputCount, OutputCount  )
        self.statsCreature = Creature( NeuronCount, InputCount, OutputCount  )

        for outIndex in range(OutputCount):
            self.deltaCreature.output[outIndex].outbox = random()
            self.trainingCreature.output[outIndex].outbox = random()
            self.statsCreature.output[outIndex].outbox = []

        self.statsCreature.fitness = []

        #once we start adding/deleting synapses/neurons, these two will either need to be updated with changes, or removed from the code:
        self.synapseCount = len ( self.sigmaCreature.synapseList )
        self.neuronCount = NeuronCount

        self.populate()


    def populate( self ):
         while (len(self.creatureList) < 2): #If there is less than two creatures left, create a random new creature.
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
         while (len(self.creatureList) < self.creatureCount): #Breed until full population
              mother = choice( self.creatureList )
              father = choice( self.creatureList )
              if not (mother == father):
                child = mate( mother , father )
                self.creatureList.append( child )
         #self.mutate()

    def mutate_lessonMutation(self):
        '''
        Mutates all creatures in the populations,
            Lesson mutation gaussian based on:
                mu = current propery's value  (ex: neuron 1's threshold)
                sigma = ((1-creature.fitness)*mu+0.1)/(lessonMutateDiv)

            Percentage of traits to mutate:
                % = (1-creature.fitness)/2
        '''
        for creature in self.creatureList:
            #Calculate percentage of traits to mutate
            percentageToMutate = (1-creature.fitness)/2     #TODO: Evaluate this calculation.

            creatNeuronCount = len(creature.neuronList)
            creatSynapseCount = len(creature.synapseList)
            creatPropertyCount = creatNeurons+creatSynapses

            mutatedPerc = 0
            mutationCount = 0
            propInds =[]
            watchdog = 0

            #Determine, randomly, which properties to mutate.
            while mutatedPerc <= percentageToMutate:
                if watchdog > creatPropertyCount:
                    break
                newInd = randint(0,creatPropertyCount-1)
                if newInd not in propInds:
                    propInds.append(newInd)
                    mutationCount+=1
                    mutatedPerc=mutationCount/creatPropertyCount
                watchdog+=1

            #Calculate the sigma for each property to mutate, and mutate that property.
            for propInd in propInds:
                if propInd < creatNeuronCount:
                    mu=creature.neuronList[propInd].threshold
                    sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutateDiv)
                    creature.neuronList[propInd].threshold = gauss(mu,sigma)

                else:
                    propInd -= creatNeuronCount
                    synInd = int(propInd/4)
                    abcd = int(propInd/(synInd+1))
                    if abcd == 0:
                        mu = creature.synapseList[synInd].a
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutateDiv)
                        creature.synapseList[synInd].a = gauss(mu,sigma)
                    elif abcd == 1:
                        mu = creature.synapseList[synInd].b
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutateDiv)
                        creature.synapseList[synInd].b = gauss(mu,sigma)
                    elif abcd == 2:
                        mu = creature.synapseList[synInd].c
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutateDiv)
                        creature.synapseList[synInd].c = gauss(mu,sigma)
                    elif abcd == 3:
                        mu = creature.synapseList[synInd].d
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutateDiv)
                        creature.synapseList[synInd].d = gauss(mu,sigma)



    def mutate_generationMutation(self):
        '''
        Mutates all creatures in the populations, except the current best creature.
            Generation mutation gaussian based on:
                mu = current propery's value  (ex: creature.neuronList[0].threshold)
                sigma = sigmaCreatures's property value (ex: sigmaCreature.neuronList[0].threshold)

            Percentage of traits to mutate:
                % = (1-creature.fitness)/2
        '''
        for creature in self.creatureList:
            #Don't mutate the best creature.
            if (creature == self.creatureList[0]):
                pass
            else:
                #Calculate percentage of traits to mutate
                percentageToMutate = (1-creature.fitness)/2     #TODO: Evaluate this calculation.

                creatNeuronCount = len(creature.neuronList)
                creatSynapseCount = len(creature.synapseList)
                creatPropertyCount = creatNeuronCount+creatSynapseCount

                mutatedPerc = 0
                mutationCount = 0
                propInds =[]
                watchdog = 0

                #Determine, randomly, which properties to mutate.
                while mutatedPerc <= percentageToMutate:
                    if watchdog > creatPropertyCount:
                        break
                    newInd = randint(0,creatPropertyCount-1)
                    if newInd not in propInds:
                        propInds.append(newInd)
                        mutationCount+=1
                        mutatedPerc=mutationCount/creatPropertyCount
                    watchdog+=1

                #Calculate the sigma for each property to mutate, and mutate that property.
                for propInd in propInds:
                    if propInd < creatNeuronCount:
                        mu=creature.neuronList[propInd].threshold
                        sigma = self.sigmaCreature.neuronList[propInd].threshold
                        creature.neuronList[propInd].threshold = gauss(mu,sigma)

                    else:
                        propInd -= creatNeuronCount
                        synInd = int(propInd/4)
                        abcd = int(propInd/(synInd+1))
                        if abcd == 0:
                            mu = creature.synapseList[synInd].a
                            sigma = self.sigmaCreature.synapseList[synInd].a
                            creature.synapseList[synInd].a = gauss(mu,sigma)
                        elif abcd == 1:
                            mu = creature.synapseList[synInd].b
                            sigma = self.sigmaCreature.synapseList[synInd].b
                            creature.synapseList[synInd].b = gauss(mu,sigma)
                        elif abcd == 2:
                            mu = creature.synapseList[synInd].c
                            sigma = self.sigmaCreature.synapseList[synInd].c
                            creature.synapseList[synInd].c = gauss(mu,sigma)
                        elif abcd == 3:
                            mu = creature.synapseList[synInd].d
                            sigma = self.sigmaCreature.synapseList[synInd].d
                            creature.synapseList[synInd].d = gauss(mu,sigma)



    def setTrainingCreature( self, inList = None, outList = None ):
        '''
        Sets the values of this population's trainingCreature.
        Parameters:
            inList: (optional) A list of input values to set the inputs of trainingCreature ( len(inList) = len(inputs))
            outList: (optional) A list of output values to set the outputs of trainingCreature ( len(outList) = len(outputs))
        Returns:
            None.

        Commented out code are alternative training sets.
        '''

        '''
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
        '''
         # 6 output, 4 input
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))##random bool
        '''
##        self.trainingCreature.input[-2].inbox = randint(-5,5)
##        self.trainingCreature.input[-1].inbox = random()

        for i in range(len(inList)):
            self.trainingCreature.input[i].inbox = inList[i]
        for i in range(len(outList)):
            self.trainingCreature.output[i].outbox = outList[i]
##        #xor(I0,I1)
##        self.trainingCreature.output[0].outbox = float(bool(self.trainingCreature.input[0].inbox)^bool(self.trainingCreature.input[1].inbox))
##
##        #PAM4
####        self.trainingCreature.output[0].outbox = float(self.trainingCreature.input[0].inbox) + 0.5*float(self.trainingCreature.input[1].inbox)
##        #and(I0,I1)
##        self.trainingCreature.output[1].outbox = float(bool(self.trainingCreature.input[0].inbox)&bool(self.trainingCreature.input[1].inbox))
##        #or(I0,I1)
##        self.trainingCreature.output[2].outbox = float(bool(self.trainingCreature.input[0].inbox) or bool(self.trainingCreature.input[1].inbox))
##        #0.5*I0 + 0.5*I1
##        #self.trainingCreature.output[3].outbox = 0.5*float(self.trainingCreature.input[0].inbox) + 0.5*float(self.trainingCreature.input[1].inbox)
##        #
        '''
        self.trainingCreature.output[3].outbox = float(~(bool(self.trainingCreature.input[0].inbox) & bool(self.trainingCreature.input[1].inbox)))
        #2 * 3
        self.trainingCreature.output[4].outbox = float(bool(self.trainingCreature.input[0].inbox) or bool(self.trainingCreature.input[1].inbox)) * self.trainingCreature.input[3].inbox
        # 2 / 3
        self.trainingCreature.output[5].outbox =  float(self.trainingCreature.input[2].inbox)**self.trainingCreature.input[-2].inbox
        '''

    def compete_run(self):
        '''
        Each creature runs once
        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        sigmas = []
        for outp in self.deltaCreature.output:
            sigmas.append(outp.outbox)

        for creature in self.creatureList:
            creature.run(inputSet,self.cycles)
            creature.fitness = fitness(creature,inputSet,sigmas)
        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

    def compete_lessons(self):
        '''
        Each creature runs once for every lesson.
        Mutation occurs between each lesson, but only mutations causing improved fitness will be kept
        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        sigmas = []
        for outp in self.deltaCreature.output:
            sigmas.append(outp.outbox)

        for l in range(self.lessons):
            if l != 0:
                #Don't mutate the first lesson (generational mutation has not yet been tested)
                self.mutate_lessonMutation()

            for creature in self.creatureList:
                creature.run(inputSet,self.cycles,self.lessonMutDiv)
                creature.fitness = fitness(creature,inputSet,sigmas)
                if l == 0:
                    #On the first lesson, force creature to forget previous 'best' values, thereby keeping generational mutations
                    creature.updateBest()
                else:
                    #Don't evaluate on first lesson, this just saves computations - since updateBest() was already ran on the first lesson, evaluateBest() has no effect
                    creature.evaluateBest()

        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

    def compete_exhaustiveLessons(self, inputSets):
        '''
        Each creature runs on each input set once, for every lesson.
        Creature fitness is calculated based on it's performance on ALL input sets
        Mutation occurs between each lesson, but only mutations causing improved fitness will be kept
        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        '''
        sigmas = []
        for outp in self.deltaCreature.output:
            sigmas.append(outp.outbox)

        for l in range(self.lessons):
            if l != 0:
                #Don't mutate the first lesson (generational mutation has not yet been tested)
                self.mutate_lessonMutation()

            for creature in self.creatureList:
                #Each creature runs on each input set once
                shuffle(inputSets)
                setFits = []
                for inputSet in inputSets:
                    self.setTrainingCreature(inputSet) #Each creature sees each set in the training set, but in potentially different orders
                    creature.run(inputSet,self.cycles,self.lessonMutDiv)
                    setFits.append(fitness(creature,inputSet,sigmas))

                creature.fitness = fitness_exhaustiveCombiner(setFits)

                if l == 0:
                    #On the first lesson, force creature to forget previous 'best' values, thereby keeping generational mutations
                    creature.updateBest()
                else:
                    #Don't evaluate on first lesson, this just saves computations - since updateBest() was already ran on the first lesson, evaluateBest() has no effect
                    creature.evaluateBest()

        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

    def update_statsCreature(self):
        '''
        Appends the current best creatures outputs and fitness to the statsCreature's outputs and fitness
        '''
        for o in range(len(self.creatureList[0].output)):
            self.statsCreature.output[o].outbox.append(self.creatureList[0].output[0])
        self.statsCreature.fitness.append(self.creatureList[0].fitness)


    def update_pseudoCreatures( self ):
        '''
        Updates:
            avgTopCreature
            avgBottomCreature
            sigmaCreature
            deltaCreature
        '''

        topCreatures = []
        bottomCreatures = []


        #divide winners and losers into winners, lesser winners, and losers
        for i in range ( len(self.creatureList) ):
            if (i<len(self.creatureList)/2):
                topCreatures.append(self.creatureList[i])
            else:
                bottomCreatures.append(self.creatureList[i])

        #update sigma and delta creature NOTE: avgLosingCreature is really the average of the 'lesser winning creatures'
        self.avgTopCreature.fitness = sum( w.fitness for w in topCreatures) / len(topCreatures)
        for i in range ( self.neuronCount ):
            thresh = 0
            outb = 0
            for w in topCreatures:
                thresh += w.neuronList[i].threshold
                outb += w.neuronList[i].outbox
            self.avgTopCreature.neuronList[i].threshold = thresh/len(topCreatures)
            self.avgTopCreature.neuronList[i].outbox = outb / len(topCreatures)

            thresh = 0
            outb = 0

            for l in bottomCreatures:
                thresh += l.neuronList[i].threshold
                outb += l.neuronList[i].outbox
            self.avgBottomCreature.neuronList[i].threshold = thresh/len(bottomCreatures)
            self.avgBottomCreature.neuronList[i].outbox = outb / len(bottomCreatures)

            self.sigmaCreature.neuronList[i].threshold = (self.avgTopCreature.neuronList[i].threshold - self.avgBottomCreature.neuronList[i].threshold)/self.genMutDiv
            self.deltaCreature.neuronList[i].outbox = (self.trainingCreature.neuronList[i].outbox - self.avgTopCreature.neuronList[i].outbox)*1

        for i in range ( self.synapseCount ):
            suma = 0
            sumb = 0
            sumc=0
            sumd=0
            for w in topCreatures:
                suma += w.synapseList[i].a
                sumb += w.synapseList[i].b
                sumc += w.synapseList[i].c
                sumd += w.synapseList[i].d

            self.avgTopCreature.synapseList[i].a = suma/len(topCreatures)
            self.avgTopCreature.synapseList[i].b = sumb/len(topCreatures)
            self.avgTopCreature.synapseList[i].c = sumc/len(topCreatures)
            self.avgTopCreature.synapseList[i].d = sumd/len(topCreatures)

            suma = 0
            sumb = 0
            sumc=0
            sumd=0

            for l in bottomCreatures:
                suma += l.synapseList[i].a
                sumb += l.synapseList[i].b
                sumc += l.synapseList[i].c
                sumd += l.synapseList[i].d

            self.avgBottomCreature.synapseList[i].a = suma/len(bottomCreatures)
            self.avgBottomCreature.synapseList[i].b = sumb/len(bottomCreatures)
            self.avgBottomCreature.synapseList[i].c = sumc/len(bottomCreatures)
            self.avgBottomCreature.synapseList[i].d = sumd/len(bottomCreatures)

            self.sigmaCreature.synapseList[i].a = (self.avgTopCreature.synapseList[i].a - self.avgBottomCreature.synapseList[i].a)/self.genMutDiv
            self.sigmaCreature.synapseList[i].b = (self.avgTopCreature.synapseList[i].b - self.avgBottomCreature.synapseList[i].b)/self.genMutDiv
            self.sigmaCreature.synapseList[i].c = (self.avgTopCreature.synapseList[i].c - self.avgBottomCreature.synapseList[i].c)/self.genMutDiv
            self.sigmaCreature.synapseList[i].d = (self.avgTopCreature.synapseList[i].d - self.avgBottomCreature.synapseList[i].d)/self.genMutDiv


    def prune(self):
        '''
        Will delete bottom half of creature list. And any creatures with extremely low fitness
        '''
        startLen = len(self.creatureList)
        toBeRemoved = []
        for creature in self.creatureList:
            #print startLen,self.creatureList.index(creature),creature.fitness
            if (creature.fitness <= 0.000000001):
                #self.creatureList.remove(creature)
                toBeRemoved.append(creature)
            elif self.creatureList.index(creature)  > (startLen-1)/2:
                #self.creatureList.remove(creature)
                toBeRemoved.append(creature)

        for loser in toBeRemoved:
            self.creatureList.remove(loser)

        if len(self.creatureList)==0:
            print '======== WARNING: ALL CREATURES DIED ========'
            self.populate()
            print '======== !!RANDOMLY REPOPULATED!! ========'


    def run_generationsBasic(self,generationCount):
        '''
        Runs the population for generationCount generation, in a basic loop
        '''
        for g in generationCount:
            self.populate()
            self.setTrainingCreature()
            self.compete_run()
            self.prune()
            self.update_pseudoCreatures()
            self.mutate_generationMutation()

def fitness(creature,mus,sigmas):
    '''
    Calculates the fitness for a creature using it's outputs as points in gaussians created from the provided mus and sigmas
    Parameters:
        creature: The creature who's fitness is being calculated
        mus: A list of mu's to be used in the gaussian
        sigmas: A list of sigma's to be used in the gaussian
    Returns:
        fitness: Bounded between 0 and 1, unless the output is above MAX_VALUE, then fitness is -1
    '''
    fitMult = 1
    fitSum = 0
    for out in range(len(mus)):
        x = creature.output[out].outbox
        if abs(x) > MAX_VALUE:
            return -1
        g = myGauss(mus[out],sigmas[out],x)
        fitSum +=g
        fitMult *= g

    avgFit = fitSum/len(mus)
    fitness = (avgFit+fitMult)/2
    return fitness

def fitness_exhaustiveCombiner(setFits):
    '''
    Combines a list of fitnesses into one number. Bounded between 0 and 1. This is currently used in the exhaustive training set per lesson test
    '''
    setMult = 1
    setSum = 0
    for fit in setFits:
        setSum+=fit
        setMult*=fit
    setAvg = setSum/len(setFits)
    setFitness = (setAvg+setMult)/2
    return setFitness

def mate (mother, father):
    '''
    Pull out the wine, light some candles, and turn on the charm. Giggity giggity.
    '''
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
    '''
    Uses mu and sig to create a gaussian, then uses x as an input to the gaussian, returning the probability that x would be seen in the gaussian
    '''
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



def main():
    CreatureCount = 100
    NeuronCount = 10
    InputCount = 1
    OutputCount = 2
    Cycles = 25
    Lessons = 1
    LessonMutationDivider = 2
    GenerationMutationDivider = 20

    trainingSetInputs = [[1],[0],[1]]
    trainingSetOutputs = [[1,1],[1,0],[1,1]]

    print "Population Description:"

    demoPop =  Population(CreatureCount, NeuronCount, InputCount, OutputCount,Cycles, Lessons, LessonMutationDivider,GenerationMutationDivider)
    demoPop.populate()
    print "  Number of creatures:",len(demoPop.creatureList)

    for setInd in range(len(trainingSetInputs)):
        demoPop.setTrainingCreature(trainingSetInputs[setInd],trainingSetOutputs[setInd])
        demoPop.compete_run()
        demoPop.prune()
        demoPop.update_pseudoCreatures()
        demoPop.mutate_generationMutation()

        printTrain = []
        printDelta = []
        printBest = []

        for i in range(len(demoPop.trainingCreature.output)):
            printTrain.append(demoPop.trainingCreature.output[i].outbox)
            printDelta.append(demoPop.deltaCreature.output[i].outbox)
            printBest.append(demoPop.creatureList[0].output[i].outbox)

        print "Set ",setInd," Results:"
        print "  Number of surviving creatures:",len(demoPop.creatureList)
        print "  trainingCreature outputs:",printTrain
        print "  deltaCreature outputs:",printDelta
        print "  bestCreature outputs:",printBest
        print "  bestCreature fitness:",demoPop.creatureList[0].fitness

        demoPop.populate()

if __name__ == '__main__':
    MAX_VALUE = 15
    main()
