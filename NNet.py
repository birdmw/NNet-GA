from math import *
from random import *
from copy import *
import numpy as np
import matplotlib.pyplot as plt
from sobol_lib_NoNumpy import *

import time
#from matplotlib.ticker import NullFormatter
#from mpl_toolkits.mplot3d import Axes3D
import csv

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
         #self.mutate()

    def mutate( self):
        for creature in self.creatureList:
            #Don't mutate the two best creatures.
            if (creature == self.creatureList[0]):
                pass
            else:
                #Randomly choose a property to mutate
                percentageToMutate = (1-creature.fitness)/2

                creature.selfMutation(percentageToMutate,self.sigmaCreature)
                '''
                mutationPerc = 0
                mutationCount = 0
                mutationOptions = ['threshold','a','b','c','d']
                mutatedObjProps = []

                while mutationPerc <= percentageToMutate:
                    selectObjectToMutate = {'threshold':choice(creature.neuronList),
                                        'a':choice(creature.synapseList),
                                        'b':choice(creature.synapseList),
                                        'c':choice(creature.synapseList),
                                        'd':choice(creature.synapseList)}

                    propertyToMutate = choice(mutationOptions)

                    objectToMutate = selectObjectToMutate.get(propertyToMutate)

                    if [objectToMutate,popertyToMutate] not in mutatedObjProps:
                        mu = getattr(objectToMutate,propertyToMutate)

                        if propertuToMutate == 'threshold':
                            ind = creature.neruronList.index(objectToMutate)
                            sigma = getattr(self.sigmaCreature.neuronList[ind],propertyToMutate)
                        else:
                            ind = creature.synapseList.index(objectToMutate)
                            sigma = getattr(self.sigmaCreature.synapseList[ind],propertyToMutate)


                        newVal = gauss(mu,sigma)
                        #print "gauss(",mu,",",sigma,")=",newVal
                        setattr(objectToMutate,propertyToMutate,newVal)
                        mutationCount+=1
                        mutationPerc = mutationCount/creature.propertyCount

                        mutatedObjProps.append([objectToMutate,popertyToMutate])







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
                '''

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
        #self.trainingCreature.output[3].outbox = 0.5*float(self.trainingCreature.input[0].inbox) + 0.5*float(self.trainingCreature.input[1].inbox)
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
        lesserWinningCreatures=[]
        losingCreatures = []

        #divide winners and losers
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

        for i in range ( len(self.creatureList) ):
            if (i<len(self.creatureList)/4):
                winningCreatures.append(self.creatureList[i])
            elif(i<len(self.creatureList)/2):
                lesserWinningCreatures.append(self.creatureList[i])
            else:
                losingCreatures.append(creature)

        #build sigma and delta creature

        self.avgWinningCreature.fitness = sum( w.fitness for w in winningCreatures) / len(winningCreatures)
        for i in range ( self.neuronCount ):
            thresh = 0
            outb = 0
            for w in winningCreatures:
                thresh += w.neuronList[i].threshold
                outb += w.neuronList[i].outbox
            self.avgWinningCreature.neuronList[i].threshold = thresh/len(winningCreatures)
            self.avgWinningCreature.neuronList[i].outbox = outb / len(winningCreatures)

            thresh = 0
            outb = 0

            for l in lesserWinningCreatures:
                thresh += l.neuronList[i].threshold
                outb += l.neuronList[i].outbox
            self.avgLosingCreature.neuronList[i].threshold = thresh/len(lesserWinningCreatures)
            self.avgLosingCreature.neuronList[i].outbox = outb / len(lesserWinningCreatures)

##            for l in losingCreatures:
##                thresh += l.neuronList[i].threshold
##                outb += l.neuronList[i].outbox
##            self.avgLosingCreature.neuronList[i].threshold = thresh/len(losingCreatures)
##            self.avgLosingCreature.neuronList[i].outbox = outb / len(losingCreatures)

            self.sigmaCreature.neuronList[i].threshold = (self.avgWinningCreature.neuronList[i].threshold - self.avgLosingCreature.neuronList[i].threshold)/MUT_DIVISOR
            self.deltaCreature.neuronList[i].outbox = (self.trainingCreature.neuronList[i].outbox - self.avgWinningCreature.neuronList[i].outbox)*1

        for i in range ( self.synapseCount ):
            suma = 0
            sumb = 0
            sumc=0
            sumd=0
            for w in winningCreatures:
                suma += w.synapseList[i].a
                sumb += w.synapseList[i].b
                sumc += w.synapseList[i].c
                sumd += w.synapseList[i].d

            self.avgWinningCreature.synapseList[i].a = suma/len(winningCreatures)
            self.avgWinningCreature.synapseList[i].b = sumb/len(winningCreatures)
            self.avgWinningCreature.synapseList[i].c = sumc/len(winningCreatures)
            self.avgWinningCreature.synapseList[i].d = sumd/len(winningCreatures)

            suma = 0
            sumb = 0
            sumc=0
            sumd=0

            for l in lesserWinningCreatures:
                suma += l.synapseList[i].a
                sumb += l.synapseList[i].b
                sumc += l.synapseList[i].c
                sumd += l.synapseList[i].d

            self.avgLosingCreature.synapseList[i].a = suma/len(lesserWinningCreatures)
            self.avgLosingCreature.synapseList[i].b = sumb/len(lesserWinningCreatures)
            self.avgLosingCreature.synapseList[i].c = sumc/len(lesserWinningCreatures)
            self.avgLosingCreature.synapseList[i].d = sumd/len(lesserWinningCreatures)

##            for l in losingCreatures:
##                suma += l.synapseList[i].a
##                sumb += l.synapseList[i].b
##                sumc += l.synapseList[i].c
##                sumd += l.synapseList[i].d
##
##            self.avgLosingCreature.synapseList[i].a = suma/len(losingCreatures)
##            self.avgLosingCreature.synapseList[i].b = sumb/len(losingCreatures)
##            self.avgLosingCreature.synapseList[i].c = sumc/len(losingCreatures)
##            self.avgLosingCreature.synapseList[i].d = sumd/len(losingCreatures)

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
            self.propertyCount+=1

        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])

        for o in range (self.outputCount):
            index = self.neuronCount - self.outputCount + o
            self.output.append(self.neuronList[index])

        for n1 in self.neuronList:
            for n2 in self.neuronList:
                self.synapseList.append(Synapse(n1, n2, random() / self.neuronCount, random() / (2*pi) / self.neuronCount, random() * pi / self.neuronCount, random() / self.neuronCount))
                self.propertyCount+=4
        self.updateSelf(0)




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
            #self.selfMutation(SELF_MUTATION_PERC)
        else:
            self.updateSelf(fit)
            #self.selfMutation(SELF_MUTATION_PERC)


    def revertSelf(self):
        self.neuronList = deepcopy(self.bestNeuronList)
        self.synapseList = deepcopy(self.bestSynapseList)

    def updateSelf(self,newFitness):
        self.selfWorth = newFitness
        self.bestNeuronList = deepcopy(self.neuronList)
        self.bestSynapseList = deepcopy(self.synapseList)

    def selfMutation(self,percentageToMutate,sigmaCreature = None):
        mutationPerc = 0
        mutationCount = 0
        #mutationOptions = ['threshold','a','b','c','d']
        mutatedObjProps = []
        propInds =[]
        watchdog = 0

        while mutationPerc <= percentageToMutate:
            if watchdog > self.propertyCount:
                break
            newInd = randint(0,self.propertyCount-1)
            if newInd not in propInds:
                propInds.append(newInd)
                mutationCount+=1
                mutationPerc=mutationCount/self.propertyCount
            watchdog+=1



        for propInd in propInds:
            if propInd < self.neuronCount:
                mu=self.neuronList[propInd].threshold
                if sigmaCreature != None:
                    sigma = sigmaCreature.neuronList[propInd].threshold
                else:
                    sigma = ((1-self.selfWorth)*mu+0.1)/(LESSON_MUT_DIVIDER)
                self.neuronList[propInd].threshold = gauss(mu,sigma)

            else:
                propInd -= self.neuronCount
                synInd = int(propInd/4)
                abcd = int(propInd/(synInd+1))
                if abcd == 0:
                    mu = self.synapseList[synInd].a
                    if sigmaCreature != None:
                        sigma = sigmaCreature.synapseList[synInd].a
                    else:
                        sigma = ((1-self.selfWorth)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].a = gauss(mu,sigma)
                elif abcd == 1:
                    mu = self.synapseList[synInd].b
                    if sigmaCreature != None:
                        sigma = sigmaCreature.synapseList[synInd].b
                    else:
                        sigma = ((1-self.selfWorth)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].b = gauss(mu,sigma)
                elif abcd == 2:
                    mu = self.synapseList[synInd].c
                    if sigmaCreature != None:
                        sigma = sigmaCreature.synapseList[synInd].c
                    else:
                        sigma = ((1-self.selfWorth)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].c = gauss(mu,sigma)
                elif abcd == 3:
                    mu = self.synapseList[synInd].d
                    if sigmaCreature != None:
                        sigma = sigmaCreature.synapseList[synInd].d
                    else:
                        sigma = ((1-self.selfWorth)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].d = gauss(mu,sigma)





        '''


        while mutationPerc <= percentageToMutate:
            selectObjectToMutate = {'threshold':choice(self.neuronList),
                                'a':choice(self.synapseList),
                                'b':choice(self.synapseList),
                                'c':choice(self.synapseList),
                                'd':choice(self.synapseList)}

            propertyToMutate = choice(mutationOptions)

            objectToMutate = selectObjectToMutate.get(propertyToMutate)

            if [objectToMutate,propertyToMutate] not in mutatedObjProps:
                mu = getattr(objectToMutate,propertyToMutate)

                if sigmaCreature != None:
                    if propertyToMutate == 'threshold':
                        ind = self.neuronList.index(objectToMutate)
                        sigma = getattr(sigmaCreature.neuronList[ind],propertyToMutate)
                    else:
                        ind = self.synapseList.index(objectToMutate)
                        sigma = getattr(sigmaCreature.synapseList[ind],propertyToMutate)
                else:
                    sigma = 1-self.selfWorth #ie: 1-fitness

                newVal = gauss(mu,sigma)
                #print "gauss(",mu,",",sigma,")=",newVal
                setattr(objectToMutate,propertyToMutate,newVal)
                mutationCount+=1
                mutationPerc = mutationCount/self.propertyCount

                mutatedObjProps.append([objectToMutate,propertyToMutate])
        '''
        '''
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



            newVal = gauss(mu,sigma)
            #print "gauss(",mu,",",sigma,")=",newVal
            setattr(objectToMutate,propertyToMutate,newVal)
            mutationCount+=1
            mutationPerc = mutationCount/self.propertyCount
        '''


    def test(self,population,lessons,cycles):
        for l in range(lessons):
            if l > 0:#Skip mutation on first pass.
                self.selfMutation((1-self.selfWorth)/LESSON_MUT_DIVIDER)
            self.run(population,cycles)
            self.selfEvaluation(population) #I think this could also be skipped first pass...

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
    fitMult = 1
    fitSum = 0
    outputCount =trainingCreature.outputCount
    for out in range(outputCount):
        mu = trainingCreature.output[out].outbox
        sigma = deltaCreature.output[out].outbox
        x = creature.output[out].outbox
        if abs(x) > MAX_VALUE:
            return -1

        g = myGauss(mu,sigma,x)
        fitSum +=g
        fitMult *= g

    avgFit = fitSum/outputCount
    fitness = (avgFit+fitMult)/2
    return fitness #/outputCount


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

def calculateOverallStrength(bestOutputs,trainOutputs):
    strength = 0
    G = len(bestOutputs)
    for g in range(G):
        distance = 0
        for o in range(len(bestOutputs[-1])):
            distance += abs(trainOutputs[g][o]-bestOutputs[g][o])
        m = g/(float(G)**2)
        #m=1
        strength+=m*distance
    return strength

def generateSobolCharacterizationPoints(numDims,numPts,starts,stops,resolution,startSeed = 0):
    dim_num = numDims

    seed = 0
    while seed < startSeed:
        qs = prime_ge ( dim_num )

        [ r, seed_out ] = i4_sobol ( dim_num, seed )
        seed = seed_out


    qs = prime_ge ( dim_num )
    pts=[]
    for pt in range( 0, numPts):
        newPt = False
        while newPt == False:

            [ r, seed_out ] = i4_sobol ( dim_num, seed )
            nxtPt = []
            for i in range(len(r)):
                rng = stops[i]-starts[i]


                newVal = float(round(starts[i]+rng*r[i],resolution[i]))
                #newVal = r[i]

                nxtPt.append(newVal)
            if nxtPt not in pts:
                pts.append(nxtPt)
                newPt = True
            seed = seed_out

    return pts

if __name__ == "__main__":
    '''
    Differences from main branch:
        Fixed bug in selfMutatation() --main branch will always choose to keep NEW mutations

        Sobol characterizations
            Sobol generator (need file)
            File saving

        Sigma creature = avgWinners - avgLesserWinners (Losers completely disgarded)

        Evolution strength calculation

        Generational mutation:
            Don't mutate best. Except during next generation's lessons. (ie, only change if it can be improved.)
            Removed mutate() from populate()
                Mutation now performed before re-population (If nothing else, saves half the mutation computations)
            Percentage of traits mutated = 1-creature.fitness
            mutate() calls creature.selfMutatation(...)

        Lessonal mutation:
            mu = same as generational
            percentage of traits mutated = (1-selfWorth)/SELF_MUT_DIVIDER
            sigma = ((1-selfWorth)*mu+0.1)/(SELF_MUT_DIVIDER)

        Fitness =  {[(O1+O2+..+On)/n]+(O1*O2*...*On)}/2 = average of average fitness and multiplied fitness

    '''
    GENERATIONS = 50
    CREATURE_COUNT = 100
    INPUT_COUNT = 2
    OUTPUT_COUNT = 3
    sobolTestPts = 7
    # next seed = 109
    sobolSeed = 0 #Which sobol point to start from. Remeber, 0 indexed
    POPS_TO_TEST=40

    MAX_VALUE = 10
    #50 Gen, 100 Creat,2In, 3Out, 20 sobol, 1 pops =~ 20 to 50 min. On Chris' laptop. Depending on start/stop values

    FILE_LOCATION =r"C:\Users\chris.nelson\Desktop\NNet\Feb 4"
    #Relationships between inputs and outputs for this training set, only used in results file
    outputRelations = [r"In[0]^In[1]",r"In[0]&In[1]",r"In[0](or)In[1]"]

    #Parameters controlled by sobol points
    toSobolTest = ['Neurons','Cycles','Lessons','Lesson Mutation Divider','Gen Mutation Divider']

    #xxx_startStop = [Minimum Value, Maximum Value, Resolution (ie: decimels to the right of 0. Can be negative)]
    Neurons_startStop = [INPUT_COUNT+OUTPUT_COUNT,INPUT_COUNT+OUTPUT_COUNT+5,0]
    Cycles_startStop = [Neurons_startStop[0],Neurons_startStop[1]*2.5,0]
    Lessons_startStop = [1,7,0]
    LessMutDiv_startStop = [1,20,0]
    MutDivider_startStop = [5,100,0]

    mins = [Neurons_startStop[0],Cycles_startStop[0],Lessons_startStop[0],LessMutDiv_startStop[0],MutDivider_startStop[0]]
    maxs = [Neurons_startStop[1],Cycles_startStop[1],Lessons_startStop[1],LessMutDiv_startStop[1],MutDivider_startStop[1]]
    resolution=[Neurons_startStop[2],Cycles_startStop[2],Lessons_startStop[2],LessMutDiv_startStop[2],MutDivider_startStop[2]]

    #G
    testPoints = generateSobolCharacterizationPoints(len(toSobolTest),sobolTestPts,mins,maxs,resolution,sobolSeed)


    localtime = time.localtime(time.time())
    Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
    Time = str(localtime[3])+'_'+str(localtime[4])


    file_name=FILE_LOCATION+'\\SobolCharacterizationOfNNet_'+str(GENERATIONS)+'Gens_'+str(CREATURE_COUNT)+'Creats_'+str(INPUT_COUNT)+'Ins_'+str(OUTPUT_COUNT)+'Outs_'+str(Date)+'_'+str(Time)+'.csv'
    fdata = open(file_name,'wb')
    scribe= csv.writer(fdata, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    scribe.writerow(["Sobol Characterization of NNet"])
    scribe.writerow(["Generations:",GENERATIONS])
    scribe.writerow(["Creatures:",CREATURE_COUNT])
    scribe.writerow(["Inputs:",INPUT_COUNT,"Outputs:",OUTPUT_COUNT])
    for o in range(OUTPUT_COUNT):
        scribe.writerow(["Out["+str(o)+"]=",outputRelations[o]])
    scribe.writerow([])
    scribe.writerow([])
    scribe.writerow(toSobolTest+['Strength'])
    fdata.close()

    ''' Uncomment to force specific test points
    testPoints=[]
    testPoints.append([10,25,4,1,16])
    testPoints.append([9,7,1,4,54])
    testPoints.append([5,8,3,12,71])
    testPoints.append([8,18,1,8,38])
    testPoints.append([6,22,2,11,91])
    testPoints.append([10,6,4,19,35])
    testPoints.append([9,19,4,2,94])
    '''
    print testPoints
    for i in range(sobolTestPts):
        NEURON_COUNT= int(testPoints[i][0])
        CYCLES_PER_RUN = int(testPoints[i][1])
        LESSONS_PER_TEST = int(testPoints[i][2])
        LESSON_MUT_DIVIDER = testPoints[i][3]
        MUT_DIVISOR = testPoints[i][4]

        for k in range(len(toSobolTest)):
            if k < 3:
                print toSobolTest[k],"=",int(testPoints[i][k])
            else:
                print toSobolTest[k],"=",testPoints[i][k]


        BestFits=[]
        bestOutputs=[]
        trainOutputs=[]
        testStrength=[]

        for p in range(POPS_TO_TEST):
            details_file_name=FILE_LOCATION+'\\SobolGenerationDetails_'+str(GENERATIONS)+'Gens_'+str(CREATURE_COUNT)+'Creats_'+str(INPUT_COUNT)+'Ins_'+str(OUTPUT_COUNT)+'Outs_'+str(p)+"Evol_"+str(NEURON_COUNT)+"N_"+str(CYCLES_PER_RUN)+"Cyc_"+str(LESSONS_PER_TEST)+"L_"+str(LESSON_MUT_DIVIDER)+"LMD_"+str(MUT_DIVISOR)+"GMD_"+(Date)+'_'+str(Time)+'.csv'
            fdetails = open(details_file_name,'wb')
            nerdScribe= csv.writer(fdetails, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nerdScribe.writerow(["Sobol Generation Details"])
            nerdScribe.writerow(["Generations:",GENERATIONS])
            nerdScribe.writerow(["Creatures:",CREATURE_COUNT])
            nerdScribe.writerow(["Inputs:",INPUT_COUNT,"Outputs:",OUTPUT_COUNT])

            headers =["Generation","Best Fitness"]
            htemp = []
            for o in range(OUTPUT_COUNT):
                nerdScribe.writerow(["Out["+str(o)+"]=",outputRelations[o]])
                headers.append("Best Output "+str(o))
                htemp.append("Train Output "+str(o))

            nerdScribe.writerow(["Neurons:",NEURON_COUNT,"Cycles:",CYCLES_PER_RUN,"Lessons:",LESSONS_PER_TEST,"Lesson Mut Div:",LESSON_MUT_DIVIDER,"Generation Mut Divr:",MUT_DIVISOR])
            nerdScribe.writerow([])
            nerdScribe.writerow([])
            nerdScribe.writerow(headers+htemp)
            fdetails.close()

            BestFits.append([])
            bestOutputs.append([])
            trainOutputs.append([])
            population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )

            for G in range (GENERATIONS):
                print "|||||||||||||| POINT:",i," EVOLUTION:",p,", GENERATION:",G,"||||||||||||||"

                #printPopulation (population)
                #printCreature(population.creatureList[0])
                #printSynapse(population.creatureList[0].synapseList[0])
                population.populate()
                population.setTrainingCreature()
                population.compete( CYCLES_PER_RUN )
                population.resolve()

                bestOutputs[-1].append([])
                trainOutputs[-1].append([])
                BestFits[-1].append([])
                for c in range (len(population.creatureList[0].output)):
                    bestOutputs[-1][-1].append(population.creatureList[0].output[c].outbox)
                    trainOutputs[-1][-1].append(population.trainingCreature.output[c].outbox)
                BestFits[-1][-1].append(population.creatureList[0].fitness)


                population.mutate()

##                print "best creature outs:"
##                for c in range (len(population.creatureList[0].output)):
##                    out = population.creatureList[0].output[c].outbox
##                    print out
##                print "best creature fit:"
##                print population.creatureList[0].fitness

            testStrength.append( calculateOverallStrength(bestOutputs[-1],trainOutputs[-1]))
            print 'Final Evolution Strength:',testStrength[-1]

            fdetails = open(details_file_name,'ab')
            #nerdScribe= csv.writer(fdetails, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nerdScribe= csv.writer(fdetails, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for G in range(GENERATIONS):
                nerdScribe.writerow([G]+BestFits[p][G]+bestOutputs[p][G]+trainOutputs[p][G])
            nerdScribe.writerow(['Final Evolution Strength:',testStrength[-1]])
            nerdScribe.writerow([" "])
            fdetails.close()


        fdata = open(file_name,'ab')
        scribe= csv.writer(fdata, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for testS in testStrength:
            scribe.writerow([NEURON_COUNT,CYCLES_PER_RUN,LESSONS_PER_TEST,LESSON_MUT_DIVIDER,MUT_DIVISOR,testS])
        fdata.close()
    '''
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
                name = 'Out'+str(o)+" Str"+str(round(testStrength[i],3))+" C"+str(CREATURE_COUNT)+" N"+str(NEURON_COUNT)+" Cy"+str(CYCLES_PER_RUN)+" L"+str(LESSONS_PER_TEST)+" Mp"+str(SELF_MUTATION_PERC)+" MDiv"+str(MUT_DIVISOR)
                plt.figure(name, figsize=(8,8))
                plt.plot(trainPlotter[o])
                plt.plot(bestPlotter[o])
    ##            plt.plot(diffPlotter[o])




    ##    print "training outs:"
    ##    for o in range (len(population.trainingCreature.output)):
    ##        print population.trainingCreature.output[o].outbox
    ##    print "best creature outs:"
    ##    for c in range (len(population.creatureList[0].output)):
    ##        print population.creatureList[0].output[c].outbox
    ##    print


        plt.show()

if __name__ == "__main__":
	main()
'''