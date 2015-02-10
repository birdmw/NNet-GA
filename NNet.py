from math import *
from random import *
from copy import *
from pickle import *
from sobol_lib_NoNumpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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

    def mutate( self):
        '''
        Mutates all creatures in the populations, except the current best creature.
        '''
        for creature in self.creatureList:
            #Don't mutate the best creature.
            if (creature == self.creatureList[0]):
                pass
            else:
                #Calculate percentage of traits to mutate
                percentageToMutate = (1-creature.fitness)/2     #TODO: Evaluate this calculation.

                creature.mutateSelf(percentageToMutate,self.sigmaCreature)

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
        '''
         # 6 output, 4 input
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))##random bool

##        self.trainingCreature.input[-2].inbox = randint(-5,5)
##        self.trainingCreature.input[-1].inbox = random()
        '''
        for i in range(len(inList)):
            self.trainingCreature.input[i].inbox = inList[i]
        #xor(I0,I1)
        #self.trainingCreature.output[0].outbox = float(bool(self.trainingCreature.input[0].inbox)^bool(self.trainingCreature.input[1].inbox))

        #PAM4
        self.trainingCreature.output[0].outbox = float(self.trainingCreature.input[0].inbox) + 0.5*float(self.trainingCreature.input[1].inbox)
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
    def compete( self, cycles,lessons ):
        for creature in self.creatureList:
            creature.test(self, lessons,cycles)

    def resolve( self ):

        #Calculate fitness
        sumFitness=0.0
        for creature in self.creatureList:
            #tempFit = fitness(creature,self.trainingCreature,self.deltaCreature)
            #fitness is now updated during lessons
            if creature.fitness == -1:
                self.creatureList.remove(creature)
            else:
                sumFitness += creature.fitness

        averageFitness = sumFitness / self.creatureCount

        winningCreatures = []
        lesserWinningCreatures=[]
        losingCreatures = []

        #divide winners and losers into winners, lesser winners, and losers
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

        for i in range ( len(self.creatureList) ):
            if (i<len(self.creatureList)/4):
                winningCreatures.append(self.creatureList[i])
            elif(i<len(self.creatureList)/2):
                lesserWinningCreatures.append(self.creatureList[i])
            else:
                losingCreatures.append(creature)

        #build sigma and delta creature NOTE: avgLosingCreature is really the average of the 'lesser winning creatures'

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

            self.sigmaCreature.synapseList[i].a = (self.avgWinningCreature.synapseList[i].a - self.avgLosingCreature.synapseList[i].a)/MUT_DIVISOR
            self.sigmaCreature.synapseList[i].b = (self.avgWinningCreature.synapseList[i].b - self.avgLosingCreature.synapseList[i].b)/MUT_DIVISOR
            self.sigmaCreature.synapseList[i].c = (self.avgWinningCreature.synapseList[i].c - self.avgLosingCreature.synapseList[i].c)/MUT_DIVISOR
            self.sigmaCreature.synapseList[i].d = (self.avgWinningCreature.synapseList[i].d - self.avgLosingCreature.synapseList[i].d)/MUT_DIVISOR


        #Remove all losers, and all creatures exhibiting terrible fitness (helps remove the run-away condition)
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

        #Best parameters are used in the lessons to allow improvement only
        self.bestFitness = self.fitness
        self.bestNeuronList = []
        self.bestSynapseList = []

        #Property count is used for calculating the percentage of traits to mutate - evaluated at initialization
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




    def evaluateSelf(self,population):
        '''
        Calculates the fitness of this creature, and determines if this creature should update or revert it's neurons, synapses and fitness
        Parameters:
            trainingCreature: The populations training creature to be used in the fitness calculation.
            deltaCreature: The populations delta creature to be used in the fitness calculation.
        '''
        tempFit = 0
        fit = fitness(self,population.trainingCreature,population.deltaCreature)

        #If this new fitness is better than old fitness, keep changes, otherwise revert.
        if self.fitness > fit:
            self.revertSelf()
        else:
            self.updateSelf(fit)


    def revertSelf(self):
        '''
        Reverts this creatures fitness, neuronList, and synapseList to their previous best.
        '''
        self.fitness = self.bestFitness
        self.neuronList = deepcopy(self.bestNeuronList)
        self.synapseList = deepcopy(self.bestSynapseList)

    def updateSelf(self,newFitness):
        '''
        Updates this creatures fitness with 'newFitness' and updates the bestNeuronList and bestSynapselist.
        '''
        self.bestNeuronList = deepcopy(self.neuronList)
        self.bestSynapseList = deepcopy(self.synapseList)

    def mutateSelf(self,percentageToMutate,sigmaCreature = None):
        '''
        Mutates this creature's parameters according to a gaussian distribution.
        Parameters:
            percentageToMuate: The percentage of traits of this creature to mutate
            sigmaCreature: (optional) The sigma creature of the population, allows sigma to be defined by the population
        Returns:
            none
        '''
        mutationPerc = 0
        mutationCount = 0
        mutatedObjProps = []
        propInds =[]
        watchdog = 0

        #Determine, randomly, which properties to mutate.
        while mutationPerc <= percentageToMutate:
            if watchdog > self.propertyCount:
                break
            newInd = randint(0,self.propertyCount-1)
            if newInd not in propInds:
                propInds.append(newInd)
                mutationCount+=1
                mutationPerc=mutationCount/self.propertyCount
            watchdog+=1

        #Calculate the sigma for each property to mutate, and mutate that property.
        for propInd in propInds:
            if propInd < self.neuronCount:
                mu=self.neuronList[propInd].threshold
                if sigmaCreature != None:
                    sigma = sigmaCreature.neuronList[propInd].threshold
                else:
                    sigma = ((1-self.fitness)*mu+0.1)/(LESSON_MUT_DIVIDER)
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
                        sigma = ((1-self.fitness)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].a = gauss(mu,sigma)
                elif abcd == 1:
                    mu = self.synapseList[synInd].b
                    if sigmaCreature != None:
                        sigma = sigmaCreature.synapseList[synInd].b
                    else:
                        sigma = ((1-self.fitness)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].b = gauss(mu,sigma)
                elif abcd == 2:
                    mu = self.synapseList[synInd].c
                    if sigmaCreature != None:
                        sigma = sigmaCreature.synapseList[synInd].c
                    else:
                        sigma = ((1-self.fitness)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].c = gauss(mu,sigma)
                elif abcd == 3:
                    mu = self.synapseList[synInd].d
                    if sigmaCreature != None:
                        sigma = sigmaCreature.synapseList[synInd].d
                    else:
                        sigma = ((1-self.fitness)*mu+0.1)/(LESSON_MUT_DIVIDER)
                    self.synapseList[synInd].d = gauss(mu,sigma)



    def test(self,population,lessons,cycles):
        '''
        Let each creature run 'lessons' number of 'cycles' length tests on the current training set
        parameters:
           population: The population this creature lives in. This can be replaced with trainingCreature and deltaCreature.
           lessons: The number of iterations to, for this training set, mutate, run, and evaluate this creature. (Only keeping mutations that produces better fitnesses)
           cycles: The number of cycles that this creature is allowed to run before it's outputs are read.
        returns:
            none
        '''
        for l in range(lessons):
            if l > 0:#Skip mutation on first pass.
                self.mutateSelf((1-self.fitness)/LESSON_MUT_DIVIDER)
            self.run(population,cycles)
            self.evaluateSelf(population) #I think this could also be skipped first pass...

    def run( self, population, cycles ):
        '''
        Let each creature run 'lessons' number of 'cycles' length tests on the current training set
        parameters:
           population: The population this creature lives in. This can be replaced with trainingCreature
           cycles: The number of cycles that this creature is allowed to run before it's outputs are read.
        returns:
            none
        '''
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

        #Check for run-away condition
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

def calculateSpeciesFitness_goverG2(bestOutputs,trainOutputs):
    myfit = 0
    G = len(bestOutputs)
    for g in range(G):
        distance = 0
        for o in range(len(bestOutputs[-1])):
            distance += abs(trainOutputs[g][o]-bestOutputs[g][o])
        m = g/(float(G)**2)
        #m=1
        myfit+=m*distance
    return myfit


def calculateSpeciesFitness_binaryOER(creatOutputs,trainOutputs,outThreshList):
    OER = 0 #Output error rate
    errors = 0
    ptCnt = 0
    G = len(creatOutputs)
    for g in range(G):
        for o in range(len(creatOutputs[-1])):
            if (trainOutputs[g][o] < outThreshList[o][0]) and (creatOutputs[g][o] >outThreshList[o][0]) :
                errors+=1
            elif (trainOutputs[g][o] > outThreshList[o][1]) and (creatOutputs[g][o] < outThreshList[o][1]) :
                errors+=1

            ptCnt +=1

    OER = errors/ptCnt
    return OER



def testSpeciesFitness_exhaustiveTrainingSpaceDistance(population,inList):
    '''
    NOTE: This function will OVERWRITE all creatures fitness and, because of this, potentially change which creature is 'bestCreature'
    '''
    for creature in population.creatureList:        #For each creature

        for inp in inList:                          #Test it against training creature for all combinations of inputs
            population.setTrainingCreature(inp)
            creature.run(population, cycles)

            dist = 0
            for outp in range (len(creature.output)):
                dist +=  abs(population.trainingCreature.output[outp].outbox - creature.output[outp].outbox)

            creature.fitness = 1-dist/(1+dist) #Will be '1' when dist = 0, and approach '0' when dist = inf

    population.creatureList.sort(key = lambda x: x.fitness, reverse=True)


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

def createSobolFiles(fileLoc,fileName, gens, creats, inCount, outCount,outputRelations):
    localtime = time.localtime(time.time())
    Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
    Time = str(localtime[3])+'_'+str(localtime[4])


    file_name=fileLoc+'\\'+fileName+'_'+str(Date)+'_'+str(Time)+'.csv'
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
    fdata.close()

    return file_name


def writeSobolFileRow(fname,data):
    fdata = open(fname,'ab')
    scribe= csv.writer(fdata, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    scribe.writerow(data)
    fdata.close()
    return

def writeSobolFileMultiRows(fname,data):
    fdata = open(fname,'ab')
    scribe= csv.writer(fdata, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in data:
        scribe.writerow(row)
    fdata.close()
    return

def createFig_creature_exhaustiveTrainingSpace(population,creature,cycles,inList,ID = randint(1,100)):
    creature.run(population, cycles)
    creatOuts = []
    trainOuts = []
    for inp in inList:
        population.setTrainingCreature(inp)
        creature.run(population, cycles)
        creatOuts.append([])
        trainOuts.append([])
        for outp in range (len(creature.output)):
            creatOuts[-1].append(creature.output[outp].outbox)
            trainOuts[-1].append(population.trainingCreature.output[outp].outbox)

    creatPlots=[]
    trainPlots=[]

    for outp in range(len(creature.output)):
        creatPlots.append([])
        trainPlots.append([])
        for inp in range(len(inList)):
            creatPlots[-1].append(creatOuts[inp][outp])
            trainPlots[-1].append(trainOuts[inp][outp])

    for outp in range(len(creature.output)):
        plt.figure("ID:"+str(ID)+" Output:"+str(outp), figsize=(8,8))
        plt.plot(trainPlots[outp],'b',creatPlots[outp],'r')
    return

def createFig_DistHistogram(data,divisions,xtitle,ytitle):
    mu = np.mean(data)
    sigma = np.std(data)
    mini = min(data)
    maxi = max(data)

    plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(data, divisions, normed=1, facecolor='green', alpha=0.85)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'b--', linewidth=1)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu='+str(mu)+',\ \sigma='+str(sigma)+'$')
    plt.axis([mini, maxi, 0, max(n)*1.2])
    plt.grid(True)


def save_creature(creature,fileName):
    fCreature = open(fileName,'wb')
    dump(creature,fCreature)
    fCreature.close()

def load_creature(fileName):
    fCreature = open(fileName,'r')
    creat = load(fCreature)
    fCreature.close()
    return creat




if __name__ == "__main__":
    '''

    '''
    GENERATIONS = 10 #50
    CREATURE_COUNT = 20 #100
    INPUT_COUNT = 2
    OUTPUT_COUNT = 3
    sobolTestPts = 1
    # next seed = 9
    sobolSeed = 0 #Which sobol point to start from. Remeber, 0 indexed
    POPS_TO_TEST=3

    MAX_VALUE = 10
    #50 Gen, 100 Creat,2In, 3Out, 20 sobol, 1 pops =~ 20 to 50 min. On Chris' laptop. Depending on start/stop values

    FILE_LOCATION =r"C:\Users\chris.nelson\Desktop\NNet\ExhaustiveTrainingPerGen"
    #Relationships between inputs and outputs for this training set, only used in results file
    outputRelations = [r"In[0]+0.5*In[1]",r"In[0]&In[1]",r"In[0](or)In[1]"]

    #Both of these are optional:
    inList = [[0,0],[0,1],[1,0],[1,1]]
    outList = [[0,0,0],[1,0,1],[1,0,1],[0,1,1]]
    outThreshList = [[0.4,0.6],[0.4,0.6],[0.4,0.6]]

    #Parameters controlled by sobol points
    toSobolTest = ['Neurons','Cycles','Lessons','Lesson Mutation Divider','Gen Mutation Divider']

    #xxx_startStop = [Minimum Value, Maximum Value, Resolution (ie: decimels to the right of 0. Can be negative)]
    Neurons_startStop = [INPUT_COUNT+OUTPUT_COUNT+3,INPUT_COUNT+OUTPUT_COUNT+15,0]
    Cycles_startStop = [Neurons_startStop[0],Neurons_startStop[1]*5,0]
    Lessons_startStop = [1,7,0]
    LessMutDiv_startStop = [0.1,50,1]
    MutDivider_startStop = [5,100,0]

    mins = [Neurons_startStop[0],Cycles_startStop[0],Lessons_startStop[0],LessMutDiv_startStop[0],MutDivider_startStop[0]]
    maxs = [Neurons_startStop[1],Cycles_startStop[1],Lessons_startStop[1],LessMutDiv_startStop[1],MutDivider_startStop[1]]
    resolution=[Neurons_startStop[2],Cycles_startStop[2],Lessons_startStop[2],LessMutDiv_startStop[2],MutDivider_startStop[2]]

    #G
    testPoints = generateSobolCharacterizationPoints(len(toSobolTest),sobolTestPts,mins,maxs,resolution,sobolSeed)


    file_name='SobolCharacterizationOfNNet_'+str(GENERATIONS)+'Gens_'+str(CREATURE_COUNT)+'Creats_'+str(INPUT_COUNT)+'Ins_'+str(OUTPUT_COUNT)+'Outs'


    charFileName = createSobolFiles(FILE_LOCATION,file_name, GENERATIONS, CREATURE_COUNT, INPUT_COUNT,OUTPUT_COUNT,outputRelations)
    writeSobolFileRow(charFileName,toSobolTest+['Strength'])

    ''' Uncomment to force specific test points'''
##    testPoints=[]
##    testPoints.append([10,25,4,1,16])
####    testPoints.append([9,7,1,4,54])
####    testPoints.append([5,8,3,12,71])
####    testPoints.append([8,18,1,8,38])
####    testPoints.append([6,22,2,11,91])
####    testPoints.append([10,6,4,19,35])
##    testPoints.append([9,19,4,2,94])

    for i in range(sobolTestPts):
        NEURON_COUNT= int(testPoints[i][0])
        CYCLES_PER_RUN = int(testPoints[i][1])
        LESSONS_PER_TEST = int(testPoints[i][2])
        LESSON_MUT_DIVIDER = testPoints[i][3]
        MUT_DIVISOR = testPoints[i][4]

        for k in range(len(toSobolTest)):
            if k < 3:
                print toSobolTest[k],"=",testPoints[i][k]
            else:
                print toSobolTest[k],"=",testPoints[i][k]


        BestFits=[]
        bestOutputs=[]
        trainOutputs=[]
        testStrength=[]

        for p in range(POPS_TO_TEST):
            details_file_name='SobolGenerationDetails_'+str(GENERATIONS)+'Gens_'+str(CREATURE_COUNT)+'Creats_'+str(INPUT_COUNT)+'Ins_'+str(OUTPUT_COUNT)+'Outs_'+str(p)+"Evol_"+str(NEURON_COUNT)+"N_"+str(CYCLES_PER_RUN)+"Cyc_"+str(LESSONS_PER_TEST)+"L_"+str(LESSON_MUT_DIVIDER)+"LMD_"+str(MUT_DIVISOR)+"GMD"
            detailsFileName = createSobolFiles(FILE_LOCATION,details_file_name,GENERATIONS, CREATURE_COUNT, INPUT_COUNT,OUTPUT_COUNT,outputRelations)

            headers =["Generation","Best Fitness"]
            htemp = []
            for o in range(OUTPUT_COUNT):
                headers.append("Best Output "+str(o))
                htemp.append("Train Output "+str(o))

            toWrite =[]
            toWrite.append(["Neurons:",NEURON_COUNT,"Cycles:",CYCLES_PER_RUN,"Lessons:",LESSONS_PER_TEST,"Lesson Mut Div:",LESSON_MUT_DIVIDER,"Generation Mut Divr:",MUT_DIVISOR])
            toWrite.append([])
            toWrite.append(headers+htemp)
            writeSobolFileMultiRows(detailsFileName,toWrite)

            BestFits.append([])
            bestOutputs.append([])
            trainOutputs.append([])

            population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )

            for G in range (GENERATIONS):
                print "|||||||||||||| POINT:",i," EVOLUTION:",p,", GENERATION:",G,"||||||||||||||"

                #printPopulation (population)
                #printCreature(population.creatureList[0])
                #printSynapse(population.creatureList[0].synapseList[0])

                if G != 0: #Don't mutate the first round (no need to)
                    population.mutate()

                population.populate()

                testedPoints =[]
                for trainIndex in range(len(inList)):
                    tstPt = choice(inList)
                    if tstPt not in testedPoints:
                        testedPoints.append(tstPt)
                        population.setTrainingCreature(tstPt)
                        population.compete( CYCLES_PER_RUN , LESSONS_PER_TEST)

                        bestOutputs[-1].append([])
                        trainOutputs[-1].append([])
                        BestFits[-1].append([])
                        for c in range (len(population.creatureList[0].output)):

                            bestOutputs[-1][-1].append(population.creatureList[0].output[c].outbox)
                            trainOutputs[-1][-1].append(population.trainingCreature.output[c].outbox)

                        BestFits[-1][-1].append(population.creatureList[0].fitness)


                    else:
                        trainIndex -= 1

                population.resolve()



            testStrength.append( calculateSpeciesFitness_goverG2(bestOutputs[-1],trainOutputs[-1]))
            print 'Final Species Fitness:',testStrength[-1]


            toWrite = []
            for G in range(GENERATIONS):
                for trainInd in range(len(inList)):
                    toWrite.append([G]+BestFits[p][G+trainInd]+bestOutputs[p][G+trainInd]+trainOutputs[p][G+trainInd])

            toWrite.append(['Final Species Fitness:',testStrength[-1]])
            toWrite.append([" "])

            writeSobolFileMultiRows(detailsFileName,toWrite)

            createFig_creature_exhaustiveTrainingSpace(population,population.creatureList[0],CYCLES_PER_RUN,inList,"So"+str(i)+"Ev"+str(p))

        toWrite = []
        for testS in testStrength:
            toWrite.append([NEURON_COUNT,CYCLES_PER_RUN,LESSONS_PER_TEST,LESSON_MUT_DIVIDER,MUT_DIVISOR,testS])
        writeSobolFileMultiRows(charFileName,toWrite)

        createFig_DistHistogram(testStrength,5,'Species Fitness','Probability')

    plt.show()

    '''


    ##    print "training outs:"
    ##    for o in range (len(population.trainingCreature.output)):
    ##        print population.trainingCreature.output[o].outbox
    ##    print "best creature outs:"
    ##    for c in range (len(population.creatureList[0].output)):
    ##        print population.creatureList[0].output[c].outbox
    ##    print

if __name__ == "__main__":
	main()
'''