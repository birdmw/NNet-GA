from creature import *
from math import *
from random import *
import numpy as np
from creatureGUI import *
import time


class Population:
    def __init__(self, CreatureCount, NeuronCount, InputCount, OutputCount,Cycles, Lessons = 1, LessonMutationDivider=1,GenerationMutationDivider=1,MaxValue=10,inputSets = None,outputSets = None):
        self.creatureList = []
        self.creatureCount = CreatureCount
        self.inputCount = InputCount
        self.outputCount = OutputCount
        self.cycles = Cycles
        self.lessons = Lessons
        self.lessonMutDiv = LessonMutationDivider
        self.genMutDiv = GenerationMutationDivider
        self.speciesFitness = 0
        self.MaxValue = MaxValue
        self.inputSets = inputSets
        self.outputSets = outputSets
        self.fitness = 0
        self.fitSigma = 1

        self.bestRepeatCreature = None

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

        for inIndex in range(InputCount):
            self.statsCreature.input[inIndex].inbox = []

        self.statsCreature.fitness = []

        #once we start adding/deleting synapses/neurons, these two will either need to be updated with changes, or removed from the code:
        self.synapseCount = len ( self.sigmaCreature.synapseList )
        self.neuronCount = NeuronCount

        self.populate()


    def populate( self ):
         while (len(self.creatureList) < self.creatureCount): #If there is less than two creatures left, create a random new creature.
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))

    def repopulate( self ):
        while (len(self.creatureList) < 2): #If there is less than two creatures left, create a random new creature.
            self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
        while (len(self.creatureList) < self.creatureCount): #Breed until full population
            mother = choice( self.creatureList )
            father = choice( self.creatureList )
            if not (mother == father):
                child = mate( mother , father )
                self.creatureList.append( child )


    def repopulate_randInjections( self ):
        while (len(self.creatureList) < 2): #If there is less than two creatures left, create a random new creature.
            self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
        while (len(self.creatureList) < self.creatureCount): #Breed until full population
            if randint(1,50) == 1:
                self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
            else:
                mother = choice( self.creatureList )
                father = choice( self.creatureList )
                if not (mother == father):
                    child = mate( mother , father )
                    self.creatureList.append( child )


    def mutate_lesson(self):
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
                    sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutDiv)
                    creature.neuronList[propInd].threshold = gauss(mu,sigma)

                else:
                    propInd -= creatNeuronCount
                    synInd = int(propInd/4)
                    abcd = int(propInd/(synInd+1))
                    if abcd == 0:
                        mu = creature.synapseList[synInd].a
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutDiv)
                        creature.synapseList[synInd].a = gauss(mu,sigma)
                    elif abcd == 1:
                        mu = creature.synapseList[synInd].b
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutDiv)
                        creature.synapseList[synInd].b = gauss(mu,sigma)
                    elif abcd == 2:
                        mu = creature.synapseList[synInd].c
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutDiv)
                        creature.synapseList[synInd].c = gauss(mu,sigma)
                    elif abcd == 3:
                        mu = creature.synapseList[synInd].d
                        sigma = ((1-creature.fitness)*mu+0.1)/(self.lessonMutDiv)
                        creature.synapseList[synInd].d = gauss(mu,sigma)



    def mutate_generation(self):
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
                #percentageToMutate = (1-creature.fitness)/2     #TODO: Evaluate this calculation.
                percentageToMutate = 0.5

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
                        #sigma = self.sigmaCreature.neuronList[propInd].threshold*self.genMutDiv
                        sigma = 1/self.genMutDiv
                        creature.neuronList[propInd].threshold = gauss(mu,sigma)

                    else:
                        propInd -= creatNeuronCount
                        synInd = int(propInd/4)
                        abcd = int(propInd/(synInd+1))
                        if abcd == 0:
                            mu = creature.synapseList[synInd].a
                            #sigma = self.sigmaCreature.synapseList[synInd].a*self.genMutDiv
                            sigma =  1/self.genMutDiv
                            creature.synapseList[synInd].a = gauss(mu,sigma)
                        elif abcd == 1:
                            mu = creature.synapseList[synInd].b
                            #sigma = self.sigmaCreature.synapseList[synInd].b*self.genMutDiv
                            sigma =  1/self.genMutDiv
                            creature.synapseList[synInd].b = gauss(mu,sigma)
                        elif abcd == 2:
                            mu = creature.synapseList[synInd].c
                            #sigma = self.sigmaCreature.synapseList[synInd].c*self.genMutDiv
                            sigma =  1/self.genMutDiv
                            creature.synapseList[synInd].c = gauss(mu,sigma)
                        elif abcd == 3:
                            mu = creature.synapseList[synInd].d
                            #sigma = self.sigmaCreature.synapseList[synInd].d*self.genMutDiv
                            sigma =  1/self.genMutDiv
                            creature.synapseList[synInd].d = gauss(mu,sigma)

    def mutate_generation_random(self):
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
                #print "I'm the best! Suck it."
                pass
            elif (creature == self.creatureList[1]):
                #print "I'm the first worst! Suck it."
                pass
            elif (creature == self.bestRepeatCreature):
                #print "    If it works, don't fix it"
                pass
            else:
                #Calculate percentage of traits to mutate
                percentageToMutate = random()*0.1
                #percentageToMutate = 1-(creature.fitness/self.creatureList[0].fitness)*(random())

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
                        #sigma = self.sigmaCreature.neuronList[propInd].threshold*self.genMutDiv
                        #sigma = 1/self.genMutDiv
                        sigma = random()/self.genMutDiv
                        creature.neuronList[propInd].threshold = gauss(mu,sigma)

                    else:
                        propInd -= creatNeuronCount
                        synInd = int(propInd/4)
                        abcd = int(propInd/(synInd+1))
                        if abcd == 0:
                            mu = creature.synapseList[synInd].a
                            #sigma = self.sigmaCreature.synapseList[synInd].a*self.genMutDiv
                            sigma =  random()/self.genMutDiv
                            creature.synapseList[synInd].a = gauss(mu,sigma)
                        elif abcd == 1:
                            mu = creature.synapseList[synInd].b
                            #sigma = self.sigmaCreature.synapseList[synInd].b*self.genMutDiv
                            sigma =  random()/self.genMutDiv
                            creature.synapseList[synInd].b = gauss(mu,sigma)
                        elif abcd == 2:
                            mu = creature.synapseList[synInd].c
                            #sigma = self.sigmaCreature.synapseList[synInd].c*self.genMutDiv
                            sigma =  random()/self.genMutDiv
                            creature.synapseList[synInd].c = gauss(mu,sigma)
                        elif abcd == 3:
                            mu = creature.synapseList[synInd].d
                            #sigma = self.sigmaCreature.synapseList[synInd].d*self.genMutDiv
                            sigma =  random()/self.genMutDiv
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

##        for i in range(len(inList)):
##            self.trainingCreature.input[i].inbox = inList[i]

##        for i in range(len(outList)):
##            self.trainingCreature.output[i].outbox = outList[i]

        if inList != None:
            for i in range(len(self.trainingCreature.input)):
                self.trainingCreature.input[i].inbox = inList[i]
        else:
            for i in self.trainingCreature.input:
                i.inbox = float(bool(getrandbits(1)))##random bool

        if outList != None:
            print outList

            for o in range(len(self.trainingCreature.output)):
                self.trainingCreature.output[o].outbox = outList[o]
        else:
            self.trainingCreature.output[0].outbox = 1-self.trainingCreature.input[0].inbox
            self.trainingCreature.output[1].outbox = self.trainingCreature.input[0].inbox

        #xor(I0,I1)
        #self.trainingCreature.output[0].outbox = float(bool(self.trainingCreature.input[0].inbox)^bool(self.trainingCreature.input[1].inbox))

        #PAM4
##        self.trainingCreature.output[0].outbox = float(self.trainingCreature.input[0].inbox) + 0.5*float(self.trainingCreature.input[1].inbox)
        #and(I0,I1)
        #self.trainingCreature.output[1].outbox = float(bool(self.trainingCreature.input[0].inbox)&bool(self.trainingCreature.input[1].inbox))
        #or(I0,I1)
        #self.trainingCreature.output[0].outbox = float(bool(self.trainingCreature.input[0].inbox) ^ bool(self.trainingCreature.input[1].inbox))
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
        Updates statsCreature
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        outputSet = []
        for outp in self.trainingCreature.output:
            outputSet.append(outp.outbox)

        sigmas = []
        for outp in self.deltaCreature.output:
            sigmas.append(outp.outbox)

        for creature in self.creatureList:
            creature.run(inputSet,self.cycles)
            creature.fitness = fitness(creature,outputSet,sigmas,self.MaxValue)
        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        #self.creatureList.sort(key = lambda x: x.fitness, reverse=False)
        self.update_statsCreature()

    def compete_similarity(self):
        '''
        Each creature runs once
        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        outputSet = []
        for outp in self.trainingCreature.output:
            outputSet.append(outp.outbox)

        for creature in self.creatureList:
            creature.run(inputSet,self.cycles)
            creature.fitness = fitness_similarity(creature,outputSet,self.MaxValue)
        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        #self.creatureList.sort(key = lambda x: x.fitness, reverse=False)
        self.update_statsCreature()

    def compete_similarity_runUntilConverged(self):
        '''
        Each creature runs once
        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        outputSet = []
        for outp in self.trainingCreature.output:
            outputSet.append(outp.outbox)

        for creature in self.creatureList:
            creature.run_untilConverged(inputSet,self.cycles)
            creature.fitness = fitness_similarity(creature,outputSet,self.MaxValue)
        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        #self.creatureList.sort(key = lambda x: x.fitness, reverse=False)
        self.update_statsCreature()

    def compete_similarity_runUntilConverged_repeatabilityFitness(self):
        '''


        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        '''
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        outputSet = []
        for outp in self.trainingCreature.output:
            outputSet.append(outp.outbox)
        '''
        for creature in self.creatureList:
            #creature.fitness = fitness_repeatability(creature,self.inputSets,2,self.cycles,self.outputSets)
            creature.fitness = fitness_repeatability_withCyclePenalty(creature,self.inputSets,2,self.cycles,self.outputSets)
        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        #self.creatureList.sort(key = lambda x: x.fitness, reverse=False)
        self.update_statsCreature()

    def compete_similarity_runUntilConverged_randHybridFitness(self):
        '''


        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        '''
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        outputSet = []
        for outp in self.trainingCreature.output:
            outputSet.append(outp.outbox)
        '''

        #On average, only run repeatability test once every 10 generations. Otherwise use simple similarity
        repVal = 5
        repeatDecider = randint(1,repVal)
        if repeatDecider == repVal:
            print '      using repeatability fitness'
            percPerCreat = 1.0/(len(self.creatureList))
            runs = 3
            counter = 0
            for creature in self.creatureList:
                if counter%(int(len(self.creatureList)*0.1)) == 0:
                    print '          ',counter*percPerCreat*100,'%'
                inSetCopy = deepcopy(self.inputSets)
                outSetCopy = deepcopy(self.outputSets)
            #creature.fitness = fitness_repeatability(creature,self.inputSets,2,self.cycles,self.outputSets)
                creature.fitness = fitness_repeatability_withCyclePenalty(creature,inSetCopy,runs,self.cycles,outSetCopy)
                creature.age+=1
                counter+=1
        else:
            inputSet = choice(self.inputSets)
            print '      using input set:',inputSet
            for creature in self.creatureList:
                creature.run_untilConverged(inputSet,self.cycles)
                creature.fitness = fitness_similarity_withCyclePenalty(creature,self.outputSets[self.inputSets.index(inputSet)],self.cycles)
                creature.age+=1

        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        #self.creatureList.sort(key = lambda x: x.fitness, reverse=False)
        if repeatDecider == repVal:
            self.bestRepeatCreature = self.creatureList[0]
        self.update_statsCreature()


    def compete_randGDRepFitness(self):
        '''


        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        '''

        percPerCreat = 100.0/(len(self.creatureList))
        repVal = int(len(self.inputSets)*1.5)
        repeatDecider = randint(1,repVal)
        if repeatDecider == repVal:
            print '      using repeatability fitness'
            runs = 3
            counter = 0
            for creature in self.creatureList:
                if counter%(int(len(self.creatureList)*0.1)) == 0:
                    if counter*percPerCreat != 0:
                        print '          ',counter*percPerCreat,'%'
                inSetCopy = deepcopy(self.inputSets)
                outSetCopy = deepcopy(self.outputSets)
            #creature.fitness = fitness_repeatability(creature,self.inputSets,2,self.cycles,self.outputSets)
                creature.fitness = fitness_repeatability_withCyclePenalty(creature,inSetCopy,runs,self.cycles,outSetCopy)
                creature.age+=1
                counter+=1
        else:
            inputSet = choice(self.inputSets)
            print '      using input set:',inputSet
            counter = 0
            for creature in self.creatureList:
                if counter%(int(len(self.creatureList)*0.25)) == 0:
                    if counter*percPerCreat != 0:
                        print '          ',counter*percPerCreat,'%'
                creature.run_untilConverged(inputSet,self.cycles)
                creature.fitness = fitness_gaussianDistance_withCyclePenalty(creature,self.outputSets[self.inputSets.index(inputSet)],self.cycles)
                creature.age+=1

        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        #self.creatureList.sort(key = lambda x: x.fitness, reverse=False)
        if repeatDecider == repVal:
            self.bestRepeatCreature = self.creatureList[0]
        self.update_statsCreature()


    def compete_gaussDistFit(self):
        '''


        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        '''

        percPerCreat = 1.0/(len(self.creatureList))
        inputSet = choice(self.inputSets)
        print '      using input set:',inputSet
        counter = 0
        for creature in self.creatureList:
            if counter%(int(len(self.creatureList)*0.20)) == 0:
                print '          ',counter*percPerCreat*100,'%'

            creature.run_untilConverged(inputSet,self.cycles)
            creature.fitness = fitness_gaussianDistance_withCyclePenalty(creature,self.outputSets[self.inputSets.index(inputSet)],self.cycles)
            creature.age+=1
            counter+=1

        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        self.update_statsCreature()





    def compete_lessons(self):
        '''
        Each creature runs once for every lesson.
        Mutation occurs between each lesson, but only mutations causing improved fitness will be kept
        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        '''
        inputSet = []
        for inp in self.trainingCreature.input:
            inputSet.append(inp.inbox)

        outputSet = []
        for outp in self.trainingCreature.output:
            outputSet.append(outp.outbox)

        sigmas = []
        for outp in self.deltaCreature.output:
            sigmas.append(outp.outbox)


        for l in range(self.lessons):
            if l != 0:
                #Don't mutate the first lesson (generational mutation has not yet been tested)
                self.mutate_lesson()

            for creature in self.creatureList:
                creature.run(inputSet,self.cycles)
                creature.fitness = fitness(creature,inputSet,sigmas,self.MaxValue)
                if l == 0:
                    #On the first lesson, force creature to forget previous 'best' values, thereby keeping generational mutations
                    creature.updateBest()
                else:
                    #Don't evaluate on first lesson, this just saves computations - since updateBest() was already ran on the first lesson, evaluateBest() has no effect
                    creature.evaluateBest()

        #Sort creatures based on fitness
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        self.update_statsCreature()

    def compete_exhaustiveLessons(self, inputSets,outputSets= None):
        '''
        Each creature runs on each input set once, for every lesson.
        Creature fitness is calculated based on it's performance on ALL input sets
        Mutation occurs between each lesson, but only mutations causing improved fitness will be kept
        Updates creatures fitness
        Updates creatureList (sorts based on fitness)
        Updates statsCreature
        Sets trainingCreature's inputs, optionally outputs
        '''
        sigmas = []
        for outp in self.deltaCreature.output:
            sigmas.append(outp.outbox)

        for l in range(self.lessons):
            if l != 0:
                #Don't mutate the first lesson (generational mutation has not yet been tested)
                self.mutate_lesson()

            for creature in self.creatureList:
                #Each creature runs on each input set once
                shuffle(inputSets)
                setFits = []
                for inpInd in range(len(inputSets)):

                    #Each creature sees each set in the training set, but in potentially different orders
                    if outputSets != None:
                        self.setTrainingCreature(inputSets[inpInd],outputSets[inpInd])
                    else:
                        self.setTrainingCreature(inputSets[inpInd])
                        outputSet = [] #Since target output set not provided, must build it.
                        for outp in self.trainingCreature.output:
                            outputSet.append(outp.outbox)

                    creature.run(inputSets[inpInd],self.cycles)

                    if outputSets != None:
                        setFits.append(fitness(creature,outputSets[inpInd],sigmas,self.MaxValue))
                    else:
                        setFits.append(fitness(creature,outputSet,sigmas,self.MaxValue))


                    #Track each training set for the current best creature with statsCreature
                    if self.creatureList.index(creature)==0:
                        self.update_statsCreature()

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
        Appends the current status of the population to statsCreature
            statsCreature.input = appends trainingCreature's inputs
            statsCreature.output = appends [trainingCreature's outputs, best creature's outputs]
            statsCreature.fitness = appends best creature's fitness
        '''
        for i in range(len(self.trainingCreature.input)):
            self.statsCreature.input[i].inbox.append(self.trainingCreature.input[i].inbox)
        for o in range(len(self.creatureList[0].output)):
            self.statsCreature.output[o].outbox.append([self.trainingCreature.output[o].outbox,self.creatureList[0].output[o].outbox])
        self.statsCreature.fitness.append(self.creatureList[0].fitness)
        #print self.statsCreature.fitness


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
            #self.deltaCreature.neuronList[i].outbox = (self.trainingCreature.neuronList[i].outbox - self.avgTopCreature.neuronList[i].outbox)*1
            self.deltaCreature.neuronList[i].outbox = 0.4

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
        percentToPrune = 0.8
        '''
        for creature in self.creatureList:
            #print startLen,self.creatureList.index(creature),creature.fitness
##            if (creature.fitness < 0):
##                print '!!!!! Creature has exceeded MaxValue. !!!!!'
##                #self.creatureList.remove(creature)
##                toBeRemoved.append(creature)
##            elif (creature.fitness < 0.05): #2e-6
##                #print '!!!!! Creature with fitness of ',creature.fitness,'will be removed !!!!!'
##                toBeRemoved.append(creature)
            if self.creatureList.index(creature)  > 9*(startLen-1)/10: #Kill bottom 10%
                #self.creatureList.remove(creature)
                toBeRemoved.append(creature)


        for loser in toBeRemoved:
            self.creatureList.remove(loser)

        '''
        self.creatureList = self.creatureList[:int(percentToPrune*(startLen))]


        if len(self.creatureList)==0:
            print '======== WARNING: ALL CREATURES DIED ========'
            self.populate()
            print '======== !!RANDOMLY REPOPULATED!! ========'


    def updatePopulationFitness(self):
        fitList = []
        for creature in self.creatureList:
            fitList.append(creature.fitness)

        self.fitness = sum(fitList)/len(fitList)

        sumSquares = sum((fit - self.fitness)**2 for fit in fitList)

        self.fitSigma = sumSquares/len(fitList)

    def run_generationBasic(self,inputSet = None,outputSet = None):
        '''
        Runs the population for one generation in the basic (populate)-(run)-(prune)-(mutate) sequence
        '''
        self.repopulate()
        self.mutate_generation()
        self.setTrainingCreature(inputSet,outputSet)
        #self.compete_run()
        self.compete_similarity()
        self.update_pseudoCreatures()
        self.prune()


    def run_generation_runUntilConverged(self,inputSets = None,outputSets = None):
        '''
        Runs the population for one generation in the basic (populate)-(run)-(prune)-(mutate) sequence
        '''
        inputSet = None
        outputSet = None
        self.repopulate()
        #self.mutate_generation()
        self.mutate_generation_random()
        if inputSets != None:
            inputSet = choice(inputSets)
        if outputSets != None:
            outputSet = outputSets[inputSets.index(inputSet)]
        self.setTrainingCreature(inputSet,outputSet)
        #self.compete_run()
        self.compete_similarity_runUntilConverged()
        self.update_pseudoCreatures()
        self.prune()

    def run_generation_runUntilConverged_repeatabilityFitness(self,inputSets,outputSets):
        '''
        Runs the population for one generation in the basic (populate)-(run)-(prune)-(mutate) sequence
        '''
        inputSet = None
        outputSet = None
        self.inputSets = inputSets
        self.outputSets = outputSets
        self.repopulate()
        #self.mutate_generation_random()
        self.mutate_generation()
        '''
        if inputSets != None:
            inputSet = choice(inputSets)
        if outputSets != None:
            outputSet = outputSets[inputSets.index(inputSet)]
        '''
        #self.setTrainingCreature(inputSet,outputSet)
        #self.compete_run()
        self.compete_similarity_runUntilConverged_repeatabilityFitness()
        self.update_pseudoCreatures()
        self.prune()

    def run_generation_runUntilConverged_randHybridFitness(self,inputSets,outputSets):
        '''
        Runs the population for one generation in the basic (populate)-(run)-(prune)-(mutate) sequence
        '''

        self.inputSets = inputSets
        self.outputSets = outputSets
        print '   populating...'
        self.repopulate()
        print '   mutating...'
        self.mutate_generation_random()
        #self.mutate_generation()
        print '   competing...'
        self.compete_similarity_runUntilConverged_randHybridFitness()
        print '   updating pseudos...'
        self.update_pseudoCreatures()
        print '   pruning...'
        self.prune()
        print 'remaining creatures:',len(self.creatureList)
        print "best creature's fitness:",self.creatureList[0].fitness
        print "best creature's cycles:",self.creatureList[0].cycles

    def run_generation_randHybridFitness_randCreatInjection(self,inputSets,outputSets):
        '''
        Runs the population for one generation in the basic (populate)-(run)-(prune)-(mutate) sequence
        '''
        self.inputSets = inputSets
        self.outputSets = outputSets
        print '   populating...'
        self.repopulate_randInjections()
        print '   mutating...'
        self.mutate_generation_random()
        #self.mutate_generation()
        print '   competing...'
        self.compete_similarity_runUntilConverged_randHybridFitness()
        print '   updating pseudos...'
        self.update_pseudoCreatures()
        print '   pruning...'
        self.prune()
        print '   updating pop fitness...'
        self.updatePopulationFitness()

        print "population fitness: mu=",self.fitness," sig=",self.fitSigma
        print "best creature's age:",self.creatureList[0].age
        print "best creature's fitness:",self.creatureList[0].fitness
        print "best creature's cycles:",self.creatureList[0].cycles
        toPrint = []
        for outp in self.creatureList[0].output:
            toPrint.append(outp.outbox)

        print "best creature's outputs:",toPrint

        for inputSet in inputSets:
            print '   Input:',inputSet,'Expected Output:',outputSets[inputSets.index(inputSet)]
            self.creatureList[0].run_untilConverged(inputSet,self.cycles)
            simFit = fitness_similarity_withCyclePenalty(self.creatureList[0],outputSets[inputSets.index(inputSet)],self.cycles)
            toPrint = []
            for outp in self.creatureList[0].output:
                toPrint.append(outp.outbox)
            print '      Outputs: ',toPrint
            print '      Fitness = ',simFit
        brep = fitness_repeatability_printer(self.creatureList[0],inputSets,5,self.cycles,outputSets)
        if brep > 5:
            localtime = time.localtime(time.time())
            Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
            Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])

            fileName = r"C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\midEvolutionExcellent_"+str(brep)+"_"+str(self.creatureList[0].cycles)+'_'+Date+'_'+Time
            save_creature(self.creatureList[0],fileName)
            print '=====EXCELLENT CREATURE SAVED ======'

        print 'remaining creatures:',len(self.creatureList)



    def run_generation_gaussDistFitness_randCreatInjection(self,inputSets,outputSets):
        '''
        Runs the population for one generation in the basic (populate)-(run)-(prune)-(mutate) sequence
        '''
        fitness_gaussianDistance_withCyclePenalty
        self.inputSets = inputSets
        self.outputSets = outputSets
        print '   populating...'
        self.repopulate_randInjections()
        print '   mutating...'
        self.mutate_generation_random()
        #self.mutate_generation()
        print '   competing...'
        self.compete_gaussDistFit()
        print '   updating pseudos...'
        self.update_pseudoCreatures()
        print '   pruning...'
        self.prune()
        print '   updating pop fitness...'
        self.updatePopulationFitness()

        print "population fitness: mu=",self.fitness," sig=",self.fitSigma
        print "best creature's age:",self.creatureList[0].age
        print "best creature's GD fitness:",self.creatureList[0].fitness
        print "best creature's cycles:",self.creatureList[0].cycles
        toPrint = []
        for outp in self.creatureList[0].output:
            toPrint.append(outp.outbox)

        print "best creature's outputs:",toPrint

        for inputSet in inputSets:
            print '   Input:',inputSet,'Expected Output:',outputSets[inputSets.index(inputSet)]
            self.creatureList[0].run_untilConverged(inputSet,self.cycles)
            simFit = fitness_similarity_withCyclePenalty(self.creatureList[0],outputSets[inputSets.index(inputSet)],self.cycles)
            toPrint = []
            for outp in self.creatureList[0].output:
                toPrint.append(outp.outbox)
            print '      Outputs: ',toPrint
            print '      Sim Fitness = ',simFit
        brep = fitness_repeatability_printer(self.creatureList[0],inputSets,5,self.cycles,outputSets)
        if brep > 4:
            localtime = time.localtime(time.time())
            Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
            Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])

            fileName = r"C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\midEvolutionExcellent_"+str(brep)+"_"+str(self.creatureList[0].cycles)+'_'+Date+'_'+Time
            save_creature(self.creatureList[0],fileName)
            print '=====EXCELLENT CREATURE SAVED ======'

        print 'remaining creatures:',len(self.creatureList)


    def run_generation_randGDRepFitness_randCreatInjection(self,inputSets,outputSets):
        '''
        Runs the population for one generation in the basic (populate)-(run)-(prune)-(mutate) sequence
        '''
        fitness_gaussianDistance_withCyclePenalty
        self.inputSets = inputSets
        self.outputSets = outputSets
        print '   populating...'
        self.repopulate_randInjections()
        print '   mutating...'
        self.mutate_generation_random()
        #self.mutate_generation()
        print '   competing...'
        self.compete_randGDRepFitness()
        print '   updating pseudos...'
        self.update_pseudoCreatures()
        print '   pruning...'
        self.prune()
        print '   updating pop fitness...'
        self.updatePopulationFitness()

        print "population fitness: mu=",self.fitness," sig=",self.fitSigma
        print "best creature's age:",self.creatureList[0].age
        print "best creature's GD fitness:",self.creatureList[0].fitness
        print "best creature's cycles:",self.creatureList[0].cycles
        toPrint = []
        for outp in self.creatureList[0].output:
            toPrint.append(outp.outbox)

        print "best creature's outputs:",toPrint

        for inputSet in inputSets:
            print '   Input:',inputSet,'Expected Output:',outputSets[inputSets.index(inputSet)]
            self.creatureList[0].run_untilConverged(inputSet,self.cycles)
            simFit = fitness_similarity_withCyclePenalty(self.creatureList[0],outputSets[inputSets.index(inputSet)],self.cycles)
            toPrint = []
            for outp in self.creatureList[0].output:
                toPrint.append(outp.outbox)
            print '      Outputs: ',toPrint
            print '      Sim Fitness = ',simFit
        brep = fitness_repeatability_printer(self.creatureList[0],inputSets,5,self.cycles,outputSets)
        if brep > 4:
            localtime = time.localtime(time.time())
            Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
            Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])

            fileName = r"C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\midEvolutionExcellent_"+str(brep)+"_"+str(self.creatureList[0].cycles)+'_'+Date+'_'+Time
            save_creature(self.creatureList[0],fileName)
            print '=====EXCELLENT CREATURE SAVED ======'

        print 'remaining creatures:',len(self.creatureList)


    def run_generationLessons(self,inputSet = None,outputSet = None):
        '''
        Runs the population for one generation in the (populate)-(run lessons)-(prune)-(mutate) sequence
        '''
        self.repopulate()
        self.mutate_generation()
        self.setTrainingCreature(inputSet,outputSet)
        self.compete_lessons()
        self.prune()
        self.update_pseudoCreatures()

    def run_generationExhaustiveLessons(self,inputSets,outputSets=None):
        '''
        Runs the population for one generation in the (populate)-(run lessons)-(prune)-(mutate) sequence
        '''
        self.repopulate()
        self.mutate_generation()
        self.compete_exhaustiveLessons(inputSets,outputSets)
        self.prune()
        self.update_pseudoCreatures()


#END POPULATION CLASS


def fitness(creature,mus,sigmas,MaxValue):
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
        if abs(x) > MaxValue:
            return -1
        g = myGauss(mus[out],sigmas[out],x)
        fitSum +=g
        fitMult *= g

    avgFit = fitSum/len(mus)
    fitness = (avgFit+fitMult)/2
    return fitness


def fitness_repeatability(creature,inputSets,runs,maxCycles,outputSets):
    repDist = 0
    shuffle(inputSets)
    for inputSet in inputSets:
        for r in range(runs):
            creature.run_untilConverged(inputSet,maxCycles)
            #creature.run(inputSet,cycles)
            outputs = []
            for outInd in range(len(creature.output)):
                outputs.append(creature.output[outInd].outbox)
                repDist+= distance_calculator(outputSets[inputSets.index(inputSet)][outInd],outputs[-1])


    normalizer = runs*len(outputSets) #Convert total distance to average distance per output set
    if repDist <= 0.0001:
        sim = 10000
    else:
        sim = normalizer/repDist
    return sim

def fitness_repeatability_withCyclePenalty(creature,inputSets,runs,maxCycles,outputSets):
    halfPenalty = 1
    endPenalty = 0.95

    mhalf = (1-halfPenalty)/(-maxCycles/2)
    mend = (endPenalty-halfPenalty)/(maxCycles/2)
    bhalf = 1
    bend = -1*mend*maxCycles/2+halfPenalty

    repDist = 0

    for r in range(runs):
        shuffle(inputSets)
        runDist = 0
        for inputSet in inputSets:
            setDist = 0
            creature.run_untilConverged(inputSet,maxCycles)
            #creature.run(inputSet,cycles)
            outputs = []
            for outInd in range(len(creature.output)):
                outputs.append(creature.output[outInd].outbox)
                setDist+= distance_calculator(outputSets[inputSets.index(inputSet)][outInd],outputs[-1])

            if creature.cycles >= maxCycles/2:
                cyclePen = creature.cycles*mend + bend
            else:
                cyclePen = creature.cycles*mhalf + bhalf

            runDist += setDist/cyclePen

        repDist+=runDist

    normalizer = runs*len(outputSets) #Convert total distance to average distance per output set
    if repDist <= 0.0001:
        sim = 10000
    else:
        sim = normalizer/repDist

    return sim

def fitness_repeatability_printer(creature,inputSets,runs,maxCycles,outputSets):
    halfPenalty = 1
    endPenalty = 0.95

    mhalf = (1-halfPenalty)/(-maxCycles/2)
    mend = (endPenalty-halfPenalty)/(maxCycles/2)
    bhalf = 1
    bend = -1*mend*maxCycles/2+halfPenalty

    repDist = 0

    outputResults = []
    for outpI in outputSets[0]:
        outputResults.append([])

    print 'Repeatability Printer:'
    for r in range(runs):
        print '    Run',r
        shuffle(inputSets)
        runDist = 0
        for inputSet in inputSets:
            setDist = 0
            creature.run_untilConverged(inputSet,maxCycles)
            #creature.run(inputSet,cycles)
            outputs = []
            for outInd in range(len(creature.output)):
                outputs.append(creature.output[outInd].outbox)
                setDist+= distance_calculator(outputSets[inputSets.index(inputSet)][outInd],outputs[-1])

            if creature.cycles >= maxCycles/2:
                cyclePen = creature.cycles*mend + bend
            else:
                cyclePen = creature.cycles*mhalf + bhalf

            runDist += setDist/cyclePen
            print '        Input:',inputSet,' Output:',outputs,' Cycles:',creature.cycles
        repDist+=runDist

    normalizer = runs*len(outputSets) #Convert total distance to average distance per output set
    if repDist <= 0.0001:
        sim = 10000
    else:
        sim = normalizer/repDist

    print 'Repeatability fitness = ',sim
    print ''
    return sim

def fitness_repeatability_printer_noCyclePenalty_badOrder_worthless_noGood_dirty_rotten(creature,inputSets,runs,maxCycles,outputSets):
    repDist = 0
    for inputSet in inputSets:
        print 'Inputs: ',inputSet,' Expected Outputs:',outputSets[inputSets.index(inputSet)]
        for r in range(runs):
            creature.run_untilConverged(inputSet,maxCycles)
            #creature.run(inputSet,cycles)
            outputs = []
            for outInd in range(len(creature.output)):
                outputs.append(creature.output[outInd].outbox)
                repDist+= distance_calculator(outputSets[inputSets.index(inputSet)][outInd],outputs[-1])

            print '  Run',r,' Outputs: ',outputs,' Cycles:',creature.cycles

    normalizer = runs*len(outputSets)
    if repDist <= 0.0001:
        sim = 10000
    else:
        sim = normalizer/repDist
    print 'Repeatability fitness = ',sim
    print ''
    return sim

def fitness_similarity_withCyclePenalty(creature,targets,maxCycles):
    '''
    Calculates the fitness for a creature using similarity. (eg: 1/distance)
    Parameters:
        creature: The creature who's fitness is being calculated
        targets: A list of target outputs (usually from training creature)
    Returns:
        fitness: Bounded between 0 and 10000 via hardcoded check
    '''
    halfPenalty = 1
    endPenalty = 0.95

    mhalf = (1-halfPenalty)/(-maxCycles/2)
    mend = (endPenalty-halfPenalty)/(maxCycles/2)
    bhalf = 1
    bend = -1*mend*maxCycles/2+halfPenalty

    distance = 0
    for outInd in range(len(targets)):
        creatOut = creature.output[outInd].outbox
        distance+=distance_calculator(targets[outInd],creatOut)

    if creature.cycles >= maxCycles/2:
        cyclePen = creature.cycles*mend + bend
    else:
        cyclePen = creature.cycles*mhalf + bhalf

    #Put an upper cap on similarity
    if distance/cyclePen <= 0.0001:
        return 10000

    return cyclePen/distance

def fitness_gaussianDistance_withCyclePenalty(creature,targets,maxCycles):
    '''
    Calculates the fitness for a creature using similarity. (eg: 1/distance)
    Parameters:
        creature: The creature who's fitness is being calculated
        targets: A list of target outputs (usually from training creature)
    Returns:
        fitness: Bounded between 0 and 10000 via hardcoded check
    '''
    halfPenalty = 1
    endPenalty = 0.95

    mhalf = (1-halfPenalty)/(-maxCycles/2)
    mend = (endPenalty-halfPenalty)/(maxCycles/2)
    bhalf = 1
    bend = -1*mend*maxCycles/2+halfPenalty

    distance = 0
    for outInd in range(len(targets)):
        creatOut = creature.output[outInd].outbox
        distance+=distance_calculator(targets[outInd],creatOut)

    if creature.cycles >= maxCycles/2:
        cyclePen = creature.cycles*mend + bend
    else:
        cyclePen = creature.cycles*mhalf + bhalf

    #Put an upper cap on similarity
    if distance/cyclePen <= 0.0001:
        return 10000

    return cyclePen/distance


def fitness_similarity(creature,targets,MaxValue):
    '''
    Calculates the fitness for a creature using similarity. (eg: 1/distance)
    Parameters:
        creature: The creature who's fitness is being calculated
        targets: A list of target outputs (usually from training creature)
    Returns:
        fitness: Bounded between 0 and 1000 via hardcoded check
    '''
    distance = 0
    for outInd in range(len(targets)):
        creatOut = creature.output[outInd].outbox
        if abs(creatOut) > MaxValue:
            return -1

        distance+= distance_calculator(targets[outInd],creatOut)

    #Put an upper cap on similarity
    if distance <= 0.0001:
        return 10000

    fit = myGauss(0,1,distance)
    return fit


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

def distance_calculator(a,b):
    return (abs(a-b)+1)**2
    #return abs(a-b)

def mate (mother, father):
    '''
    Pull out the wine, light some candles, and turn on the charm. Giggity giggity.
    '''
    child = deepcopy( mother )
    child.age = 0
    for i in range(len(child.neuronList)):
        if getrandbits(1):
            child.neuronList[i].threshold =  deepcopy(father.neuronList[i].threshold)
    for i in range(len(child.synapseList)):
        if getrandbits(1):
            child.synapseList[i].a = deepcopy( father.synapseList[i].a)
        if getrandbits(1):
            child.synapseList[i].b = deepcopy( father.synapseList[i].b)
        if getrandbits(1):
            child.synapseList[i].c = deepcopy( father.synapseList[i].c)
        if getrandbits(1):
            child.synapseList[i].d = deepcopy( father.synapseList[i].d)
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

    p1 = -np.power(x-mu,2.)
    p2 = 2*np.power(sig,2.)

    g = np.exp(p1/p2)
    return g



def main():
    CreatureCount = 80000
    NeuronCount = 3
    MaxCycles = 600
    Lessons = 1
    LessonMutationDivider = 1
    GenerationMutationDivider = 10
    MaxValue=2000000

    runs = 5

##    trainingSetInputs = [[0,0],[0,1],[1,0],[1,1]]
##    trainingSetOutputs = [[0,0],[0,1],[1,0],[1,1]]
    trainingSetInputs = [[-2],[0],[2]]
    trainingSetOutputs = [[-2],[0],[2]]


    InputCount = len(trainingSetInputs[0])
    OutputCount = len(trainingSetOutputs[0])

    inSetCopies = deepcopy(trainingSetInputs)
    outSetCopies = deepcopy(trainingSetOutputs)

    demoPop =  Population(CreatureCount, NeuronCount, InputCount, OutputCount,MaxCycles, Lessons, LessonMutationDivider,GenerationMutationDivider,MaxValue)
    '''
    badStart = True
    print 'Finding valid starting population...'
    while(badStart):
        demoPop =  Population(CreatureCount, NeuronCount, InputCount, OutputCount,MaxCycles, Lessons, LessonMutationDivider,GenerationMutationDivider,MaxValue)

        #demoPop.run_generation_runUntilConverged_randHybridFitness(trainingSetInputs,trainingSetOutputs)
        #demoPop.run_generation_randHybridFitness_randCreatInjection(trainingSetInputs,trainingSetOutputs)
        demoPop.run_generation_gaussDistFitness_randCreatInjection(trainingSetInputs,trainingSetOutputs)



        if (demoPop.fitness > 1e-5):
            if (len(demoPop.creatureList) > CreatureCount/100):
                badStart = False
            else:
                print 'Failed. Retrying...'
        else:
            print 'Failed. Retrying...'
    '''

    bRepFit = 0
    genCount = 650
    for g in range(genCount):
        #if ((g) % 10) == 0:
        print ""
        print ""
        print "GENERATION: ",g+1

        #demoPop.run_generation_runUntilConverged(trainingSetInputs,trainingSetOutputs)
        #demoPop.run_generation_runUntilConverged_repeatabilityFitness(trainingSetInputs,trainingSetOutputs)
        #demoPop.run_generation_runUntilConverged_randHybridFitness(trainingSetInputs,trainingSetOutputs)
        #demoPop.run_generation_randHybridFitness_randCreatInjection(trainingSetInputs,trainingSetOutputs)
        #demoPop.run_generation_gaussDistFitness_randCreatInjection(trainingSetInputs,trainingSetOutputs)
        demoPop.run_generation_randGDRepFitness_randCreatInjection(trainingSetInputs,trainingSetOutputs)
        '''
        if ((g) % 10) == 0:
            print "Best Creature:"
            print "  Cycles:",demoPop.creatureList[0].cycles
            #testCreatureRepeatability(demoPop.creatureList[0],trainingSetInputs,runs,MaxCycles)
            bRepFit = fitness_repeatability_printer(demoPop.creatureList[0],inSetCopies,runs,MaxCycles,outSetCopies)
            print ""
            print "Worst Creature (to survive pruning):"
            print "  Cycles:",demoPop.creatureList[-1].cycles
            #testCreatureRepeatability(demoPop.creatureList[-1],trainingSetInputs,runs,MaxCycles)
            fitness_repeatability_printer(demoPop.creatureList[-1],inSetCopies,runs,MaxCycles,outSetCopies)

        if bRepFit > 10:
            print "Found exceptional creature. Discontinuing evolution"
            break
        '''


    localtime = time.localtime(time.time())
    Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
    Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])

    filename = r"C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\bestie4lyfe_"+Date+'_'+Time
    save_creature(demoPop.creatureList[0],filename)


    print '--FINISHED--'
    '''


    print "  Number of creatures:",len(demoPop.creatureList)

    for setInd in range(len(trainingSetInputs)):
        demoPop.setTrainingCreature(trainingSetInputs[setInd],trainingSetOutputs[setInd])
        demoPop.compete_run()
        demoPop.prune()
        demoPop.update_pseudoCreatures()
        demoPop.mutate_generation()

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

        demoPop.repopulate()

    '''
    seeCreature(demoPop.creatureList[0] )


if __name__ == '__main__':
    main()
