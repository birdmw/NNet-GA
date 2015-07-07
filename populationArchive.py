class MutationArchive:
    def mutateByError (self):
        for creature in self.creatureList:
            error = 1-creature.fitness
            for n in range( len ( creature.neuronList ) ):
                creature.neuronList[n].threshold = gauss( creature.neuronList[n].threshold ,error )
            for s in range ( len ( creature.synapseList ) ):
                creature.synapseList[s].a = gauss( creature.synapseList[s].a , error )
                creature.synapseList[s].b = gauss( creature.synapseList[s].b , error )
                creature.synapseList[s].c = gauss( creature.synapseList[s].c , error )
                creature.synapseList[s].d = gauss( creature.synapseList[s].d , error )

    def mutateAbs( self, absVal=1 ):
        for creature in self.creatureList:
            for n in creature.neuronList:
                n.threshold = max(min(gauss( n.threshold , absVal),1000000),-1000000)
            for s in creature.synapseList:
                s.a = max(min(gauss( s.a , absVal ),1000000),-1000000)
                s.b = max(min(gauss( s.b , absVal ),1000000),-1000000)
                s.c = max(min(gauss( s.c , absVal ),1000000),-1000000)
                s.d = max(min(gauss( s.d , absVal ),1000000),-1000000)

    def mutateAbsHalf( self, absVal ):
        half = len(self.creatureList)/2
        for creature in self.creatureList[:half]:
            for n in range( len ( creature.neuronList ) ):
                creature.neuronList[n].threshold = gauss( creature.neuronList[n].threshold , absVal )
            for s in range ( len ( creature.synapseList ) ):
                creature.synapseList[s].a = gauss( creature.synapseList[s].a , absVal )
                creature.synapseList[s].b = gauss( creature.synapseList[s].b , absVal )
                creature.synapseList[s].c = gauss( creature.synapseList[s].c , absVal )
                creature.synapseList[s].d = gauss( creature.synapseList[s].d , absVal )

    def mutateByError (self, multiplier):
        for creature in self.creatureList:
            error = (1-creature.fitness)*multiplier
            for n in range( len ( creature.neuronList ) ):
                creature.neuronList[n].threshold = max(min(gauss( creature.neuronList[n].threshold ,error ),1000000),-1000000)
            for s in range ( len ( creature.synapseList ) ):
                creature.synapseList[s].a = max(min(gauss( creature.synapseList[s].a , error ),1000000),-1000000)
                creature.synapseList[s].b = max(min(gauss( creature.synapseList[s].b , error ),1000000),-1000000)
                creature.synapseList[s].c = max(min(gauss( creature.synapseList[s].c , error ),1000000),-1000000)
                creature.synapseList[s].d = max(min(gauss( creature.synapseList[s].d , error ),1000000),-1000000)


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

class PruneArchive:
    
    def pruneByMu (self):
        print '======== WARNING: CREATURE LIST BEING SORTED  ========'
        self.sortByMu()
        half = len(self.creatureList)/2
        for k in range(half):
            self.creatureList.pop()
            
    def pruneByFitness(self):
        '''
        Will delete bottom half of creature list. And any creatures with extremely low fitness
        '''
        print '======== WARNING: CREATURE LIST BEING SORTED  ========'
        self.sortByFitness()
        startLen = len(self.creatureList)
        toBeRemoved = []
        percentToPrune = 0.5 #Can be adjusted to kill more or less creatures

        self.creatureList = self.creatureList[:int(percentToPrune*(startLen))]


        if len(self.creatureList)==0:
            print '======== WARNING: ALL CREATURES DIED ========'
            self.populate()
            print '======== !!RANDOMLY REPOPULATED!! ========'


class TrainingArchive:
    def trainBySimpleRun(self):
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

    def trainBySimilarity(self):
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

    def trainBySimilarity_runUntilConverged(self):
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

    def trainBySimilarity_runUntilConverged_repeatabilityFitness(self):
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

    def trainBySimilarity_runUntilConverged_randHybridFitness(self):
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


    def trainByRandGDRepFitness(self):
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


    def trainByGaussDistFit(self):
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





    def trainByLessons(self):
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

    def trainByExhaustiveLessons(self, inputSets,outputSets= None):
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

    def setTrainingTrack( self, inputRange= 1):
        for i in self.trainingCreature.input:
            i.inbox = [float(random()*inputRange)]
        for o in self.trainingCreature.output:
            o.outbox = self.trainingCreature.input[0].inbox[0]

    def setTrainingMultiply ( self, multiplier ):
        for i in self.trainingCreature.input:
            i.inbox = random()*multiplier
        for o in self.trainingCreature.output:
            self.trainingCreature.output[0].outbox = float( self.trainingCreature.input[0].inbox * self.trainingCreature.input[1].inbox)

    def setTrainingBools ( self ):
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))
        for o in self.trainingCreature.output:
            if self.trainingCreature.output.index(o)%4==0:
                self.trainingCreature.output[0].outbox = float(  bool(self.trainingCreature.input[0].inbox) ^ bool(self.trainingCreature.input[1].inbox))##<---xor for inputs 0 and 1
            elif self.trainingCreature.output.index(o)%4==1:
                self.trainingCreature.output[1].outbox = float(  bool(self.trainingCreature.input[0].inbox) & bool(self.trainingCreature.input[1].inbox))##<---and for inputs 0 and 1
            elif self.trainingCreature.output.index(o)%4==2:
                self.trainingCreature.output[2].outbox = float(  bool(self.trainingCreature.input[0].inbox) or bool(self.trainingCreature.input[1].inbox))##<---or for inputs 0 and 1
            elif self.trainingCreature.output.index(o)%4==3:
                self.trainingCreature.output[3].outbox = float(~(bool(self.trainingCreature.input[0].inbox) & bool(self.trainingCreature.input[1].inbox)))##<---nand for inputs 0 and 1

class FitnessArchive:    
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

    def setFitnessGauss( self ):
        outputDifferenceList = []
        for creature in self.creatureList:
            totalCreatureOutputDifference = 0.0
            for Out in range(len(creature.output)):
                tOut = self.trainingCreature.output[Out].outbox
                cOut = creature.output[Out].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)
            outputDifferenceList.append(totalCreatureOutputDifference)
        std = np.std(np.array(outputDifferenceList))
        for i in range( len( self.creatureList ) ):
            creature.fitness = myGauss(0,std,outputDifferenceList[i])

    def setFitnessAbs ( self ):
        outputDifferenceList = []
        for creature in self.creatureList:
            totalCreatureOutputDifference = 0.0
            for Out in range(len(creature.output)):
                tOut = self.trainingCreature.output[Out].outbox
                cOut = creature.output[Out].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)
            creature.fitness = totalCreatureOutputDifference

    def normalizeFitness ( self ):
        fitList = []
        for c in self.creatureList:
            fitList.append(c.fitness)
        norm = [float(i)/sum(fitList) for i in fitList]
        norm = [float(i)/max(fitList) for i in fitList]
        for i in range(len(self.creatureList)):
            self.creatureList[i].fitness = norm[i]

class MiscellaneousArchive:
    def sortByFitness( self ):
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

    def clearPopulationData(self):
        for C in self.creatureList:
                C.clearNet()

    def exhaustiveTrial( population ):
        shuffle(population.creatureList)
        for creature1 in population.creatureList:
            for creature2 in population.creatureList:
                if not(creature1 == creature2):
                    population.setTrainingBools()
                    creature1.run( population, CYCLES_PER_RUN)
                    creature2.run( population, CYCLES_PER_RUN)
                    creature1.setFitness(population)
                    creature2.setFitness(population)
                    updateELO(creature1, creature2)
