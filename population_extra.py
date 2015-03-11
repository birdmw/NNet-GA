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

def pruneByMu ( self ):
    lowMuCreatureList = []
    half = len(self.creatureList)/2
    self.sortByMu()
    for k in range(half):
        self.creatureList.pop()

def pruneByFitness ( self ):
    population.sortByFitness()
    creatureCount = len(self.creatureList)
    for i in range (creatureCount/4):
        self.creatureList.pop()

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
