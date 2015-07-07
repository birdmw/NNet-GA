
    def run_maxCycles(self):
        for cyc in range( self.maxCycles):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

        totalCreatureOutputDifference=0
        if self.expectedOutputs != None:
            for OutInd in range(len(self.output)):
                tOut = self.expectedOutputs[OutInd]
                cOut = self.output[OutInd].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)

            mu=0
            stdev = 1
            self.fitness = (self.fitness + cHelp.myGauss(mu,stdev,totalCreatureOutputDifference) ) / 2

    def run_pattern_1cycle(self):
        for cyc in range( self.maxCycles):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

        totalCreatureOutputDifference=0
        if self.expectedOutputs != None:
            for OutInd in range(len(self.output)):
                tOut = self.expectedOutputs[OutInd]
                cOut = self.output[OutInd].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)

            mu=0
            stdev = 1
            self.fitness = (self.fitness + cHelp.myGauss(mu,stdev,totalCreatureOutputDifference) ) / 2

    def run_untilConverged(self):
        runFitness = 0.0
        outputTracker = []
        self.cycles = 0

        for cyc in range( self.maxCycles):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

            self.cycles +=1

            creatureOutputDifference = 0.0
            if self.expectedOutputs != None:
                for OutInd in range(len(self.output)):
                    tOut = self.expectedOutputs[OutInd]
                    cOut = self.output[OutInd].outbox
                    creatureOutputDifference += abs(tOut-cOut)


                #10 is a magic number. Determine experimentally  THIS IS THE PID ALTERNATIVE TO PLAY WITH!!!
                #runFitness = (runFitness + cHelp.myGauss(0,10,totalCreatureOutputDifference) ) / 2
                mu=0
                stdev = 5
                self.fitness = cHelp.myGauss(mu,stdev,round(creatureOutputDifference,4))
                self.fitnessList.append(self.fitness)


                #Number of starting cycles is magic number. Determine experimentally
                if cyc <= (self.neuronCount**2+10):#*(2.0/3.0):
                    outputTracker.append([])
                    for o in self.output:
                        outputTracker[-1].append(o.outbox)
                else:
                    newVal=[]
                    for o in self.output:
                        newVal.append(o.outbox)
                    outputTracker = outputTracker[1:]+outputTracker[:1]
                    outputTracker[-1] = newVal

                    if cHelp.checkConvergence(outputTracker):
                        break
        # THIS IS THE PID ALTERNATIVE TO PLAY WITH!!!
        #self.fitness = (self.fitness + runFitness ) / 2
