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

        dist = 0
        for inp in inList:                          #Test it against training creature for all combinations of inputs
            population.setTrainingCreature(inp)
            creature.run(inp, cycles)

            for outp in range (len(creature.output)):
                dist +=  abs(population.trainingCreature.output[outp].outbox - creature.output[outp].outbox)

        creature.fitness = 1-dist/(1+dist) #Will be '1' when dist = 0, and approach '0' when dist = inf

    population.creatureList.sort(key = lambda x: x.fitness, reverse=True)












def evolveSpecies_basic(CreatCount, NeurCount, InCount, OutCount, Gens, Cycles, Lessons,lessonMutateDiv,genMutateDiv,InputSet=None,OutputSet=None):

    population = Population(CreatCount, NeurCount, InCount, OutCount,Cycles, Lessons, lessonMutateDiv,genMutateDiv)
    bestOutputs = []
    trainOutputs = []
    bestFits = []
    testStrength = []

    for G in range (Gens):
        population.run_generationBasic(InputSet,OutputSet)

    for testPt in range(len(population.statsCreature.fitness)):
        bestOutputs.append([])
        trainOutputs.append([])
        bestFits.append(population.statsCreature.fitness[testPt])
        for outInd in range(len(population.trainingCreature.output)):
            trainOutputs[-1].append(population.statsCreature.output[outInd].outbox[testPt][0])
            bestOutputs[-1].append(population.statsCreature.output[outInd].outbox[testPt][1])

    testStrength= calculateSpeciesFitness_goverG2(bestOutputs,trainOutputs)

    return bestFits, bestOutputs, trainOutputs, testStrength


def evolveSpecies_lesson(CreatCount, NeurCount, InCount, OutCount, Gens, Cycles, Lessons,lessonMutateDiv,genMutateDiv,InputSet=None,OutputSet=None):

    population = Population(CreatCount, NeurCount, InCount, OutCount,Cycles, Lessons, lessonMutateDiv,genMutateDiv)
    bestOutputs = []
    trainOutputs = []
    bestFits = []
    testStrength = []

    for G in range (Gens):
        population.run_generationLessons(InputSet,OutputSet)

    for testPt in range(len(population.statsCreature.fitness)):
        bestOutputs.append([])
        trainOutputs.append([])
        bestFits.append(population.statsCreature.fitness[testPt])
        for outInd in range(len(population.trainingCreature.output)):
            trainOutputs[-1].append(population.statsCreature.output[outInd].outbox[testPt][0])
            bestOutputs[-1].append(population.statsCreature.output[outInd].outbox[testPt][1])

    testStrength= calculateSpeciesFitness_goverG2(bestOutputs,trainOutputs)

    return bestFits, bestOutputs, trainOutputs, testStrength

def evolveSpecies_exhaustiveLesson(CreatCount, NeurCount, InCount, OutCount, Gens, Cycles, Lessons,lessonMutateDiv,genMutateDiv, InputSets, OutputSets = None):

    population = Population(CreatCount, NeurCount, InCount, OutCount,Cycles, Lessons, lessonMutateDiv,genMutateDiv)
    bestOutputs = []
    trainOutputs = []
    bestFits = []
    testStrength = []

    for G in range (Gens):
        print "|||||||||||||| GENERATION:",G,"||||||||||||||"
        population.run_generationExhaustiveLessons(InputSets,OutputSets)

    for testPt in range(len(population.statsCreature.fitness)):
        bestOutputs.append([])
        trainOutputs.append([])
        bestFits.append([population.statsCreature.fitness[testPt]])
        for outInd in range(len(population.trainingCreature.output)):
            trainOutputs[-1].append(population.statsCreature.output[outInd].outbox[testPt][0])
            bestOutputs[-1].append(population.statsCreature.output[outInd].outbox[testPt][1])

    testStrength= calculateSpeciesFitness_goverG2(bestOutputs,trainOutputs)


    localtime = time.localtime(time.time())
    Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
    Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])

    bestFileLocation = save_creature(population.creatureList[0],FILE_LOCATION+"\\MaNigga"+Date+'_'+Time)
    print bestFileLocation

    #seeCreature(population, population.creatureList[0])

    createFig_creature_exhaustiveTrainingSpace(population,population.creatureList[0],Cycles,InputSets)
    return bestFits, bestOutputs, trainOutputs, testStrength, bestFileLocation

def run_sobol_exhaustiveLesson(evolNum,sobolTestPoint,Gens, CreatCount, InCount,OutCount,charFileName,inputSets,outputRelations=None):
    NeurCount= int(sobolTestPoint[0])
    Cycles = int(sobolTestPoint[1])
    Lessons = int(sobolTestPoint[2])
    lessonMutateDiv = sobolTestPoint[3]
    genMutateDiv = sobolTestPoint[4]


    toSobolTest = ['Neurons','Cycles','Lessons','Lesson Mutation Divider','Gen Mutation Divider']
    for k in range(len(toSobolTest)):
        print "      ",toSobolTest[k],"=",sobolTestPoint[k]

    BestFits=[]
    bestOutputs=[]
    trainOutputs=[]
    testStrength=[]

    for p in range(POPS_TO_TEST):

        print "|||||||||| POPULATION:",p,"||||||||||"
        details_file_name='SobolGenerationDetails_'+str(Gens)+'Gens_'+str(CreatCount)+'Creats_'+str(InCount)+'Ins_'+str(OutCount)+'Outs_'+str(evolNum)+"Evol_"+str(NeurCount)+"N_"+str(Cycles)+"Cyc_"+str(Lessons)+"L_"+str(lessonMutateDiv)+"LMD_"+str(genMutateDiv)+"GMD"
        detailsFileName = createSobolFiles(FILE_LOCATION,details_file_name,Gens, CreatCount, InCount,OutCount,outputRelations)

        headers =["Generation","Best Fitness"]
        htemp = []
        for o in range(OutCount):
            headers.append("Best Output "+str(o))
            htemp.append("Train Output "+str(o))

        toWrite =[]
        toWrite.append(["Neurons:",NeurCount,"Cycles:",Cycles,"Lessons:",Lessons,"Lesson Mut Div:",lessonMutateDiv,"Generation Mut Divr:",genMutateDiv])
        toWrite.append([])
        toWrite.append(headers+htemp)
        writeSobolFileMultiRows(detailsFileName,toWrite)

        results = evolveSpecies_exhaustiveLesson(CreatCount, NeurCount, InCount, OutCount, Gens, Cycles, Lessons,lessonMutateDiv,genMutateDiv,inputSets)

        BestFits.append(results[0])
        bestOutputs.append(results[1])
        trainOutputs.append(results[2])
        testStrength.append(results[3])
        bestFileLocation = results[4]

        print 'Final Species Fitness:',testStrength[-1]


        toWrite = []
        for G in range(GENERATIONS):
            for trainInd in range(len(inputSets)):
##                print p,G,trainInd
##                print BestFits[p]
##                print bestOutputs[p]
##                print trainOutputs[p]
                toWrite.append([G]+BestFits[p][G*len(inputSets)+trainInd]+bestOutputs[p][G*len(inputSets)+trainInd]+trainOutputs[p][G*len(inputSets)+trainInd])

        toWrite.append(['Final Species Fitness:',testStrength[-1]])
        toWrite.append([" "])
        toWrite.append(["Best creature file:",bestFileLocation])

        writeSobolFileMultiRows(detailsFileName,toWrite)

    toWrite = []
    for testS in testStrength:
        toWrite.append([NeurCount,Cycles,Lessons,lessonMutateDiv,genMutateDiv,testS])
    writeSobolFileMultiRows(charFileName,toWrite)



def main():
    '''
    IMPORTANT INVESTIGATION AVENUES:
        ELO's

        Covariance should determine WHICH properties to mutate, NOT how much to mutate by:

        Remove generational mutation divisor, each property doesn't necessarily need to mutate the same amount
            Perhaps have each creature know it's own mutation rates for each property.
                Start with constants, perhaps evolve/PID them later.
            Alternatively:
                Current sigmaCreature becomes covarianceCreature
                sigmaCreature is still used for mutational sigmas, but they are NOT computed through covariance (See above)


        PID each input set individually

        To penalize long cycle count, integrate differences over cycles.

        Population fitness/fitnesSigma determined by all un-pruned creature's fitnesses

        Prune more based on 'similarity of population'

    '''

    #Relationships between inputs and outputs for this training set, only used in results file
    sobolTestPts = 10
    # next seed = 9
    sobolSeed = 0 #Which sobol point to start from. Remeber, 0 indexed

    outputRelations = [r"In[0]+0.5*In[1]",r"In[0]&In[1]",r"In[0](or)In[1]"]

    #These are optional: (unless running exhaustive lessons, then inList is required)
    inList = [[0,0],[0,1],[1,0],[1,1]]
    #inList = [[0],[1]]
    #outList = [[0,0,0],[1,0,1],[1,0,1],[0,1,1]]
    #outThreshList = [[0.4,0.6],[0.4,0.6],[0.4,0.6]]
    #outSigmas = [0.3,0.3,0.3]
    #outSigmas = [0.5]

    #Parameters controlled by sobol points
    toSobolTest = ['Neurons','Cycles','Lessons','Lesson Mutation Divider','Gen Mutation Divider']


    #Create sobol characterization overview file
    file_name='SobolCharacterizationOfNNet_'+str(GENERATIONS)+'Gens_'+str(CREATURE_COUNT)+'Creats_'+str(INPUT_COUNT)+'Ins_'+str(OUTPUT_COUNT)+'Outs'
    charFileName = createSobolFiles(FILE_LOCATION,file_name, GENERATIONS, CREATURE_COUNT, INPUT_COUNT,OUTPUT_COUNT,outputRelations)
    writeSobolFileRow(charFileName,toSobolTest+['Strength'])


    #Specfic mins, maxes, and resolution for sobol test points
    #xxx_startStop = [Minimum Value, Maximum Value, Resolution (ie: decimels to the right of 0. Can be negative)]
    Neurons_startStop = [INPUT_COUNT+OUTPUT_COUNT+3,INPUT_COUNT+OUTPUT_COUNT+10,0]
    Cycles_startStop = [Neurons_startStop[0],Neurons_startStop[1]*4,0]
    Lessons_startStop = [1,7,0]
    LessMutDiv_startStop = [0.1,50,1]
    MutDivider_startStop = [5,100,0]

    mins = [Neurons_startStop[0],Cycles_startStop[0],Lessons_startStop[0],LessMutDiv_startStop[0],MutDivider_startStop[0]]
    maxs = [Neurons_startStop[1],Cycles_startStop[1],Lessons_startStop[1],LessMutDiv_startStop[1],MutDivider_startStop[1]]
    resolution=[Neurons_startStop[2],Cycles_startStop[2],Lessons_startStop[2],LessMutDiv_startStop[2],MutDivider_startStop[2]]


    #Determine points to test.
    testPoints = generateSobolCharacterizationPoints(len(toSobolTest),sobolTestPts,mins,maxs,resolution,sobolSeed)

    ''' Uncomment to force specific test points'''
##    testPoints=[]
##    testPoints.append([10,25,4,1,16])
##    testPoints.append([4,25,4,1,16])
##    testPoints.append([9,7,1,4,54])
##    testPoints.append([5,8,3,12,71])
##    testPoints.append([8,18,1,8,38])
##    testPoints.append([6,22,2,11,91])
##    testPoints.append([10,6,4,19,35])
##    testPoints.append([9,19,4,2,94])

    for i in range(sobolTestPts):
        print "|||||| POINT:",i,"||||||"
        run_sobol_exhaustiveLesson(i,testPoints[i],GENERATIONS, CREATURE_COUNT, INPUT_COUNT,OUTPUT_COUNT,charFileName,inList,outputRelations)

    plt.show()
