from math import *
from random import *
from copy import *
from pickle import *
from sobol_lib_NoNumpy import *
from population import *
from creature import *
from creatureGUI import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import time
#from matplotlib.ticker import NullFormatter
#from mpl_toolkits.mplot3d import Axes3D
import csv


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

        dist = 0
        for inp in inList:                          #Test it against training creature for all combinations of inputs
            population.setTrainingCreature(inp)
            creature.run(inp, cycles)

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
    Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])


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
    #creature.run(population, cycles)
    creatOuts = []
    trainOuts = []
    for inp in inList:
        population.setTrainingCreature(inp)
        creature.run(inp, cycles)
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

def evolve_species(CreatCount, NeurCount, InCount, OutCount, Gens, Cycles, Lessons,lessonMutateDiv,genMutateDiv):

    population = Population(CreatCount, NeurCount, InCount, OutCount,Cycles, Lessons, lessonMutateDiv,genMutateDiv)
    bestOutputs = []
    trainOutputs = []
    bestFits = []
    testStrength = []

    for G in range (Gens):







        print "|||||||||||||| GENERATION:",G,"||||||||||||||"
        if G != 0: #Don't mutate the first round (no need to)
            population.mutate(lessonMutateDiv)

        population.populate()

        testedPoints =[]
        for trainIndex in range(len(inList)):
            tstPt = choice(inList)
            if tstPt not in testedPoints:
                testedPoints.append(tstPt)
                population.setTrainingCreature(tstPt)
                population.compete( Cycles , Lessons,lessonMutateDiv)

                bestOutputs.append([])
                trainOutputs.append([])
                bestFits.append([])
                for c in range (len(population.creatureList[0].output)):

                    bestOutputs[-1].append(population.creatureList[0].output[c].outbox)
                    trainOutputs[-1].append(population.trainingCreature.output[c].outbox)

                bestFits[-1].append(population.creatureList[0].fitness)


            else:
                trainIndex -= 1

        population.resolve(genMutateDiv)

    testStrength= calculateSpeciesFitness_goverG2(bestOutputs,trainOutputs)

    return bestFits, bestOutputs, trainOutputs, testStrength


def run_sobol(evolNum,testPoint,Gens, Creats, InCount,OutCount,charFileName):
    NEURON_COUNT= int(testPoint[0])
    CYCLES_PER_RUN = int(testPoint[1])
    LESSONS_PER_TEST = int(testPoint[2])
    LESSON_MUT_DIVIDER = testPoint[3]
    MUT_DIVISOR = testPoint[4]

    for k in range(len(toSobolTest)):
        if k < 3:
            print toSobolTest[k],"=",testPoint[k]
        else:
            print toSobolTest[k],"=",testPoint[k]


    BestFits=[]
    bestOutputs=[]
    trainOutputs=[]
    testStrength=[]

    for p in range(POPS_TO_TEST):

        print "|||||||||||||| POPULATION:",p,"||||||||||||||"
        details_file_name='SobolGenerationDetails_'+str(Gens)+'Gens_'+str(Creats)+'Creats_'+str(InCount)+'Ins_'+str(OutCount)+'Outs_'+str(evolNum)+"Evol_"+str(NEURON_COUNT)+"N_"+str(CYCLES_PER_RUN)+"Cyc_"+str(LESSONS_PER_TEST)+"L_"+str(LESSON_MUT_DIVIDER)+"LMD_"+str(MUT_DIVISOR)+"GMD"
        detailsFileName = createSobolFiles(FILE_LOCATION,details_file_name,Gens, Creats, InCount,OutCount,outputRelations)

        headers =["Generation","Best Fitness"]
        htemp = []
        for o in range(OutCount):
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

        results = evolve_species(Creats, NEURON_COUNT, InCount, OutCount, Gens, CYCLES_PER_RUN, LESSONS_PER_TEST,LESSON_MUT_DIVIDER,MUT_DIVISOR)

        BestFits[-1].append(results[0])
        bestOutputs[-1].append(results[1])
        trainOutputs[-1].append(results[2])
        testStrength.append(results[3])

        print 'Final Species Fitness:',testStrength[-1]


        toWrite = []
        for G in range(GENERATIONS):
            for trainInd in range(len(inList)):
                print p, G, trainInd
                toWrite.append([G]+BestFits[p][G+trainInd]+bestOutputs[p][G+trainInd]+trainOutputs[p][G+trainInd])

        toWrite.append(['Final Species Fitness:',testStrength[-1]])
        toWrite.append([" "])

        writeSobolFileMultiRows(detailsFileName,toWrite)

        #createFig_creature_exhaustiveTrainingSpace(population,population.creatureList[0],CYCLES_PER_RUN,inList,"So"+str(i)+"Ev"+str(p))

    toWrite = []
    for testS in testStrength:
        toWrite.append([NEURON_COUNT,CYCLES_PER_RUN,LESSONS_PER_TEST,LESSON_MUT_DIVIDER,MUT_DIVISOR,testS])
    writeSobolFileMultiRows(charFileName,toWrite)

    #createFig_DistHistogram(testStrength,5,'Species Fitness','Probability')


if __name__ == "__main__":
    '''

    '''
    GENERATIONS = 50 #50
    CREATURE_COUNT = 100 #100
    INPUT_COUNT = 1
    OUTPUT_COUNT = 1
    sobolTestPts = 1
    # next seed = 9
    sobolSeed = 0 #Which sobol point to start from. Remeber, 0 indexed
    POPS_TO_TEST=1

    MAX_VALUE = 10
    #50 Gen, 100 Creat,2In, 3Out, 20 sobol, 1 pops =~ 20 to 50 min. On Chris' laptop. Depending on start/stop values

    FILE_LOCATION =r"C:\Users\chris.nelson\Desktop\NNet\ExhaustiveTrainingPerGen"
    #Relationships between inputs and outputs for this training set, only used in results file
    outputRelations = [r"In[0]+0.5*In[1]",r"In[0]&In[1]",r"In[0](or)In[1]"]

    #Both of these are optional:
    #inList = [[0,0],[0,1],[1,0],[1,1]]
    inList = [[0],[1]]
    outList = [[0,0,0],[1,0,1],[1,0,1],[0,1,1]]
    outThreshList = [[0.4,0.6],[0.4,0.6],[0.4,0.6]]
    #outSigmas = [0.3,0.3,0.3]
    outSigmas = [0.5]

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
    testPoints=[]
    #testPoints.append([10,25,4,1,16])
    testPoints.append([4,25,4,1,16])
####    testPoints.append([9,7,1,4,54])
####    testPoints.append([5,8,3,12,71])
####    testPoints.append([8,18,1,8,38])
####    testPoints.append([6,22,2,11,91])
####    testPoints.append([10,6,4,19,35])
##    testPoints.append([9,19,4,2,94])

    for i in range(sobolTestPts):
        print testPoints[i]

        print "|||||||||||||| POINT:",i,"||||||||||||||"

        run_sobol(i,testPoints[i],GENERATIONS, CREATURE_COUNT, INPUT_COUNT,OUTPUT_COUNT,charFileName)
        '''
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

        '''
        '''
            population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN,LESSONS_PER_TEST )

            for G in range (GENERATIONS):
                print "|||||||||||||| POINT:",i," EVOLUTION:",p,", GENERATION:",G,"||||||||||||||"

                #printPopulation (population)
                #printCreature(population.creatureList[0])
                #printSynapse(population.creatureList[0].synapseList[0])

                if G != 0: #Don't mutate the first round (no need to)
                    population.mutate()

                population.populate()



                population.setTrainingCreature()
                population.compete( CYCLES_PER_RUN , LESSONS_PER_TEST)

                bestOutputs[-1].append([])
                trainOutputs[-1].append([])
                BestFits[-1].append([])
                for c in range (len(population.creatureList[0].output)):

                    bestOutputs[-1][-1].append(population.creatureList[0].output[c].outbox)
                    trainOutputs[-1][-1].append(population.trainingCreature.output[c].outbox)

                BestFits[-1][-1].append(population.creatureList[0].fitness)

        '''
        '''
                testedPoints = []
                while (len(testedPoints) != len(inList)):
                    tstPt = choice(inList)
                    if tstPt not in testedPoints:
                        print 'Training Inputs:',tstPt

                        testedPoints.append(tstPt)
                        population.setTrainingCreature(tstPt)
                        population.compete( CYCLES_PER_RUN , LESSONS_PER_TEST)

                        bestOutputs[-1].append([])
                        trainOutputs[-1].append([])
                        BestFits[-1].append([])
                        for c in range (len(population.creatureList[0].output)):

                            bestOutputs[-1][-1].append(population.creatureList[0].output[c].outbox)
                            trainOutputs[-1][-1].append(population.trainingCreature.output[c].outbox)

                        print 'Training Outputs:',trainOutputs[-1][-1]
                        print 'Best Outputs:',bestOutputs[-1][-1]
                        BestFits[-1][-1].append(population.creatureList[0].fitness)



                population.compete_exhaustiveLessons(inList,outSigmas)

                population.resolve()

        '''
        '''
            results = evolve_species(CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, GENERATIONS, CYCLES_PER_RUN, LESSONS_PER_TEST)
            BestFits.append(results[0])
            bestOutputs.append(results[1])
            trainInd.append(results[2])
            testStrength.append(results[3])

            #testStrength.append( calculateSpeciesFitness_goverG2(bestOutputs[-1],trainOutputs[-1]))
            print 'Final Species Fitness:',testStrength[-1]


            toWrite = []
            for G in range(GENERATIONS):
                for trainInd in range(len(inList)):
                    toWrite.append([G]+BestFits[p][G+trainInd]+bestOutputs[p][G+trainInd]+trainOutputs[p][G+trainInd])

            toWrite.append(['Final Species Fitness:',testStrength[-1]])
            toWrite.append([" "])

            writeSobolFileMultiRows(detailsFileName,toWrite)

            #createFig_creature_exhaustiveTrainingSpace(population,population.creatureList[0],CYCLES_PER_RUN,inList,"So"+str(i)+"Ev"+str(p))

        toWrite = []
        for testS in testStrength:
            toWrite.append([NEURON_COUNT,CYCLES_PER_RUN,LESSONS_PER_TEST,LESSON_MUT_DIVIDER,MUT_DIVISOR,testS])
        writeSobolFileMultiRows(charFileName,toWrite)

        #createFig_DistHistogram(testStrength,5,'Species Fitness','Probability')
    '''
    #plt.show()

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
