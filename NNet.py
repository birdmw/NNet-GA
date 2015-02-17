from math import *
from random import *
#from copy import *
#from pickle import *
from sobol_lib_NoNumpy import *
from population import *
from creature import *
from creatureGUI import *
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from Tkinter import *
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

def createSobolFiles(fileLoc,fileName, gens, creats, inCount, outCount,outputRelations=None):
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
    if outputRelations !=None:
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

def createFig_creature_exhaustiveTrainingSpace(population,creature,cycles,inList,ID = randint(1,1000)):
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

    #createFig_DistHistogram(testStrength,5,'Species Fitness','Probability')


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


if __name__ == "__main__":

    FILE_LOCATION =r"C:\Users\chris.nelson\Desktop\NNet\ExhaustiveTrainingPerGen"

    GENERATIONS = 50 #50
    CREATURE_COUNT = 100 #100
    INPUT_COUNT = 2
    OUTPUT_COUNT = 1
    POPS_TO_TEST=1

    main()

