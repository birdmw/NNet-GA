

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