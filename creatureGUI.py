#==========================
#
# VERSION A02 2_12_2015
#
# THIS IS A PRE-ALPHA VERSION FOR DEMO PURPOSES
# NOT INTENDED FOR PRACTICAL APPLICATION YET
#
# EXAMPLE OF USE:
#
# from creatureGUI import *
#
# seeCreature(population, creature)
#
#==========================

from Tkinter import *
from math import *
master = Tk()
screenWidth = master.winfo_screenwidth()
screenHeight = master.winfo_screenheight()

def seeCreature(population, creature):

    w = Canvas(master, width=screenWidth*.75, height=screenHeight*.75)
    canvasWidth = int(w.cget("width"))
    canvasHeight = int(w.cget("height"))
    canvasCenter = (canvasWidth/2,canvasHeight/2)
    w.pack()
    neuronCount=len (creature.neuronList)
    circleRadius = canvasHeight/3
    neuronRadius=1.5 * circleRadius / neuronCount
    neuronOval = []
    neuronCenter = []
    entry = []
    neuronLabel = []
    outputLabel = []
    outStringVar = []

    for n in range ( neuronCount ):

        x0 = circleRadius*cos(n*2*pi/neuronCount)+canvasWidth/2-neuronRadius
        y0 = circleRadius*sin(n*2*pi/neuronCount)+canvasHeight/2-neuronRadius
        x1 = circleRadius*cos(n*2*pi/neuronCount)+canvasWidth/2+neuronRadius
        y1 = circleRadius*sin(n*2*pi/neuronCount)+canvasHeight/2+neuronRadius

        neuronCenter.append([(x0+x1)/2,(y0+y1)/2])
        neuronOval.append((x0,y0,x1,y1))

        if creature.neuronList[n] in creature.input:
            entry.append(Entry(master))
            neuronLabel.append(Label(master,text = "input " + str(n)))
            entryX=(x0+x1)/2
            entryY=(y0+y1)/2
            labelX=neuronCenter[n][0]+neuronRadius*cos(n*2*pi/neuronCount)+30*cos(n*2*pi/neuronCount)
            labelY=neuronCenter[n][1]+neuronRadius*sin(n*2*pi/neuronCount)+30*sin(n*2*pi/neuronCount)
            entry[n].place(x=entryX,y=entryY, anchor = CENTER, width = min(max((x1-x0)/2,20),100))
            neuronLabel[n].place(x=labelX,y=labelY, anchor = CENTER)
            outX=neuronCenter[n][0]
            outY=neuronCenter[n][1]
        elif creature.neuronList[n] in creature.output:
            neuronLabel.append(Label(master,text = "output " + str(creature.output.index(creature.neuronList[n]))))
            labelX=neuronCenter[n][0]+neuronRadius*cos(n*2*pi/neuronCount)+30*cos(n*2*pi/neuronCount)
            labelY=neuronCenter[n][1]+neuronRadius*sin(n*2*pi/neuronCount)+30*sin(n*2*pi/neuronCount)

            outStringVar.append(StringVar())
            outStringVar[-1].set(str(round(creature.neuronList[n].outbox,2)))


            outputLabel.append(Label(master,textvariable = outStringVar[-1]))
            outX=neuronCenter[n][0]
            outY=neuronCenter[n][1]

            outputLabel[n-(len(creature.neuronList)-len(creature.output))].place(x=outX,y=outY,anchor=CENTER)
            neuronLabel[n].place(x=labelX,y=labelY, anchor = CENTER)

        else:
            outX=neuronCenter[n][0]
            outY=neuronCenter[n][1]
            neuronLabel.append(Label(master,text = "n " + str(creature.neuronList.index(creature.neuronList[n]))))
            labelX=neuronCenter[n][0]+neuronRadius*cos(n*2*pi/neuronCount)+30*cos(n*2*pi/neuronCount)
            labelY=neuronCenter[n][1]+neuronRadius*sin(n*2*pi/neuronCount)+30*sin(n*2*pi/neuronCount)
            neuronLabel[n].place(x=labelX,y=labelY, anchor = CENTER)
        w.create_oval(x0,y0,x1,y1)

    def callback():
        inlist = []
        outlist = creature.output
        for i in range(len(entry)):
            try:
                inlist.append( float(entry[i].get()) )
            except:
                inlist.append(0.0)
                print "input strange or empty - using 0.0"
        #local_setTrainingCreature(population,inList=inlist, outList = outlist)
        #local_run(inlist,creature,1)
        print inlist
        creature.run(inlist,1)
        updateGraph( creature )
        for o in range(len(creature.output)):
            outStringVar[o].set(str(round(creature.output[o].outbox,2)))

    def local_run( inlist, creature, cycles ):
        for r in range( cycles ):
            for i in range ( creature.inputCount ):
                creature.input[i].inbox += population.trainingCreature.input[i].inbox
            for n in creature.neuronList:
                n.run()
            for s in creature.synapseList:
                s.run()
        updateGraph( creature )

    def local_setTrainingCreature( population, inList = None, outList = None ):
        for i in range(len(inList)):
            population.trainingCreature.input[i].inbox = inList[i]
        for i in range(len(outList)):
            population.trainingCreature.output[i].outbox = outList[i]

    def updateGraph(creature):
        findMaxValues ( creature )

    def findMaxValues ( creature ):
        maxA=-999
        maxB=-999
        maxC=-999
        maxD=-999
        maxT=-999
        for s in creature.synapseList:
            maxA = max(maxA,s.a)
            maxB = max(maxA,s.b)
            maxC = max(maxA,s.c)
            maxD = max(maxA,s.d)
        for n in creature.neuronList:
            maxT = max(maxA,n.threshold)

        return maxA,maxB,maxC,maxD,maxT

    step = Button(master, text="Step", command=callback)
    step.place(x=canvasCenter[0],y=0,anchor=N)
    mainloop()
