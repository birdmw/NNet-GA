#==========================
#
# VERSION A03 2_13_2015
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

from creature import *

class CreatureGUI:
    def __init__(self ,creature):
        self.creature = creature

    def seeCreature():
        w = Canvas(master, width=screenWidth*.75, height=screenHeight*.75)
        canvasWidth = int(w.cget("width"))
        canvasHeight = int(w.cget("height"))
        canvasCenter = (canvasWidth/2,canvasHeight/2)
        w.pack()
        neuronCount=len (self.creature.neuronList)
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

            if self.creature.neuronList[n] in self.creature.input:
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
            elif self.creature.neuronList[n] in self.creature.output:
                neuronLabel.append(Label(master,text = "output " + str(self.creature.output.index(self.creature.neuronList[n]))))
                labelX=neuronCenter[n][0]+neuronRadius*cos(n*2*pi/neuronCount)+30*cos(n*2*pi/neuronCount)
                labelY=neuronCenter[n][1]+neuronRadius*sin(n*2*pi/neuronCount)+30*sin(n*2*pi/neuronCount)
                outStringVar.append(StringVar())
                outStringVar[-1].set(str(round(self.creature.neuronList[n].outbox,2)))
                outputLabel.append(Label(master,textvariable = outStringVar[-1]))
                outX=neuronCenter[n][0]
                outY=neuronCenter[n][1]
                outputLabel[n-(len(self.creature.neuronList)-len(self.creature.output))].place(x=outX,y=outY,anchor=CENTER)
                neuronLabel[n].place(x=labelX,y=labelY, anchor = CENTER)
            else:
                outX=neuronCenter[n][0]
                outY=neuronCenter[n][1]
                neuronLabel.append(Label(master,text = "n " + str(self.creature.neuronList.index(self.creature.neuronList[n]))))
                labelX=neuronCenter[n][0]+neuronRadius*cos(n*2*pi/neuronCount)+30*cos(n*2*pi/neuronCount)
                labelY=neuronCenter[n][1]+neuronRadius*sin(n*2*pi/neuronCount)+30*sin(n*2*pi/neuronCount)
                neuronLabel[n].place(x=labelX,y=labelY, anchor = CENTER)
            w.create_oval(x0,y0,x1,y1)
        synapseLines = []
        maxA = 0
        minA = 0
        for s in self.creature.synapseList:
            maxA = max(abs(s.a),maxA)
            #print maxA
            minA = 0
        for s in self.creature.synapseList:
            neuron1index=self.creature.neuronList.index(s.n1)
            neuron2index=self.creature.neuronList.index(s.n2)
            sWidth = 40/len (self.creature.neuronList)*(abs(s.a)-minA)/(maxA-minA)
            #print sWidth
            w.create_line( neuronCenter[neuron1index][0], neuronCenter[neuron1index][1], neuronCenter[neuron2index][0], neuronCenter[neuron2index][1], width=sWidth )
            if neuron1index==neuron2index:
                w.create_oval(neuronOval[neuron1index][0],neuronOval[neuron1index][1],neuronOval[neuron1index][2],neuronOval[neuron1index][3], width = sWidth)


        w.create_line(canvasWidth-200,canvasHeight-100,canvasWidth-180,canvasHeight-100,width=40/len (self.creature.neuronList)*(maxA-minA)/(maxA-minA))
        scaleAvar=StringVar()
        L1=Label(master,text = "a = ")
        L2=Label(master,textvariable=scaleAvar)
        L1.place(x=canvasWidth-160,y=canvasHeight-100, anchor = CENTER)
        L2.place(x=canvasWidth-150,y=canvasHeight-100, anchor = W)
        scaleAvar.set(str(round(maxA,2)))


        step = Button(master, text="Step", command=callback)
        step.place(x=canvasCenter[0],y=0,anchor=N)
        mainloop()


    def callback():
        inlist = []
        outlist = self.creature.output
        for i in range(len(entry)):
            try:
                inlist.append( float(entry[i].get()) )
            except:
                inlist.append(0.0)
                print "input strange or empty - using 0.0"
##        local_setTrainingCreature(population,inList=inlist, outList = outlist)
##        local_run(inlist,creature,1)
        self.creature.run()
        updateGraph()
        for o in range(len(self.creature.output)):
            outStringVar[o].set(str(round(self.creature.output[o].outbox,2)))

    def local_run( cycles ):
        for r in range( cycles ):
            for i in range ( self.creature.inputCount ):
                self.creature.input[i].inbox += population.trainingCreature.input[i].inbox
            for n in self.creature.neuronList:
                n.run()
            for s in self.creature.synapseList:
                s.run()
        updateGraph(  )

    def local_setTrainingCreature( population, inList = None, outList = None ):
        for i in range(len(inList)):
            population.trainingCreature.input[i].inbox = inList[i]
        for i in range(len(outList)):
            population.trainingCreature.output[i].outbox = outList[i]

    def updateGraph():
        findMaxValues()

    def findMaxValues():

        maxNeuronProps = []
        maxSynapseProps = []

        for p in self.creature.synapseList[0].propertyList:
            maxNeuronProps.append(p)

        for p in self.creature.neuronList[0].propertyList:
            maxSynapseProps.append(p)

        for s in self.creature.synapseList:
            for i in range(len(maxSynapseProps)):
                maxSynapseProps[i] = max(maxSynapseProps[i],s.propertyList[i])

        for n in self.creature.neuronList:
            for i in range(len(maxNeuronProps)):
                maxSynapseProps[i] = max(maxSynapseProps[i],n.propertyList[i])

        return maxSynapseProps+maxNeuronProps








def main():

    neuronCount =3

    inputCount = 1
    outputCount = 1
    MaxCycles = 1


    demoCreature = Creature(neuronCount, inputCount, outputCount,MaxCycles)
    demoCreature.expectedOutputs = [1]

    seeCreature(demoCreature)


if __name__ == '__main__':
    main()
