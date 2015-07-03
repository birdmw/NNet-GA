#!/usr/bin/python
# -*- coding: utf-8 -*-


#from Tkinter import Tk, Canvas, Frame, BOTH, Label,W, StringVar, CENTER,E,S,N,NE, SUNKEN,RAISED,GROOVE,FLAT,RIDGE, DoubleVar, Button, Menu, VERTICAL,SOLID, HORIZONTAL, ALL
from Tkinter import *
import ttk
from ttk import Scrollbar
import tkFileDialog
from math import *
from random import *
import sys
sys.path.append(r'C:\Users\chris.nelson\Desktop\NNet\NNet-GA')
from sobol_lib_NoNumpy import *
import creatureHelper as chelp
from creature import *
from trainer import *
import time
#import ScrollBarClass
#from PIL import Image, ImageTk

from numpy import arange, sin, pi, fft,asarray
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib import pyplot


from matplotlib.figure import Figure

class CreatureGUI_Beta(Frame):

    def __init__(self, parent, Creature=None,inputPatterns=[]):
        Frame.__init__(self, parent)

        self.screenWidth = parent.winfo_screenwidth()
        self.screenHeight = parent.winfo_screenheight()

        self.stopRunning = 0

        self.parent = parent
        self.myCreature = Creature
        self.inputPatterns = inputPatterns
##        print 'self.inputPatterns in gui init',self.inputPatterns
        self.patternPosition = []
        for i in range(len(self.inputPatterns)):
            self.patternPosition.append(0)

        self.inboxStringVars=[]
        self.neurValueVars = []
        self.neurValuePercVars=[]
        self.outputValueVars = []
        self.avgInboxStringVars=[]
        self.neurGridCoords=[]
        self.creatureCanvas=''

        self.canvasList = []
        self.myScope=None
        #self.scopeList=[]

##        self.NeuronsPerRow = 8
##        self.NeuronsPerRow = 11
        self.maxSyn = 16
        self.columnsPerNeuron = 3

        self.roundValue= 3

        self.widthOfColumnsRightOfCreature=0
        self.numColumnsRightOfCreature=0
        self.heigthOfRowsBelowCreature=0
        self.numRowsBelowCreature=0

        self.inputColor = '#055'
        self.outputColor ='#606'
        self.hiddenColor ='#666'
        self.backgroundColor = '#000'
        self.textColor = '#FFFF00'
        self.parent.config(background=self.backgroundColor)
        self.config(background=self.backgroundColor)


        pStyle = ttk.Style()
        progbarThickness = 75
        progBarColor = "#0000DF"
        pStyle.theme_use('classic')
        pStyle.configure("input.Vertical.TProgressbar", troughcolor =self.inputColor, background=progBarColor, thickness=progbarThickness,relief=FLAT)
        pStyle.configure("output.Vertical.TProgressbar", troughcolor =self.outputColor, background=progBarColor, thickness=progbarThickness,relief=FLAT)
        pStyle.configure("hidden.Vertical.TProgressbar", troughcolor =self.hiddenColor , background=progBarColor, thickness=progbarThickness,relief=FLAT)


        self.parent.title("Creature GUI 2.0")
        self.pack(fill=BOTH, expand=1)

        self.initUI()

    def initUI(self):

        if self.myCreature != None:
            if len(self.myCreature.neuronList[-1].synapseList)>self.maxSyn:
                self.NeuronsPerRow = int(floor(self.screenWidth/168.0)) #168 is the width of the neuron GUI without synapses
            else:
                self.NeuronsPerRow = int(floor(self.screenWidth/215.0)) #215 is the width of the neuron GUI with synapses
        else:
            self.NeuronsPerRow = int(floor(self.screenWidth/215.0)) #215 is the width of the neuron GUI with synapses (IE: The minimum possible neurons per row)

        numCols=15
        runCol = 1
        stepCol = 3
        stopCol = 2
        delayCol=0
        creatureCol = 0
        creatureColSpan = numCols-1

        for i in range(numCols):
            self.columnconfigure(i, pad=0)

        numRows=4
        runRow=0
        stepRow=0
        stopRow=0
        delayRow=0
        creatureRow=2
        for i in range(numRows):
            self.rowconfigure(i, pad=0)

        #Create menu
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        menubar.add_cascade(label="File", menu=fileMenu)
        toolMenu = Menu(menubar)
        toolMenu.add_command(label="NeuroloScope", command=self.startScope)
        toolMenu.add_command(label="Pattern Generator", command=self.patternGen)
        menubar.add_cascade(label="Tools", menu=toolMenu)

        #Create buttons
        runCanvas = Canvas(self,background=self.backgroundColor)
        run = Button(runCanvas, text="  Run  ", command=self.runCreature,background=self.backgroundColor, foreground=self.textColor)
        run.pack()
##        run.place(relx=0.5,rely=0.5,anchor=W)
##        irunIcon = Image.open(r'C:\Users\chris.nelson\Desktop\Tkinter\images\green-play-button.png')
##        runIcon = ImageTk.PhotoImage(irunIcon)
##        runCanvas.create_image(5, 5, image=runIcon, anchor=CENTER)
        runCanvas.grid(row=runRow, column=runCol)

        stop = Button(self, text="  Stop  ", command=self.stopCreature,background=self.backgroundColor, foreground=self.textColor)
        stop.grid(row=stopRow, column=stopCol)
        step = Button(self, text="  Step  ", command=self.stepCreature,background=self.backgroundColor, foreground=self.textColor)
        step.grid(row=stepRow, column=stepCol)

        #Create run delay scale
        minDelay=0
        ticks = 0.25
        resolution = 0.01
        maxDelay=0.5 #10 seconds/cycle
        delayBarColor = "#DF0000"
        delayBarLength=self.screenWidth/numCols+10
        self.delayBarVar = DoubleVar()
        delayScale = Scale(self,variable = self.delayBarVar,from_=minDelay,to=maxDelay,width=10,length = delayBarLength,tickinterval=ticks,resolution=resolution,sliderlength=15,orient=HORIZONTAL,cursor='spider',troughcolor = self.textColor,background=self.backgroundColor, foreground=self.textColor, highlightbackground=self.backgroundColor)
        delayScale.grid(row=delayRow,column=delayCol)


        #The next 9000 lines replace 2 lines, just for the sake of... A SCROLL BAR...  but it is a pretty scroll bar..

        #TODO: Look at moving all of the below into displayCreature
        superFrame=Frame(self,borderwidth=0,relief=RAISED,background=self.backgroundColor,highlightbackground=self.backgroundColor)
        superFrame.grid(row=creatureRow,column=creatureCol,columnspan=creatureColSpan, sticky=N+S+E+W)

        self.creatureCanvas = Canvas(superFrame,background = self.backgroundColor, highlightbackground=self.backgroundColor)#,width=self.screenWidth-50,height=self.screenHeight-380)
        self.creatureCanvas.grid(row=0, column=0, sticky=N+S+E+W)
        vscroll = Scrollbar(superFrame, orient=VERTICAL, command=self.creatureCanvas.yview)
        hscroll = Scrollbar(superFrame, orient=HORIZONTAL, command=self.creatureCanvas.xview)
        vscroll.grid(row=0, column=1, sticky=N+S)
        hscroll.grid(row=1, column=0, sticky=E+W)
        self.creatureCanvas["xscrollcommand"] = hscroll.set
        self.creatureCanvas["yscrollcommand"] = vscroll.set

        creatureFrame=Frame(self.creatureCanvas,borderwidth=0,relief=RAISED,background=self.backgroundColor, highlightbackground=self.backgroundColor)
        self.frame_id = self.creatureCanvas.create_window(0,0,window=creatureFrame, anchor=N+W)#,width=self.screenWidth-50,height=self.screenHeight-50)

        for column in range(self.NeuronsPerRow*self.columnsPerNeuron):
            creatureFrame.columnconfigure(column, pad=0)

        if self.myCreature != None:
            self.numNeurons = self.myCreature.neuronCount
            self.numSynapses = len(self.myCreature.synapseList)
            self.generateColors()
            self.displayCreature(creatureFrame)

        creatureFrame.update_idletasks()
        self.creatureCanvas["scrollregion"]=self.creatureCanvas.bbox(ALL)
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        w_offset = 20+self.widthOfColumnsRightOfCreature*self.numColumnsRightOfCreature
        h_offset = 78+self.heigthOfRowsBelowCreature+self.numRowsBelowCreature
        w,h = event.width-w_offset, event.height-h_offset
        self.creatureCanvas.config(width=w, height=h)

    def generateColors(self):
        self.neurColorDict={}
        starts=[]
        stops=[]
        for i in range(3): #There are three colors...RGB
            stops.append(0)
            starts.append(0xff)

        rawColorPoints, endSeed = generateSobolCharacterizationPoints(3,self.numNeurons,starts,stops,0,0)

        for i in range(self.numNeurons):
            newColor = ""
            for rgb in range(3):
                tempNewColor = str(hex(int(rawColorPoints[i][rgb])))
                tempNewColor = tempNewColor[2:]
                if len(tempNewColor) == 1:
                    tempNewColor = "0"+tempNewColor
##                elif len(tempNewColor)==2:
##                    tempNewColor = "0"+tempNewColor
                newColor += tempNewColor

            newColor = "#"+newColor
            self.neurColorDict[str(self.myCreature.neuronList[i].ID)]=(newColor)

    def displayCreature(self,myFrame):
        placedNeurons=0
        row = 0
        while placedNeurons < self.numNeurons:
            myFrame.rowconfigure(row, pad=0)
            for column in range(self.NeuronsPerRow):
                if placedNeurons == self.numNeurons:
                    break
                else:
                    self.addNeuron(myFrame,row+1,column*self.columnsPerNeuron,self.myCreature.neuronList[placedNeurons])
                    placedNeurons+=1
            row+=1

    def onOpen(self):

        ftypes = [('All files', '*')]
        dlg = tkFileDialog.Open(self, filetypes = ftypes)
        fl = dlg.show()

        if fl != '':
##            neuronCount = 17
##            inputCount =3
##            outputCount = 1
##            MaxCycles = 1
##            self.myCreature = Creature(neuronCount, inputCount, outputCount,MaxCycles)

            #Load new creature
            self.myCreature = chelp.load_creature(fl)

            #Create some input pattern.
            patternLength=10
            inputSet = []
            for i in range(len(self.myCreature.input)):
                inputSet.append([])
                amp = randint(1,10)
                for j in range(patternLength):
                    #inputSet[-1].append(randint(0,10))
                    inputSet[-1].append(sin((pi*j)/patternLength)*amp)
                    #inputSet[-1].append(0)

            #Set the input set
            for i in range(len(inputSet)):
                self.myCreature.input[i].inbox = [inputSet[i][0]]

            self.inputPatterns = inputSet
            print 'self.inputPatterns in onOpen',self.inputPatterns
            self.patternPosition = []
            for i in range(len(self.inputPatterns)):
                self.patternPosition.append(0)

            #Wipe all of the creature variables.
            self.inboxStringVars=[]
            self.neurValueVars = []
            self.neurValuePercVars=[]
            self.outputValueVars = []
            self.avgInboxStringVars=[]
            self.neurGridCoords=[]
            self.canvasList=[]

            #Delete the old canvas
            self.creatureCanvas.destroy()

            #Reload UI with new creature
            self.initUI()

    def runCreature(self):
        while(self.stopRunning == 0):
        #for i in range(20):
            self.stepCreature()
            time.sleep(self.delayBarVar.get())
        self.stopRunning = 0

    def stopCreature(self):
        self.stopRunning = 1

    def stepCreature(self):
##        for i in range(len(self.inputPatterns)):
##            if self.patternPosition[i] >= len(self.inputPatterns[i]):
##                self.patternPosition[i] = 0
            #print [self.inputPatterns[i][self.patternPosition[i]]]
            #self.myCreature.input[i].inbox = [self.inputPatterns[i][self.patternPosition[i]]]
            #print self.inputPatterns[i][self.patternPosition[i]]


##            self.myCreature.setInputs([self.inputPatterns[i][self.patternPosition[i]]])
##            self.patternPosition[i] += 1

        if self.patternPosition[0] >= len(self.inputPatterns[0])-1:
            self.patternPosition[0] = 0
        else:
            self.patternPosition[0]+=1
        #run( self, cycles = 1, inputs = [0], ForceTracking = True )
        nextInputs=[]
        for inp in range(len(self.inputPatterns)):
            nextInputs.append(self.inputPatterns[inp][self.patternPosition[0]])

        #self.myCreature.run(1, [self.inputPatterns[0][self.patternPosition[0]]])
        self.myCreature.run(1, nextInputs)
        self.updateVars()
        if self.myScope != None:
            self.myScope.scopeUpdateVars()

    def updateVars(self):
        for n in range(len(self.myCreature.neuronList)):
            neuron = self.myCreature.neuronList[n]
            self.avgInboxStringVars[n].set(round(neuron.avgInput,self.roundValue))
            self.neurValueVars[n].set(round(neuron.value,self.roundValue))
            self.neurValuePercVars[n].set(int(self.calc_progressBarValue((neuron))))
            self.outputValueVars[n].set(round(neuron.outbox,self.roundValue))
            if len(neuron.synapseList)<=self.maxSyn:
                for s in range(len(neuron.synapseList)):
                    sout = neuron.synapseList[s].output
                    if abs(sout) > 1000:
                        self.inboxStringVars[n][s].set("{:.1e}".format(sout))
                    else:
                        self.inboxStringVars[n][s].set(str(round(sout,self.roundValue)))

##            gridCoords = self.neurGridCoords[n]
##            print gridCoords
##            gSlaves = self.parent.grid_slaves(row=gridCoords[0],column=gridCoords[1])
##            print gSlaves

            if neuron.fired==1:
                self.canvasList[n].config(background="#DF0000")
            else:
                if neuron.isInput:
                    self.canvasList[n].config(background=self.inputColor)
                elif neuron.isOutput:
                    self.canvasList[n].config(background=self.outputColor)
                else:
                    self.canvasList[n].config(background=self.hiddenColor)

            self.parent.update()




    def addNeuron(self,myFrame, gridX, gridY, neuron): #, synapseCount):

        self.neurGridCoords.append((gridX,gridY))

        h_inbox = 25# h / self.maxSyn
        w_inbox = 35
        w_inbox_text = 4
        inbox_list = []

        self.inboxStringVars.append([])

        #Create the inbox average label
        self.avgInboxStringVars.append(StringVar())
        self.avgInboxStringVars[-1].set(str(round(neuron.avgInput,2)))

        if neuron.isInput:
            myBodyColor = self.inputColor
        elif neuron.isOutput:
            myBodyColor = self.outputColor
        else:
            myBodyColor = self.hiddenColor

        #If we can fit the inboxes, add them.
        if (len(neuron.synapseList) <= self.maxSyn) and (len(neuron.synapseList) != 0):
            w = 209
            h = 200
            h_body = 115
            w_body = 84

            neuronGUI = Canvas(myFrame, width=w, height=h,background=myBodyColor,highlightthickness=0)#, highlightbackground=self.backgroundColor)

            if neuron.inputSynapseCount <= 8:
                if neuron.inputSynapseCount % 2 == 0:
                    yStart = int(h/2) - int(h_inbox*float(neuron.inputSynapseCount/2+0.5))
                else:
                    yStart = int(h/2) - int(h_inbox*float(neuron.inputSynapseCount/2+1))
            else:
                yStart = int(h/2) - int(h_inbox*float(8/2+0.5))

            yStart+=1
            yPosition = yStart
            xStart = 0
            xPosition = 0

            for synInd in range(len(neuron.synapseList)):
                if synInd <= 7:
                    yPosition += h_inbox
                    self.inboxStringVars[-1].append(StringVar())

                    #Place holder for real values
                    self.inboxStringVars[-1][-1].set(str(round(neuron.synapseList[synInd].output,2)))

                    newSynIn = Label(neuronGUI,width = w_inbox_text, textvariable = self.inboxStringVars[-1][-1],background='#fff')
                    newSynIn.place(x=4,y=yPosition,anchor=W)

                    synColor = self.neurColorDict[str(neuron.synapseList[synInd].n1.ID)]
                    neuronGUI.create_rectangle(xPosition, int(yPosition-h_inbox/2), w_inbox+4, int(yPosition+h_inbox/2), outline=synColor, fill=synColor)
                    #inbox_list.append(newSynIn)

                elif synInd <=11:
                    if synInd == 8:
                        yStart+=h_inbox
                    xPosition+=(w_inbox+3)
                    self.inboxStringVars[-1].append(StringVar())

                    #Place holder for real values
                    self.inboxStringVars[-1][-1].set(str(round(neuron.synapseList[synInd].output,2)))

                    newSynIn = Label(neuronGUI,width = w_inbox_text, textvariable = self.inboxStringVars[-1][-1],background='#fff')
                    newSynIn.place(x=xPosition+4,y=yStart,anchor=W)

                    synColor = self.neurColorDict[str(neuron.synapseList[synInd].n1.ID)]
                    neuronGUI.create_rectangle(xPosition+2, int(yStart-h_inbox/2), xPosition+w_inbox+5, int(yStart+h_inbox/2), outline=synColor, fill=synColor)

                elif synInd <=15:
                    if synInd == 12:
                        xPosition = 0

                    xPosition+=(w_inbox+4)
                    self.inboxStringVars[-1].append(StringVar())

                    #Place holder for real values
                    self.inboxStringVars[-1][-1].set(str(round(neuron.synapseList[synInd].output,2)))

                    newSynIn = Label(neuronGUI,width = w_inbox_text, textvariable = self.inboxStringVars[-1][-1],background='#fff')
                    newSynIn.place(x=xPosition+2,y=yPosition,anchor=W)

                    synColor = self.neurColorDict[str(neuron.synapseList[synInd].n1.ID)]
                    neuronGUI.create_rectangle(xPosition, int(yPosition-h_inbox/2), xPosition+w_inbox+3, int(yPosition+h_inbox/2), outline=synColor, fill=synColor)


            xMarker =int(w_inbox)+6

        else:
            w = 168
            h = 100
            h_body = 83
            w_body = 84
            xMarker =0

            neuronGUI = Canvas(myFrame, width=w, height=h,background=myBodyColor, highlightbackground=self.backgroundColor)

        IDLabel = Label(neuronGUI,width = w_inbox_text-1,text=str(neuron.ID),foreground='#FFFF00',background='#000')
        IDLabel.place(x=xMarker+5+w_inbox/2,y=ceil(h-h_body)/2+2,anchor=N)
        neuronGUI.create_rectangle(xMarker, ceil(int((h-h_body)/2-2)), xMarker+w_inbox+4, ceil(int((h-h_body)/2+h_inbox+4)), fill=self.neurColorDict[str(neuron.ID)],width=0)

        avgSynIn = Label(neuronGUI,width = w_inbox_text, textvariable = self.avgInboxStringVars[-1], background=myBodyColor,foreground='#fff')
        avgY = int(h/2)
        avgSynIn.place(x=xMarker+5+w_inbox/2,rely=0.5,anchor=CENTER)
        neuronGUI.create_rectangle(xMarker, int(avgY-h_inbox/2.0), xMarker+w_inbox+4, int(avgY+h_inbox/2.0+6), fill=self.neurColorDict[str(neuron.ID)],width=0)

        xMarker+=w_inbox+4

        neurBod =self.create_neuronBody(neuronGUI,neuron,w_body,h_body,w_inbox)
        neurBod.place(x=xMarker, rely=0.5, anchor=W)

        xMarker+=w_body+4

        #neuronGUI.create_rectangle(w-w_inbox-8,int(avgY-h_inbox/2.0)-2,w,int(avgY+h_inbox/2.0+6),fill=self.neurColorDict[str(neuron.ID)])
        neuronGUI.create_rectangle(xMarker,int(avgY-h_inbox/2.0),xMarker+w_inbox+8,int(avgY+h_inbox/2.0+6),fill=self.neurColorDict[str(neuron.ID)],width=0)
        self.outputValueVars.append(DoubleVar())
        self.outputValueVars[-1].set(round(neuron.outbox,2))
        outLabel = Label(neuronGUI,width=w_inbox_text,textvariable=str(self.outputValueVars[-1]), background=myBodyColor,foreground='#fff')
        outLabel.place(x=xMarker+5 , rely=0.5, anchor=W)

        neuronGUI.grid(row=gridX, column=gridY, columnspan=self.columnsPerNeuron)

        self.canvasList.append(neuronGUI)

    def calc_progressBarValue(self,neuron):
        if len(neuron.propertyList) == 2:
            neurThreshold=neuron.propertyList[0]
            neurThreshold2=neuron.propertyList[1]

            topPt = neurThreshold
            botPt = neurThreshold2
            if neuron.value > topPt:
                percVal = 1.0
            elif neuron.value < botPt:
                percVal = 0.0
##            elif neuron.value > neurThreshold:
##                percVal=((.5/(topPt-neurThreshold))*(neuron.value-neurThreshold)+1)
            else:
                if topPt == botPt:
                    percVal = 1
                else:
                    percVal = (1.0/(topPt-botPt))*(neuron.value-botPt)
        else:
            neurThreshold=neuron.propertyList[0]

            botPt = neurThreshold - 2*abs(neurThreshold)
            topPt = neurThreshold + abs(neurThreshold)

            if neuron.value > topPt:
                percVal = 1.50
            elif neuron.value < botPt:
                percVal = 0
            elif neuron.value > neurThreshold:
                percVal=((.5/(topPt-neurThreshold))*(neuron.value-neurThreshold)+1)
            else:
                percVal = (1.0/(neurThreshold-botPt))*(neuron.value-botPt)

        percVal=percVal*100
        return percVal

    def create_neuronBody(self,neuronGUI,neuron,w_body,h_body,w_inbox):
        #Create the neuron body
        neuronBody = Canvas(neuronGUI,width=w_body, height=h_body, bd=0)
        neuronBody.config(highlightbackground=self.neurColorDict[str(neuron.ID)], highlightthickness=5)
        if neuron.isInput:
            myBodyColor = self.inputColor
            myStyle = "input.Vertical.TProgressbar"
        elif neuron.isOutput:
            myBodyColor = self.outputColor
            myStyle = "output.Vertical.TProgressbar"
        else:
            myBodyColor = self.hiddenColor
            myStyle = "hidden.Vertical.TProgressbar"

        neuronBody.create_rectangle(0,0,w_body+4,h_body+4, fill=myBodyColor)
        #neuronBody.create_rectangle(5,5,w_body+2,h_body+2, fill="#000")

        self.neurValueVars.append(DoubleVar())
        self.neurValuePercVars.append(DoubleVar())

        neurThreshold = (neuron.propertyList[0])
        self.neurValueVars[-1].set(round(neuron.value,2))
        self.neurValuePercVars[-1].set(int(self.calc_progressBarValue(neuron)))
        threshLineRatio = 1.0/3.0

        #ttk.Progressbar(frame, style="neuron.Vertical.TProgressbar", orient="horizontal", length=600,mode="determinate", maximum=4, value=1).grid(row=1, column=1)


        if len(neuron.propertyList) == 2:
            progBar = ttk.Progressbar(neuronBody, style=myStyle, orient=VERTICAL, length=h_body, mode='determinate', variable=self.neurValuePercVars[-1], maximum=100)
            progBar.place(relx=0.5, rely=0.5, anchor=CENTER)
            threshLineHeight = 30
            valpos = 0.5
        else:
            progBar = ttk.Progressbar(neuronBody, style=myStyle, orient=VERTICAL, length=h_body, mode='determinate', variable=self.neurValuePercVars[-1], maximum=150)
            progBar.place(relx=0.5, rely=0.5, anchor=CENTER)
            threshLineHeight = int(threshLineRatio*h_body+3)
            valpos = 0.55
##        neuronBody.create_line(0,threshLineHeight,w_body,threshLineHeight,fill="#00FFFF",dash=(3,),width=2)
##        neuronBody.pack()

        valLabel = Label(neuronBody,width=5,textvariable=str(self.neurValueVars[-1]),foreground="#00FF00", background="#000", padx=0, pady=0)
        valLabel.place(relx=0.5, rely=valpos, anchor=CENTER)

        threshFrame=Frame(neuronBody,borderwidth=0,relief=SOLID,background='#00FFFF')
        threshFrame.config(highlightbackground='#00FFFF',highlightthickness=2)
        threshLabel = Label(threshFrame,width=5,text=(str(round(neurThreshold,2))),foreground="#00FFFF", background="#000", padx=0, pady=0,bd=1).pack()
        threshFrame.place(relx=0.5, y=threshLineHeight, anchor=S)

        if len(neuron.propertyList) == 2:
            neurThreshold2 = (neuron.propertyList[1])
            thresh2Frame=Frame(neuronBody,borderwidth=0,relief=SOLID,background='#00FFFF')
            thresh2Frame.config(highlightbackground='#00FFFF',highlightthickness=2)
            thresh2Label = Label(thresh2Frame,width=5,text=(str(round(neurThreshold2,2))),foreground="#00FFFF", background="#000", padx=0, pady=0,bd=1).pack()
            thresh2Frame.place(relx=0.5, y=h_body+3, anchor=S)

        return neuronBody

    def startScope(self):
        scopeWindow =Toplevel(self)
        scopeDefX = 1300
        scopeDefY = 800
        scopeWindow.geometry(str(scopeDefX)+'x'+str(scopeDefY))
        self.myScope = NeuroloScope(scopeWindow,self.myCreature,self.inputPatterns,useGUIRun=True,colorDict=self.neurColorDict,GUI=self)


    def patternGen(self):
        PGWindow=Toplevel(self)
        PGDefX = 600
        PGDefY = 400
        PGWindow.geometry(str(PGDefX)+'x'+str(PGDefY))
        self.myPattGen = PatternGenerator(PGWindow,self.myCreature)



class PatternGenerator(Frame):
    def __init__(self, parent, Creature):
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Pattern Generator")
        self.screenWidth = parent.winfo_screenwidth()
        self.screenHeight = parent.winfo_screenheight()
        self.myCreature = Creature

        self.initPGen()

    def initPGen(self):
        self.patternGridColSpan=100
        WaveshapeButtonColumn=4

        for i in range(self.patternGridColSpan):
            self.parent.columnconfigure(i,pad=0)

        #Length of pattern
        pattLCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        pattLTitle = Label(pattLCanvas,text="Cycle Length")
        self.pattLengthInput = Entry(pattLCanvas,width=10,text="1")#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        pattLTitle.grid(row=0,column=0)
        self.pattLengthInput.grid(row=1,column=0)
        pattLCanvas.grid(row=0,column=0)

        inpIDs=[]
        self.inpIDVar = StringVar(self.parent)
        # initial value
        for i in range(len(self.myCreature.input)):
            inpIDs.append(self.myCreature.input[i].ID)

        self.inpIDVar.set(str(inpIDs[0]))
##        choices = ['red', 'green', 'blue', 'yellow','white', 'magenta']
        #option = tk.OptionMenu(root, var, *choices)
        #Which Neuron to set
        neurToSetCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        neurToSetTitle = Label(neurToSetCanvas,text="Input To Set")
        #self.neurToSetInput = Entry(neurToSetCanvas,width=10)#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        self.neurToSetInput = OptionMenu(neurToSetCanvas, self.inpIDVar, *inpIDs)
        neurToSetTitle.grid(row=0,column=0)
        self.neurToSetInput.grid(row=1,column=0)
        neurToSetCanvas.grid(row=1,column=0)

        #Amplitude
        AmpCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        AmpTitle = Label(AmpCanvas,text="Amplitude")
        self.AmpInput = Entry(AmpCanvas,width=10)#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        AmpTitle.grid(row=0,column=0)
        self.AmpInput.grid(row=1,column=0)
        AmpCanvas.grid(row=0,column=1)

        #Offset
        OffCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        OffTitle = Label(OffCanvas,text="Offset")
        self.OffInput = Entry(OffCanvas,width=10)#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        OffTitle.grid(row=0,column=0)
        self.OffInput.grid(row=1,column=0)
        OffCanvas.grid(row=1,column=1)

        #Frequency
        FreqCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        FreqTitle = Label(FreqCanvas,text="Frequency")
        self.FreqInput = Entry(FreqCanvas,width=10)#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        FreqTitle.grid(row=0,column=0)
        self.FreqInput.grid(row=1,column=0)
        FreqCanvas.grid(row=0,column=2)

        #Phase
        PhaseCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        PhaseTitle = Label(PhaseCanvas,text="Phase")
        self.PhaseInput = Entry(PhaseCanvas,width=10)#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        PhaseTitle.grid(row=0,column=0)
        self.PhaseInput.grid(row=1,column=0)
        PhaseCanvas.grid(row=1,column=2)

        #Pad Length
        PadLCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        PadLTitle = Label(PadLCanvas,text="Pad Length")
        self.PadLInput = Entry(PadLCanvas,width=10)#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        PadLTitle.grid(row=0,column=0)
        self.PadLInput.grid(row=1,column=0)
        PadLCanvas.grid(row=0,column=3)

        #Pad Value
        PadValCanvas = Canvas(self.parent,borderwidth=0,relief=RAISED)
        PadValTitle = Label(PadValCanvas,text="Pad Value")
        self.PadValInput = Entry(PadValCanvas,width=10)#,background="#555")#,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5)
        PadValTitle.grid(row=0,column=0)
        self.PadValInput.grid(row=1,column=0)
        PadValCanvas.grid(row=1,column=3)

        #Waveshape buttons:
        sineButton = Button(self.parent, text="Sine",command=self.generateSine)
        sineButton.grid(row=0,column=WaveshapeButtonColumn)
        squareButton = Button(self.parent, text="Square", command=self.generateSquare)
        squareButton.grid(row=0,column=WaveshapeButtonColumn+1)
        squareButton = Button(self.parent, text="Constant", command=self.generateConstant)
        squareButton.grid(row=1,column=WaveshapeButtonColumn)
        squareButton = Button(self.parent, text="Random", command=self.generateRandom)
        squareButton.grid(row=1,column=WaveshapeButtonColumn+1)

        patternLength=10
        self.addPGTable(patternLength)


    def addPGTable(self,patternLength):
        patternGridCol = 0
        patternGridRow = 2
        PGFrameExt=Frame(self.parent,borderwidth=0,relief=RAISED)
        PGFrameExt.grid(row=patternGridRow,column=patternGridCol,columnspan=self.patternGridColSpan,sticky=N+S+E+W)
        PGFrameExt.rowconfigure(0, pad=0)
        PGFrameExt.rowconfigure(1, pad=0)
        PGFrameExt.columnconfigure(0, pad=0)

        self.PGCanvas = Canvas(PGFrameExt,width=self.screenWidth-50,height=self.screenHeight-50)
        self.PGCanvas.grid(row=0, column=0, sticky=N+S+E+W)
        PGvscroll = Scrollbar(PGFrameExt, orient=VERTICAL, command=self.PGCanvas.yview)
        PGhscroll = Scrollbar(PGFrameExt, orient=HORIZONTAL, command=self.PGCanvas.xview)
        PGvscroll.grid(row=0, column=1, sticky=N+S)
        PGhscroll.grid(row=1, column=0, sticky=E+W)
        self.PGCanvas["xscrollcommand"] = PGhscroll.set
        self.PGCanvas["yscrollcommand"] = PGvscroll.set

        PGFrame=Frame(self.PGCanvas,borderwidth=0,relief=RAISED)
        self.frame_id = self.PGCanvas.create_window(0,0,window=PGFrame, anchor=N+W)#,width=self.screenWidth-50,height=self.screenHeight-50)

        textColor = '#FFFF00'
        bgColor = "#000"


        PGFrame.rowconfigure(0, pad=0)
        PGFrame.columnconfigure(0, pad=0)

        PGDefPattLength = patternLength
        for column in range(1,PGDefPattLength+1,1):
            PGFrame.columnconfigure(column, pad=1)
            pattLabel = Label(PGFrame,text=str(column),foreground=textColor,background=bgColor, highlightcolor="#fff")
            pattLabel.grid(row=0,column=column)

        for row in range(1,len(self.myCreature.input)+1,1):
            PGFrame.rowconfigure(row, pad=1)
            neurLabel = Label(PGFrame,text=str(self.myCreature.input[row-1].ID),foreground=textColor,background=bgColor, highlightcolor="#fff")
            neurLabel.grid(row=row,column=0)

        self.patternEntryList=[]
        self.patternVars=[]

        for row in range(1,len(self.myCreature.input)+1,1):
            self.patternEntryList.append([])
            self.patternVars.append([])
            for column in range(1,PGDefPattLength+1,1):
                self.patternVars[-1].append(StringVar())
                self.patternVars[-1][-1].set('1')
                pattInput = Entry(PGFrame,foreground=textColor,background=bgColor, highlightcolor="#fff",width=5,textvariable=self.patternVars[-1][-1])
                self.patternEntryList[-1].append(pattInput)
                pattInput.grid(row=row,column=column)

        PGFrame.grid(row=0,column=0)

        PGFrame.update_idletasks()
        self.PGCanvas["scrollregion"]=self.PGCanvas.bbox(ALL)
        self.bind("<Configure>", self.on_resize)
        return

    def on_resize(self, event):
        w_offset = 80#+self.widthOfColumnsRightOfCreature*self.numColumnsRightOfCreature
        h_offset = 78#+self.heigthOfRowsBelowCreature+self.numRowsBelowCreature
        w,h = event.width-w_offset, event.height-h_offset
        self.PGCanvas.config(width=w, height=h)

    def generateSine(self):
        neuronToSet = int(self.inpIDVar.get())
        patternLength = int(self.pattLengthInput.get())
        padLen = int(self.PadLInput.get())

        self.PGCanvas.destroy()
        self.addPGTable(patternLength+padLen)

        amp = float(self.AmpInput.get())
        off = float(self.OffInput.get())
        freq = float(self.FreqInput.get())
        phase = float(self.PhaseInput.get())
        padVal = int(self.PadValInput.get())

        for cycle in range(patternLength+padLen):
            print 'cycle:',cycle
            if cycle < patternLength:
                self.patternVars[-1][-1].set(str(amp*sin(freq*pi*((cycle+phase)/patternLength))+off))
                print 'sin:',str(amp*sin(freq*pi*((cycle+phase)/patternLength))+off)
            else:
                self.patternVars[-1][-1].set(str(padVal))
                print 'pad',padVal

        self.parent.update()

        return

    def generateSquare(self):
        return

    def generateRandom(self):
        return

    def generateConstant(self):
        return

class NeuroloScope(Frame):
    def __init__(self, parent, Creature=None,inputPatterns=[],expOutputs=[],useGUIRun=False,colorDict=None,GUI=None):
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("NeuroloScope")
        self.screenWidth = parent.winfo_screenwidth()
        self.screenHeight = parent.winfo_screenheight()
        self.myCreature = Creature
        self.outputVarList=[]
        self.useGUIRun=useGUIRun
        self.stopRunning=0
        self.AddedIDList=[]
        self.scopeNeurValueVars=[]
        self.scopeValuePercVars=[]
        self.scopeValueList=[]
        self.scopeRoundValue= 3
        self.patternPosition=[]
        self.myGUI=GUI
        self.inputPatterns = inputPatterns
        for i in range(len(self.inputPatterns)):
            self.patternPosition.append(0)

        self.FFTCheckVars={}
        self.CycleCheckVars={}
        if colorDict == None:
            self.myColorDict={}
            self.scopeGenerateColors()

        else:
            self.myColorDict=colorDict


        #if not useGUIRun:
        self.inputColor = '#055'
        self.outputColor ='#606'
        self.hiddenColor ='#666'
        pStyle = ttk.Style()
        progbarThickness = 73
        progBarColor = "#0000DF"
        pStyle.theme_use('classic')
        pStyle.configure("input.Vertical.TProgressbar", troughcolor =self.inputColor, background=progBarColor, thickness=progbarThickness,relief=FLAT)
        pStyle.configure("output.Vertical.TProgressbar", troughcolor =self.outputColor, background=progBarColor, thickness=progbarThickness,relief=FLAT)
        pStyle.configure("hidden.Vertical.TProgressbar", troughcolor =self.hiddenColor , background=progBarColor, thickness=progbarThickness,relief=FLAT)

        self.parent.config(background='#000')
        self.config(background='#000')

        self.initScope()
##        newScope=Toplevel(self)
##        newScope.wm_title("")
    def initScope(self):
        textColor ='#FFFF00'
        bgColor ='#000'
        numCols = 3
        numRows = 2
        plotCol = 1


        self.scopeCanvas = Canvas(self.parent,width=self.screenWidth, height=self.screenHeight )
        self.scopeCanvas.create_rectangle(0,0,self.screenWidth,self.screenHeight,fill=bgColor)

        for c in range(numCols):
            self.scopeCanvas.columnconfigure(c,pad=3)
        for r in range(numRows):
            self.scopeCanvas.rowconfigure(r,pad=3)


        self.neurSummaryCanvas=Canvas(self.scopeCanvas,width=120,height=100,highlightbackground=bgColor)
        self.neurSummaryCanvas.create_rectangle(0,0,124,104,fill=bgColor)
        self.neurSummaryCanvas.grid(row=1,column=0)

        #New neuron canvas/button/input
        newNeurCanvas=Canvas(self.scopeCanvas,width=130,height=50,highlightbackground=bgColor)
        newNeurCanvas.create_rectangle(0,0,130,50,fill=bgColor)
        self.idInput = Entry(newNeurCanvas,foreground=textColor,background=bgColor, highlightcolor=bgColor,width=5)
        idLabel = Label(newNeurCanvas,text="Neuron ID:",foreground=textColor,background=bgColor, highlightcolor=bgColor)
        addNeurButton = Button(newNeurCanvas, text=" Add Neuron ", foreground=textColor,background=bgColor,command=self.addNeurSignal,wraplength=45)

        self.idInput.place(x=7,y=35,anchor=W)#, expand=True, padx=100, pady=100)
        idLabel.place(x=5,y=13,anchor=W)#, expand=True, padx=100, pady=100)
        addNeurButton.place(x=72,y=25,anchor=W)
        newNeurCanvas.grid(row=0,column=0)

        #Run/stop/step canvas
        buttonCanvas=Canvas(self.scopeCanvas,width=650,height=50,highlightbackground=bgColor,relief=FLAT)
        buttonCanvas.create_rectangle(0,0,655,50,fill=bgColor)
        numButtonCols = 8
        for i in range(numButtonCols):
            buttonCanvas.columnconfigure(i,pad=3)
        buttonCanvas.rowconfigure(0,pad=3)

        if self.useGUIRun:
            run = Button(buttonCanvas, text="  Run  ", command=self.myGUI.runCreature, foreground=textColor,background=bgColor)
            stop = Button(buttonCanvas, text="  Stop  ", command=self.myGUI.stopCreature, foreground=textColor,background=bgColor)
            step = Button(buttonCanvas, text="  Step  ", command=self.myGUI.stepCreature, foreground=textColor,background=bgColor)
        else:
            run = Button(buttonCanvas, text="  Run  ", command=self.scopeRunCreature, foreground=textColor,background=bgColor)
            stop = Button(buttonCanvas, text="  Stop  ", command=self.scopeStopCreature, foreground=textColor,background=bgColor)
            step = Button(buttonCanvas, text="  Step  ", command=self.scopeStepCreature, foreground=textColor,background=bgColor)

        close = Button(buttonCanvas, text=" Close Scope ", command=self.scopeClose, foreground=textColor,background=bgColor)

        run.grid(row=0, column=1)
        stop.grid(row=0, column=2)
        step.grid(row=0, column=3)
        close.grid(row=0,column=7)

        #Max cycle entry
        CycCanvas = Canvas(buttonCanvas,borderwidth=0,relief=FLAT,highlightbackground=bgColor,background=bgColor)
        CycTitle = Label(CycCanvas,text="Max Length",background=bgColor,foreground=textColor)
        self.CycEntry = Entry(CycCanvas,width=6,background=bgColor,foreground=textColor, highlightcolor=bgColor)#,background=bgColor, highlightcolor="#fff",width=5)
        CycTitle.grid(row=0,column=0)
        self.CycEntry.grid(row=1,column=0)
        CycCanvas.grid(row=0,column=0)

        #Close Scope Button

        #FFT Log scale
        self.FFTLogToggle=IntVar()
        FFTLogCheck = Checkbutton(buttonCanvas,text="FFT Log Scale",variable=self.FFTLogToggle,foreground=textColor,background='#000',selectcolor='#000')
        FFTLogCheck.grid(row=0,column=5)

        self.FFTSupprDC=IntVar()
        FFTDCCheck = Checkbutton(buttonCanvas,text="FFT Squash DC",variable=self.FFTSupprDC,foreground=textColor,background='#000',selectcolor='#000')
        FFTDCCheck.grid(row=0,column=6)

        buttonCanvas.grid(row=0,column=1)

        #Plot window
        self.plotWidth = self.screenWidth - 0.25*self.screenWidth
        self.plotHeight = self.screenHeight - 0.25*self.screenHeight

        self.plotCanvas = Canvas(self.scopeCanvas,width=self.plotWidth,height=self.plotHeight)
        self.plotCanvas.create_rectangle(0,0,self.plotWidth,self.plotHeight,fill=bgColor)
        self.plotCanvas.columnconfigure(0,pad=3)
        self.plotCanvas.columnconfigure(1,pad=3)
        self.plotCanvas.rowconfigure(0,pad=3)
        self.plotCanvas.rowconfigure(1,pad=3)


        #Cycle scope:
        self.timeSignalCanvas = Canvas(self.plotCanvas,width=self.plotWidth/1, height=self.plotHeight/2)
        self.timeSignalCanvas.create_rectangle(0,0,self.plotWidth/1,self.plotHeight/2,fill='#500')
        self.timeSignalCanvas.rowconfigure(0)
        self.timeSignalCanvas.columnconfigure(0)
        #self.timePlot = Figure(figsize=(12,3.5), dpi=100)
        self.timePlot = pyplot.figure(figsize=(12,3), dpi=100,facecolor='black')
        tplot = self.timePlot.add_subplot(111,axisbg='black')
        #t = arange(0.0,3.0,0.01)
        #s = sin(2*pi*t)
        t=[0,1,2,3,4,5]
        s=[0,1,0,1,0,1]
        tplot.spines['bottom'].set_color('yellow')
        tplot.spines['top'].set_color('yellow')
        tplot.tick_params(axis='x',colors='yellow')
        tplot.tick_params(axis='y',colors='yellow')
        tplot.plot(t,s)

        self.timeSignalPlotCanvas = FigureCanvasTkAgg(self.timePlot, master=self.timeSignalCanvas)
        self.timeSignalPlotCanvas.show()
        self.timeSignalPlotCanvas.get_tk_widget().grid(row=0,column=0)#.pack(side=TOP, fill=BOTH, expand=20)
        self.timeToolbar = NavigationToolbar2TkAgg( self.timeSignalPlotCanvas, self.timeSignalCanvas  )
        self.timeToolbar.update()
        self.timeSignalPlotCanvas._tkcanvas.pack()#.pack(side=TOP, fill=BOTH, expand=1)


        #FFT Scope:
        self.FFTMagCanvas = Canvas(self.plotCanvas,width=self.plotWidth, height=self.plotHeight/2)
        self.FFTMagCanvas.create_rectangle(0,0,self.plotWidth,self.plotHeight/2,fill='#050')

##        self.FFTPhaseCanvas = Canvas(self.plotCanvas,width=self.plotWidth/2, height=self.plotHeight/2)
##        self.FFTPhaseCanvas.create_rectangle(0,0,self.plotWidth/2,self.plotHeight/2,fill='#005')

        self.fftPlot = pyplot.figure(figsize=(12,3), dpi=100,facecolor='black')
        fftMagPlot = self.fftPlot.add_subplot(121,axisbg='black')
        fftPhasePlot = self.fftPlot.add_subplot(122,axisbg='black')

        fftMagPlot.spines['bottom'].set_color('yellow')
        fftMagPlot.spines['top'].set_color('yellow')
        fftMagPlot.tick_params(axis='x',colors='yellow')
        fftMagPlot.tick_params(axis='y',colors='yellow')
        fftMagPlot.set_yscale('log')
        fftPhasePlot.spines['bottom'].set_color('yellow')
        fftPhasePlot.spines['top'].set_color('yellow')
        fftPhasePlot.tick_params(axis='x',colors='yellow')
        fftPhasePlot.tick_params(axis='y',colors='yellow')
        fftPhasePlot.set_yscale('log')

        npCycles = asarray(t)
        fftFreq = fft.fftfreq(npCycles.shape[-1])
        npOutputs =asarray(s)
        fftMagAndPhase = fft.fft(npOutputs)
        ##plt.plot(freq, sp.real)#, freq, sp.imag)
        fftMagPlot.plot(fftFreq,fftMagAndPhase.real)
        fftPhasePlot.plot(fftFreq,fftMagAndPhase.imag)

        self.fftSignalPlotCanvas = FigureCanvasTkAgg(self.fftPlot, master=self.FFTMagCanvas)
        self.fftSignalPlotCanvas.show()
        self.fftSignalPlotCanvas.get_tk_widget().grid(row=0,column=0)#.pack(side=TOP, fill=BOTH, expand=20)
        fftToolbar = NavigationToolbar2TkAgg( self.fftSignalPlotCanvas, self.FFTMagCanvas  )
        fftToolbar.update()
        self.fftSignalPlotCanvas._tkcanvas.pack()#.pack(side=TOP, fill=BOTH, expand=1)


        self.FFTMagCanvas.grid(row=1,column=0,columnspan=2)
##        self.FFTPhaseCanvas.grid(row=1,column=1)
        self.timeSignalCanvas.grid(row=0,column=0,columnspan=2)

        self.plotCanvas.grid(row=1,column=plotCol,sticky=N)

        #self.timeSignalCanvas = Canvas(self.plotCanvas,width=plotWidth/4, height=plotHeight/4)


        #l = Label(scopeCanvas, text=" SIKE!!! No scope yet sucka. ",foreground='#f00',background ='#000' )
        #l.place(relx=0.5,rely=0.5,anchor=CENTER)#, expand=True, padx=100, pady=100)

        self.scopeCanvas.pack()
        #self.scopeList.append(newScope)

    def scopeClose(self):
        if self.myGUI != None:
            self.myGUI.myScope = None
        pyplot.close('all')
        self.destroy()
        self.parent.destroy()


    def scopeRunCreature(self):
        while(self.stopRunning == 0):
        #for i in range(20):
            self.scopeStepCreature()
            time.sleep(0.0005)
            #time.sleep(self.delayBarVar.get())
        self.stopRunning = 0

    def scopeStopCreature(self):
        self.stopRunning = 1

    def scopeStepCreature(self):
        if self.inputPatterns != []:
            for i in range(len(self.inputPatterns)):
                if self.patternPosition[i] >= len(self.inputPatterns[i]):
                    self.patternPosition[i] = 0
                self.myCreature.input[i].inbox = [self.inputPatterns[i][self.patternPosition[i]]]
                self.patternPosition[i] += 1

        self.myCreature.run()
        self.scopeUpdateVars()

    def addNeurSignal(self):
        IDToAdd = int(self.idInput.get())
        if IDToAdd in self.AddedIDList:
            return
        else:
            self.refreshNeurSummary(IDToAdd)

    def refreshNeurSummary(self,IDToAdd=None):
        neursToAdd=[]
        for neur in self.myCreature.neuronList:
            if neur.ID in self.AddedIDList:
                neursToAdd.append(neur)
            elif neur.ID == IDToAdd:
                neursToAdd.append(neur)
                self.AddedIDList.append(IDToAdd)

        if IDToAdd != None:
            if IDToAdd not in self.AddedIDList:
                return


        self.neurSummaryCanvas.destroy()
        self.neurSummaryCanvas=Canvas(self.scopeCanvas,width=120,height=self.screenHeight - 50,highlightbackground='#000')
        self.neurSummaryCanvas.create_rectangle(0,0,120,self.screenHeight/2,fill='#000')
        self.neurSummaryCanvas.columnconfigure(0,pad=1)
        row = 0
        for neur in neursToAdd:
            self.neurSummaryCanvas.rowconfigure(row,pad=1)
            newNeurBody = self.scopeNeurSummaryGUI(self.neurSummaryCanvas,neur,self.myColorDict[str(neur.ID)])
            newNeurBody.grid(row=row,column=0)
            self.scopeValueList.append([])
            row+=1

        self.neurSummaryCanvas.grid(row=1,column=0)
        return

    def scopeNeurSummaryGUI(self,canvas,neuron,color):
        w= 120
        h = 84
        w_body = 83
        h_body = 84
        w_inbox=25
        nbodyRowSpan = 10
        textColor='#FFFF00'
        #Create the neuron body
        neuronCanvas = Canvas(canvas,width=w, height=h, bd=0,background=color,highlightbackground=color)
##        neuronCanvas.rowconfigure(0,pad=3)
        neuronCanvas.columnconfigure(0,pad=3)

        IDLabel = Label(neuronCanvas,width = 3,text=str(neuron.ID),foreground=textColor,background='#000',highlightbackground=textColor)
        IDLabel.grid(row=0,column=0)#.place(x=5,y=2,anchor=N)
##        FFTLabel = Label(neuronCanvas,width = 3,text="FFT",foreground=textColor,background='#000',highlightbackground=textColor)
##        FFTLabel.grid(row=1,column=2)#.place(x=5,y=2,anchor=N)
##        CycLabel = Label(neuronCanvas,width = 3,text="Cyc",foreground=textColor,background='#000',highlightbackground=textColor)
##        CycLabel.grid(row=6,column=2)#.place(x=5,y=2,anchor=N)
        deleteButton = Button(neuronCanvas, text="X", command=lambda: self.removeNeuron(neuron.ID), foreground=textColor,background='#000')
        deleteButton.grid(row=0,column=2)

        self.FFTCheckVars[str(neuron.ID)]=IntVar()
        FFTCheck = Checkbutton(neuronCanvas,text="FFT",variable=self.FFTCheckVars[str(neuron.ID)],foreground=textColor,background='#000',selectcolor='#000')
        FFTCheck.grid(row=7,column=2)
        FFTCheck.toggle()
        self.CycleCheckVars[str(neuron.ID)]=IntVar()
        CycCheck = Checkbutton(neuronCanvas,text="Cycle",variable=self.CycleCheckVars[str(neuron.ID)],foreground=textColor,background='#000',selectcolor='#000')
        CycCheck.toggle()
        CycCheck.grid(row=3,column=2)

        neuronBody = Canvas(neuronCanvas,width=w_body, height=h_body, bd=0)
        neuronBody.config(highlightbackground=color, highlightthickness=5)
        if neuron.isInput:
            myBodyColor = self.inputColor
            myStyle = "input.Vertical.TProgressbar"
        elif neuron.isOutput:
            myBodyColor = self.outputColor
            myStyle = "output.Vertical.TProgressbar"
        else:
            myBodyColor = self.hiddenColor
            myStyle = "hidden.Vertical.TProgressbar"

        neuronBody.create_rectangle(0,0,w_body+4,h_body+4, fill=myBodyColor)
        #neuronBody.create_rectangle(5,5,w_body+2,h_body+2, fill="#000")

        self.scopeNeurValueVars.append(DoubleVar())
        self.scopeValuePercVars.append(DoubleVar())

        neurThreshold = (neuron.propertyList[0])
        neurIndex = self.AddedIDList.index(neuron.ID)
        self.scopeNeurValueVars[neurIndex].set(round(neuron.value,2))
        if self.useGUIRun:
            self.scopeValuePercVars[neurIndex].set(int(self.myGUI.calc_progressBarValue(neuron)))
        else:
            self.scopeValuePercVars[neurIndex].set(int(self.scope_calc_progressBarValue(neuron)))

        threshLineRatio = 1.0/3.0

        #ttk.Progressbar(frame, style="neuron.Vertical.TProgressbar", orient="horizontal", length=600,mode="determinate", maximum=4, value=1).grid(row=1, column=1)

        if len(neuron.propertyList) == 2:
            progBar = ttk.Progressbar(neuronBody, style=myStyle, orient=VERTICAL, length=h_body, mode='determinate', variable=self.scopeValuePercVars[neurIndex], maximum=100)
            progBar.place(relx=0.5, rely=0.5, anchor=CENTER)
            threshLineHeight = 30
            valpos = 0.5
        else:
            progBar = ttk.Progressbar(neuronBody, style=myStyle, orient=VERTICAL, length=h_body, mode='determinate', variable=self.scopeValuePercVars[neurIndex], maximum=150)
            progBar.place(relx=0.5, rely=0.5, anchor=CENTER)
            threshLineHeight = int(threshLineRatio*h_body+3)
            valpos = 0.55
##        neuronBody.create_line(0,threshLineHeight,w_body,threshLineHeight,fill="#00FFFF",dash=(3,),width=2)
##        neuronBody.pack()

        valLabel = Label(neuronBody,width=5,textvariable=str(self.scopeNeurValueVars[neurIndex]),foreground="#00FF00", background="#000", padx=0, pady=0)
        valLabel.place(relx=0.5, rely=valpos, anchor=CENTER)

        threshFrame=Frame(neuronBody,borderwidth=0,relief=SOLID,background='#00FFFF')
        threshFrame.config(highlightbackground='#00FFFF',highlightthickness=2)
        threshLabel = Label(threshFrame,width=5,text=(str(round(neurThreshold,2))),foreground="#00FFFF", background="#000", padx=0, pady=0,bd=1).pack()
        threshFrame.place(relx=0.5, y=threshLineHeight, anchor=S)

        if len(neuron.propertyList) == 2:
            neurThreshold2 = (neuron.propertyList[1])
            thresh2Frame=Frame(neuronBody,borderwidth=0,relief=SOLID,background='#00FFFF')
            thresh2Frame.config(highlightbackground='#00FFFF',highlightthickness=2)
            thresh2Label = Label(thresh2Frame,width=5,text=(str(round(neurThreshold2,2))),foreground="#00FFFF", background="#000", padx=0, pady=0,bd=1).pack()
            thresh2Frame.place(relx=0.5, y=h_body+3, anchor=S)

        #neuronBody.place(relx=0.5,rely=0.5,anchor=CENTER)
        neuronBody.grid(row=0,column=1,rowspan=nbodyRowSpan)


        return neuronCanvas

    def removeNeuron(self,nID):
        self.AddedIDList.remove(nID)
        self.refreshNeurSummary()

    def scopeUpdateVars(self):
        cyclesToPlot =int(self.CycEntry.get())
        for neur in self.myCreature.neuronList:
            if neur.ID in self.AddedIDList:
                #self.avgInboxneurStringVars[n].set(round(neur.avgInput,self.roundValue))
                self.scopeNeurValueVars[self.AddedIDList.index(neur.ID)].set(round(neur.value,self.scopeRoundValue))
                self.scopeValuePercVars[self.AddedIDList.index(neur.ID)].set(int(self.scope_calc_progressBarValue((neur))))
                if len(self.scopeValueList[self.AddedIDList.index(neur.ID)])>=cyclesToPlot:
                    self.scopeValueList[self.AddedIDList.index(neur.ID)].pop(0)
                    if len(self.scopeValueList[self.AddedIDList.index(neur.ID)])>cyclesToPlot:
                        delta = len(self.scopeValueList[self.AddedIDList.index(neur.ID)]) - cyclesToPlot
                        self.scopeValueList[self.AddedIDList.index(neur.ID)] = self.scopeValueList[self.AddedIDList.index(neur.ID)][delta-1:]
                else:
##                    self.scopeValueList[self.AddedIDList.index(neur.ID)][len(self.scopeValueList[self.AddedIDList.index(neur.ID)])-1:] = [0 for ind in range(0,int(self.CycEntry.get())-len(self.scopeValueList[self.AddedIDList.index(neur.ID)]))]
                    self.scopeValueList[self.AddedIDList.index(neur.ID)] = [0 for ind in range(0,(cyclesToPlot-len(self.scopeValueList[self.AddedIDList.index(neur.ID)])))]+self.scopeValueList[self.AddedIDList.index(neur.ID)]
                self.scopeValueList[self.AddedIDList.index(neur.ID)].append(neur.outbox)

        self.updatePlots()
        self.parent.update()


    def updatePlots(self):
##        pyplot.clf()
        self.timePlot.clf()
        tplot = self.timePlot.add_subplot(111,axisbg='black')
        tplot.spines['bottom'].set_color('yellow')
        tplot.spines['top'].set_color('yellow')
        tplot.tick_params(axis='x',colors='yellow')
        tplot.tick_params(axis='y',colors='yellow')


        self.fftPlot.clf()
        fftMagPlot = self.fftPlot.add_subplot(121,axisbg='black')
        fftPhasePlot = self.fftPlot.add_subplot(122,axisbg='black')

        fftMagPlot.spines['bottom'].set_color('yellow')
        fftMagPlot.spines['top'].set_color('yellow')
        fftMagPlot.tick_params(axis='x',colors='yellow')
        fftMagPlot.tick_params(axis='y',colors='yellow')
        fftPhasePlot.spines['bottom'].set_color('yellow')
        fftPhasePlot.spines['top'].set_color('yellow')
        fftPhasePlot.tick_params(axis='x',colors='yellow')
        fftPhasePlot.tick_params(axis='y',colors='yellow')
        if self.FFTLogToggle.get()==1:
            fftMagPlot.set_yscale('log')
            fftPhasePlot.set_yscale('log')
        else:
            fftMagPlot.set_yscale('linear')
            fftPhasePlot.set_yscale('linear')


        cycles=[]
        for cyc in range(len(self.scopeValueList[0])):
            cycles.append(cyc)

        npCycles = asarray(cycles)
        fftFreq = fft.fftfreq(npCycles.shape[-1])
        fftFreqList = fftFreq.tolist()

        timePlotString=''
        fftMagPlotString = ''
        fftPhasePlotString = ''
        for ID in self.AddedIDList:
            if self.FFTCheckVars[str(ID)].get() == 1:
                npOutputs =asarray(self.scopeValueList[self.AddedIDList.index(ID)])
                fftMagAndPhase = fft.fft(npOutputs)

                if self.FFTSupprDC.get()==1:
                    fftMagAndPhase.real[0] = 0
                    fftMagAndPhase.imag[0] = 0

            if ID == self.AddedIDList[-1]:
                if self.CycleCheckVars[str(ID)].get()==1:
                    timePlotString+=str(cycles)+','+str(self.scopeValueList[self.AddedIDList.index(ID)])+",'"+self.myColorDict[str(ID)]+"'"

                if self.FFTCheckVars[str(ID)].get() == 1:
                    fftMagPlotString+=str(fftFreqList)+','+str(fftMagAndPhase.real.tolist())+",'"+self.myColorDict[str(ID)]+"'"
                    fftPhasePlotString+=str(fftFreqList)+','+str(fftMagAndPhase.imag.tolist())+",'"+self.myColorDict[str(ID)]+"'"
            else:
                if self.CycleCheckVars[str(ID)].get()==1:
                    timePlotString+=str(cycles)+','+str(self.scopeValueList[self.AddedIDList.index(ID)])+",'"+self.myColorDict[str(ID)]+"',"

                if self.FFTCheckVars[str(ID)].get() == 1:
                    fftMagPlotString+=str(fftFreqList)+','+str(fftMagAndPhase.real.tolist())+",'"+self.myColorDict[str(ID)]+"',"
                    fftPhasePlotString+=str(fftFreqList)+','+str(fftMagAndPhase.imag.tolist())+",'"+self.myColorDict[str(ID)]+"',"

        timePlotString = "tplot.plot("+timePlotString+")"
        exec(timePlotString)
        fftMagPlotString = "fftMagPlot.plot("+fftMagPlotString+")"
        exec(fftMagPlotString)
        fftPhasePlotString = "fftPhasePlot.plot("+fftPhasePlotString+")"
        exec(fftPhasePlotString)

        canv = self.timeSignalPlotCanvas
        self.timePlot.canvas.draw()
        canv = self.fftSignalPlotCanvas
        self.fftPlot.canvas.draw()

##
##        npOutputs =asarray(s)
##        fftMagAndPhase = fft.fft(npOutputs)
##        ##plt.plot(freq, sp.real)#, freq, sp.imag)
##        fftMagPlot.plot(npCycles,fftMagAndPhase.real)
##        fftPhasePlot.plot(npCycles,fftMagAndPhase.imag)
##
##        self.fftSignalPlotCanvas = FigureCanvasTkAgg(self.fftPlot, master=self.FFTMagCanvas)
##        self.fftSignalPlotCanvas.show()
##        self.fftSignalPlotCanvas.get_tk_widget().grid(row=0,column=0)#.pack(side=TOP, fill=BOTH, expand=20)
##        fftToolbar = NavigationToolbar2TkAgg( self.fftSignalPlotCanvas, self.FFTMagCanvas  )
##        fftToolbar.update()
##        self.fftSignalPlotCanvas._tkcanvas.pack()#.pack(side=TOP, fill=BOTH, expand=1)






        return

    def scope_calc_progressBarValue(self,neuron):
        if len(neuron.propertyList) == 2:
            neurThreshold=neuron.propertyList[0]
            neurThreshold2=neuron.propertyList[1]

            topPt = neurThreshold
            botPt = neurThreshold2
            if neuron.value > topPt:
                percVal = 1.0
            elif neuron.value < botPt:
                percVal = 0.0
##            elif neuron.value > neurThreshold:
##                percVal=((.5/(topPt-neurThreshold))*(neuron.value-neurThreshold)+1)
            elif topPt == botPt:
                percVal = 1
            else:
                percVal = (1.0/(topPt-botPt))*(neuron.value-botPt)
        else:
            neurThreshold=neuron.propertyList[0]

            botPt = neurThreshold - 2*abs(neurThreshold)
            topPt = neurThreshold + abs(neurThreshold)

            if neuron.value > topPt:
                percVal = 1.50
            elif neuron.value < botPt:
                percVal = 0
            elif neuron.value > neurThreshold:
                percVal=((.5/(topPt-neurThreshold))*(neuron.value-neurThreshold)+1)
            else:
                if neurThreshold == botPt:
                    perVal = 1
                else:
                    percVal = (1.0/(neurThreshold-botPt))*(neuron.value-botPt)

        percVal=percVal*100
        return percVal

#by limiting yourself to doing only that which you can understand, you necessarily excuse yourself from the discussion of the greatest possibilities
#This simple epigram lies at the heart of every artist, poet, and musician, but is terminally complex to the typical scientist

    def scopeGenerateColors(self):
        starts=[]
        stops=[]
        numNeurons = len(self.myCreature.neuronList)
        for i in range(3): #There are three colors...RGB
            stops.append(0)
            starts.append(0xff)

        #starts[0] = 55
        rawColorPoints, endSeed = generateSobolCharacterizationPoints(3,numNeurons,starts,stops,0,0)

        for i in range(numNeurons):
            newColor = ""
            for rgb in range(3):
                tempNewColor = str(hex(int(rawColorPoints[i][rgb])))
                tempNewColor = tempNewColor[2:]
                if len(tempNewColor) == 1:
                    tempNewColor = "0"+tempNewColor
##                elif len(tempNewColor)==2:
##                    tempNewColor = "0"+tempNewColor
                newColor += tempNewColor

            newColor = "#"+newColor
            self.myColorDict[str(self.myCreature.neuronList[i].ID)] = (newColor)

def main():
##    neuronCount =100
##    inputCount =20
##    outputCount = 10
    neuronCount =10
    inputCount =1
    outputCount = 1
    MaxCycles = 1
    patternLength=10
    inputSet=[]
##    expOut=[]
##    for i in range(inputCount):
##        inputSet.append([])
##        amp = randint(1,10)
##        for j in range(patternLength):
##            #inputSet[-1].append(randint(0,10))
##            inputSet[-1].append(sin((pi*j)/patternLength)*amp)
##            #inputSet[-1].append(0)
####
##    for o in range(outputCount):
##        expOut.append(randint(0,10))
    trainData = docy()
    cycleCount=360
    a=1
    b=1
    c=0
    trainData.generateSinTracker(inputCount, outputCount,cycleCount,a,b,c)
    #trainData.generateConstant(len(population.creatureList[0].input), len(population.creatureList[0].output), constantIn=1, constantOut=5)
    print "ins"
    print trainData.data[0][0]
    print "outs"
    print trainData.data[0][1]

    #demoCreature = Creature(neuronCount, inputCount, outputCount)#,MaxCycles)
    demoCreature = DummyCreature(neuronCount, inputCount, outputCount)#,MaxCycles)

    for i in range(inputCount):
        demoCreature.input[i].inbox = [trainData.data[0][0][0][i]]

##    demoCreature.expectedOutputs = expOut

    root = Tk()
    #newScope = NeuroloScope(root,demoCreature,inputSet)
    ex = CreatureGUI_Beta(root,demoCreature,trainData.data[0][0])

    root.geometry("900x500+300+300")
    root.mainloop()

'''
def main(creature = None, trainData = None):
    if creature == None:
        creature = Creature(neuronCount=45)
    if trainData == None:
        trainData = docy()
        trainData.generateSin(len(creature.input), len(creature.output))

    inputCount = len(trainData.data[0][0])  # trainingSet 0 input
    patternLength = len(trainData.data[0][0][0])
    inputSet=trainData.data[0][0]
    print inputSet

    for i in range(len(inputSet)):
        creature.input[i].inbox = [inputSet[i][0]]

    root = Tk()

    ex = CreatureGUI_Beta(root,creature,inputSet)


    root.geometry("900x500+300+300")
    root.mainloop()
'''

if __name__ == '__main__':
    main()
