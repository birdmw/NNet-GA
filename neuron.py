from random import *
class Neuron:
    def __init__(self,ID=0):
        stdeviation = 10
        gaussCenter = 0
        dblThreshOffset = 0
        self.inbox = [gauss(gaussCenter,stdeviation)]
        self.avgInput = self.inbox[0]
        self.value, self.threshold, self.outbox, self.prevOutbox, self.doubleThreshold = gauss(gaussCenter,stdeviation), gauss(gaussCenter,stdeviation), gauss(gaussCenter,stdeviation), gauss(gaussCenter,stdeviation),gauss(gaussCenter-dblThreshOffset,stdeviation)
        self.isInput, self.isOutput= 0,0
        self.maxVal = 10000
        self.propertyList=[0,0]
        self.propertyList[0]=max(self.threshold,self.doubleThreshold)
        self.propertyList[1]=min(self.threshold,self.doubleThreshold)
        self.synapseList=[]
        self.ID = ID
        self.fired = randint(0,1)
        self.inputSynapseCount = 0

    def run(self):
        if self.propertyList[0]<self.propertyList[1]:
            self.propertyList[0],self.propertyList[1] = self.propertyList[1],self.propertyList[0]

        if self.isInput or self.isOutput:
            self.propertyList[0]=.001
            self.propertyList[1]=-.001

        if len(self.inbox) != 0:
            self.avgInput = sum(self.inbox)#/float(len(self.inbox))
            self.value = min(self.maxVal,max(self.avgInput+self.value,-1*self.maxVal))

        if (self.value >= self.propertyList[0]):
            self.prevOutbox = self.outbox
            self.outbox = self.value - self.propertyList[0]
            if len(self.propertyList)==2:
                self.value = self.propertyList[1]
            else:
                self.value = 0
            self.fired = 1

        elif len(self.propertyList)==2:
            if (self.value <= self.propertyList[1]):
                self.prevOutbox = self.outbox
                self.outbox = self.value - self.propertyList[1]
                self.value = self.propertyList[0]
                self.fired = 1
            else:
                self.fired = 0
                if not self.isOutput:
                    self.outbox = self.outbox/2
        else:
            self.fired = 0
            self.outbox = self.outbox/2

        if not self.isInput:
            self.inbox = []

    def setNeuronInput(self,value):
        #print "setting neuron input   ", value
        self.inbox = [value]
        if self.inbox < .5 and self.inbox>-.5:
            print self.inbox
        if value != self.inbox[0]:
            print "not same"

class DummyNeuron:
    def __init__(self,ID=0):
        self.inbox = [0]
        self.avgInput = self.inbox[0]
        self.value=0
        self.threshold = 1
        self.outbox = 0
        self.prevOutbox =0
        self.doubleThreshold = -1
        self.isInput, self.isOutput= 0,0
        self.maxVal = 10000
        self.propertyList=[0,0]
        self.propertyList[0]=max(self.threshold,self.doubleThreshold)
        self.propertyList[1]=min(self.threshold,self.doubleThreshold)
        self.synapseList=[]
        self.ID = ID
        self.fired = 0
        self.inputSynapseCount = 0

    def run(self):
        if self.propertyList[0]<self.propertyList[1]:
            self.propertyList[0],self.propertyList[1] = self.propertyList[1],self.propertyList[0]

        if self.isInput or self.isOutput:
            self.propertyList[0]=0
            self.propertyList[1]=0

        if len(self.inbox) != 0:
            self.avgInput = sum(self.inbox)/float(len(self.inbox))
            self.value = min(self.maxVal,max(self.avgInput+self.value,-1*self.maxVal))

        if (self.value >= self.propertyList[0]):
            self.prevOutbox = self.outbox
            self.outbox = self.value - self.propertyList[0]
            if len(self.propertyList)==2:
                self.value = self.propertyList[1]
            else:
                self.value = 0
            self.fired = 1

        elif len(self.propertyList)==2:
            if (self.value <= self.propertyList[1]):
                self.prevOutbox = self.outbox
                self.outbox = self.value - self.propertyList[1]
                self.value = self.propertyList[0]
                self.fired = 1
            else:
                self.fired = 0
##                if not self.isOutput:
##                    self.outbox = 0
        else:
            self.fired = 0
            #self.outbox = 0

        if not self.isInput:
            self.inbox = []

    def setNeuronInput(self,value):
        #print "setting neuron input   ", value
        self.inbox = [value]
        #print self.inbox
        if value != self.inbox[0]:
            print "not same. Inbox not set correctly"