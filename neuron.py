from random import *
class Neuron:
    def __init__(self,ID=0):
        stdeviation = 10
        gaussCenter = 0
        dblThreshOffset = 0
        self.inbox = [gauss(gaussCenter,stdeviation)]
        self.avgInput = self.inbox[0]
        self.value, self.threshold, self.outbox, self.prevOutbox,self.doubleThreshold = gauss(gaussCenter,stdeviation), gauss(gaussCenter,stdeviation), gauss(gaussCenter,stdeviation), gauss(gaussCenter,stdeviation),gauss(gaussCenter-dblThreshOffset,stdeviation)
        self.isInput, self.isOutput, self.inputSynapseCount = 0, 0,0
        self.maxVal = 10000
        self.propertyList=[0,0]

        self.propertyList[0]=max(self.threshold,self.doubleThreshold)
        self.propertyList[1]=min(self.threshold,self.doubleThreshold)
        self.synapseList=[]
        self.ID = ID
        self.fired = randint(0,1)
    '''
    def run(self):

        self.prevOutbox = self.outbox
        if self.isOutput == 0:#If not an output neuron
            self.outbox = 0.0
        self.value += sum(self.inbox) / (float(len(self.inbox)))
        if (self.value >= self.threshold):
            self.outbox = max(min(self.value,1000000),-1000000)
            self.value = 0.0
        if self.isInput == 0:#If not an input neuron
            self.inbox = [0.0]

        #self.outbox += sum(self.inbox) / (float(len(self.inbox)))

    '''
    def run(self):
        if len(self.inbox) != 0:
            self.avgInput = sum(self.inbox)/float(len(self.inbox))
            self.value = min(self.maxVal,max(self.avgInput+self.value,-1*self.maxVal))

        if (self.value >= self.propertyList[0]):
            self.prevOutbox = self.outbox
            self.outbox = self.value
            if len(self.propertyList)==2:
                self.value = self.propertyList[1]
            else:
                self.value = 0
            self.fired = 1

        elif len(self.propertyList)==2:
            if (self.value <= self.propertyList[1]):
                self.prevOutbox = self.outbox
                self.outbox = self.value
                self.value = self.propertyList[0]
                self.fired = 1
            else:
                self.fired = 0
                self.outbox=0
        else:
            self.fired=0
            self.outbox=0

        if not self.isInput:
            self.inbox = []