from random import *
from math import *
class Synapse:
    def __init__(self, n1, n2, neuronCount, ID=0):
        stdeviation = 2
        gaussCenter =0
        self.a = gauss(gaussCenter,stdeviation)# / neuronCount
        self.b = gauss(gaussCenter,stdeviation)
        self.c = gauss(gaussCenter,stdeviation)
        self.d = gauss(gaussCenter,stdeviation)
        self.e = gauss(gaussCenter,stdeviation)
        self.n1, self.n2 = n1, n2
        self.propertyList=[self.a,self.b,self.c,self.d,self.e]
        #self.propertyList=[self.a]
        self.output = gauss(0,stdeviation)
        self.ID = ID

    def run(self):
        if self.n1.isOutput == 0 and self.n2.isInput == 0: #If not an input and not an output
            sinFxn = max(min(self.propertyList[0] * sin(self.propertyList[1] * self.n1.outbox + self.propertyList[2]),1000000),-1000000)
            diffFxn = max(min(self.propertyList[3] * (self.n1.prevOutbox - self.n1.outbox),1000000),-1000000)
            self.output = sinFxn + diffFxn + self.propertyList[4]

            #self.output = self.propertyList[0] * self.n1.outbox
            #self.output = 0
            self.n2.inbox.append(self.output)
