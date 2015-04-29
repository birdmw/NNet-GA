from random import *
from math import *
class Synapse:
    def __init__(self, n1, n2, neuronCount):
        self.a = gauss(0,1) / neuronCount
        self.b = gauss(0,1) / neuronCount
        self.c = gauss(0,1) / neuronCount
        self.d = gauss(0,1) / neuronCount
        self.e = gauss(0,1) / neuronCount
        self.n1, self.n2 = n1, n2
        self.propertyList=[self.a,self.b,self.c,self.d,self.e]
        #self.propertyList=[self.a]

    def run(self):
        if self.n2.isInput == 0: #If not an input and not an output
            sinFxn = max(min(self.propertyList[0] * sin(self.propertyList[1] * self.n1.outbox + self.propertyList[2])+ self.propertyList[3],1000),-1000)
            #diffFxn = max(min(self.propertyList[4] * (self.n1.prevOutbox - self.n1.outbox),1000),-1000)
            self.n2.inbox.append( sinFxn) #+ diffFxn )
            #self.n2.inbox.append(self.propertyList[0] * self.n1.outbox )
