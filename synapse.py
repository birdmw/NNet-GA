from random import *
from math import *
class Synapse:
    def __init__(self, n1, n2, neuronCount):
        self.a = gauss(0,1) / neuronCount
        self.b = gauss(0,1) / neuronCount
        self.c = gauss(0,1) / neuronCount
        self.d = gauss(0,1) / neuronCount
        self.n1, self.n2 = n1, n2

    def run(self):
        if self.n1.isOutput == 0 and self.n2.isInput == 0: #If not an input and not an output
            sinFxn = max(min(self.a * sin(self.b * self.n1.outbox + self.c),1000000),-1000000)
            diffFxn = max(min(self.d * (self.n1.prevOutbox - self.n1.outbox),1000000),-1000000)
            self.n2.inbox.append( sinFxn + diffFxn )
