from random import *
class Neuron:
    def __init__(self):
        self.inbox = [gauss(0,1)]
        self.value, self.threshold, self.outbox, self.prevOutbox = gauss(0,1), gauss(0,1), gauss(0,1), gauss(0,1)
        self.isInput, self.isOutput = 0, 0
        self.maxVal = 10000
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
        self.prevOutbox = self.outbox
        avgInput = sum(self.inbox)/float(len(self.inbox)+1)
        self.value += min(self.maxVal,max(avgInput,-1*self.maxVal))
        if (self.value >= self.threshold):
            self.outbox = self.value
            self.value = 0.0
        self.inbox = []