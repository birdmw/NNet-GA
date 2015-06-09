from random import *
from math import *
import numpy as np

#note: hunting is where each value, at each cycle, each neuron, is allowed to seach through the local expected outputs for a best value
#this is different from phase searching where we wait till the end and "shift" our data looking for a best result. Personally I like the
#idea of hunting a little more. It is more flexible and it solves the "punished for overflow" issue. Only hunting works right now.


class docy:
    def __init__(self, data=[]): #for now it will only work with a hunt window, working on phase shift
        #docy.data[set][io][neuron][cycle]
        self.data = data

    def addSet(self, tset):
        #[io][put][cycle]
        self.data.append(tset)
        
    def removeSet(self, index = -1 ):
        #index
        if (abs(index) > len(self.data)):
            print "set does not exist, cannot remove"
        else:
            self.data.pop(index)
            
    def randomSet(self):
        return choice(self.data)
    
    def generateSin(self, inputCount, outputCount, cycleCount=360, a=1, b=1, c=1, reps=1):
        inputList, outputList = [] , []
        inputs , outputs = [] , []
        for x in range(int(cycleCount * reps)):
                inputs.append(x*pi/180)
                outputs.append(a*sin(b*inputs[-1]+c))
        for y in range(inputCount):
                inputList.append(inputs)
                outputList.append(outputs)
        self.data = [[inputList, outputList]]
