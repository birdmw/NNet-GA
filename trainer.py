from random import *
from math import *

#note: hunting is where each value, at each cycle, each neuron, is allowed to seach through the local expected outputs for a best value
#this is different from phase searching where we wait till the end and "shift" our data looking for a best result. Personally I like the
#idea of hunting a little more. It is more flexible and it solves the "punished for overflow" issue. Only hunting works right now.


class docy:
    def __init__(self, data=[], huntWindow = 4): #for now it will only work with a hunt window, working on phase shift
        #docy.data[set][io][put][cycle]
        self.data = data

        #cycleWindow is the window in which outputs are allowed to hunt for a best result
        self.huntWindow = huntWindow

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

    def generateSinTracker(self, inputCount=1, outputCount=1, cycleCount=360, a=[1,1], b=[1,1], c=[0,0], d=[0,0], reps=1):
        inputList, outputList = [] , []
        inputs , outputs = [] , []
        for x in range(int(cycleCount * reps)):
            valin = a[0]*sin(b[0]*(x*pi/180)+c[0])+d[0]
            inputs.append(valin)
            valout = a[1]*sin(b[1]*(x*pi/180)+c[1])+d[1]
            outputs.append(valout)
        for y in range(inputCount):
            inputList.append(inputs)
        for z in range(outputCount):
            outputList.append(outputs)
        self.data = [[inputList, outputList]]


    def generateSquareTracker(self, inputCount=1, outputCount=1, cyclesPerRep=360, mag=[1,1], period=[360,360], phase_off=[0,0], dc_off=[0,0], reps=1):
        inputList, outputList = [] , []
        posCounter = phase_off
        putList=[]

        for put in range(inputCount+outputCount):
            putList=[]
            for x in range(int(cyclesPerRep * reps)):
                if posCounter[put] > period[put]:
                    posCounter[put] = 0

                if posCounter[put]<(float(period[put])/2.0):
                    val = mag[put]+dc_off[put]
                else:
                    val = -1*(mag[put])+dc_off[put]

                putList.append(val)
                posCounter[put]+=1

            if put < inputCount:
                inputList.append(putList)
            else:
                outputList.append(putList)

        self.data = [[inputList, outputList]]

    def generateSinAdder(self, inputCount=1, outputCount=1, cyclesPerRep=360, inp_abcd=[[1,1,0,0]], reps=1):
        inputList, outputList = [] , []
        outputs = []
        for inp in range(inputCount):
            inputs=[]
            [a,b,c,d] = inp_abcd[inp]
            for cyc in range(int(cyclesPerRep * reps)):
                valin = a*sin(b*(cyc*pi/180)+c)+d
                inputs.append(valin)
            inputList.append(inputs)

        for cyc in range(int(cyclesPerRep * reps)):
            outputs.append(0)
            for inp in range(inputCount):
                outputs[-1] += inputList[inp][cyc]

        for z in range(outputCount):
                outputList.append(outputs)
        self.data = [[inputList, outputList]]

    def generateConstant(self, inputCount, outputCount, constantIn=1, constantOut=10, cycleCount=360, reps=1):
        inputList, outputList = [] , []
        inputs , outputs = [] , []
        for x in range(int(cycleCount * reps)):
                valin = constantIn
                valout = constantOut
                #valout=valin
                inputs.append(valin)
                outputs.append(valout)
        for y in range(inputCount):
                inputList.append(inputs)
        for z in range(outputCount):
                outputList.append(outputs)
        self.data = [[inputList, outputList]]


def arrayAbsSum(array):
    total = 0.0
    for a in array:
        total += abs(array)
    return total

def arrayAbsDifference (arrayOne,arrayTwo):
    array=[]
    for i in range( len(arrayTwo) ):
        array.append( abs(arrayOne[i] - arrayTwo[i]) )
    return array

