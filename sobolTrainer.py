from population import *
from sobolInterface import *


class sobolTrainer(object):
    """description of class"""
    def __init__(self, PopSize, NeuronCount, InputCount, OutputCount,MaxCycles):
        self.popSize = PopSize
        self.population = Population(PopSize,NeuronCount,InputCount,OutputCount,MaxCycles)
        
