import numpy as np
import pandas as pd
import torch.nn


class Optimize():
    grad=[]
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.dim=len(parameters)
        self.lr = lr
        self.gradient_type=None
        grad=[0]*self.dim

    """______Stochastic Gradient Descent_______
                General speak,it's an optimizor to improve the model
                which has  kinds of parameters,so that training a
                perfect parameters to fit the data

            :param parameters:model_parameters
            :param lr:learning_rate (To change the parameter's weight and bias)
            :return:fitted parameters
    """

    class SGD():
        def __init__(self):
            pass




    def Adam(cls):
        cls.gradient_type="Adam"

        """_____clear_grad________
            Clear the gradient to prepare for next backward pass
            and avoid gradient accumulation from subsequent computations.
        
        """
    def zero_grad(self):
        for i in range(self.dim):
            self.grad[i]=0





        """______step_function_________
                utilize the gradient to optimize
                 parameter step by step
                formula: parameter =parameter-lr*gradient

        :return:
        """

    def step(self,):
        if (self.gradient_type=="SGD"):
            for i in range(self.dim):
                self.parameters[i]-=self.lr*self.grad[i]
        elif(self.gradient_type=="Adam"):
            for i in range(self.dim):
                self.parameters[i]-=self.lr*self.grad[i]

