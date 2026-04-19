import numpy as np
import typing
class Function():
    def forward(self,*input):
        raise NotImplementedError
    def backward(self,*grid_output):
        raise NotImplementedError


class Add(Function):
    def forward(self,input1,input2):
        return input2+input1
    def backward(self,grad_output):
        return grad_output,grad_output

class Mul(Function):
    def forward(self,input1,input2):
        self.input1=input1
        self.input2=input2
        return self.input1*self.input2

    def backward(self,grad_output):
        return self.input2*grad_output,self.input1*grad_output

class Matmul(Function):
    def forward(self,input1,input2):
        self.input1=input1
        self.input2=input2
        return np.matmul(self.input1,self.input2)
    def backward(self,grad_output):
        return np.matmul(grad_output,self.input2.T),np.matmul(self.input1.T,grad_output)


class Tensor():
    def __init__(self,data,require_grad=False):
        self.data=np.array(data)
        self.grad=None
        self.grad_fn=None
        self.require_grad=require_grad
        self.parent=()

    def __add__(self,other):
        calculate=Add()
        result=Tensor(calculate.forward(self.data,other.data),require_grad=self.require_grad or other.require_grad)
        result.grad_fn=calculate
        result.parent=(self,other)
        return result

    def __mul__(self, other):
        calculate=Mul()
        result=Tensor(calculate.forward(self.data,other.data),require_grad=self.require_grad or other.require_grad)
        result.grad_fn=calculate
        result.parent=(self,other)
        return result

    def matmul(self,other):
        calculate=Matmul()
        result=Tensor(calculate.forward(self.data,other.data),require_grad=self.require_grad or other.require_grad)
        result.grad_fn=calculate
        result.parent=(self,other)
        return result


class Optimize():
    def step(self):
        return NotImplementedError
    def zero_grad(self):
        return NotImplementedError


"""
        _________________Stochastic Gradient Descent__________________
                General speak,it's an optimizor to improve the model
                which has  kinds of parameters,so that training a
                perfect parameters to fit the data

            :param parameters:model_parameters
            :param lr:learning_rate (To change the parameter's weight and bias)
            :return:fitted parameters
"""

class SGD(Optimize):
    """
        parameters: [weight, bias]
        weight.shape = (n_feature, n_output)
        bias.shape = (n_output,) or (1, n_output)
        lr: learning rate
    """
    def __init__(self,parameters, lr) ->None:
        self.parameters=parameters
        self.lr=lr

    """________step_function_________
               
        utilize the gradient to optimize
        parameter step by step
        formula: parameter =parameter-lr*gradient
     :return:
    """

    def step(self):
        for value in self.parameters:
            if value.grad is not None:
                value.data-=self.lr*value.grad
    def zero_grad(self):
        for value in self.parameters:
            value.grad=None




class Adam(Optimize):
    """______clear_grad________
                Clear the gradient to prepare for next backward pass
                and avoid gradient accumulation from subsequent computations.

    """
    def __init__(self,parameters,lr,beta=(0.999,0.99),eps=1e-8):
        pass

def zero_grad(self):
    for i in range(self.dim):
        self.grad[i]=0





if __name__=="__main__":
    pass
