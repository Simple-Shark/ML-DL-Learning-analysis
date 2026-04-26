import numpy as np
""" __________tensor_graph_forward____________
    


"""
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

class Transpose(Function):

    def forward(self,input):
        self.input=input
        return self.input.T
   
    def backward(self,grad_output):
        return grad_output.T

    """ ________Create the Tensor class_________
        Imitating the machenism of the tensor,and making the autograd come true
                Just a simplified version
    
    """

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
    def Transpose(self):
        calculate=Transpose()
        result=Tensor(calculate.forward(self.data),require_grad=self.require_grad)
        result.grad_fn=calculate
        result.parent=(self,)
        return result
    """_______________ Tensor backward ________________
     
        

    """
    def backward(self,grad):
        
        if not self.require_grad :

            return 
        
        if grad is None:
            grad=np.ones_like(self.data)
    
        if self.grad is None:
            self.grad=grad
        else :
            self.grad += grad
        
        if self.grad_fn is not None:
            grads=self.grad_fn.backward(grad)
            if not isinstance(grads,tuple):
                grads=(grads,)
            
            for parent,parent_grad in zip(self.parent,grads):
                parent.backward(parent_grad)

    def backward_help(self): 
        pass

class Loss()   :
    def forward(self,*y):
        raise NotImplementedError
    def backwaed(self):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    def log_softmax(self,z):
        self.log_sum=np.log(np.sum(np.exp(z)))
        Temp=z
        for index,value in enumerate(Temp):
            Temp[index]=value-self.log_sum
        return Temp

    def forward(self,y_predictation,y_right):
        """
            CrossEntropyLoss 
                Formula : -Σ y_right *log(np.exp(z)/np.sum(np.exp(z)))
            
        """
        self.y_right=y_right
        self.y_predictation=y_predictation
        self.values=self.log_softmax(self.y_predictation)
        Loss=0
        for index,value in enumerate(self.values):
            Loss-=y_right[index]*value
        self.Loss=Loss
        return loss(self.y_predictation,self.y_right,self.Loss)
        #return a loss class to update the parameters
        
    
class loss():
    def __init__(self,y_predication,y_right,loss_):
        self.y_predication=y_predication
        self.y_right=y_right
        self.Loss=loss_.data.item()

    def softmax(self,x):
        x=np.array(x)
        sum_x=sum(np.exp(x))
        for index,value in enumerate(x):            
            x[index]=np.exp(value)/sum_x
        return x
            

    def item(self): 
        return self.Loss
    
    def backward(self):
        """
            update the gradient
            integrate the CrossEntropyLoss to obtain the new parameters
            
            V: the number of label 
            Formula : -Σyi*(zi-ln(Σe^zi))
            gradient: ∂L/∂zi =yi -(Σy)*(zi)*softmax(zi)
            if you want to simplify the gradient ,you should take one-hot ,the 
            it will be yi-(zi)*softmax(zi)

            In this function ,we need to calculate the gradient of CrossEntroyLoss
            and put it into global variable ,so that the optimize can use it to update 
            the parameters of model 

        """
        grad=self.y_right-self.softmax(self.y_predication)
        self.y_predication.backward(grad)       
        




class Optimize():
    def step(self):
        return NotImplementedError
    def zero_grad(self):
        return NotImplementedError
"""
        _________________Stochastic Gradient Descent__________________
                Generally speak,it's an optimizor to improve the model
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
    def __init__(self,parameters, lr):
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
    
    def __init__(self,parameters,lr,beta=(0.999,0.99),eps=1e-8):
        self.parameters=parameters
        self.lr =lr
        self.beta=beta
        self.eps=eps
        self.v_last=None
        self.m_last=None
        self.values=None
    """Instead of SGD ,the Adam use moment estimation to help update something like weight,bias and so on ,

        Formula:mt=m{t-1}*beta1-(1-beta1)*gradient
                vt=v{t-1}*beta2-(1-beta2)*gradient^2
        parameters update formula: w=w-lr*m_hat/(sqrt(v_hat)+eps)
    
    """
  
    def step(self):
        time=0
        for value in self.parameters:
            mt=self.m_last*self.beta[0]+(1-self.beta[0])*value.grad
            vt=self.v_last*self.beta[1]+(1-self.beta[1])*value.grad*value.grad
            m_hat=mt/(1-(self.beta[0]**time))
            v_hat=vt/(1-(self.beta[1]**(time*2)))
            self.m_last=mt
            self.v_last=vt
            value.data-=self.lr*(m_hat/(np.sqrt(v_hat)+self.eps))
            time+=1


             
             
  
    """______clear_grad________
        Clear the gradient to prepare for next backward pass
        and avoid gradient accumulation from subsequent computations.

     Adopt a moment updater to refreash the parameters 
        between the last parameters and the new paramerters to be given a kinds weight
    """
    def zero_grad(self):
        for value in self.parameters:
            value.grad=None





if __name__=="__main__":
    data=[1,2,3,4,5]
    data1=[[2,3,4,5,6]]
    A=Tensor(data,require_grad=True)
    B=Tensor(data1,require_grad=True)
    C=A+B
    print(C.data)