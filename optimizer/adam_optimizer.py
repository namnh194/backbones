# ref: https://github.com/enochkan
import numpy as np
import torch.nn as nn

class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr # learning rate
        self.beta1 = beta1 # momentum  beta 1
        self.beta2 = beta2 # momentum beta 2
        self.epsilon = epsilon # smooth param
        self.m_dw, self.m_db = 0, 0 # init weight and gradient value by m
        self.v_dw, self.v_db = 0, 0 # init weight and gradient value by v
    def adam_update(self, t, w, b, dw, db):
        # t: current step
        # w, b: current weight value
        # dw, db: current gradient value of w, b

        # with momentum beta 1:
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        # with momentum beta 2:
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

        # bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)

        # update weight and bias
        w = w - self.lr*m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon)
        b = b - self.lr*m_db_corr/(np.sqrt(v_db_corr)+self.epsilon)
        return w, b
    def update(self, t, w, b, dw, db):
        w = w - self.lr*dw
        b = b - self.lr*db
        return w, b

# define some loss function and gradient function value
def loss_function(w):
    return w**2-2*w+1
def grad_function(w): # global minimum is 1
    return 2*w-2
def check_convergence(w0, w1): #
    return w0==w1

if __name__ == "__main__":
    w0, b0 = 0, 0
    adam = AdamOptimizer()
    step = 1
    convergence = False

    while not convergence:
        dw = grad_function(w0)
        db = grad_function(b0)
        w0_old = w0
        #w0, b0 = adam.update(step, w=w0, b=b0, dw=dw, db=db)
        w0, b0 = adam.adam_update(step, w=w0, b=b0, dw=dw, db=db)
        if check_convergence(w0, w0_old):
            print('converged after ' + str(step) + ' iterations')
            print('weight:',w0,'. bias:', b0)
            break
        else:
            print('iteration ' + str(step) + ': weight=' + str(w0))
            step += 1