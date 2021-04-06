import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from optimize import newton, gradient_descent


class LR(object):
    computed = False

    def __init__(self, solver='newton', *args, **kwargs):
        self.solver = solver


    def fit(self, X, y):
        x_aug = self._augment(X)
        h = lambda beta, x, y: 1/(1+np.exp(-x.dot(beta)))
        loss = lambda beta, x, y: np.sum(-y*x.dot(beta)+np.log(1+np.exp(x.dot(beta))))
        grad = lambda beta, x, y: -np.sum(x*(y-h(beta, x, y)), axis=0, ).reshape((x.shape[-1], 1))

        np.random.seed(2021)
        if self.solver == 'newton':
            def grad_hess(beta, x, y):
                f = h(beta, x, y)
                g = grad(beta, x, y)
                p1 = h(beta, x, y)
                p = np.diag((p1*(1-p1)).reshape(y.shape[0]))
                hess = x.T.dot(p).dot(x)
                return g, hess

            self.beta = newton(grad_hess, loss, grad, 1*np.random.randn(x_aug.shape[-1], 1), args=(x_aug, y))
        elif self.solver == 'gd':
            self.beta = gradient_descent(grad, loss, 1*np.random.randn(x_aug.shape[-1], 1), args=(x_aug, y), maxiter=5000)

    def _augment(self, x):
        return np.hstack((x, np.ones((x.shape[0], 1))))
    
def classification_plot(beta, x, y, save=''):
    w1, w2, b = beta
    y = y.flatten()
    x1, y1, x0, y0 = x[y==1,0], x[y==1,1], x[y==0,0], x[y==0,1]
    # print(x1, y1, x0, y0, sep='\n')
    # w1 x + w2 y + b = 0 ==> y = -w1/w2 x - b/w2
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(x1, y1, c='r', marker='o', label='positive cases')
    ax.scatter(x0, y0, c='b', marker='+', label='negative cases')
    xpos = np.linspace(x[:,0].min(), x[:,0].max(), endpoint=False)
    ypos = -w1/w2*xpos - b/w2
    ax.plot(xpos, ypos, lw=2, c='k', label='classification line')
    ax.set_xlabel('density')
    ax.set_ylabel('sugar content')
    ax.legend()
    fig.tight_layout()
    fig.show()
    fig.savefig(r'../fig/%s.png' % save, dpi=600)

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data.csv', )
    X = df.iloc[:,1:3].to_numpy()
    y = df.iloc[:,3].to_numpy()[:,None]
    print(X.shape, y.shape)
    print(np.hstack((X, y)))
    
    model = LR()
    model.fit(X, y)
    beta_newton = model.beta
    classification_plot(model.beta, X, y, save='newton')

    model.solver = 'gd'

    model.fit(X, y)
    beta_gd = model.beta
    classification_plot(model.beta, X, y, save='gd')

    print('Estimated arguments by newton method:', beta_newton, sep='\n')
    print('Estimated arguments by gd:', beta_gd, sep='\n')
