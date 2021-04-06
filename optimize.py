import numpy as np
import warnings

def newton(grad_hessian, func, grad, x0, tol=1e-5, maxiter=200, args=(), log_each=10, log=True):
    xk = np.asarray(x0).reshape((x0.size, 1))
    k = 0
    l_old = func(xk, *args)
    absgrad = None
    while k < maxiter:
        fgrad, fhess = grad_hessian(x0, *args)
        absgrad = np.abs(fgrad)


        try:
            xk -= np.linalg.inv(fhess).dot(fgrad)
        except np.linalg.LinAlgError:
            pass
        l_new = func(xk, *args)
        if absgrad.max() <= tol:
            break
        if log and k % log_each == 0:
            print('Newton at Iteration %d: loss=%f grad=%f loss_difference=%f' %
                  (k+1, func(xk, *args), absgrad.max(), np.abs(l_new - l_old)))
        l_old = l_new
        k += 1
    if k >= maxiter:
        warnings.warn('Newton method did not converge after %d iterations!' % k)
    if log:
        print('Ultimate argument after %d iterations:\n%s\nloss=%f grad=%f' %
              (k, xk, l_old, absgrad.max()))
    return xk

def gradient_descent(grad, func, x0, alpha=.3, maxiter=200, tol=1e-5, args=(), log_each=10, log=True):
    xk = np.asarray(x0)
    k = 0
    l_old = func(xk, *args)
    absgrad = None
    while k < maxiter:
        fgrad = grad(x0, *args)
        absgrad = np.abs(fgrad)

        xk -= alpha*fgrad
        l_new = func(xk, *args)
        if absgrad.max() <= tol:
            break
        if log and k % log_each == 0:
            print('GD at Iteration %d: loss=%f grad=%f loss_difference=%f' %
                  (k+1, func(xk, *args), absgrad.max(), np.abs(l_new - l_old)))
        l_old = l_new
        k += 1
    if k >= maxiter:
        warnings.warn('Gradient descent did not converge after %d iterations!' % k)
    if log:
        print('Ultimate argument after %d iterations:\n%s\nloss=%f grad=%f' %
              (k, xk, l_old, absgrad.max()))
    return xk