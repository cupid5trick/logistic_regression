# 用牛顿法和梯度下降实现对率回归（Logistic Regerssion）

Logistic Regression实际上是用来完成分类任务的，是一种广义线性模型：通过一个单调可微函数把分类任务的标签$y_i$和线性回归模型的预测值联系起来。

对率回归使用的单调可微函数是：
$$
z=ln \frac{y}{1-y}
$$
，叫做对数几率函数。
把这个函数反解就得到一种Sigmoid函数：
$$
y=\frac{1}{1+e^{-z}}
$$
对率回归使用线性函数来逼近这个对数几率函数
$$
ln\frac{y}{1-y}=\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}+b
$$


## 1. 公式推导

### 参数估计

增广向量$\boldsymbol{w}$,$\boldsymbol{x}$为$\boldsymbol{\beta}=(\boldsymbol{w};b)$,$\boldsymbol{\hat{x}}=(\boldsymbol{x}, 1)$,极大似然估计的目的是最大化数据集的后验概率$p(y_i|\boldsymbol{x_i};\boldsymbol{\beta})$。
$$
p(y_i|\boldsymbol{x_i};\boldsymbol{\beta})=\begin{cases}
p(y=1|\boldsymbol{x_i};\boldsymbol{\beta}),\quad y_i=1\\
p(y=0|\boldsymbol{x_i};\boldsymbol{\beta}),\quad y_i=0\\

\end{cases}\\
=\left(p(y=1|\boldsymbol{x_i};\boldsymbol{\beta})\right)^{y_i}\times
\left(p(y=0|\boldsymbol{x_i};\boldsymbol{\beta})\right)^{1-y_i}
$$
定义$h_{\boldsymbol{\beta}}(\boldsymbol{x_i})=p(y=1|\boldsymbol{x_i};\boldsymbol{\beta})$, $p(y=0|\boldsymbol{x_i};\boldsymbol{\beta})=1-h_{\boldsymbol{\beta}}(\boldsymbol{x_i})$

定义似然函数
$$
\begin{equation}
\boldsymbol{L}(\boldsymbol{\beta})=\prod_{i=1}^{m}p(y_i|\boldsymbol{x_i};\boldsymbol{\beta})\\

\end{equation}
$$
取似然函数的负对数$\ell(\boldsymbol{\beta})=-\ln\boldsymbol{L}(\boldsymbol{\beta})$, 最大化$\boldsymbol{L}(\boldsymbol{\beta})$等价于最小化$\ell(\boldsymbol{\beta})$:
$$
\begin{align*}
&\ell\boldsymbol(\beta)=-\ln\boldsymbol{L}(\boldsymbol{\beta})\\
&=-\sum_{i=1}^{m}\,y_i\ln p(y=1|\boldsymbol{x_i};\boldsymbol{\beta})+(1-y_i)\ln p(y=0|\boldsymbol{x_i};\boldsymbol{\beta})\\
&=-\sum_{i=1}^{m}\,y_i\ln h_{\boldsymbol{\beta}}(\boldsymbol{x_i})+(1-y_i)\ln\left(1-h_{\boldsymbol{\beta}}(\boldsymbol{x_i})\right)\\
&=\sum_{i=1}^{m}\, -y_i\boldsymbol{\beta}^{\mathrm{T}}\boldsymbol{\hat{x_i}}+\ln(1+e^{\boldsymbol{\beta}^{\mathrm{T}}\boldsymbol{\hat{x_i}}})
\end{align*}
$$
参数的估计值为：
$$
\boldsymbol{\beta^{*}}=\mathop{argmin}_{\boldsymbol{\beta}}\,\ell(\boldsymbol{\beta})
$$

### 牛顿法迭代公式

$$
\begin{align}
&\frac{\partial{\ell(\boldsymbol{\beta})}}{\partial{\boldsymbol{\beta}}}=\sum_{i=1}^{m}\,-y_i\boldsymbol{\hat{x_i}}+\frac{e^{\boldsymbol{\beta}^{\mathrm{T}}\boldsymbol{\hat{x_i}}}}{1+e^{\boldsymbol{\beta}^{\mathrm{T}}\boldsymbol{\hat{x_i}}}}\boldsymbol{\hat{x_i}}\\
&=\sum_{i=1}^{m}\,-y_i\boldsymbol{\hat{x_i}}+h_{\boldsymbol{\beta}}(\boldsymbol{{x_i}})\boldsymbol{\hat{x_i}}\notag
\end{align}
$$

$$
\begin{align}
&\frac{\partial^2{\ell(\boldsymbol{\beta})}}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta^\mathrm{T}}}=\frac{e^{-\boldsymbol{\beta}^\mathrm{T}\boldsymbol{\hat{x_i}}}}{(1+e^{-\boldsymbol{\beta}^\mathrm{T}\boldsymbol{\hat{x_i}}})^2}\boldsymbol{\hat{x_i}}\boldsymbol{\hat{x_i}^\mathrm{T}}\\
&=h_{\boldsymbol{\beta}}(\boldsymbol{x_i})\left(1-h_{\boldsymbol{\beta}}(\boldsymbol{x_i})\right)\boldsymbol{\hat{x_i}}\boldsymbol{\hat{x_i}^\mathrm{T}}\notag
\end{align}
$$

$$
\boldsymbol{\beta^{t+1}}=\boldsymbol{\beta^{t}}-\left[\frac{\partial^2{\ell(\boldsymbol{\beta})}}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta^\mathrm{T}}}\right]^{-1}\frac{\partial{\ell(\boldsymbol{\beta})}}{\partial{\boldsymbol{\beta}}}
$$



### 梯度下降法迭代公式

$$
\boldsymbol{\beta^{t+1}}=\boldsymbol{\beta^{t}}-\alpha\frac{\partial{\ell(\boldsymbol{\beta})}}{\partial{\boldsymbol{\beta}}}
$$



