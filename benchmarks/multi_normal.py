# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
r'''
{xrst_begin multi_normal_benchmark ben}
{xrst_spell
    cholesky
}

Fitting a multivariate normal distribution
##########################################

Loss Function
*************
Let :math:`\Sigma \in \mathbb{R}^{m \times m}` be a
symmetric positive definite matrix and :math:`\mu \in \mathbb{R}^m` .
Suppose :math:`y_i \in \mathbb{R}^m` for :math:`i = 0 , \ldots , n-1`
is distributed normally with mean :math:`\mu` and variance :math:`\Sigma` .
The probability density for the sequence :math:`\{ y_i \}` given
its mean and variance is

.. math::

    \mathbb{p} ( \{ y_i \} | \mu , \Sigma )
    =
    ( 2 \pi )^{- n m / 2 } \det( \Sigma )^{ - n / 2 }
    \cdot
    \exp \left[ -\frac{1}{2}
        \sum_{i=0}^{n-1} ( y_i - \mu)^T \Sigma^{-1} (y_i - \mu)
    \right]

The negative log density is

.. math::

    - \log [ \mathbb{p} ( \{ y_i \} | \mu , \Sigma ) ]
    =
    \frac{n m}{2} \log(2 \pi) + \frac{n}{2} \log \det( \Sigma )
    +
    \frac{1}{2} \sum_{i=0}^{n-1} ( y_i - \mu)^T \Sigma^{-1} (y_i - \mu)

We express :math:`\Sigma^{-1}` in terms of its Cholesky factor
:math:`L` ; i.e., :math:`L` is lower triangular,
:math:`\Sigma^{-1} = L L^T` , and

.. math::

    - \log [ \mathbb{p} ( \{ y_i \} | \mu , L ) ]
    =
    \frac{n m}{2} \log(2 \pi) - \frac{n}{2} \log \det( L L^T  )
    +
    \frac{1}{2} \sum_{i=0}^{n-1} ( y_i - \mu)^T L L^T (y_i - \mu)

We drop the constant :math:`n m \log(2 \pi)` and the factor of one half.
In addition, we scale the objective by dividing by the number of data points
:math:`n`.
The resulting loss function is
:math:`f : \mathbb{R}^m \times \mathbb{R}^{m \times m} \rightarrow \mathbb{R}`
where

.. math::
    f( \mu, L )
    =
    - \log \det( L L^T )
    +
    \frac{1}{n} \sum_{i=0}^{n-1} ( y_i - \mu)^T L L^T (y_i - \mu)

and :math:`L` is restricted to the lower triangular matrices
with positive entries on the diagonal.
The minimizer of this loss function is

.. math::

    \hat{\mu}    & =
        \frac{1}{n} \sum_{i=0}^{n-1} y_i
    \\
    \hat{\Sigma} & =
        \frac{1}{n} \sum_{i=0}^{n-1} (y_i - \hat{\mu}) (y_i - \hat{\mu})^T

see the `proof <https://statproofbook.github.io/P/mvn-mle.html>`_ .

{xrst_end multi_normal_benchmark}
'''
import time
import torch
#
start_time            = time.perf_counter()
number_learning_steps = 5000
learning_rate         = 1e-4
#
def f(mu, L, y) :
    (m, n)     = y.size()
    Lt         = L.transpose(0, 1)
    Sigma_inv  = torch.matmul(L, Lt)
    term_det   = torch.logdet(Sigma_inv)
    #
    res        = y - mu.repeat(1, n)
    Lt_res     = Lt.matmul(res)
    term_res   = (Lt_res * Lt_res).sum() / n
    #
    return term_res - term_det
#
#   m, n, L_min
m = 2
n = 1000
#
# L_min
L_min = torch.full( (m,m), float('-inf') )
for j in range(m) :
    L_min[j, j] = 0.01
L_min = torch.tril( L_min )
#
# y, mu, L
y    = torch.randn( (m, n) )
mu   = torch.randn(m).view((m, 1))
L    = torch.tril( torch.randn( (m, m) ) )
L    = torch.maximum(L, L_min)
print("L =", L)
print("mu =", mu)
#
initial_loss  = None
for t in range(number_learning_steps) :
    mu.requires_grad_(True)
    mu.grad = None
    #
    L.requires_grad_(True)
    L.grad  = None
    #
    loss = f(mu, L, y)
    loss.backward()
    if t == 0 :
        initial_loss = loss.item()
    relative_loss = loss.item() / initial_loss
    if t % 500 == 0  :
        print( f'relative_loss = {relative_loss}' )
    with torch.no_grad() :
        mu = mu - learning_rate * mu.grad
        L  = torch.tril( L - learning_rate * L.grad )
        L  = torch.maximum(L, L_min)
elapsed_time = time.perf_counter() - start_time
print( f'elapsed time = {elapsed_time}' )
Lt       = L.transpose(0, 1)
res      = y - mu.repeat(1, n)
Lt_res   = Lt.matmul(res)
term_res = (Lt_res * Lt_res).sum() / (m * n)
print( f'average weighted residual squared = {term_res}' )
