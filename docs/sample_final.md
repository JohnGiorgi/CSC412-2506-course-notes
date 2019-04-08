# Sample Final

These are question pulled from the lecture and the sample midterm, alongside questions that were written based on the study guide. These were _not_ given by an instructor and are merely guesses as to what kind of questions might be on the final.

!!! note
    See the study guide [here](http://www.cs.toronto.edu/~jessebett/CSC412/content/Final_Topics/final_topics.pdf).

## Week 1

## Tutorial 1

### Question 1

Recall that the definition of an exponential family model is:

\[
f(x | \eta) = h(x)g(\eta)\exp(\eta^TT(x))
\]

where

- \(\eta\) are the parameters
- \(T(x)\) are the sufficient statistics
- \(h(x)\) is the base measure
- \(g(\eta)\) is the normalizing constant

Consider the univariate Gaussian, with mean \(\mu\) and precision \(\lambda = \frac{1}{\sigma^2}\)

\[
p(D | \mu \lambda) = \prod^N_{i=1}(\frac{\lambda}{2\pi})^{\frac{1}{2}}\exp(-\frac{\lambda}{2}(x_i - \mu)^2)
\]

What are \(\eta\) and \(T(x)\) for this distribution when represented in exponential family form?

__ANSWER__

Start by expanding the terms in the exponent

\[
= \prod^N_{i=1}(\frac{\lambda}{2\pi})^{\frac{1}{2}} \exp(\sum_{i=1}^N -\frac{\lambda}{2}x_i^2 + \lambda u x_i - \frac{\lambda}{2}\mu^2) \\
\]

from here, we can rearrange the exponent into \(\eta^TT(x)\)

\[
= \prod^N_{i=1}(\frac{\lambda}{2\pi})^{\frac{1}{2}}\exp(\sum_{i=1}^N - \frac{\lambda}{2}\mu^2)\exp(\begin{bmatrix}\lambda u & -\frac{\lambda}{2} & \dotsc & \lambda u & -\frac{\lambda}{2}\end{bmatrix} \begin{bmatrix}x_1 \\ x_1^2 \\ \vdots \\ x_N \\ x_N^2\end{bmatrix}) \\
\]


where

- \(\eta^T = \begin{bmatrix}\lambda u & -\frac{\lambda}{2} & \dotsc & \lambda u & -\frac{\lambda}{2}\end{bmatrix}\)
- \(T(x) = \begin{bmatrix}x_1 \\ x_1^2 \\ \vdots \\ x_N \\ x_N^2\end{bmatrix}\)

## Week 2
## Week 3

### Question 1

Consider the Hidden Markov Model

![](img/sample_midterm_4.png)

(a) Assume you are able to sample from these conditional distributions, i.e.

\[
x_i \sim p(X_i \ | \ \text{parents of } X_i)
\]

Write down a step-by-step process to produce a sample observation from this model, i.e. \((x_1, x_2, x_3, ..., x_T)\) in terms of samples from the individual factors.

__ANSWER__

We want to sample a sequence of observations \(x_1, x_2, x_3, ..., x_T\) from the model according to

\[
x_{1:T} \sim \prod_{t=1}^T p(X_t \ | \ \text{parents of } X_t)
\]

since observations \(x_t\) are independent of one another. Notice that this forms a chain, with probability

\[
p(x_{1:T}) \sim \bigg [ \prod_{t=1}^T p(X_t | z_t) \bigg ] \bigg [ p(z_1) \prod_{t=2}^T p(Z_t | z_{t-1}) \bigg ]
\]

_Step-by-step_

1. Start with \(t=1\)
2. Sample \(z_t\) according to \(z_t \sim p(z_1) \prod_{i=2}^t p(Z_i | z_{i-1})\)
3. Given the sampled \(z_t\), sample \(x_t\) according to \(x_t \sim \ p(X_t | z_t)\)
4. Increment \(t\) by 1
5. Repeat steps 2-4 until \(t=T\)

## Week 4
## Week 5
## Week 6
## Week 8
## Week 9: Hidden Markov Models

### Question 1

Say we have the following simple chain

![](../img/lecture_8_3.png)

where

- \(x_t \in [N, Z, A]\)
- \(z_t \in [H, S]\)

where our observed states are whether or not we are watching Netflix (\(N\)), sleeping (\(Z\)), or working on the assignment (\(A\)) and our hidden states are whether we are happy (\(H\)) or sad (\(S\)). Say futher that we are given the initial (\(\pi\)), transition (\(T\)), and emission probabilities (\(\varepsilon\))

<center>

| \(\pi\) |      |
| ------- | ---- |
| H       | 0.70 |
| S       | 0.30 |

</center>

<center>

| \(\varepsilon\) | N    | Z    | A    |
| --------------- | ---- | ---- | ---- |
| H               | 0.40 | 0.50 | 0.10 |
| S               | 0.10 | 0.30 | 0.60 |

</center>

<center>


| T   | H    | S    |
| --- | ---- | ---- |
| H   | 0.80 | 0.20 |
| S   | 0.10 | 0.90 |

</center>

!!! note
    It is the _rows_ of these tables that need to sum to 1, not the columns!

From these conditional probabilities, compute

- \(p(z_3 = H | z_1 = S) = ?\)
- \(p(x_3 = A | z_1 = S) = ?\)

__ANSWER__

\[
p(z_3 = H | z_1 = S) = p(z_3 = H | z_2 = H)p(z_2 = H | z_1 = S) + p(z_3 = H | z_2 = S)p(z_2 = S | z_1 = S) \\
= (0.80)(0.1) + (0.10)(0.90) \\
= 0.17 \\
\]

and

\[
p(x_3 = A | z_1 = S) = p(x_3 = A | z_3 = H)p(z_3 = H | z_1 = S) + p(x_3 = A | z_3 = S)p(z_3 = S | z_1 = S) \\
= (0.10)(0.17) + (0.60)(1 - 0.17) \\
= 0.515
\]

### Question 2

Consider the following directed graphical model:

![](img/sample_midterm_1.png)

(a) List all variables that are independent of \(A\) given evidence on \(B\)

![](img/sample_midterm_2.png)

By Bayes' Balls, no variables are conditionally independent of \(A\) given evidence on \(B\).

(b) Write down the factorized normalized joint distribution that this graphical model represents.

\[
p(A, ..., I) = p(A | B, C)P(B | D)P(C | E, F)P(D | G)P(E | G)P(F | H)P(G)P(H)P(I | G, H)
\]

(c) If each node is a single discrete random variable in \({1, ..., K}\) how many distinct joint states can the model take? That is, how many different configurations can the variables in this model be set?

For each node (random variable) there is \(k\) states. There are \(k^n\) possible configurations where \(k\) is the number of states and \(n\) the number of nodes (\(x_{\pi_i}\))

\[
\therefore \text{number of possible configurations} = k^9
\]

### Question 3

__ANSWER__

a) The size of the maximum factor is 3.
b) The size of the maximum clique is 3.
c) The size of the maximum factor is 4.
d) The size of the maximum clique is 4.

## Week 10: Stochastic Variational Inference / Automatic Differentiation Variation Inference (SAD VI)

### Question 1

Starting from the Kullback–Leibler divergence (\(D_{KL}\)), derive the Evidence Lower Bound (ELBO) for a true distribution \(p_\theta (z | x)\) and approximate distribution \(q_\phi(z|x)\) and show that maximizing the ELBO is equivalent to minimizing \(D_{KL}(q_\phi (z | x) || p_\theta (z | x))\).

__ANSWER__

\begin{align}
  D_{KL}(q_\phi (z | x) || p_\theta (z | x)) &= E_{z_\phi \sim q_\phi} \log \frac{q_\phi(z | x)}{p_\theta(z | x)} \\
  &= E_{z_\phi \sim q_\phi} \Bigg [ \log \Bigg ( q_\phi(z | x) \cdot \frac{p_\theta(x)}{p_\theta(z, x)} \Bigg ) \Bigg ] \\
  &= E_{z_\phi \sim q_\phi} \log \frac{q_\phi(z | x)}{p_\theta(z, x)}  + E_{z_\phi \sim q_\phi} \log p_\theta(x) \\
  &= -\mathcal L(\theta, \phi ; x)  + \log p_\theta(x) \\
\end{align}

Where \(\mathcal L(\theta, \phi ; x)\) is the **ELBO**. Rearranging,

\begin{align}
  D_{KL}(q_\phi (z | x) || p_\theta (z | x)) &= -\mathcal L(\theta, \phi ; x)  + \log p_\theta(x) \\
  \Rightarrow \mathcal L(\theta, \phi ; x) + D_{KL}(q_\phi (z | x) || p_\theta (z | x)) &= \log p_\theta(x) \\
\end{align}

because \(D_{KL}(q_\phi (z | x) || p_\theta (z | x)) \ge 0\)

\[
\mathcal L(\theta, \phi ; x) \le \log p_\theta(x)
\]

\(\therefore\) maximizing the ELBO \(\Rightarrow\) minimizing \(D_{KL}(q_\phi (z | x) || p_\theta (z | x))\).
