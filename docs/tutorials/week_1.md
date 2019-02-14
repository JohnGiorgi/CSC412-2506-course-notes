# Tutorial 1: Introduction to Advanced Probability for Graphical Models

### Overview

- Basics
- Probability rules
- Exponential family models
- Maximum likelihood
- Conjugate Bayesian inference (time permitting)

## Notation

A random variable, \(X\) represents outcomes or states of the world. In this class, we will write \(p(x)\) to mean \(p(X=x)\), the probability of the random variable \(X\) taking on state \(x\).

The __sample space__ is the space of all possible outcomes, which may be discrete, continuous, or mixed.

\(p(x)\) is the __probability mass (or density) function__ (PMF/PDF), and assigns a non-negative number to each point in the  sample space. The PMF/PDF _must_ sum (or integrate) to 1. Intuitively, we can understand the PMF/PDF at \(x\) as representing how often \(x\) occurs, or how much we _believe_ in \(x\).

!!! note
    There is no requirement however, that the PMF/PDF cannot take values greater than 1. A commonly cited and intuitive example is the uniform distribution on the interval \([0, \frac{1}{2}]\). While the value of the pdf \(f_X(x)\) is 2 for \(0 \le x \le \frac{1}{2}\), the area under the graph of \(f_X(x)\) is rectangular, and therefore equal to base \(\times\) width \(= \frac{1}{2} * 2 = 1\).

## Probability Distributions

##### 1. [**Joint probability distribution**](https://en.wikipedia.org/wiki/Joint_probability_distribution)

The joint probability distribution for random variables \(X\), \(Y\) is a probability distribution that gives the probability that each of \(X\), \(Y\) falls in any particular range or discrete set of values specified for that variable.

\[
P(X=x, Y=y)
\]

which is read as "the probability of \(X\) taking on \(x\) and \(Y\) taking on \(y\)"

#### 2. [**Conditional Probability Distribution**](https://en.wikipedia.org/wiki/Conditional_probability_distribution)

 The conditional probability distribution of \(Y\) given \(X\) is the probability distribution of \(Y\) when \(X\) is known to be a particular value.

\[
P(Y=y | X=x)
= \frac{p(x, y)}{p(x)}
\]

which is read as "the probability of \(Y\) taking on \(y\) given that \(X\) is \(x\)"

#### 3. [**Marginal Probability Distribution**](https://en.wikipedia.org/wiki/Conditional_probability_distribution)

The marginal distribution of a subset of a collection of random variables is the probability distribution of the variables contained in the subset

\[
P(X=x) ; P(y=y)
\]

if \(X\), \(Y\) are discrete

\[
p(X=x) = \sum_Yp(X=x,Y=y) \; ; \; p(Y=y) = \sum_Xp(X=x,Y=y)
\]

if \(X\), \(Y\) are continuous

\[
p(X=x) = \int_Yp(X=x,Y=y) \; ; \; p(Y=y) = \int_Xp(X=x,Y=y)
\]

which is read as "the probability of \(X\) taking on \(x\)" or "the probability of \(Y\) taking on \(y\)".

!!! warning
    Skipped some slides here. Come back and finish them.

## Exponential Family

An [**exponential family**](https://en.wikipedia.org/wiki/Exponential_family) is a set of probability distributions of a certain form, specified below. This special form is chosen for mathematical convenience, based on some useful algebraic properties, as well as for generality, as exponential families are in a sense very natural sets of distributions to consider.

Most of the commonly used distributions are in the exponential family, including

- Bernoulli
- Binomial/multinomial
- Normal (Gaussian)

_def_. __Eponential family__: The exponential family of distributions over \(x\), given parameter \(\eta\) (eta) is the set of distributions of the form

\[
p(x | \eta) = h(x)g(\eta)exp\{\eta^T\mu(x)\}
\]
