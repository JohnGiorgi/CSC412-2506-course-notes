# Tutorial 1: Introduction to Advanced Probability for Graphical Models

### Overview

- Basics
- Probability rules
- Exponential family models
- Maximum likelihood
- Conjugate Bayesian inference (time permitting)

## Notation

A random variable, \(X\) represents outcomes or states of the world. In this class, we will write \(p(x)\) to mean \(p(X=x)\), the probability of the random variable \(X\) taking on state \(x\).

!!! tip
    See [here](https://en.wikipedia.org/wiki/Notation_in_probability_and_statistics#Probability_theory) for a helpful list of notational norms in probability.

The __sample space__ is the space of all possible outcomes, which may be discrete, continuous, or mixed.

\(p(x)\) is the __[probability mass](https://en.wikipedia.org/wiki/Probability_mass_function) (or [density](https://en.wikipedia.org/wiki/Probability_density_function)) function__ (PMF/PDF), and assigns a non-negative number to each point in the  sample space. The PMF/PDF _must_ sum (or integrate) to 1. Intuitively, we can understand the PMF/PDF at \(x\) as representing how often \(x\) occurs, or how much we _believe_ in \(x\).

!!! note
    There is no requirement however, that the PMF/PDF cannot take values greater than 1. A commonly cited and intuitive example is the uniform distribution on the interval \([0, \frac{1}{2}]\). While the value of the pdf \(f_X(x)\) is 2 for \(0 \le x \le \frac{1}{2}\), the area under the graph of \(f_X(x)\) is rectangular, and therefore equal to base \(\times\) width \(= \frac{1}{2} * 2 = 1\).

## Probability Distributions

##### 1. [**Joint probability distribution**](https://en.wikipedia.org/wiki/Joint_probability_distribution)

The joint probability distribution for random variables \(X\), \(Y\) is a probability distribution that gives the probability that each of \(X\), \(Y\) falls in any particular range or discrete set of values specified for that variable.

\[
P(X=x, Y=y)
\]

which is read as "the probability of \(X\) taking on \(x\) and \(Y\) taking on \(y\)".

#### 2. [**Conditional Probability Distribution**](https://en.wikipedia.org/wiki/Conditional_probability_distribution)

 The conditional probability distribution of \(Y\) given \(X\) is the probability distribution of \(Y\) when \(X\) is known to be a particular value.

\[
P(Y=y | X=x)
= \frac{p(x, y)}{p(x)}
\]

which is read as "the probability of \(Y\) taking on \(y\) given that \(X\) is \(x\)".

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

### Probability Rules

Some important rules in probability include

#### 1. [Rule of Sum](https://brilliant.org/wiki/probability-rule-of-sum/) (marginalization)

Gives the situations in which the probability of a _union_ of events can be calculated by _summing_ probabilities together. It is often used on mutually exclusive events, meaning events that cannot both happen at the same time.

\[
p(x) = \sum_Y p(X=x, Y=y)
\]

#### 2. [Chain Rule](https://en.wikipedia.org/wiki/Chain_rule_%28probability%29)

Permits the calculation of any member of the joint distribution of a set of random variables using only conditional probabilities. The rule is useful in the study of Bayesian networks, which describe a probability distribution in terms of conditional probabilities.

#### 3. [Bayes' Rule](https://en.wikipedia.org/wiki/Chain_rule_%28probability%29)

Bayes' theorem is a formula that describes how to update the probabilities of hypotheses when given evidence. It follows simply from the axioms of conditional probability, but can be used to powerfully reason about a wide range of problems involving belief updates.

\[
p(x | y) = \frac{p(y | x)p(x)}{p(y)} = \frac{p(y | x)p(x)}{\sum_{x'}p(y | x')p(x')}
\]

which is read as "the posterior is equal to the likelihood times the prior divided by the evidence".

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
p(x | \eta) = h(x)g(\eta)exp\{\eta^TT(x)\}
\]

where

- \(x\) is a scalar or a vector and is either continuous or discrete
- \(\eta\) are the __natural parameter__ or __canonical parameters__
- \(T(x)\) is a vector of __sufficient statistics__
- \(h(x)\) is the __scaling constant__ or __base measure__
- \(g(\eta)\) is the normalizing constant that guarantees the sum/integral equals 1.

Lets show some examples of rearranging distributions into their exponential family forms

### Bernoulli

The Bernoulli distribution is given by

\[
p(x | \mu) = \mu^x(1 - \mu)^{1-x}
\]

re-arranging into exponential family form, we get

\[
= \exp\{\ln(\mu^x(1 - \mu)^{1-x})\} \\
= \exp\{\ln(\mu^x) + \ln((1 - \mu)^{1-x})\} \\
= \exp\{x\ln(\mu) + (1-x)\ln(1 - \mu)\} \\
= \exp\{\ln(1 - \mu) + x\ln(\mu) - x\ln(1 - \mu)\} \\
= (1 - \mu)\exp\{\ln(\frac{\mu}{1 - \mu})x\} \\
\]

from here, it is clear that

- \(\eta = \ln(\frac{\mu}{1 - \mu})\)
- \(T(x) = x\)
- \(h(x) = 1\)

noting that

\[
\eta = \ln(\frac{\mu}{1 - \mu}) \\
\Rightarrow \exp(\eta) = \exp(\ln(\frac{\mu}{1 - \mu})) \\
\Rightarrow 0 = e^\eta - ue^\eta - u \\
\Rightarrow u = \frac{e^\eta}{e^\eta + 1} \\
= \frac{1}{1 + e^{-\eta}} \\
= \sigma(\eta) \\
\]

we can see that \(g(\eta) = (1 - \mu) = \sigma(-\eta)\).

### Multinomial

The multinomial distribution is given by

\[
p(x_1, ..., x_M | \mu) = \prod_{k=1}^M \mu_k^{x_k}
\]

re-arranging into exponential family form, we get

\[
= \exp(\ln(\prod_{k=1}^M \mu_k^{x_k})) \\
= \exp(\sum_{k=1}^M x_k \ln\mu_k) \\
\]

from here, it is clear that

- \(\eta = \begin{bmatrix}\ln(u_1) & \cdots & \ln(u_M)\end{bmatrix}\)
- \(T(x) = x\)
- \(h(x) = 1\)


### Gaussian

The univariate normal or Gaussian distribution is given by

\[
p(x | \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma}\exp\{-\frac{1}{2\sigma^2}(x-\mu)^2\} \\
= \frac{1}{\sqrt{2 \pi} \sigma} \exp \{-\frac{1}{2\sigma^2}x^2 + \frac{1}{\sigma^2}xu - \frac{1}{2\sigma^2}u^2\} \\
= \frac{1}{\sqrt{2 \pi} \sigma} \exp\{- \frac{1}{2\sigma^2}u^2\} \exp \{  \begin{bmatrix}\frac{1}{\sigma^2}u -\frac{1}{2\sigma^2}\end{bmatrix} \begin{bmatrix}x \\\ x^2\end{bmatrix} \} \\
\]

from here, it is clear that

- \(\eta = \begin{bmatrix}\frac{1}{\sigma^2}u -\frac{1}{2\sigma^2}\end{bmatrix}\)
- \(T(x) = \begin{bmatrix}x \\\ x^2\end{bmatrix}\)

re-writing in terms of \(\eta\)

\[
= (\sqrt{2 \pi})^{-\frac{1}{2}} \cdot (-2\eta_2)^{\frac{1}{2}} \cdot \exp\{\frac{\eta_1^2}{4\eta_2}\} \cdot \exp \{  \begin{bmatrix}\frac{1}{\sigma^2}u -\frac{1}{2\sigma^2}\end{bmatrix} \begin{bmatrix}x \\\ x^2\end{bmatrix} \} \\
\]

noting that

- \(h(x) = (\sqrt{2 \pi})^{-\frac{1}{2}}\)
- \(g(\eta) = (-2\eta_2)^{\frac{1}{2}} \cdot \exp\{\frac{\eta_1^2}{4\eta_2}\}\)

!!! tip
    Chapter 9.2 of [K. Murphy, Machine Learning a Probabilistic Perspective](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20[Murphy%202012-08-24].pdf) fleshes out these examples in more detail.
