---
layout: post
title: "On the Criticism of Priors in Bayesian Methods"
date: 2026-01-13
author: Pavlo Mysak
tags: [journal]
---

If you’ve spent any time lurking on LinkedIn, statistics forums, or the sacred r/datascience subreddit, you’ve probably heard something like: “The validity of Bayesian methods hinge entirely on priors.” Well, priors *do* bias your estimates, right? So how can we trust the results?

Before we grab the pitchforks, let’s take a closer look. Are these criticisms genuinely damning or are they mostly a misunderstanding of what priors actually do? In the sections that follow, we’ll explore why, in practice, thoughtfully chosen priors often make your inferences better, not worse. 

# 1. Let's talk bias

Most frequentist guarantees, like unbiasedness or even the central limit theorem, are asymptotic. That means they only truly kick in when your sample size goes to infinity. In practice, “infinity” is usually larger than what you actually have. More importantly, the sample size required for those guarantees depends heavily on the specifics of your data and the problem you’re solving. So even if you tell me you have an MVUE (minimum variance unbiased estimator)... do you really?

But what if your sample size truly is massive?

Let's look at the posterior mean in a Bayesian normal model (with known variance, for simplicity): 
$$ \mathbb{E}[\mu|x] = \frac{\kappa_0}{\kappa_0 + n} \mu_0 + \frac{n}{\kappa_0 + n} \bar{y} $$

where $n$ is our sample size, $\bar{y}$ is the sample mean (MLE here), $\mu_0$ is the prior mean, and $\kappa_0$ can be thought of as a pseudo sample size for the prior here.

- As $ n \rightarrow \infty $:
    - The prior's weight $\frac{\kappa_0}{\kappa_0 + n}$ goes to 0.
    - The posterior mean converges to the MLE.

In this asymptotic limit, Bayesian and frequentist answers are effectively identical. But we rarely live in the land of asymptotics. We live in the land of small $n$, where relying solely on the data yields high-variance, unstable estimates. In this context, the posterior mean is biased, but in a useful way. When samples are small, we lean on the prior for stability; as evidence accumulates, the likelihood takes over and the posterior converges to frequentist estimates. Here, the bias acts as a stabilizer rather than a flaw.

Furthermore, this idea of “bias for stability” isn’t unique to Bayesians. Frequentist methods often handle the same problem with regularization: penalizing large coefficients, shrinking estimates toward zero, or adding other constraints to prevent overfitting.

Bayes offers an elegant counterpart. Instead of shrinking everything toward zero, you can shrink toward informed values like the prior mean or, in hierarchical models, the mean of a related group. This is the magic of partial pooling: each group borrows strength from others, balancing individual data with the larger structure.

# 2. Let's talk information

When it comes to priors, we not only control the location of the density, but also the *strength* of the encoded information.

Take our Gaussian example from before. The prior’s “strength” ($\kappa_0$, the pseudo sample size) essentially controls the precision of the prior. A taller, narrower prior (large $\kappa_0$) pulls the posterior more strongly toward $\mu_0$, while a flatter, wider prior (small $\kappa_0$) lets the data dominate. Mechanistically, this is the same as adjusting the “weight” of our prior relative to the data.

But remember, the prior isn’t the only player. The data also has a "strength" (think Fisher's Information). If the data tells a compelling story, it will dominate the posterior, and the prior barely matters.

Yes, you can use diffuse (uninformative) priors if you truly want a “raw” posterior. But the associated cost is that uninformative priors often lead to inefficient sampling and slower convergence if you're using MCMC, especially in complex models. Even a small amount of well-encoded information can dramatically improve performance without biasing results in any practically meaningful way. 

# 3. Priors Make Assumptions Explicit

One of the best things about Bayesian methods is that priors force you to be explicit about your assumptions. In Frequentist settings, parameters are treated as fixed, unknown constants that are often determined entirely by the likelihood and can range from $-\infty$ to $\infty$. That’s not assumption-free; it's a deliberate choice to leave plausibility unmodeled. 

Making assumptions explicit is both philosophically nice and practically safer. When priors are clearly specified, you can test them with prior predictive checks, reason about their impact, and adjust them if they’re implausible. When assumptions are implicit, it’s easier to overlook them and commit statistical malpractice without realizing it. 

At the end of the day, priors make your assumptions visible, your estimates more stable, and your modeling more transparent. Sure, misuse is possible... but it’s a human problem, not a flaw in the framework itself.