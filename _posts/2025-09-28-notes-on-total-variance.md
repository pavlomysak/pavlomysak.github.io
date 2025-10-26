---
layout: post
title: "Notes on Total Variance"
date: 2025-09-28
author: Pavlo Mysak
tags: [journal]
---

Within the first month of working toward my master’s degree at Boston University, I noticed a pattern of the same ideas reappearing across difference classes, but under different names and from different angles. 

Two of those ideas were the law of total variance and the bias-variance tradeoff. And while the math itself is straightforward enough, what caught my attention is how different fields frame and interpret them. I haven’t seen a post tying these perspectives together, so I figured I’d take a shot at formalizing this connection.

## Law of Total Variance (Eve's Law)
Let’s start with the probability theory side of things. The law of total variance tells us:
$$ Var(Y) = \mathbb{E}[Var(Y|X)] + Var(\mathbb{E}[Y|X]) $$

If you’ve never seen this before it may look a little cryptic or suspicious, but it’s called a law for a reason. And you don’t just have to take my (or Eve’s) word for it: with the law of iterated expectations (Adam’s Law) and some algebra, you can prove it yourself. Turns out Adam & Eve’s contributions to probability theory are vastly underappreciated.

I’ll skip the [proof](https://statproofbook.github.io/P/var-tot) here, but the key idea is that the total variance of Y can be split into two pieces:
- the *within-group variance*, \( \mathbb{E}[Var(Y|X)] \)
- the *between-group variance*, \( Var(\mathbb{E}[Y|X]) \)

If you’ve ever seen ANOVA, this probably rings a bell. In that world, the whole point is to compare between-group variance to within-group variance: if the former is substantially larger, it’s evidence that group means are likely different.

That’s one way to read this formula. But it can also be shown in a perspective that puts this decomposition in the [language of uncertainty](https://andrewcharlesjones.github.io/journal/epi-ali-uncertainty.html). Specifically:
- The first term, \( \mathbb{E}[Var(Y|X)] \), corresponds to aleatoric uncertainty: the inherent randomness and noise built into the data-generating process. That is, it’s the “noise we’re stuck with.”
- The second term, \( Var(\mathbb{E}[Y|X]) \), corresponds to epistemic uncertainty: uncertainty that comes from not knowing enough about the data-generating process. In other words, uncertainty that could shrink if we gathered more data or adjusted model parameters.

In Bayesian inference, this split isn’t just a neat decomposition; it’s baked into how we think about likelihoods and posteriors. Aleatoric uncertainty is the expected noise from the sampling model. Epistemic uncertainty is the variation induced by the posterior over parameters. And crucially, as we collect more data, epistemic uncertainty tends to fade away, which lines up perfectly with the intuition.

## Bias-Variance Tradeoff

Now let’s switch gears to machine learning. A classic idea here is the [bias–variance tradeoff](https://mlu-explain.github.io/bias-variance/). You’ve probably heard the high-level version before: models with too much flexibility tend to have high variance, while models that are too simple/rigid tend to have high bias. But let’s actually derive where this decomposition comes from.

Suppose we want to measure the error of a predictive model $y(x; D)$, where $D$ is a dataset. A natural way to do this is to look at the squared deviation from the true signal $h(x)$:  

$$ [y(x;D) - h(x)]^2 $$

Here:  

- $y(x;D)$ is our prediction given a particular dataset $D$  
- $h(x)$ is the “ground truth” or optimal prediction  

The trick is to rewrite this error by pivoting around
the expected prediction, $\mathbb{E}_D[y(x;D)]$.
In other words, add and subtract the same thing:

$$ [y(x;D) - \mathbb{E}_D[y(x;D)] + \mathbb{E}_D[y(x;D)] - h(x)]^2 $$

Now expand it with $(a+b)^2 = a^2 + b^2 + 2ab$:

$$ [y(x;D) - \mathbb{E}_D[y(x;D)]]^2 + [\mathbb{E}_D[y(x;D)] - h(x)]^2 + 2(y(x;D) - \mathbb{E}_D[y(x;D)])(\mathbb{E}_D[y(x;D)] - h(x)) $$

To turn this into a proper measure of error, we take the expectation
over dataset $D$ (think of it as averaging over many possible training samples):

$$ \mathbb{E}_D[(y(x;D) -\mathbb{E}_D[y(x;D)])^2] + \mathbb{E}_D[(\mathbb{E}_D[y(x;D)] - h(x))^2] + \mathbb{E}_D[\text{cross term}] $$

Here’s the nice part: with a bit of algebra, you can show the cross term
vanishes. (Quick hint: it’s the covariance between $y(x;D) - \mathbb{E}_D[y(x;D)]$ and the constant term $\mathbb{E}_D[y(x;D)] - h(x)$, which is zero.)

So we’re left with:

$$ \mathbb{E}_D[(y(x;D) - \mathbb{E}_D[y(x;D)])^2] + [\mathbb{E}_D[y(x;D)] - h(x)]^2 $$

And now we can name the two pieces: 

- $\mathbb{E}_D[(y(x;D) -\mathbb{E}_D[y(x;D)])^2]$ is the **model variance**
- $[\mathbb{E}_D[y(x;D)] - h(x)]^2$ is the **model bias squared**

That’s the bias–variance decomposition. It tells us that model error isn’t just one homogenous thing, rather it's the sum of two different sources.  

... doesn't this look kind of familiar?

## The Comparison

We know that:  

- \( \mathbb{E}[Var(Y|X)] \rightarrow \) **aleatoric uncertainty**
- \( Var(\mathbb{E}[Y|X]) \rightarrow \) **epistemic uncertainty**
- $\mathbb{E}_D[(y(x;D) -\mathbb{E}_D[y(x;D)])^2] \rightarrow $ **model variance**
- $[\mathbb{E}_D[y(x;D)] - h(x)]^2 \rightarrow $ **model bias squared**

How can we match up these terms based on similarity? Well, let’s start with the easy one. 

Epistemic uncertainty and the variance term are clearly long-lost cousins. Both are expectations of a squared deviation, and if you stare at the bias–variance expression long enough, you’ll notice $\mathbb{E}_D[(y(x;D) -\mathbb{E}_D[y(x;D)])^2]$ is literally the variance of $y(x;D)$. And what is $y(x;D)$, really, if not “$Y$ given $X$” in disguise?

So if we let $D$ go to infinity: 

$$ \mathbb{E}_D[(y(x;D) - \mathbb{E}_D[y(x;D)])^2] = Var(\mathbb{E}_D[Y|X]) $$

Ergo, **model variance = epistemic uncertainty**.  
Nice and tidy.

But what about the other half? Does that mean the bias term should line up with aleatoric uncertainty?

Let’s think it through. Our bias, 
$$ [\mathbb{E}_D[y(x;D)] - h(x)]^2 $$
measures the systematic gap between the average prediction (across datasets) and the true signal $h(x)$. Even if we trained on infinitely many datasets, if our model is misspecified, this term won’t vanish. That’s the heart of bias: it’s structural error baked into the model. And specifically, it comes from the model being too rigid or inflexible.

Aleatoric uncertainty, by contrast, is the part of our variance that *cannot* vanish with more information. It reflects the randomness in the data-generating process, and with more data (or a more flexible model) it won't shrink.

Mathematically, aleatoric uncertainty is written as:
$$ \mathbb{E}[Var(Y|X)] = \mathbb{E}\left[\mathbb{E}\!\left[(Y|X - \mathbb{E}[Y|X])^2\right]\right] $$
This doesn’t look much like bias at all. In fact,

$$\mathbb{E}\left[\mathbb{E}\!\left[(Y|X - \mathbb{E}[Y|X])^2\right]\right] \neq (\mathbb{E}[Y|X] - Y)^2 $$

and unfortunately, Adam & Eve can’t rescue us from that algebra.

Furthermore, theoretically this makes a lot of sense. Why would the model bias line up with the inherent noise in the data-generating process? But while we're thinking theoretically, why doesn't bias line up with epistemic uncertainty at all? I mean, it *is* redicible. 

It turns out, it's all in how you define epistemic uncertainty. 

### Statistical Learning Perspective
If we take the **statistical learning** definition, epistemic uncertainty only refers to uncertainty that comes from *dataset* randomness. In that framework, bias is not part of epistemic uncertainty, because it’s not random at all. It’s a deterministic shift caused by a misspecified model class. You could collect infinite data and the model would still be wrong in the same way.

Under this view:
- Epistemic Uncertainty = Model Variance
- Aleatoric Uncertainty = Data Noise

and bias is a separate, non-uncertainty term seen as a kind of structural error. This is exactly what we proved above. 

### Uncertainty Quantification Perspective
if we broaden the lens to the **uncertainty quantification (UQ)** perspective (which is a little more theoretical), then epistemic uncertainty is anything stemming from lack of knowledge. That includes both:
- the variability in the model’s predictions across training sets (**variance**), and
- our uncertainty about whether the model form is even correct (**bias**).

## Conclusion

And that’s the punchline. Bias–variance decomposition and aleatoric–epistemic uncertainty aren’t rival frameworks but the same story told in two different dialects.

The former slices error by statistical origin (noise, variance, bias), while the latter slices it by philosophical origin (randomness vs ignorance).  
