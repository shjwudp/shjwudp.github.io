---
layout: post
title: 'Math for Megatron Mixture-of-Experts (MoE)'
date: 2023-08-22 20:00:00 +0800
tags: Mixture-of-Experts
categories: note
giscus_comments: true
related_posts: false
---

notations:

- $$s$$ - sequence length
- $$b$$ - micro-batch size
- $$h$$ - hidden dimension size
- $$L$$ - number of transformer layers
- $$P$$ - number of parameters
- $$p_{etp}$$ - degree of expert tensor parallelism
- $$p_{tp}$$ - degree of tensor parallelism
- $$p_{dp}$$ - degree of data parallelism
- $$p_{pp}$$ - degree of pipeline parallelism
- $$p_{ep}$$ - degree of expert parallelism
- $$e_{local\_n}$$ - num of local experts
- $$e_i$$ - interval of experts in transformer layers
- $$e_{top-k}$$ - the top-k number configured in the MoE algorithm

## Memory Estimation

The following memory consumption is based on the [Megatron-LM](https://github.com/shjwudp/megatron-moe-for-sharing.git) GPT Model with experts and distributed optimizer.

The activation memory consumption of the model is:

$$
M_{full\_activation} = sbhL * (13 + 19\frac{e_{local\_n} + e_i - 1}{e_i} ) + 2sbh + 4sbv
$$

With tensor parallelism, sequence parallelism and expert tensor parallelism, the activation memory consumption of the model is:

$$
M_{activation} = \frac{M_{full\_activation}}{p_{tp}}
$$

With tensor parallelism, sequence parallelism and expert tensor parallelism, the static memory consumption of the model is:

$$
\begin{aligned}
M_{static} &= M_{grad} + M_{model\_state} + M_{optimizer\_{state}} \\
    &= \frac{P_{dense} + P_{MoE}}{p_{tp}} * 8 + \frac{P_{dense} + p_{ep}P_{MoE}}{p_{tp}p_{dp}} * 10 \\
    &= \frac{P_{dense}}{p_{tp}p_{dp}}(8p_{dp}+10) + \frac{P_{MoE}}{p_{tp}p_{dp}}(8p_{dp}+10p_{ep})
\end{aligned}
$$

$$
\begin{aligned}
P_{dense} &= M_{embedding} + (M_{attn} + M_{mlp} * \frac{e_i - 1}{e_i}) * L \\
    &= hv + (4h^2 + 3hh_{ff}\frac{e_i - 1}{e_i})L \\
    &= 12h^2L(\frac{v}{12hL} + \frac{1}{3} + \frac{e_i - 1}{e_i})
\end{aligned}
$$

$$
\begin{aligned}
P_{MoE} &= M_{mlp} * \frac{e_{local\_n}}{e_i} * L \\
    &= 2hh_{ff} * \frac{e_{local\_n}}{e_i} * L \\
    &= 8h^2L * \frac{e_{local\_n}}{e_i}
\end{aligned}
$$

$$
\begin{aligned}
P &= M_{embedding} + (M_{attn} + M_{mlp}) * L \\
    &= hv + (4h^2 + 3hh_{ff})L \\
    &= hv + 16h^2L
\end{aligned}
$$

The total memory consumption of the model is:

$$
M_{total} = M_{activation} + M_{static}
$$

## FLOPs Calculation

model FLOPs per iteration:

$$
48sbh^2L(\frac{e_{top-k} + e_i - 1}{e_i} + \frac{1}{2} + \frac{s}{4h} + \frac{v}{8hL})
$$

For the explanation of this formula, the calculation Flops of each transformer layer in GPT model is $$72sbh^2$$, $$48sbh^2$$ for MLP and $$24sbh^2 + 12s^2bh$$ for attention.

For the MoE model, the calculation in the MLP part changes, where the computing Flops of MoE layer become $$e_{top-k}$$ times, and the formula becomes

$$
\begin{aligned}
C &= C_{MoE} \frac{L}{e_i} * e_{top-k} + C_{dense\_mlp} \frac{L(e_i - 1)}{e_i} + C_{attention}L + C_{embedding} \\
    &= 48sbh^2L \frac{e_{top-k}}{e_i} + 48sbh^2L \frac{e_i-1}{e_i} + 24sbh^2L + 12s^2bhL + 6sbhv \\
    &= 48sbh^2L(\frac{e_{top-k} + e_i - 1}{e_i} + \frac{1}{2} + \frac{s}{4h} + \frac{v}{8hL}) \\
\end{aligned}
$$

# References

- [Korthikanti, V.A., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M. and Catanzaro, B., 2023. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5.](https://proceedings.mlsys.org/paper_files/paper/2023/hash/e851ca7b43815718fbbac8afb2246bf8-Abstract-mlsys2023.html)
