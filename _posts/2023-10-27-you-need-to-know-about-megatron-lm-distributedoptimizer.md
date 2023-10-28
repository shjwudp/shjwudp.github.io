---
layout: post
title: 'A few things you need to know about Megatron-LM DistributedOptimizer'
date: 2023-10-27 16:00:00 +0800
tags: Megatron-LM
categories: note
giscus_comments: true
related_posts: false
---

The Megatron-LM distributed optimizer is a code with a complex interface but good performance. There are some stories hidden under the complex interface, as well as the elegant theoretical design of ZeRO-1. This note is to record the stories behind these complex interfaces.

## I. Dauntingly complex interface

DistributedOptimizer has a dauntingly complex interface. I have organized them as follows

Basic optimizer functions:
- get_parameters
- zero_grad
- state_dict
- load_state_dict
- step

TP/SP+PP:
- get_main_grads_for_grad_norm
- get_model_parallel_group
- gather_model_params
- allreduce_word_embedding_grads
- allreduce_position_embedding_grads
- allreduce_embedding_grads
- allreduce_layernorm_grads
- allreduce_router_grads

ZeRO:
- build_model_gbuf_param_range_map
- build_model_gbuf_range
- build_model_gbuf_range_map
- build_model_param_gbuf_map
- build_optimizer_group_ranges
- build_model_and_main_param_groups
- get_model_param_range_map
- save_parameter_state
- load_parameter_state
- get_model_buffer_dp_views
- get_model_grad_buffer_dp_views
- get_model_param_buffer_dp_views
- reduce_model_grads

Mixprecision:
- clip_grad_norm
- get_loss_scale
- scale_loss
- reload_model_params

helpers:
- count_zeros

## II. How is ZeRO-1 implemented?

It is strongly recommended to read [the documentation in the Megatron-LM code repository](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/distrib_optimizer.md) first. You will find that the theory is very elegant.

![data flow](/assets/posts/2023-10-27-you-need-to-know-about-megatron-lm-distributedoptimizer/data-flow.png){: width="100%" }

We can see an essential design. The gradient and parameters share a memory space. This buffer is named "gbuf" in the code and has four functions with the word "gbuf". `build_model_gbuf_param_range_map`, `build_model_gbuf_range`, `build_model_gbuf_range_map`, and `build_model_param_gbuf_map`.

![sharding scheme](/assets/posts/2023-10-27-you-need-to-know-about-megatron-lm-distributedoptimizer/sharding-scheme.png){: width="100%" }

In the function `build_model_gbuf_range_map`, the global and local indexes of all parameters/gradients are constructed, as shown in Figure above. These core operations are implemented in `build_model_gbuf_param_range_map`, and the remaining two functions are help functions.

There are also some functions associated with ZeRO-1 implementation. List and briefly explain their functions below:

- `build_optimizer_group_ranges` - Build mapping of model parameter and group index for all parameters.
- `build_model_and_main_param_groups` - Cut out the parameters retained locally based on the index. There will be an operation of making fp32 copy for the fp16/bf16 parameters.
- `get_model_param_range_map` - Given a model param, get the index sub-range of the param that this data-parallel rank owns. 
- `save_parameter_state` - Copy parameters and optimizer shards and gather on DP rank 0 and save to disk.
- `load_parameter_state` - Reverse operation of `save_parameter_state`.
- `get_model_buffer_dp_views` - A gbuf reader that indexes separately by model id and data type.
- `get_model_grad_buffer_dp_views` - An application of `get_model_buffer_dp_views`, used to read gradient buffer.
- `get_model_param_buffer_dp_views` - An application of `get_model_buffer_dp_views`, used to read parameter buffer.

## III. The story about the gradient of MoE router that needs to be processed independently

# References

- [Megatron-LM Distributed Optimizer Document](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/distrib_optimizer.md)
