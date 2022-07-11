# Scalable-ViT-flax

"The vanilla self-attention mechanism inherently relies on pre-defined and steadfast computational dimensions. Such inflexibility restricts it from possessing context-oriented generalization that can bring more contextual cues and global representations. To mitigate this issue, we propose a Scalable Self-Attention (SSA) mechanism that leverages two scaling factors to release dimensions of query, key, and value matrix while unbinding them with the input. This scalability fetches context-oriented generalization and enhances object sensitivity, which pushes the whole network into a more effective trade-off state between accuracy and cost. Furthermore, we propose an Interactive Window-based Self-Attention (IWSA), which establishes interaction between non-overlapping regions by re-merging independent value tokens and aggregating spatial information from adjacent windows. By stacking the SSA and IWSA alternately, the Scalable Vision Transformer (ScalableViT) achieves state-of-the-art performance in general-purpose vision tasks. For example, ScalableViT-S outperforms Twins-SVT-S by 1.4% and Swin-T by 1.8% on ImageNet-1K classification." - Rui Yang, Hailong Ma, Jie Wu, Yansong Tang, Xuefeng Xiao, Min Zheng, Xiu Li

## Acknowledgement:
I have been greatly inspired by the work of [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

## Usage:
```python
import numpy

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (1, 256, 256, 3))

v = ScalableViT(
    num_classes = 1000,
    dim = 64,                               # starting model dimension. at every stage, dimension is doubled
    heads = (2, 4, 8, 16),                  # number of attention heads at each stage
    depth = (2, 2, 20, 2),                  # number of transformer blocks at each stage
    ssa_dim_key = (40, 40, 40, 32),         # the dimension of the attention keys (and queries) for SSA. in the paper, they represented this as a scale factor on the base dimension per key (ssa_dim_key / dim_key)
    reduction_factor = (8, 4, 2, 1),        # downsampling of the key / values in SSA. in the paper, this was represented as (reduction_factor ** -2)
    window_size = (64, 32, None, None),     # window size of the IWSA at each stage. None means no windowing needed
    dropout = 0.1,                          # attention and feedforward dropout
)

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2), 
            'emb_dropout': jax.random.PRNGKey(3)}

params = v.init(init_rngs, img)
output = v.apply(params, img, rngs=init_rngs)
print(output.shape)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}")
```

## Developer Updates
Developer updates can be found on: 
- https://twitter.com/EnricoShippole
- https://www.linkedin.com/in/enrico-shippole-495521b8/

## Citation:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2203.10790,
  doi = {10.48550/ARXIV.2203.10790},
  
  url = {https://arxiv.org/abs/2203.10790},
  
  author = {Yang, Rui and Ma, Hailong and Wu, Jie and Tang, Yansong and Xiao, Xuefeng and Zheng, Min and Li, Xiu},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {ScalableViT: Rethinking the Context-oriented Generalization of Vision Transformer},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```