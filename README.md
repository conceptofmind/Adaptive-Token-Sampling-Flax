## Adaptive-Token-Sampling-Flax

<img src="ats.png" width="400px"></img>

"While state-of-the-art vision transformer models achieve promising results for image classification, they are computationally expensive and require many GFLOPs. Although the GFLOPs of a vision transformer can be decreased by reducing the number of tokens in the network, there is no setting that is optimal for all input images. In this work, we, therefore, introduce a differentiable parameter-free Adaptive Token Sampling (ATS) module, which can be plugged into any existing vision transformer architecture. ATS empowers vision transformers by scoring and adaptively sampling significant tokens. As a result, the number of tokens is not constant anymore and varies for each input image. By integrating ATS as an additional layer within current transformer blocks, we can convert them into much more efficient vision transformers with an adaptive number of tokens. Since ATS is a parameter-free module, it can be added to off-the-shelf pre-trained vision transformers as a plug-and-play module, thus reducing their GFLOPs without any additional training. Moreover, due to its differentiable design, one can also train a vision transformer equipped with ATS. We evaluate our module on both image and video classification tasks by adding it to multiple SOTA vision transformers. Our proposed module improves the SOTA by reducing the computational cost (GFLOPs) by 2x while preserving the accuracy of SOTA models on ImageNet, Kinetics-400, and Kinetics-600 datasets." - Mohsen Fayyaz, Soroush Abbasi Koohpayegani, Farnoush Rezaei Jafari, Sunando Sengupta, Hamid Reza Vaezi Joze, Eric Sommerlade, Hamed Pirsiavash, Juergen Gall

## Acknowledgement:
This repository has been created in collaboration with [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

### Research Paper:
- https://arxiv.org/abs/2111.15667

### Usage:
```python
v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    max_tokens_per_depth = (256, 128, 64, 32, 16, 8), # a tuple that denotes the maximum number of tokens that any given layer should have. if the layer has greater than this amount, it will undergo adaptive token sampling
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (4, 256, 256, 3))

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2), 
            'emb_dropout': jax.random.PRNGKey(3)}

params = v.init(init_rngs, img)
output = v.apply(params, img, rngs=init_rngs)
print(output.shape)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}")
```

### Citations:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2111.15667,
  doi = {10.48550/ARXIV.2111.15667},
  
  url = {https://arxiv.org/abs/2111.15667},
  
  author = {Fayyaz, Mohsen and Koohpayegani, Soroush Abbasi and Jafari, Farnoush Rezaei and Sengupta, Sunando and Joze, Hamid Reza Vaezi and Sommerlade, Eric and Pirsiavash, Hamed and Gall, Juergen},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Adaptive Inverse Transform Sampling For Efficient Vision Transformers},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```