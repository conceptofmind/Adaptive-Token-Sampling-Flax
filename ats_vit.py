import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange, repeat
from typing import Callable, Any
import numpy as np

import tensorflow as tf

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# adaptive token sampling functions and classes

def log(t, eps = 1e-6):
    return jnp.log(t + eps)

def sample_gumbel(shape, dtype, eps = 1e-6):
    u = jax.random.uniform(key, shape, dtype = dtype)
    return -log(-log(u, eps), eps)

def torch_gather(x, indices, gather_axis):
    # if pytorch gather indices are
    # [[[0, 10, 20], [0, 10, 20], [0, 10, 20]],
    #  [[0, 10, 20], [0, 10, 20], [0, 10, 20]]]
    # tf nd_gather needs to be
    # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20], [0,2,0], [0,2,10], [0,2,20],
    #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20], [1,2,0], [1,2,10], [1,2,20]]

    indices = jnp.int64(indices)
    # create a tensor containing indices of each element
    all_indices = jnp.where(np.ndarray.fill(indices.shape))
    gather_locations = jnp.reshape(indices, [indices.shape.num_elements()])

    # splice in our pytorch style index at the correct axis
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(gather_locations)
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = jnp.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = jnp.reshape(gathered, indices.shape)
    return reshaped

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = jnp.tile(indices, reps=[1] * len(indices_shape) + [*value_dims])
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    dim += value_expand_len

    values = torch_gather(values, indices, dim)
    return values

def jax_unstack(x, axis = 0):
    return jnp.moveaxis(x, axis, 0)

class AdaptiveTokenSampling(nn.Module):
    output_num_tokens: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, attn, value=None, mask=None):

        eps = self.eps
        output_num_tokens = self.output_num_tokens

        heads, output_num_tokens, eps, dtype = attn.shape[1], self.output_num_tokens, self.eps, attn.dtype

        # first get the attention values for CLS token to all other tokens
        cls_attn = attn[..., 0, 1:]

        # calculate the norms of the values, for weighting the scores, as described in the paper
        value_norms = jnp.linalg.norm(value[..., 1:, :], axis=-1)

        # weigh the attention scores by the norm of the values, sum across all heads
        cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms)

        # normalize to 1
        normed_cls_attn = cls_attn / (jnp.sum(cls_attn, axis = -1, keepdims = True) + eps)

        # instead of using inverse transform sampling, going to invert the softmax and use gumbel-max sampling instead
        pseudo_logits = log(normed_cls_attn)

        # mask out pseudo logits for gumbel-max sampling
        mask_without_cls = mask[:, 1:]
        mask_value = -np.finfo(attn).max / 2
        pseudo_logits = jnp.where(~mask_without_cls, mask_value, pseudo_logits)

        # expand k times, k being the adaptive sampling number
        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k = output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, dtype = dtype)

        # gumble-max and add one to reserve 0 for padding / mask
        sampled_token_ids = jnp.argmax(pseudo_logits, axis=-1) + 1

        # calculate unique using torch.unique and then pad the sequence from the right
        unique_sampled_token_ids_list = []
        unstack = jax_unstack(sampled_token_ids, axis = 0)
        for t in unstack:
            t = jnp.int32(t)
            t = jnp.unique(t)
            x = jnp.sort(t)
            unique_sampled_token_ids_list.append(x)


        unique_sampled_token_ids = tf.keras.preprocessing.sequence.pad_sequences(unique_sampled_token_ids_list)

        # calculate the new mask, based on the padding
        new_mask = unique_sampled_token_ids != 0

        # CLS token never gets masked out (gets a value of True)
        new_mask = jnp.pad(new_mask, pad_width=[[0, 0], [1, 0]], constant_values=True)

        # prepend a 0 token id to keep the CLS attention scores
        unique_sampled_token_ids = jnp.pad(unique_sampled_token_ids, pad_width=[[0, 0], [1, 0]])
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h=heads)

        # gather the new attention scores
        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim=2)

        # return the sampled attention scores, new mask (denoting padding), as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids


class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features = self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Dense(features = self.dim)(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        return x

class Attention(nn.Module):
    dim: int 
    heads: int = 8 
    dim_head: int = 64 
    dropout: float = 0.0 
    output_num_tokens: Any = None

    @nn.compact
    def __call__(self, x, mask = None):
        output_num_tokens = self.output_num_tokens
        ats = AdaptiveTokenSampling(output_num_tokens) if exists(output_num_tokens) else None
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head ** -0.5

        num_tokens = x.shape[1]

        to_qkv = nn.Dense(features = inner_dim * 3, use_bias = False)(x)
        qkv = jnp.split(to_qkv, 3, axis = -1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(mask):
            mask_f = mask #tf.cast(mask, tf.float32)
            dots_mask = rearrange(mask_f, 'b i -> b 1 i 1') * rearrange(mask_f, 'b j -> b 1 1 j')
            dots_mask = dots_mask #tf.cast(dots_mask, tf.bool)
            mask_value = -jnp.finfo(dots).max
            dots = jnp.where(~dots_mask, mask_value, dots)

        attn = nn.softmax(dots, axis = -1)

        sampled_token_ids = None

        # if adaptive token sampling is enabled
        # and number of tokens is greater than the number of output tokens
        if exists(output_num_tokens) and (num_tokens - 1) > output_num_tokens:
            attn, mask, sampled_token_ids = ats(attn, v, mask = mask)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = nn.Dense(features = self.dim)(out)
        out = nn.Dropout(rate = self.dropout)(out, deterministic = False)

        return out, mask, sampled_token_ids

class Transformer(nn.Module): 
    dim: int 
    depth: int 
    max_tokens_per_depth: tuple 
    heads: int 
    dim_head: int 
    mlp_dim: int 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        assert len(self.max_tokens_per_depth) == self.depth, 'max_tokens_per_depth must be a tuple of length that is equal to the depth of the transformer'
        assert sorted(self.max_tokens_per_depth, reverse=True) == list(self.max_tokens_per_depth), 'max_tokens_per_depth must be in decreasing order'
        assert min(self.max_tokens_per_depth) > 0, 'max_tokens_per_depth must have at least 1 token at any layer'

        layers = []

        for _, output_num_tokens in zip(range(self.depth), self.max_tokens_per_depth):
            layers.append([
                PreNorm(Attention(self.dim, output_num_tokens = output_num_tokens, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout)),
                PreNorm(FeedForward(self.dim, self.mlp_dim, dropout = self.dropout))
            ])

        b, n = x.shape[:2]

        # use mask to keep track of the paddings when sampling tokens
        # as the duplicates (when sampling) are just removed, as mentioned in the paper
        mask = jnp.ones([b, n], dtype = np.bool)

        token_ids = jnp.arange(n)
        token_ids = repeat(token_ids, 'n -> b n', b = b)

        for attn, ff in layers:
            attn_out, mask, sampled_token_ids = attn(x, mask=mask)

            # when token sampling, one needs to then gather the residual tokens with the sampled token ids
            if exists(sampled_token_ids):
                x = batched_index_select(x, sampled_token_ids, dim=1)
                token_ids = batched_index_select(token_ids, sampled_token_ids, dim=1)

            x = x + attn_out

            x = ff(x) + x

        return x, token_ids

class ViT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    max_tokens_per_depth: tuple
    heads: int
    mlp_dim: int
    dim_head = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0

    @nn.compact
    def __call__(self, img, return_sampled_token_ids=False, training=True, **kwargs):

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 
        assert image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        x = rearrange(img, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        x = nn.Dense(features = self.dim)(x)

        b, n, _ = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += pos_embedding[:, :(n + 1)]

        x = nn.Dropout(rate = self.emb_dropout)(x, deterministic = False)

        x, token_ids = Transformer(self.dim, self.depth, self.max_tokens_per_depth, self.heads, self.dim_head, self.mlp_dim, self.dropout)(x)

        mlp_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(features = self.num_classes)
        ])

        logits = mlp_head(x[:, 0])

        if return_sampled_token_ids:
            # remove CLS token and decrement by 1 to make -1 the padding
            token_ids = token_ids[:, 1:] - 1
            return logits, token_ids

        return logits

if __name__ == '__main__':

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