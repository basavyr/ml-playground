import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Final, Union, Tuple, Literal, Optional, Type, List, Dict, Callable
from functools import partial
import math


from timm.layers.attention import Attention
from timm.layers.diff_attention import DiffAttention
from timm.layers.patch_embed import PatchEmbed
from timm.layers.attention_pool import AttentionPoolLatent
from timm.layers.attention_pool import AttentionPoolLatent
from timm.layers.mlp import Mlp
from timm.layers.norm import LayerNorm
from timm.layers.drop import calculate_drop_path_rates
from timm.layers.patch_dropout import PatchDropout
from timm.layers.weight_init import trunc_normal_
from timm.layers.typing import LayerType
from timm.layers.create_act import get_act_layer
from timm.layers.create_norm import get_norm_layer
from timm.layers.layer_scale import LayerScale
from timm.layers.drop import DropPath


ATTN_LAYERS = {
    '': Attention,
    'attn': Attention,
    'diff': DiffAttention,
}


def _create_attn(
        attn_layer: LayerType,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Optional[Type[nn.Module]] = None,
        depth: int = 0,
        **kwargs,
) -> nn.Module:
    if isinstance(attn_layer, str):
        attn_layer = ATTN_LAYERS.get(attn_layer, None)
        assert attn_layer is not None, f'Unknown attn_layer: {attn_layer}'

    # Only pass depth to attention layers that use it
    if issubclass(attn_layer, DiffAttention):
        kwargs['depth'] = depth

    return attn_layer(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        scale_norm=scale_norm,
        proj_bias=proj_bias,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        norm_layer=norm_layer,
        **kwargs,
    )


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            attn_layer: LayerType = Attention,
            depth: int = 0,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
            attn_layer: Attention layer type (class or string).
            depth: Block index, passed to attention layer for depth-dependent init.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        self.norm1 = norm_layer(dim, **dd)
        self.attn = _create_attn(
            attn_layer,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            depth=depth,
            **dd,
        )
        self.ls1 = LayerScale(dim, init_values=init_values,
                              **dd) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            **dd,
        )
        self.ls2 = LayerScale(dim, init_values=init_values,
                              **dd) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # dis
        x = x + \
            self.drop_path1(
                self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x


def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module: nn.Module, name: str = '', needs_reset: bool = True) -> None:
    """ViT weight initialization, original timm impl (for reproducibility).

    Args:
        module: Module to initialize.
        name: Module name for context.
        needs_reset: If True, call reset_parameters() on modules that have it.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
    elif needs_reset and hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0, needs_reset: bool = True) -> Callable:
    return partial(init_weights_vit_timm, needs_reset=needs_reset)


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_channels: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax',
                                 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            pool_include_prefix: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'reset',
                                 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            embed_norm_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            attn_layer: LayerType = Attention,
            device=None,
            dtype=None,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_channels: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in (
            'avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.global_pool = global_pool
        # for consistency with other models
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,  # we need to keep `in_chans` as argument here
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
            **dd,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(
            self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.empty(
            1, 1, embed_dim, **dd)) if class_token else None
        self.reg_token = nn.Parameter(torch.empty(
            1, reg_tokens, embed_dim, **dd)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(
                torch.empty(1, embed_len, embed_dim, **dd))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(
            embed_dim, **dd) if pre_norm else nn.Identity()

        # stochastic depth decay rule
        dpr = calculate_drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                scale_attn_norm=scale_attn_norm,
                scale_mlp_norm=scale_mlp_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                attn_layer=attn_layer,
                depth=i,
                **dd,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(
            embed_dim, **dd) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(
            embed_dim, **dd) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes,
                              **dd) if num_classes > 0 else nn.Identity()

        self.weight_init_mode = 'reset' if weight_init == 'skip' else weight_init
        self.fix_init = fix_init
        if weight_init != 'skip' and not next(self.parameters()).is_meta:
            self.init_weights(needs_reset=False)

    def fix_init_weight(self) -> None:
        """Apply weight initialization fix (scaling w/ layer index)."""
        with torch.no_grad():
            for layer_id, layer in enumerate(self.blocks):
                scale = math.sqrt(2.0 * (layer_id + 1))
                layer.attn.proj.weight.div_(scale)
                layer.mlp.fc2.weight.div_(scale)

    def init_weights(self, mode: str = '', needs_reset: bool = True) -> None:
        """Initialize model weights.

        Args:
            mode: Weight initialization mode ('jax', 'jax_nlhb', 'moco', or '').
            needs_reset: If True, call reset_parameters() on modules that have it.
                Set to False when modules have already self-initialized in __init__.
        """
        mode = mode or self.weight_init_mode
        assert mode in ('jax', 'jax_nlhb', 'moco', 'reset', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)

        named_apply(get_init_weights_vit(
            mode, head_bias, needs_reset=needs_reset), self)

        if self.fix_init:
            self.fix_init_weight()

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for a single module (compatibility method)."""
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update the input image resolution and patch size.

        Args:
            img_size: New input resolution, if None current resolution is used.
            patch_size: New patch size, if None existing patch size is used.
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(
            img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=self.patch_embed.grid_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    verbose=True,
                ))

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            # If mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        """Apply pooling to feature tokens.

        Args:
            x: Feature tensor.
            pool_type: Pooling type override.

        Returns:
            Pooled features.
        """
        if self.attn_pool is not None:
            if not self.pool_include_prefix:
                x = x[:, self.num_prefix_tokens:]
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix
        )
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier.

        Returns:
            Output tensor.
        """
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.forward_head(x)
        return x
