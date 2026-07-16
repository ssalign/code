# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


from PIL import Image
def visualize_dift_feature_matching(
    target_image: Image.Image,
    reference_image: Image.Image,
    F_target: torch.Tensor,      # (N, C)
    F_reference: torch.Tensor,   # (N, C)
    target_token_idx: int,       # p_t
    save_path: str,
    point_color: str = "red",
    alpha: float = 0.55,
):
    
    """
    Visualize DIFT-style feature matching:
    Given a target token, show where it matches in the reference image.

    Args:
        target_image: PIL image (H,W)
        reference_image: PIL image (H,W)
        F_target: (N, C) target features
        F_reference: (N, C) reference features
        target_token_idx: index in [0, N)
        save_path: where to save PNG
        point_color: 'red' or 'green'
        alpha: heatmap overlay strength
    """
    import matplotlib.pyplot as plt

    assert F_target.shape == F_reference.shape
    N, C = F_target.shape
    H, W = target_image.size[1], target_image.size[0]
    assert N == H * W, "Token count must match H*W"

    device = F_target.device

    # 1. Normalize features (DIFT)
    Ft = torch.nn.functional.normalize(F_target, dim=-1)
    Fr = torch.nn.functional.normalize(F_reference, dim=-1)

    # 2. Pick query feature
    f_q = Ft[target_token_idx]                # (C,)

    # 3. Cosine similarity (DIFT matching)
    sim = torch.matmul(Fr, f_q)               # (N,)
    sim_map = sim.view(H, W)

    # 4. Normalize for visualization
    sim_map_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-6)
    sim_map_np = sim_map_norm.detach().cpu().numpy()

    # 5. Best match location
    best_idx = sim.argmax().item()
    y = best_idx // W
    x = best_idx % W

    # 6. Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # ---- Target image with query point ----
    ax[0].imshow(target_image)
    ty = target_token_idx // W
    tx = target_token_idx % W
    ax[0].scatter(tx, ty, c=point_color, s=80)
    ax[0].set_title("Target (Query Point)")
    ax[0].axis("off")

    # ---- Reference image with heatmap + match ----
    ax[1].imshow(reference_image)
    ax[1].imshow(sim_map_np, cmap="magma", alpha=alpha)
    ax[1].scatter(x, y, c=point_color, s=80)
    ax[1].set_title("Reference (DIFT Matching)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

import math
class FluxAttnProcessorF_E4E:

    _attention_backend = None
    _parallel_config   = None

    def __init__(self, default_mode="KV", topk=1, temperature=0.05):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")
        # Interpolation defaults.
        self.default_mode   = default_mode
        self.default_topk   = int(topk)
        self.default_temp   = float(temperature)
        # Attention mask statistics.
        self._cross_attn_sum = None
        self._cross_attn_count = 0
        # Per-layer mask statistics.
        self._layer_cross_attn_sum = {}
        self._layer_cross_attn_count = {}

    @staticmethod
    def _slerp_batch(x: torch.Tensor, y: torch.Tensor, t: float = 0.5, eps: float = 1e-12, log_slerp: bool = False):
        """
        Token-wise SLERP over channels.
        Direction follows the sphere; magnitude is linear or logarithmic.
        x, y: matching (..., C) tensors.
        """
        # Direction.
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        y_norm = y.norm(p=2, dim=-1, keepdim=True)
        x_unit = x / (x_norm + eps)
        y_unit = y / (y_norm + eps)

        dot = (x_unit * y_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        # Avoid division by zero.
        sin_safe = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta)
        s1 = torch.sin((1.0 - t) * theta) / sin_safe
        s2 = torch.sin(t * theta) / sin_safe
        dir_interp = s1 * x_unit + s2 * y_unit

        # Magnitude.
        if not log_slerp:
            mag = (1.0 - t) * x_norm + t * y_norm
        else:
            mag = (x_norm ** (1.0 - t)) * (y_norm ** t)

        out = mag * dir_interp
        # Degenerate cases.
        out = torch.where((theta.abs() < 1e-7) | (y_norm < eps), x, out)
        return out

    @staticmethod
    def _slerp_fixed_length_batch(x: torch.Tensor, y: torch.Tensor, t: float = 0.5, eps: float = 1e-12):
        """SLERP with the output norm fixed to the norm of x."""
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        y_norm = y.norm(p=2, dim=-1, keepdim=True)
        y_unit = y / (y_norm + eps)
        y_same_len = y_unit * x_norm

        # Spherical interpolation with norm kept at |x|.
        dot = (x * y_same_len).sum(dim=-1, keepdim=True) / (x_norm.pow(2) + eps)
        dot = dot.clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        sin_safe = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta)

        s1 = torch.sin((1.0 - t) * theta) / sin_safe
        s2 = torch.sin(t * theta) / sin_safe
        out = s1 * x + s2 * y_same_len
        out = torch.where((theta.abs() < 1e-7) | (y_norm < eps), x, out)
        return out

    @staticmethod
    def _l2norm(x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    @staticmethod
    def _infer_hw_from_len(n: int):
        h = int(math.sqrt(n))
        if h * h != n:
            # Fall back to an approximate square grid.
            w = n // h
            if h * w != n:
                w = h
        else:
            w = h
        return h, w

    @staticmethod
    def _down_mask_to_grid(mask: torch.Tensor, H: int, W: int, dev, dtype):
        """
        mask: (H0, W0) or (1,1,H0,W0) or (H0,W0,1)
        return: (H, W, 1) float in {0,1}
        """
        if mask.dim() == 2:
            m = mask[None, None]  # (1,1,H0,W0)
        elif mask.dim() == 3:      # (H0,W0,1)
            m = mask.permute(2, 0, 1)[None]
        elif mask.dim() == 4:      # (1,1,H0,W0) or (B,1,H0,W0), use the first item
            m = mask[:1]
        else:
            raise ValueError("Unsupported mask shape")
        m = m.to(device=dev, dtype=dtype)
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        m = (m > 0.5).float()
        return m[0,0][..., None]   # (H,W,1)
    
    def _dift_match(
        self,
        F_ref: torch.Tensor,          # (1, N, C)
        F_out: torch.Tensor,          # (B-1, N, C)
        *,
        topk: int,
        temperature: float,
        mask_pair=None,               # Optional (struct_mask, style_mask)
        return_indices: str = None,     # Optional index return: 'ref' | 'out' | None
    ):
        """
        Return F_ref_perm (B-1, N, C), matched or softly reconstructed from reference features.
        """
        Bm1, N, C = F_out.shape
        # L2 normalization.
        F_ref_n = self._l2norm(F_ref, dim=-1)      # (1,N,C)
        F_out_n = self._l2norm(F_out, dim=-1)      # (B-1,N,C)

        # Optional masks remove irrelevant areas from F_out and F_ref.
        # mask_pair comes from joint kwargs and is resized to HxW internally.
        if mask_pair is not None:
            struct_mask, style_mask = mask_pair   # (H0,W0)..
            H, W = self._infer_hw_from_len(N)
            m_struct = self._down_mask_to_grid(struct_mask, H, W, F_out.device, F_out.dtype).view(-1, 1) # (N,1)
            m_style  = self._down_mask_to_grid(style_mask,  H, W, F_ref.device, F_ref.dtype).view(-1, 1) # (N,1)
            # Apply masks per token.
            F_out_n = F_out_n * m_struct.view(1, N, 1)
            F_ref_n = F_ref_n * m_style.view(1, N, 1)

        # Similarity matrix (B-1, N, N).
        S = torch.matmul(F_out_n, F_ref_n.transpose(1, 2))   # cosine

        if topk <= 1:
            idx = S.argmax(dim=-1)                           # (B-1, N)
            # Gather reference features.
            F_ref_exp = F_ref.expand(Bm1, -1, -1)            # (B-1,N,C)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)    # (B-1,N,C idx)
            F_ref_perm = torch.gather(F_ref_exp, 1, idx_exp) # (B-1,N,C)
        else:
            k = min(topk, S.shape[-1])
            val, idx_topk = torch.topk(S / max(temperature, 1e-6), k=k, dim=-1)  # (B-1,N,k)
            w = F.softmax(val, dim=-1)                                           # (B-1,N,k)

            F_ref_exp  = F_ref.expand(Bm1, -1, -1)                               # (B-1,N,C)
            F_ref_bank = F_ref_exp.unsqueeze(1).expand(-1, N, -1, -1)           # (B-1,N,N,C)
            idx_exp    = idx_topk.unsqueeze(-1).expand(-1, -1, -1, C)           # (B-1,N,k,C)
            F_k        = torch.gather(F_ref_bank, 2, idx_exp)                    # (B-1,N,k,C)
            F_ref_perm = (w.unsqueeze(-1) * F_k).sum(dim=2)                      # (B-1,N,C)

        idx_ret = None
        if return_indices is not None:
            if return_indices == "out":
                # Reference index for each output position.
                idx_ret = S.argmax(dim=-1)               # (B-1, N_out)
            elif return_indices == "ref":
                # Best output row for each reference position.
                idx_ret = S.argmax(dim=1)                # (B-1, N_ref)
            else:
                raise ValueError("return_indices must be in {'ref','out',None}")

        if idx_ret is not None:
            return F_ref_perm, idx_ret
        else:
            return F_ref_perm


    def _attn_mean_and_accumulate(
        self,
        query_3d: torch.Tensor,         # (B, L, C)
        key_3d: torch.Tensor,           # (B, L, C)
        *,
        heads: int,
        enc_len: int,                   # Text segment length
        tgt_start: int,
        tgt_len: int,
        txt_token_ids: Optional[torch.Tensor],
        head_chunk: int,
        layer_index: int,
    ):
        """
        Return:
            attn_map_mean: (B, Nt, Ne), averaged over heads.
            tok_mask_mean: (B, Nt), averaged over selected text tokens.
        Also accumulates attn_map_mean globally and per layer.
        """
        if enc_len <= 0 or tgt_len <= 0:
            return None, None

        B, Lq, Cq = query_3d.shape
        Bk, Lk, Ck = key_3d.shape
        assert B == Bk and Cq == Ck and Lk >= enc_len

        Dh = Cq // heads
        assert Cq % heads == 0

        # Convert to (B,H,L,Dh).
        q_all = query_3d.view(B, Lq, heads, Dh).permute(0, 2, 1, 3)   # (B,H,L,Dh)
        k_all = key_3d.view(B, Lk, heads, Dh).permute(0, 2, 1, 3)     # (B,H,L,Dh)

        q_img = q_all[:, :, tgt_start:tgt_start + tgt_len, :]         # (B,H,Nt,Dh)
        k_txt = k_all[:, :, :enc_len, :]                              # (B,H,Ne,Dh)

        H = q_img.shape[1]
        head_chunk = min(int(head_chunk), int(H))
        scale = 1.0 / math.sqrt(Dh)

        # Accumulate attention probabilities before head averaging.
        attn_sum = torch.zeros(B, H, tgt_len, enc_len, device=q_img.device, dtype=q_img.dtype)

        for j in range(0, H, head_chunk):
            q_blk = q_img[:, j:j+head_chunk]                                  # (B,h,Nt,Dh)
            k_blk = k_txt[:, j:j+head_chunk]                                  # (B,h,Ne,Dh)
            scores = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * scale     # (B,h,Nt,Ne)
            probs  = torch.softmax(scores, dim=-1)
            attn_sum[:, j:j+head_chunk] = probs

        # Average over heads -> (B,Nt,Ne).
        attn_map_mean = attn_sum.mean(dim=1)

        # Global and per-layer accumulation.
        # Global.
        self._cross_attn_sum = (attn_map_mean if self._cross_attn_sum is None
                                else self._cross_attn_sum + attn_map_mean)
        self._cross_attn_count += 1

        # Per layer.
        if layer_index not in self._layer_cross_attn_sum:
            self._layer_cross_attn_sum[layer_index] = attn_map_mean.clone()
            self._layer_cross_attn_count[layer_index] = 1
        else:
            self._layer_cross_attn_sum[layer_index] += attn_map_mean
            self._layer_cross_attn_count[layer_index] += 1

        # Select text tokens and reduce to a token-level mask.
        if txt_token_ids is not None and len(txt_token_ids) > 0:
            idx = torch.as_tensor(txt_token_ids, device=attn_map_mean.device, dtype=torch.long)
            idx = idx.clamp_min(0).clamp_max(enc_len - 1)
            sel = attn_map_mean.index_select(-1, idx)   # (B,Nt,|idx|)
        else:
            sel = attn_map_mean                          # (B,Nt,Ne)

        tok_mask_mean = sel.mean(dim=-1)                 # (B,Nt)

        # Normalize to 0..1 per batch.
        tmin = tok_mask_mean.amin(dim=1, keepdim=True)
        tmax = tok_mask_mean.amax(dim=1, keepdim=True)
        tok_mask_mean = (tok_mask_mean - tmin) / (tmax - tmin + 1e-6)

        return attn_map_mean, tok_mask_mean

    def _broadcast_token_mask_for_targets(
        self,
        token_mask: torch.Tensor,   # (B, Nt)
        take_from_batch: int,       # Template mask row
        repeat_for: int,            # Number of target rows
    ) -> torch.Tensor:
        """
        Take one token-mask row and expand it to (repeat_for, Nt, 1).
        """
        m1 = token_mask[take_from_batch:take_from_batch+1]        # (1,Nt)
        return m1.unsqueeze(-1).expand(repeat_for, -1, 1).contiguous()

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,                      # (B, N_all, C)
        encoder_hidden_states: torch.Tensor = None,       # (B, N_enc, C_enc) or None
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        *,
        # --- Token lengths ---
        enc_len: int = None,
        kv_tgt_len: int = None,
        kv_ctx_len: int = None,
        # --- Appearance and structure controls ---
        kv_alpha: float = 0,                            # Current-step KV strength
        e4e_mode: str = None,                             # "F" | "KV"
        e4e_topk: int = None,
        e4e_temperature: float = None,
        e4e_masks = None,                                 # (struct_mask, style_mask) or None
        e4e_swap_k: bool = True,                          # Also swap K in KV mode
        # --- Interpolation ---
        e4e_interp: str = "slerp",                        # "lerp" | "slerp" | "log_slerp"
        # --- Structure alignment ---
        e4e_struct_align: bool = False,          # Enable target-context structure alignment
        q_beta: float = 0,             # Cost blend weight: cosine vs. index prior
        e4e_step: int = None,                    # Current step from the pipeline
        e4e_align_start: int = 0,                # Used only when e4e_step is set
        e4e_align_end: int = 0,                  # Used only when e4e_step is set
        # --- mask ---
        e4e_mask_from: str = "attn",        # "attn" | "none"
        e4e_mask_txt_ids: torch.Tensor = None,              
        e4e_mask_threshold: float = 0.10,
        e4e_mask_binary: bool = True
    ):
        B, N_all, C = hidden_states.shape

        # Use joint kwargs or fall back to defaults.
        mode = (e4e_mode or self.default_mode).upper()
        topk = int(self.default_topk if e4e_topk is None else e4e_topk)
        temperature = float(self.default_temp if e4e_temperature is None else e4e_temperature)

        # Segment boundaries; dual-stream Kontext hidden states omit text.
        is_dual = (encoder_hidden_states is not None) and (attn.added_kv_proj_dim is not None)
        if is_dual:
            n_enc = 0
            s_tgt, n_tgt = 0, int(kv_tgt_len or 0)
            s_ctx, n_ctx = s_tgt + n_tgt, int(kv_ctx_len or 0)
        else:
            n_enc = (encoder_hidden_states.shape[1]
                     if (encoder_hidden_states is not None and attn.added_kv_proj_dim is not None)
                     else int(enc_len or 0))
            s_tgt = n_enc
            n_tgt = int(kv_tgt_len or 0)
            s_ctx = s_tgt + n_tgt
            n_ctx = int(kv_ctx_len or 0)

        # Enable alignment injection only when all required inputs exist.
        do_align = (mode in ("F","KV")) and (n_tgt > 0) and (B >= 2) and (kv_alpha > 0.0)

        # on-F: align features before projection.
        if do_align and mode == "F" and not is_dual:
            F_all   = hidden_states
            F_tgt   = F_all[:, s_tgt:s_tgt+n_tgt, :]     # (B, N_tgt, C)
            F_ref   = F_tgt[:1]                          # (1, N, C)
            F_out   = F_tgt[1:]                          # (B-1, N, C)
            
            F_ref_perm = self._dift_match(F_ref, F_out, topk=topk, temperature=temperature, mask_pair=e4e_masks)
            F_out_new  = (1.0 - kv_alpha) * F_out + kv_alpha * F_ref_perm
           
            # Write back only the target segment of output samples.
            hidden_states = torch.cat([
                F_all[:1, :, :],
                torch.cat([
                    F_all[1:, :s_tgt, :],
                    F_out_new,
                    F_all[1:, s_ctx:s_ctx+n_ctx, :],
                    F_all[1:, s_ctx+n_ctx:, :],
                ], dim=1)
            ], dim=0)
            
        # Standard Q/K/V projection.
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        # Compute the mask.
        mask_from_attn   = e4e_mask_from 
        txt_token_ids    = e4e_mask_txt_ids
        mask_threshold   = e4e_mask_threshold
        use_mask_binary  = e4e_mask_binary
        token_mask_targets = None
        if txt_token_ids is not None and (mask_from_attn == "attn") and (n_enc > 0) and (n_tgt > 0) and (query.shape[0] >= 2):
            # Compute and accumulate attention maps.
            attn_map_mean, m_tok = self._attn_mean_and_accumulate(
                query[1:], key[1:],
                heads=attn.heads,
                enc_len=n_enc,
                tgt_start=s_tgt, tgt_len=n_tgt,
                txt_token_ids=txt_token_ids,
                head_chunk=4,
                layer_index=getattr(attn, "layer_index", -1),
            )  # attn_map_mean: (B,Nt,Ne), m_tok: (B,Nt) in [0,1]

            # Optional thresholding.
            if m_tok is not None and use_mask_binary:
                m_tok = (m_tok > mask_threshold).to(query.dtype)
            
            if m_tok is not None:
                B_eff = m_tok.shape[0]          # = B-1
                # Content-anchor row in m_tok is 0 after dropping style batch[0].
                take_from = 0
                repeat_for = B_eff          # Match the K_out/V_out batch dimension.
                token_mask_targets = self._broadcast_token_mask_for_targets(
                    token_mask=m_tok, take_from_batch=take_from, repeat_for=repeat_for
                )   # -> (B-1, Nt, 1)
            # token_mask_targets = m_tok

        # Structural alignment between target and context.
        # Requires the switch and both target/context segments.
        do_struct_align = (
            e4e_struct_align and (kv_tgt_len or 0) > 0
        )

        if do_struct_align and e4e_step is not None and (e4e_align_start <= int(e4e_step) <= e4e_align_end):

            if query.shape[0] > 1:
                q_out = query[1:, s_tgt:s_tgt + n_tgt, :]
                q_con_ctx = query[1:, s_tgt + n_tgt:s_tgt + n_tgt + n_ctx, :]
                q_out_aligned = self._dift_match(
                    q_out, q_con_ctx, topk=topk, temperature=temperature, mask_pair=e4e_masks
                )
                query[1:, s_tgt:s_tgt + n_tgt, :] = (1.0 - q_beta) * q_out + q_beta * q_out_aligned
            else:
                q_out = query[:1, s_tgt:s_tgt + n_tgt, :]
                q_con_ctx = query[:1, s_tgt + n_tgt:s_tgt + n_tgt + n_ctx, :]
                q_out_aligned = self._dift_match(
                    q_out, q_con_ctx, topk=topk, temperature=temperature, mask_pair=e4e_masks
                )
                query[:1, s_tgt:s_tgt + n_tgt, :] = (1.0 - q_beta) * q_out + q_beta * q_out_aligned
                
        # on-KV: modify V, and optionally K, before attention.
        if do_align and mode == "KV" and not is_dual:
            # 1) Slice target K/V.
            K_all, V_all = key, value
            K_tgt = K_all[:, s_tgt:s_tgt+n_tgt, :]
            V_tgt = V_all[:, s_tgt:s_tgt+n_tgt, :]
            K_ref, K_out = K_tgt[:1], K_tgt[1:]
            V_ref, V_out = V_tgt[:1], V_tgt[1:]

            # 2) Match in F space and keep indices only.
            #    F_tgt is the target segment of hidden_states, shape (B, N, C).
            F_tgt = hidden_states[:, s_tgt:s_tgt+n_tgt, :]
            F_ref, F_out = F_tgt[:1], F_tgt[1:]

            # 2.1 Nearest-neighbor out->ref indices.
            _, idx_out2ref = self._dift_match(
                F_ref, F_out,
                topk=1, temperature=0.07,
                mask_pair=e4e_masks,
                return_indices='out'
            )  # (B-1, N)

            # Optional ref->out reorder path:
            # _, idx_ref2out = self._dift_match(F_ref, F_out, topk=1, temperature=0.07, return_indices='ref')  # (B-1, N)

            # 3) Reorder reference K/V to align with output positions.
            idxC = K_ref.shape[-1]
            K_ref_exp = K_ref.expand_as(K_out)  # (B-1, N, C)
            V_ref_exp = V_ref.expand_as(V_out)
            K_ref_perm = torch.gather(K_ref_exp, 1, idx_out2ref.unsqueeze(-1).expand(-1, -1, idxC))
            V_ref_perm = torch.gather(V_ref_exp, 1, idx_out2ref.unsqueeze(-1).expand(-1, -1, idxC))
            
            # 4) Interpolate K/V with the aligned reference tensors.
            if e4e_interp == "lerp":
                V_mix = (1.0 - kv_alpha) * V_out + kv_alpha * V_ref_perm
                if e4e_swap_k:
                    K_mix = (1.0 - kv_alpha) * K_out + kv_alpha * K_ref_perm
            else:
                V_mix = self._slerp_batch(V_out, V_ref_perm, t=float(kv_alpha), log_slerp=(e4e_interp == "log_slerp"))
                if e4e_swap_k:
                    K_mix = self._slerp_fixed_length_batch(K_out, K_ref_perm, t=float(kv_alpha))

            # Optional token-level mask.
            V_out_new = token_mask_targets * V_mix + (1.0 - token_mask_targets) * V_out if token_mask_targets is not None else V_mix
            K_out_new = (token_mask_targets * K_mix + (1.0 - token_mask_targets) * K_out) if (e4e_swap_k and token_mask_targets is not None) else (K_mix if e4e_swap_k else K_out)

            # 5) Write back.
            key = torch.cat([K_all[:1], torch.cat([K_all[1:, :s_tgt], K_out_new, K_all[1:, s_ctx:s_ctx+n_ctx], K_all[1:, s_ctx+n_ctx:]], dim=1)], dim=0)
            value = torch.cat([V_all[:1], torch.cat([V_all[1:, :s_tgt], V_out_new, V_all[1:, s_ctx:s_ctx+n_ctx], V_all[1:, s_ctx+n_ctx:]], dim=1)], dim=0)

        # Continue with the official attention flow.
        query = query.unflatten(-1, (attn.heads, -1))
        key   = key.unflatten(  -1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key   = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key   = encoder_key.unflatten(  -1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key   = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key   = torch.cat([encoder_key,   key],   dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key   = apply_rotary_emb(key,   image_rotary_emb, sequence_dim=1)

        attn_out = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_out = attn_out.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            enc_eff = encoder_hidden_states.shape[1]
            enc_out, hid_out = attn_out.split_with_sizes([enc_eff, attn_out.shape[1] - enc_eff], dim=1)
            hid_out = attn.to_out[0](hid_out); hid_out = attn.to_out[1](hid_out)
            enc_out = attn.to_add_out(enc_out)
            return hid_out, enc_out
        else:
            return attn_out


def _norm_spatial(x, eps=1e-6, dim=-2):
    # dim=-2 normalizes over the sequence dimension to remove spatial appearance statistics.
    mean = x.mean(dim=dim, keepdim=True)
    std  = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


class FluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxIPAdapterAttnProcessor(torch.nn.Module):
    """Flux Attention processor for IP-Adapter."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self, hidden_size: int, cross_attention_dim: int, num_tokens=(4,), scale=1.0, device=None, dtype=None
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
                for _ in range(len(num_tokens))
            ]
        )


    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[List[torch.Tensor]] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
        *,  # Force the following controls to be keyword-only.
        # --- Appearance transfer ---
        kv_alpha: float = 0.5,
        q_alpha: float = 0.5,
        enc_len: Optional[int]=None,
        kv_tgt_len: Optional[int]=None,
        kv_ctx_len: Optional[int]=None,
        # --- Structure alignment ---
        e4e_struct_align: bool = False, # Enable target-context structure alignment
        e4e_align_beta: float = 1,      # Cost blend weight: cosine vs. index prior
        e4e_step: int = None,           # Current step from the pipeline
        e4e_align_start: int = 0,       # Used only when e4e_step is set
        e4e_align_end: int = 0,            
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        ip_query = query

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-adapter
            ip_attn_output = torch.zeros_like(hidden_states)
            
            # Gating.
            gate = hidden_states.new_zeros((batch_size, 1, 1))
            if batch_size > 2:
                gate[1:2] = 1.0
            else:
                gate[:] = 1.0

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                current_ip_hidden_states = dispatch_attention_fn(
                    ip_query,
                    ip_key,
                    ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
                current_ip_hidden_states = current_ip_hidden_states.reshape(batch_size, -1, attn.heads * attn.head_dim)
                current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)
                
                # Pass only output images.
                current_ip_hidden_states = current_ip_hidden_states * gate   # (B,1,1) broadcasts to (B, N, D)
                
                ip_attn_output += scale * current_ip_hidden_states

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            return hidden_states


class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [
        FluxAttnProcessor,
        FluxIPAdapterAttnProcessor,
    ]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
