import os
import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.linalg as LA

try:
    import pymetis
except ImportError:  # pragma: no cover - only needed for the METIS ablation path.
    pymetis = None

from basicts.runners.callback import AddAuxiliaryLoss
from basicts.utils.serialization import get_data_file_path, load_adj

from ..config.staar_config import STAARConfig


def _calculate_pearson_correlation(data: np.ndarray) -> np.ndarray:
    """Calculate node-wise Pearson correlation and map it to [0, 1]."""

    with np.errstate(divide="ignore", invalid="ignore"):
        pearson_corr = np.corrcoef(data.T)
    pearson_corr = np.nan_to_num(pearson_corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (pearson_corr + 1.0) / 2.0


def _load_or_create_pearson_correlation(dataset_name: str) -> np.ndarray:
    """Load pearson_corr.pkl, or create it from train_data.npy when missing."""

    corr_path = get_data_file_path(dataset_name, "pearson_corr.pkl")
    train_data_path = get_data_file_path(dataset_name, "train_data.npy")

    if os.path.exists(corr_path):
        with open(corr_path, "rb") as f:
            return pickle.load(f).astype(np.float32)

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Missing both Pearson correlation and training data: {corr_path}, {train_data_path}"
        )

    train_data = np.load(train_data_path).astype(np.float32)
    pearson_corr = _calculate_pearson_correlation(train_data)

    with open(corr_path, "wb") as f:
        pickle.dump(pearson_corr, f)

    return pearson_corr


def _adjacency_matrix_to_metis_format(adj_matrix: np.ndarray, threshold: float = 0.0):
    """Convert a dense adjacency matrix to METIS CSR-style arrays."""

    num_nodes = adj_matrix.shape[0]
    xadj = [0]
    adjncy = []
    adjwgt = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = float(adj_matrix[i, j])
            if weight <= threshold:
                continue
            adjncy.append(j)
            adjwgt.append(max(int(weight * 1000), 1))
        xadj.append(len(adjncy))

    return np.array(xadj, dtype=np.int32), np.array(adjncy, dtype=np.int32), np.array(adjwgt, dtype=np.int32)


def _partition_graph_with_metis(adj_matrix: np.ndarray, num_regions: int) -> np.ndarray:
    """Partition graph nodes with METIS and return node-to-region assignments."""

    if pymetis is None:
        raise ImportError("pymetis is required when region_assignment_mode='metis'.")

    xadj, adjncy, adjwgt = _adjacency_matrix_to_metis_format(adj_matrix, threshold=0.0)
    _, assignment = pymetis.part_graph(num_regions, xadj=xadj, adjncy=adjncy, eweights=adjwgt)
    return np.array(assignment, dtype=np.int32)


def _load_or_create_metis_partition(dataset_name: str, adj_matrix: np.ndarray, num_regions: int) -> np.ndarray:
    """Load cached METIS partitions, or calculate and save them when missing."""

    cache_path = get_data_file_path(dataset_name, f"metis_partition_{num_regions}.npy")
    expected_shape = (adj_matrix.shape[0],)

    if os.path.exists(cache_path):
        partition = np.load(cache_path)
        if partition.shape == expected_shape:
            return partition.astype(np.int32)

    partition = _partition_graph_with_metis(adj_matrix, num_regions)
    np.save(cache_path, partition)
    return partition


def _calculate_laplacian_positional_embedding(adj_matrix: np.ndarray, pe_dim: int) -> np.ndarray:
    """Calculate Laplacian positional embeddings from an adjacency matrix."""

    num_nodes = adj_matrix.shape[0]
    if pe_dim <= 0:
        return np.zeros((num_nodes, 0), dtype=np.float32)

    adjacency = torch.from_numpy(np.maximum(adj_matrix, adj_matrix.T).astype(np.float32))
    degree = adjacency.sum(dim=1).clamp_min(1e-6)
    degree_inv_sqrt = torch.diag(torch.rsqrt(degree))
    laplacian = torch.eye(num_nodes, dtype=adjacency.dtype) - degree_inv_sqrt @ adjacency @ degree_inv_sqrt

    _, eigvecs = LA.eigh(laplacian)
    lap_pe = eigvecs[:, 1 : min(num_nodes, pe_dim + 1)]
    if lap_pe.shape[1] < pe_dim:
        padding = torch.zeros(num_nodes, pe_dim - lap_pe.shape[1], dtype=lap_pe.dtype)
        lap_pe = torch.cat([lap_pe, padding], dim=1)

    return lap_pe.numpy().astype(np.float32)


def _load_or_create_laplacian_positional_embedding(dataset_name: str, adj_matrix: np.ndarray, pe_dim: int) -> np.ndarray:
    """Load cached LapPE, or calculate and save it when missing."""

    cache_path = get_data_file_path(dataset_name, f"laplacian_pe_{pe_dim}.npy")
    expected_shape = (adj_matrix.shape[0], pe_dim)

    if os.path.exists(cache_path):
        laplacian_pe = np.load(cache_path)
        if laplacian_pe.shape == expected_shape:
            return laplacian_pe.astype(np.float32)

    lap_pe = _calculate_laplacian_positional_embedding(adj_matrix, pe_dim)
    np.save(cache_path, lap_pe)
    return lap_pe


def _normalize_adjacency_for_loss(adj_matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and scale adjacency to [0, 1] for the link prediction loss."""

    adjacency = np.maximum(adj_matrix, adj_matrix.T).astype(np.float32)
    max_value = float(adjacency.max()) if adjacency.size else 0.0
    if max_value > 1.0:
        adjacency = adjacency / max_value
    return np.clip(adjacency, 0.0, 1.0).astype(np.float32)


class MLPBlock(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        return hidden + input_data


class AdaptiveRegionAssignment(nn.Module):
    """Adaptive Region Assignment from node states and graph-topological information."""

    def __init__(
        self,
        node_feature_dim: int,
        laplacian_pe: torch.Tensor,
        num_regions: int,
        hidden_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("assignment_temperature must be positive.")

        self.temperature = temperature
        self.register_buffer("laplacian_pe", laplacian_pe, persistent=False)

        laplacian_pe_dim = laplacian_pe.shape[-1]
        input_dim = node_feature_dim + laplacian_pe_dim
        self.mlp_norm = nn.LayerNorm(input_dim)
        self.mlp = MLPBlock(input_dim, hidden_dim, dropout=dropout)
        self.proj = nn.Linear(input_dim, num_regions, bias=True)

    def forward(self, node_feature: torch.Tensor) -> torch.Tensor:
        if node_feature.dim() == 3:
            batch_size = node_feature.shape[0]
            laplacian_pe = self.laplacian_pe.unsqueeze(0).expand(batch_size, -1, -1)
        elif node_feature.dim() == 4:
            batch_size, num_patches = node_feature.shape[:2]
            laplacian_pe = self.laplacian_pe.view(1, 1, *self.laplacian_pe.shape).expand(
                batch_size,
                num_patches,
                -1,
                -1,
            )
        else:
            raise ValueError("node_feature must have shape [B, N, D] or [B, T, N, D].")

        assignment_input = torch.cat([node_feature, laplacian_pe], dim=-1)

        logits = self.proj(self.mlp(self.mlp_norm(assignment_input)))
        return F.softmax(logits / self.temperature, dim=-1)


class SelfAttention(nn.Module):
    """Multi-head self-attention over a token sequence."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout_p = dropout

        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask.unsqueeze(0).unsqueeze(0))
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(out)


class DiffusionPathBlock(nn.Module):
    """Diffusion Path block with Spatio-Temporal Attention over Regions."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.use_attention = num_heads > 0

        if self.use_attention:
            self.pre_attn_norm = nn.LayerNorm(hidden_size)
            self.spatio_temporal_attention = SelfAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.pre_attn_norm = None
            self.spatio_temporal_attention = None

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def _pool_to_regions(self, hidden: torch.Tensor, region_assignment: torch.Tensor) -> torch.Tensor:
        if region_assignment.dim() == 2:
            region_tokens = torch.einsum("btnd,nr->btrd", hidden, region_assignment)
            counts = region_assignment.sum(dim=0).clamp_min(1.0).view(1, 1, -1, 1)
        elif region_assignment.dim() == 3:
            region_tokens = torch.einsum("btnd,bnr->btrd", hidden, region_assignment)
            counts = region_assignment.sum(dim=1).clamp_min(1.0).view(hidden.shape[0], 1, -1, 1)
        elif region_assignment.dim() == 4:
            region_tokens = torch.einsum("btnd,btnr->btrd", hidden, region_assignment)
            counts = region_assignment.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        else:
            raise ValueError("region_assignment must have shape [N, R], [B, N, R], or [B, T, N, R].")

        return region_tokens / counts

    def _broadcast_to_nodes(self, region_tokens: torch.Tensor, region_assignment: torch.Tensor) -> torch.Tensor:
        if region_assignment.dim() == 2:
            return torch.einsum("btrd,nr->btnd", region_tokens, region_assignment)
        if region_assignment.dim() == 3:
            return torch.einsum("btrd,bnr->btnd", region_tokens, region_assignment)
        if region_assignment.dim() == 4:
            return torch.einsum("btrd,btnr->btnd", region_tokens, region_assignment)
        raise ValueError("region_assignment must have shape [N, R], [B, N, R], or [B, T, N, R].")

    def forward(self, hidden: torch.Tensor, region_assignment: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_attention:
            if region_assignment is None:
                raise ValueError("region_assignment is required when Spatio-Temporal Attention is enabled.")

            batch_size, num_patches, num_regions = hidden.shape[0], hidden.shape[1], region_assignment.shape[-1]
            region_tokens = self._pool_to_regions(self.pre_attn_norm(hidden), region_assignment)
            region_time_tokens = region_tokens.reshape(batch_size, num_patches * num_regions, hidden.shape[-1])
            region_time_tokens = self.spatio_temporal_attention(region_time_tokens)
            region_tokens = region_time_tokens.reshape(batch_size, num_patches, num_regions, hidden.shape[-1])
            hidden = hidden + self.dropout(self._broadcast_to_nodes(region_tokens, region_assignment))

        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


class STAAR(nn.Module):
    """
    STAAR model for region-level spatio-temporal traffic forecasting.
    """

    _required_callbacks: List[type] = [AddAuxiliaryLoss]

    def __init__(self, config: STAARConfig):
        super().__init__()

        self.input_len = config.input_len
        self.output_len = config.output_len
        self.patch_len = config.patch_len or config.input_len
        self.embed_dim = config.embed_dim
        self.if_spatial = config.if_spatial
        self.spatial_hidden_dim = config.spatial_hidden_dim
        self.tid_hidden_dim = config.tid_hidden_dim
        self.diw_hidden_dim = config.diw_hidden_dim
        self.time_of_day_size = config.num_time_in_day
        self.day_of_week_size = config.num_day_in_week

        if self.input_len is None or self.output_len is None or self.embed_dim is None:
            raise ValueError("input_len, output_len, and embed_dim must be specified.")
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive.")
        if self.input_len % self.patch_len != 0:
            raise ValueError(
                f"input_len ({self.input_len}) must be divisible by patch_len ({self.patch_len})."
            )
        self.num_patches = self.input_len // self.patch_len

        self.if_time_in_day = config.if_time_in_day
        self.if_day_in_week = config.if_day_in_week

        self.num_inherent_blocks = config.num_inherent_blocks
        self.num_diffusion_blocks = config.num_diffusion_blocks
        self.num_prediction_blocks = config.num_prediction_blocks
        self.num_spatio_temporal_attention_heads = config.num_spatio_temporal_attention_heads
        self.use_gating = config.use_gating
        self.dropout = config.dropout

        self.region_assignment_mode = config.region_assignment_mode
        self.use_adaptive_region_assignment = (
            config.use_adaptive_region_assignment and self.region_assignment_mode == "adaptive"
        )
        self.laplacian_pe_dim = config.laplacian_pe_dim
        self.assignment_mlp_dropout = config.assignment_mlp_dropout
        self.assignment_temperature = config.assignment_temperature
        self.assignment_blend_start = config.assignment_blend_start
        self.assignment_blend_end = config.assignment_blend_end

        self.link_pred_loss_weight = config.link_pred_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.balance_loss_weight = config.balance_loss_weight

        self.use_pearson_correlation = config.use_pearson_correlation
        self.pearson_corr_weight = config.pearson_corr_weight
        self.num_regions = config.num_regions

        _, adj_matrix = load_adj(config.dataset_name, "original")
        num_nodes = adj_matrix.shape[0]
        if config.num_nodes is not None and config.num_nodes != num_nodes:
            raise ValueError(
                f"config.num_nodes ({config.num_nodes}) does not match adjacency size ({num_nodes})."
            )
        self.num_nodes = num_nodes

        if self.use_pearson_correlation:
            pearson_corr = _load_or_create_pearson_correlation(config.dataset_name)
            weight = self.pearson_corr_weight
            adj_matrix = adj_matrix ** (1 - weight) + pearson_corr ** weight

        adj_for_loss = _normalize_adjacency_for_loss(adj_matrix)
        self.register_buffer("adj_matrix", torch.from_numpy(adj_for_loss), persistent=False)

        self.patch_projection = nn.Linear(self.patch_len, self.embed_dim, bias=True)

        if self.if_spatial:
            self.node_embedding = nn.Parameter(torch.empty(num_nodes, self.spatial_hidden_dim))
            nn.init.xavier_uniform_(self.node_embedding)

        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(config.num_time_in_day, self.tid_hidden_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(config.num_day_in_week, self.diw_hidden_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.hidden_size = (
            self.embed_dim
            + self.spatial_hidden_dim * int(self.if_spatial)
            + self.tid_hidden_dim * int(self.if_time_in_day)
            + self.diw_hidden_dim * int(self.if_day_in_week)
        )
        self.embedding_norm = nn.LayerNorm(self.hidden_size)

        self.inherent_blocks = nn.Sequential(
            *[
                MLPBlock(self.hidden_size, self.hidden_size, dropout=self.dropout)
                for _ in range(self.num_inherent_blocks)
            ]
        )

        self.fusion_gate = nn.Linear(self.hidden_size * 2, self.hidden_size) if self.use_gating else None

        self.adaptive_region_assignment_layers = nn.ModuleList()
        if self.num_spatio_temporal_attention_heads > 0:
            if self.num_regions < 1:
                raise ValueError("num_regions must be positive when Spatio-Temporal Attention is enabled.")
            if self.num_regions > num_nodes:
                raise ValueError(f"num_regions ({self.num_regions}) cannot exceed number of nodes ({num_nodes}).")

            if self.region_assignment_mode not in {"adaptive", "metis"}:
                raise ValueError("region_assignment_mode must be one of {'adaptive', 'metis'}.")

            if self.region_assignment_mode in {"adaptive", "metis"}:
                region_indices = _load_or_create_metis_partition(config.dataset_name, adj_matrix, self.num_regions)
                static_assignment = np.zeros((num_nodes, self.num_regions), dtype=np.float32)
                static_assignment[np.arange(num_nodes), region_indices] = 1.0
                self.register_buffer("static_region_assignment", torch.from_numpy(static_assignment), persistent=False)

            if self.region_assignment_mode == "adaptive" and config.use_adaptive_region_assignment:
                lap_pe = _load_or_create_laplacian_positional_embedding(
                    config.dataset_name,
                    adj_matrix,
                    self.laplacian_pe_dim,
                )
                self.adaptive_region_assignment_layers = nn.ModuleList(
                    [
                        AdaptiveRegionAssignment(
                            node_feature_dim=self.hidden_size,
                            laplacian_pe=torch.from_numpy(lap_pe),
                            num_regions=self.num_regions,
                            hidden_dim=self.hidden_size + self.laplacian_pe_dim,
                            dropout=self.assignment_mlp_dropout,
                            temperature=self.assignment_temperature,
                        )
                        for _ in range(self.num_diffusion_blocks)
                    ]
                )

        self.diffusion_blocks = nn.ModuleList(
            [
                DiffusionPathBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_spatio_temporal_attention_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_diffusion_blocks)
            ]
        )

        self.prediction_encoder = nn.Sequential(
            *[
                MLPBlock(self.hidden_size, self.hidden_size, dropout=self.dropout)
                for _ in range(self.num_prediction_blocks)
            ]
        )
        self.regression_layer = nn.Linear(self.hidden_size, self.output_len)

    def _expand_timestamps(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if inputs_timestamps is None:
            if self.if_time_in_day or self.if_day_in_week:
                raise ValueError("inputs_timestamps is required when temporal embeddings are enabled.")
            return None

        batch_size, time_steps, num_nodes = inputs.shape
        if inputs_timestamps.dim() == 3:
            if inputs_timestamps.shape[:2] != (batch_size, time_steps):
                raise ValueError("inputs_timestamps must match input batch and time dimensions.")
            return inputs_timestamps.unsqueeze(2).expand(-1, -1, num_nodes, -1)

        if inputs_timestamps.dim() == 4:
            if inputs_timestamps.shape[:3] != (batch_size, time_steps, num_nodes):
                raise ValueError("inputs_timestamps must match input batch, time, and node dimensions.")
            return inputs_timestamps

        raise ValueError("inputs_timestamps must have shape [B, T, C] or [B, T, N, C].")

    def _timestamp_to_index(self, timestamp: torch.Tensor, size: int) -> torch.Tensor:
        if timestamp.is_floating_point():
            timestamp_min = float(timestamp.min().item()) if timestamp.numel() else 0.0
            timestamp_max = float(timestamp.max().item()) if timestamp.numel() else 0.0
            if -1e-6 <= timestamp_min and timestamp_max <= 1.0 + 1e-6:
                timestamp = timestamp * float(size)
        return timestamp.long().clamp(min=0, max=size - 1)

    def _patch_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_nodes = inputs.shape
        return inputs.reshape(batch_size, self.num_patches, self.patch_len, num_nodes).permute(0, 1, 3, 2).contiguous()

    def _patch_last_timestamps(self, inputs_timestamps: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if inputs_timestamps is None:
            return None
        patch_last_index = torch.arange(
            self.patch_len - 1,
            self.input_len,
            self.patch_len,
            device=inputs_timestamps.device,
        )
        return inputs_timestamps.index_select(dim=1, index=patch_last_index)

    def _build_input_embedding(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, num_nodes = inputs.shape
        patched_inputs = self._patch_inputs(inputs)
        traffic_embedding = self.patch_projection(patched_inputs)

        embedding_parts = [traffic_embedding]
        if self.if_spatial:
            node_embedding = self.node_embedding.view(1, 1, num_nodes, -1).expand(
                batch_size,
                self.num_patches,
                -1,
                -1,
            )
            embedding_parts.append(node_embedding)

        patched_timestamps = self._patch_last_timestamps(inputs_timestamps)
        if self.if_time_in_day:
            time_index = self._timestamp_to_index(patched_timestamps[..., 0], self.time_of_day_size)
            embedding_parts.append(self.time_in_day_emb[time_index])

        if self.if_day_in_week:
            day_index = self._timestamp_to_index(patched_timestamps[..., 1], self.day_of_week_size)
            embedding_parts.append(self.day_in_week_emb[day_index])

        return torch.cat(embedding_parts, dim=-1)

    def _calculate_assignment_blend_lambda(self, block_idx: int) -> float:
        if self.num_diffusion_blocks <= 1:
            lambda_t = self.assignment_blend_end
        else:
            block_ratio = block_idx / (self.num_diffusion_blocks - 1)
            lambda_t = self.assignment_blend_start + (
                self.assignment_blend_end - self.assignment_blend_start
            ) * block_ratio
        return float(max(0.0, min(1.0, lambda_t)))

    def _build_region_assignment_bundle(self, node_feature: torch.Tensor, block_idx: int = 0) -> dict:
        if self.num_spatio_temporal_attention_heads <= 0:
            return {
                "blended_assignment": None,
                "dynamic_assignment": None,
                "static_assignment": None,
                "blend_lambda": 0.0,
            }

        if self.region_assignment_mode == "metis":
            static_assignment = getattr(self, "static_region_assignment", None)
            return {
                "blended_assignment": static_assignment,
                "dynamic_assignment": None,
                "static_assignment": static_assignment,
                "blend_lambda": 0.0,
            }

        if self.region_assignment_mode == "adaptive":
            if not self.use_adaptive_region_assignment:
                static_assignment = getattr(self, "static_region_assignment", None)
                return {
                    "blended_assignment": static_assignment,
                    "dynamic_assignment": None,
                    "static_assignment": static_assignment,
                    "blend_lambda": 0.0,
                }
            dynamic_assignment = self.adaptive_region_assignment_layers[block_idx](node_feature)
            if dynamic_assignment.dim() == 3:
                static_assignment = self.static_region_assignment.unsqueeze(0).expand(
                    dynamic_assignment.shape[0],
                    -1,
                    -1,
                )
            elif dynamic_assignment.dim() == 4:
                static_assignment = self.static_region_assignment.view(
                    1,
                    1,
                    *self.static_region_assignment.shape,
                ).expand(
                    dynamic_assignment.shape[0],
                    dynamic_assignment.shape[1],
                    -1,
                    -1,
                )
            else:
                raise ValueError("dynamic_assignment must have shape [B, N, R] or [B, T, N, R].")

            lambda_t = self._calculate_assignment_blend_lambda(block_idx)
            blended_assignment = (1.0 - lambda_t) * static_assignment + lambda_t * dynamic_assignment
            return {
                "blended_assignment": blended_assignment,
                "dynamic_assignment": dynamic_assignment,
                "static_assignment": static_assignment,
                "blend_lambda": lambda_t,
            }

        raise ValueError("region_assignment_mode must be one of {'adaptive', 'metis'}.")

    def _build_region_assignment(self, node_feature: torch.Tensor, block_idx: int = 0) -> Optional[torch.Tensor]:
        assignment_bundle = self._build_region_assignment_bundle(node_feature, block_idx=block_idx)
        self._latest_region_assignment_bundle = assignment_bundle
        return assignment_bundle["blended_assignment"]

    def _calculate_region_mass_balance_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        region_mass = assignment.mean(dim=-2).clamp_min(1e-8)
        entropy = -(region_mass * region_mass.log()).sum(dim=-1)
        max_entropy = torch.log(region_mass.new_tensor(float(self.num_regions)))
        return (max_entropy - entropy).mean()

    def _calculate_assignment_losses(
        self,
        region_assignment: torch.Tensor,
        dynamic_assignment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        regularized_assignment = dynamic_assignment if dynamic_assignment is not None else region_assignment

        if region_assignment.dim() == 2:
            region_assignment = region_assignment.unsqueeze(0)

        normalized_assignment = F.normalize(region_assignment, p=2, dim=-1)
        node_similarity = torch.matmul(normalized_assignment, normalized_assignment.transpose(-1, -2))
        adj_matrix = self.adj_matrix
        while adj_matrix.dim() < node_similarity.dim():
            adj_matrix = adj_matrix.unsqueeze(0)
        link_pred_loss = F.mse_loss(
            node_similarity,
            adj_matrix.expand_as(node_similarity),
        )

        assignment = region_assignment.clamp_min(1e-8)
        entropy_loss = -(assignment * assignment.log()).sum(dim=-1).mean()

        balance_loss = self._calculate_region_mass_balance_loss(regularized_assignment)

        return (
            self.link_pred_loss_weight * link_pred_loss
            + self.entropy_loss_weight * entropy_loss
            + self.balance_loss_weight * balance_loss
        )

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run STAAR forward propagation."""

        _, input_len, num_nodes = inputs.shape
        if input_len != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, but got {input_len}.")
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, but got {num_nodes}.")

        timestamps = self._expand_timestamps(inputs, inputs_timestamps)
        embedding = self._build_input_embedding(inputs, timestamps)
        embedding = self.embedding_norm(embedding)

        inherent_hidden = self.inherent_blocks(embedding)

        diffusion_hidden = embedding
        aux_losses = []
        for block_idx, block in enumerate(self.diffusion_blocks):
            self._latest_region_assignment_bundle = {}
            region_assignment = self._build_region_assignment(diffusion_hidden, block_idx=block_idx)
            assignment_bundle = getattr(self, "_latest_region_assignment_bundle", {})
            if self.use_adaptive_region_assignment and region_assignment is not None:
                aux_losses.append(
                    self._calculate_assignment_losses(
                        region_assignment,
                        dynamic_assignment=assignment_bundle.get("dynamic_assignment"),
                    )
                )
            diffusion_hidden = block(diffusion_hidden, region_assignment=region_assignment)

        inherent_last = inherent_hidden.mean(dim=1)
        diffusion_last = diffusion_hidden[:, -1]
        if self.use_gating:
            gate = torch.sigmoid(self.fusion_gate(torch.cat([inherent_last, diffusion_last], dim=-1)))
            hidden = torch.lerp(inherent_last, diffusion_last, gate)
        else:
            hidden = 0.5 * (inherent_last + diffusion_last)

        hidden = self.prediction_encoder(hidden)
        prediction = self.regression_layer(hidden).transpose(1, 2)

        forward_return = {"prediction": prediction}
        if aux_losses:
            forward_return["aux_loss"] = torch.stack(aux_losses).sum()
        return forward_return
