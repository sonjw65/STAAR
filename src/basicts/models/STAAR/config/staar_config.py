from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class STAARConfig(BasicTSModelConfig):
    """Config class for STAAR traffic forecasting."""

    # Data and I/O
    dataset_name: str = field(default=None, metadata={"help": "Dataset name."})
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    input_dim: int = field(default=None)
    embed_dim: int = field(default=None)
    num_nodes: int = field(default=None, metadata={"help": "Number of nodes in the graph."})
    patch_len: int = field(default=None, metadata={"help": "Temporal patch length. Defaults to input_len."})

    # Embeddings
    if_spatial: bool = field(default=True, metadata={"help": "Whether to use spatial node embedding."})
    spatial_hidden_dim: int = field(default=64, metadata={"help": "Hidden size of spatial embedding."})
    if_time_in_day: bool = field(default=False, metadata={"help": "Whether to use time-of-day embedding."})
    if_day_in_week: bool = field(default=False, metadata={"help": "Whether to use day-of-week embedding."})
    num_time_in_day: int = field(default=24, metadata={"help": "Number of time slots in a day."})
    num_day_in_week: int = field(default=7, metadata={"help": "Number of days in a week."})
    tid_hidden_dim: int = field(default=32, metadata={"help": "Hidden size of time-of-day embedding."})
    diw_hidden_dim: int = field(default=32, metadata={"help": "Hidden size of day-of-week embedding."})

    # Architecture
    num_inherent_blocks: int = field(default=3, metadata={"help": "Inherent Path MLP blocks."})
    num_diffusion_blocks: int = field(default=3, metadata={"help": "Diffusion Path blocks."})
    num_prediction_blocks: int = field(default=3, metadata={"help": "Prediction head MLP blocks."})
    num_spatio_temporal_attention_heads: int = field(
        default=4,
        metadata={"help": "Spatio-Temporal Attention heads."},
    )
    use_gating: bool = field(default=True, metadata={"help": "Use gated fusion."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})

    # Graph and region configuration
    use_pearson_correlation: bool = field(default=True, metadata={"help": "Use Pearson prior."})
    pearson_corr_weight: float = field(default=0.5, metadata={"help": "Pearson blend weight."})
    num_regions: int = field(default=16, metadata={"help": "Number of adaptive traffic regions."})
    region_assignment_mode: str = field(default="adaptive", metadata={"help": "'adaptive' or 'metis'."})

    # Adaptive Region Assignment
    use_adaptive_region_assignment: bool = field(
        default=True,
        metadata={"help": "Use adaptive soft region assignment."},
    )
    laplacian_pe_dim: int = field(default=32, metadata={"help": "Laplacian positional embedding size."})
    assignment_mlp_dropout: float = field(default=0.05, metadata={"help": "Assignment MLP dropout."})
    assignment_temperature: float = field(default=1.0, metadata={"help": "Assignment softmax temperature."})
    assignment_blend_start: float = field(default=0.1, metadata={"help": "Early assignment blend ratio."})
    assignment_blend_end: float = field(default=0.5, metadata={"help": "Late assignment blend ratio."})

    # Regularization
    link_pred_loss_weight: float = field(default=1.0, metadata={"help": "Link prediction loss weight."})
    entropy_loss_weight: float = field(default=0.5, metadata={"help": "Entropy loss weight."})
    balance_loss_weight: float = field(default=0.1, metadata={"help": "Balance loss weight."})
