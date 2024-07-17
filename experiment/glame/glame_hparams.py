from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class GLAMEHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    gnn_num_grad_steps: int
    gnn_loss_layer: int
    gnn_lr: float
    gnn_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    early_stopping_loss: float
    context_template_length_params: List[List[int]]
    subgraph_size: int
    get_repr_layer: int
    gnn_fact_token_strategy: str
    gnn_dim_factor: float
    gnn_attn_drop: float
    gnn_feat_drop: float
    compute_v_strategy: str
    use_predefined_context: bool

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
