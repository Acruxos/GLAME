import torch
import dgl
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

from .glame_hparams import GLAMEHyperParams
from glame import repr_tools
from .gnn import GNN
from util import nethook

def compute_v(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: Dict,
    hparams: GLAMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    gnn_model: GNN,
    graph_input: dgl.DGLGraph(),
    node_indices: list,
    init_rel_emb: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")
    print(graph_input.edata['r_h'].shape)
    print(graph_input.edata['etype'].shape)

    # 获取subject在EGAT中对应的节点的编号
    subject_id = node_indices[request['subject']]

    # Tokenize target into list of int token IDs
    target_ids = tokenizer(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    if target_ids[0] == tokenizer.bos_token_id or target_ids[0] == tokenizer.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tokenizer.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tokenizer(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids): ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tokenizer, hparams.fact_token, verbose=(i == 0),
            input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.gnn_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    target_init, kl_distr_init, target_delta_feature = None, None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init, target_delta_feature

        if cur_layer == hparams.mlp_module_tmp.format(layer):  # 要编辑的一层
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out):
                    # cur_out[idx, i, :] += target_delta_feature
                    cur_out[idx, i, :] += target_delta_feature
                else:
                    cur_out[i, idx, :] += target_delta_feature

        return cur_out

    logits = model(**input_tok).logits

    # Compute initial distribution for KL divergence
    kl_logits = torch.stack(
        [
            logits[i - len(kl_prompts), idx, :]
            for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
        ],
        dim=0,
    )
    kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
    kl_distr_init = kl_log_probs.detach().clone()

    # Optimizer
    opt = torch.optim.AdamW(
        [
            {'params': gnn_model.parameters(), 'lr': hparams.gnn_lr,
             'weight_decay': hparams.gnn_weight_decay},
        ]
    )

    nethook.set_requires_grad(False, model)
    gnn_model.train()

    # 记录最小值
    min_loss, min_loss_delta = None, None

    def output_loss_no_grad():
        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.mlp_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_tok).logits

                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(
                    kl_logits, dim=1)

            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)

            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100,
                            rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            loss = nll_loss + kl_loss
            print("================================")
            print(
                f"FINAL LOSS {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)}"
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )

            return loss

    # Execute optimization 优化循环
    for it in range(hparams.gnn_num_grad_steps):
        opt.zero_grad()

        # 计算GNN的输出
        outputs = gnn_model(
            graph_input, graph_input.ndata['feat'].float(), init_rel_emb.float()).to("cuda")
        subject_feature = outputs[subject_id]

        target_delta_feature = delta+subject_feature

        if (target_init is not None):
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if target_delta_feature.norm() > max_norm:
                with torch.no_grad():
                    target_delta_feature[...] = target_delta_feature * \
                        max_norm / target_delta_feature.norm()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100,
                        rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
    
        loss = nll_loss + kl_loss
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)}"
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )

        if min_loss is None or loss <= min_loss:
            min_loss = loss
            min_loss_delta = target_delta_feature

        if loss < hparams.early_stopping_loss:
            break

        if it == hparams.gnn_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

    gnn_model.eval()

    outputs = gnn_model(
        graph_input, graph_input.ndata['feat'].float(), init_rel_emb.float()).to("cuda")
    subject_feature = outputs[subject_id]
    target_delta_feature = delta+subject_feature

    if (target_init is not None):
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if target_delta_feature.norm() > max_norm:
            with torch.no_grad():
                target_delta_feature[...] = target_delta_feature * \
                    max_norm / target_delta_feature.norm()

    loss = output_loss_no_grad()

    if loss > hparams.early_stopping_loss and loss > min_loss+0.1:
        target_delta_feature = min_loss_delta
        print("USE MIN_LOSS = {} INSTEAD".format(min_loss))

    target = target_init + target_delta_feature
    
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tokenizer,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_"):]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tokenizer: PreTrainedTokenizer,
    fact_token_strategy: str,
    verbose=True,
    input_prompt=None
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = len(tokenizer.encode(input_prompt)) - 1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index(
            "subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tokenizer=tokenizer,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tokenizer.decode(tokenizer(sentence)["input_ids"][ret]),
        )

    return ret
