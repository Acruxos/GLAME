"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_mquake` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity

def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"]
                    for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_prediction_multi_hop(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question_prompts: list,
    target_new: str,
    target_true: str,
) -> dict:

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        question_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(question_prompts))],
    ]

    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new,
        target_true,
    )

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1]: cutoffs[i]]
                 for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))
    ]

    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(["question_prompts"])
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(["question_prompts"])
    }

    return ret


def compute_rewrite_quality_mquake(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the mquake dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: mquake dataset record

    :return: Dictionary containing rewriting metrics
    """

    target_new, target_true = (
        record[x] for x in ["new_answer", "answer"]
    )
    prompt = ["{} The answer to this question, most simply, is".format(
        question) for question in record["questions"]]
    prompt_for_direct_test = prompt
    answer = record["new_answer"]

    target_list = [target_new for _ in range(len(record["questions"]))]

    # Structure the restuls as a dictionary.
    print("TESTING_PORTABILITY")
    print("PORTABILITY_PROMPTS:{}".format(prompt))
    print("PORTABILITY_TARGET:{}".format(answer))

    gen_texts = generate_fast(
        model,
        tok,
        prompt,
        n_gen_per_prompt=1,
        max_out_len=60,
    )

    ret = {
        "multi_hop_result": test_prediction_multi_hop(
            model=model,
            tok=tok,
            question_prompts=prompt,
            target_new=target_new,
            target_true=target_true,
        ),
        "portability_prompt": prompt,
        "portability_target": answer,
        "portability_target_alias": record["new_answer_alias"],
        "num_requests": len(record["requested_rewrite"]),
        "num_hops": len(record["single_hops"]),
        "text": gen_texts,
    }

    return ret

