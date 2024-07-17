"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_one_hop` with the
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


def compute_rewrite_quality_one_hop(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record

    :return: Dictionary containing rewriting metrics
    """

    prompt = ["{} The answer to this question, most simply, is".format(
        record["portability"]["New Question"])]
    target = record["portability"]["New Answer"]

    print("TESTING_PORTABILITY")
    print("PORTABILITY_PROMPTS:{}".format(prompt))

    gen_texts = generate_fast(
        model,
        tok,
        prompt,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    print(type(gen_texts[0]))
    print("GEN_TEXTS:{}".format(gen_texts[0]))

    ret = {
        "portability_prompt": prompt,
        "portability_target": target,
        "text": gen_texts[0],
    }

    return ret
