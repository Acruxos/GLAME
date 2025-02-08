# GLAME

Knowledge Graph Enhanced Large Language Model Editing

## Requirements

- At least a GPU with no less than 48G memory is needed.

- For the environment, run:

```bash
conda create -n glame python=3.9.7
pip install -r requirements.txt
```

## Running the Evaluation

An example for editing GPT-J with GLAME on CounterFact dataset:
```shell
python -m experiments.evaluate_cf \
    --alg_name=GLAME \
    --model_name=[path/to/your/gpt-j/model] \
    --hparams_fname=cf/gpt-j-6b.json \
    --ds_name=cf \
    --num_edits=1
```
Computing the covariance matrix estimation $C$ locally is time consuming, but it will be stored after computing and can be directly used in the next run. It will then take a dozen hours to complete the editing and the evaluation.

To summarize the results of CounterFact dataset, use [`experiments/summarize.py`](experiments/summarize.py):

```bash
python -m experiments.summarize_cf --dir_name=GLAME --runs=run_<run1>
```

Run `evaluate_cf_plus` / `evaluate_mquake` for test results on CounterFactPlus and MQuAKE, and use corresponding `summarize_cf_plus` and `summarize_mquake` to summarize the results.

## Acknowledgement

The code we conduct our experiments is based on [`MEMIT`](https://github.com/kmeng01/memit.git).

## Citation

If you find this work helpful for your research, please kindly cite it.

```text
@misc{zhang2024knowledge,
      title={Knowledge Graph Enhanced Large Language Model Editing}, 
      author={Mengqi Zhang and Xiaotian Ye and Qiang Liu and Pengjie Ren and Shu Wu and Zhumin Chen},
      year={2024},
      eprint={2402.13593},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

