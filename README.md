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
python -m experiments.evaluate \
    --alg_name=GLAME \
    --model_name=[path/to/your/gpt-j/model] \
    --hparams_fname=cf/gpt-j-6b.json \
    --ds_name=cf \
    --num_edits=1
```
Computing the covariance matrix estimation $C$ locally is time consuming, but it will be stored after computing and can be directly used in the next run. It will then take a dozen hours to complete the editing and the evaluation.

To summarize the results of CounterFact dataset, use [`experiments/summarize.py`](experiments/summarize.py):

```bash
python -m experiments.summarize --dir_name=GLAME --runs=run_<run1>
```

Run `summarize_port` / `summarize_mquake` for test results on CounterFactPlus and MQuAKE.

## Acknowledgement

The code we conduct our experiments is based on [`MEMIT`](https://github.com/kmeng01/memit.git).

