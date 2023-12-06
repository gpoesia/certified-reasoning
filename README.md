# Certified Reasoning with Language Models

Language model guides & tools for certified reasoning.

This is the official implementation of the following paper ([arXiv link](https://arxiv.org/abs/2306.04031)):

``` bibtex
@article{poesia2023certified,
      title={Certified Reasoning with Language Models}, 
      author={Gabriel Poesia and Kanishk Gandhi and Eric Zelikman and Noah D. Goodman},
      year={2023},
      eprint={2306.04031},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Set up

This repository is forked from a frozen version of [gpoesia/peano](https://github.com/gpoesia/peano).
Peano is still being actively developed, but to keep this repository (certified-reasoning) stable,
the updates will only be merged here periodically and without breaking the experiments from the paper.

First, set up a conda environment or virtualenv (technically optional but recommended). Install dependencies with:

```sh
[certified-reasoning] $ cd learning/
[learning] $ pip install -r requirements.txt 
```

Then, set up `synchromesh` (it will be a proper Python package soon, but until then):

```sh
(base) [certified-reasoning] $ git clone https://github.com/kanishkg/synchromesh.git
[...]
(base) [certified-reasoning] $ cd synchromesh
(base) [synchromesh] $ python setup.py install
```

You'll then need to do two changes to environment variables (which you might want to do in your `~/.bashrc` or equivalent if you want to make them permanent):
- If you want to test with OpenAI models, put your OpenAI API key in the `OPENAI_API_KEY` environment variable.

Now you should compile and link the Peano environment (instructions in the section below).
After that, you should be able to run the current experiments on PrOntoQA:

```sh
[learning] $ python lm_tool.py
```

This will just run one vanilla OpenAI model (`gpt3.5-turbo` - it will use up credits!) on one PrOntoQA split. The current default experiment is quite small. Right now, what experiment to run is hardcoded in `lm_tool.py`. You can uncomment the `PeanoLMReasoner` in `run_prontoqa_experiment` to test one of the models using the Peano guide. We're recently packaged and pushed the support for Hugging Face models, so you can run with LLaMA (or other models) as well.

### Implementing a Language Model Guide

A Language Model Guide is implemented as a CSD Completion Engine (i.e., a subclass of synchromesh's `CompletionEngine` class, which computes valid completions).
LogicGuide from the paper is implemented as the `PeanoCompletionEngine` class in `completion.py`.

The main task of the guide is to implement a `complete(prefix)` method like any other
Completion Engine for Synchromesh: given a string
representing the language model's generation so far, it should return a regular expression
that only matches valid continuations from that point.
For guides, where the idea is that constrained generation is activated by a special sequence,
with the model going back to free generation after the closing sequence,
the `complete` method will generally have the following structure:

- First, we must look at `prefix` to determine if we're currently inside or outside a guided block.
- If outside a guided block, then the model should be able to output anything until it opens a guided block. It's easy to encode this as a regular expression: match any string that does not contain the start marker (like `[[` in LogicGuide), followed by the start marker itself. Check out how `PeanoCompletionEngine` does this.
- If inside a guided block, then we must return a regular expression that will constrain the model to follow the guide. In the case of `PeanoCompletionEngine`, this involves extracting all previous guided blocks from the prefix, making a Peano context with them, then querying Peano to know what inferences might be made. Different guides will have different logics here. The `PeanoCompletionEngine.get_verified_blocks` can be helpful there for extracting previous guided blocks.

A simple self-contained example of a guide is given in `guide_example.py`. This file implements a very simple guide that forces the model to output the elements of an arithmetic sequence inside the guided blocks. The given example asks the model to complete the sequence starting with 103 and adding 13 to the previous element. The example prompts `text-curie-001` with the following:

``` text
A very important and very long sequence in the science of natural numbers is the following: it starts with [[103]], then goes to [[116]], followed
```

Of course, what the sequence is supposed to be is not really described here (nor is `curie` good at arithmetic anyhow). Unconstrained generation will lead to:

``` text
(...) by [[199]], [[233]], [[283]], [[362]], [[429]], [[583]], [[737]], [[1093]], [[1345]], [[1729]], [[2187]], and [[2893]].

This sequence is known as the Fibonacci sequence. It is named after Leonardo Fibonacci, an Italian mathematician who lived in the 12th century. The Fibonacci sequence is a sequence of numbers that
```

In contrast, with the counting guide, we can force the model to output the desired sequence. It then outputs:

``` text
(...) by [[142]], [[155]], [[168]], [[
```

The guide also decides the stopping criterion: here, we decided to stop after 3 numbers for the sake of an example, but that's arbitrary.

The main experiments from the paper (PrOntoQA, ProofWriter, Syllogistic Validity) are implemented in `lm_tool.py`. Right now, the script runs them all. We're refactoring this to allow one to pass the specific experiment to be ran and using which models using a Hydra config file, to allow reproducing just a subset of the results. This is coming soon.

### Compiling and linking the Peano environment

The Peano enviroment is written in Rust and has a Python API via [PyO3](https://pyo3.rs/v0.18.2/).

To compile it, you'll first need to install the Rust toolchain. For that, use [rustup](https://rustup.rs/).

With Rust installed, you can now compile the Peano environment:

For Linux:

```sh
[peano] $ cd environment
[environment] $ cargo build --release
```
For Mac:

```sh
[peano] $ cd environment
[environment] $ cargo rustc --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup
```

This should eventually terminate without errors. It will produce a `peano` executable,
which you can test on some example proofs:

```sh
[environment]$ target/release/peano theories/cs103.p 
Loading theories/cs103.p
Verifying n_even_implies_n_squared_even... ok
Verifying two_is_even... ok
Verifying sum_of_even_is_even... ok
Verifying sum_of_odds_is_even... ok
Verifying sum_of_squares_one_is_even... ok

Verified 5 derivation(s), 5 succeeded.
```

You should also now have a dynamic library in `target/release`:
it will be called `libpeano.so` on Linux, or something like `libpeano.dylib` on Mac.
To use this library as a Python module, we'll use a simple symbolic link:

```sh
[environment] $ cd ../learning
[learning] $ ln -s ../environment/target/release/libpeano.so ./peano.so
```

Note that this must be slightly adjusted on Mac (i.e., you'll link `libpeano.dylib` instead). With that, you should be able to do the following:

```sh
[learning] $ python
>>> import peano
>>>
```

If this works, then you're ready to use Peano from Python as a library, which the `PeanoCompletionEngine` relies on.
