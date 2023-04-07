# Certified Reasoning with Language Models

Project on language model guides & tools for certified reasoning.

## Setup

This repository is technically a fork of [gpoesia/peano](https://github.com/gpoesia/peano), but Github does not allow private forks. 
Thus, this relationship is being maintained manually with merge commits. Fixes and general tweaks to the environment are commited there 
and merged here with a manual git merge (using an "upstream" remote that points to `gpoesia/peano`),
while code from this project can be directly commited here. When time comes, we'll merge this back upstream.

To set up, first pull the `synchromesh` submodule (it will be a proper Python package some day, but it still isn't):

```sh
[certified-reasoning] $ git submodule update --init
Submodule path 'learning/ext/synchromesh': checked out 'ff524f9ee20a03192efe2d3b84ae555d7c7fe88d'
[certified-reasoning] $
```

You'll then need to do two changes to environment variables (which you might want to do in your `~/.bashrc` or equivalent to make them permanent):
- Add `learning/ext/synchromesh/synchromesh` to your `PYTHONPATH`. Using the absolute path is usually more robust.
- Put your OpenAI API key in the `OPENAI_API_KEY` environment variable.

Install the Python dependencies:

```sh
[certified-reasoning] $ cd learning/
[learning] $ pip install -r requirements.txt 
```

Do that in a virtualenv / conda environment if you prefer.

Now you should compile and link the Peano environment (instructions in the section below).
After that, you should be able to run the current experiments on PrOntoQA:

```sh
[learning] $ python lm_tool.py
```

This will just run one vanilla OpenAI model (`gpt3.5-turbo` - it will use up credits!) on one PrOntoQA split. The current default experiment is quite small. Right now, what experiment to run is hardcoded in `lm_tool.py`. You can uncomment the `PeanoLMReasoner` in `run_prontoqa_experiment` to test one of the models using the Peano guide.

### Where to implement things

Some ideas we're planning on implementing:

- To add the vanilla LLaMA model as a baseline, we'd implement a corresponding `NaturalLanguageReasoner` in `lm_tool.py`. This should be trivial.
- To enable Synchromesh to run on top of LLaMA, we'd need to subclass `LanguageModel` from `ext/synchromesh/synchromesh/language_model.py`. There should be a simple and inefficient way to do it (where we repeat inference on the whole sequence in each of the methods, like we do with OpenAI models), and a faster way where we cache activations between the calls. We can start with the simple way and optimize if needed.
- To implement guides on the chat models, we need to implement the equivalent of `PeanoLMReasoner` for `OpenAIChatModelReasoner` (both in `lm_tools.py`)

### Compiling and linking the environment

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
[environment] $ cargo rustc --release --lib -- -C link-arg=-undefined -C link-arg=dynamic_lookup
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

If this works, then you're ready to use Peano from Python.

The main file to use to reproduce the Khan Academy experiments from the paper is `learning.py`, which will start an agent
to learn to solve problems using reinforcement learning and tactic induction. The config files and exact commands to run will come soon -
feel free to open an issue if you're interested in those and this hasn't been updated yet!

The Python dependencies can be installed with:

```sh
[learning] $ pip install -r requirements.txt
```
