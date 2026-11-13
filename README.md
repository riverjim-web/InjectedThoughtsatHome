# Injected Thoughts at Home

A home replication of Jack Lindsey's [injected-thoughts experiment](https://transformer-circuits.pub/2025/introspection/index.html) on an open-weights model, **gpt-oss-20b**, run on a Mac mini over 9 to 11 November 2026.

**Read the essay:** [`index.html`](index.html) (served by GitHub Pages; a print version is in [`injected-thoughts-at-home.pdf`](injected-thoughts-at-home.pdf), and a plain-text version for reading aloud is in [`report/REPORT.md`](report/REPORT.md)).

In short: concept vectors were added to gpt-oss-20b's internal state, and the model was asked whether it noticed an injected thought. Detection claims followed any disturbance of the same size, meaningful or not. Correct naming happened only while the vector was still steering the text. A small shift toward "yes" in a yes/no readout was explained by how unsure the model already was. The injected concept was still present after the vector was removed (it primed which word the model picked from a list), but the model's reports about itself never used it. Details, controls, and every trial are in the essay.

Run by RiverJim. The experiment was designed iteratively by Claude (Anthropic) and RiverJim, and run and written up by Claude. Prompt and protocol adapted from Lindsey (2025).

## What's here

| Path | What it is |
|---|---|
| `index.html` | The essay, self-contained: every trial, chart and the trial browser are embedded |
| `results/` | Raw outputs of every run: each trial's prompt settings, the model's reasoning and final answer (`*.jsonl`), probability readouts, the concept vectors (`vectors.npz`) and baseline activations (`baseline_acts.npz`) |
| `src/` | The experiment code (below) |
| `config_*.json` | Settings for each run |
| `analysis/` | Builds `data.json` (everything the page shows) from `results/`; all hand grades (`grades_all.json` is the one the page uses) and the blind matcher's items |
| `report/` | The page's sources: prose (`prose.py`), template, build scripts, and `eval/` (the comprehension check used for the explanation pass) |
| `mailbox/` | Every job script that ran, in order, with its log. Local paths in the logs are shortened to `~` |
| `media/` | *What do we do with an open model?*, the animated shanty at the end of the essay (refrain by RiverJim; verses and animation by Claude Fable 5.2) |
| `sources/` | The two conversations with gpt-oss-20b that started the project, verbatim, including the line quoted in RiverJim's note |
| `mailbox.sh` | The watcher that ran those jobs: Claude's shell on the Mac couldn't use the GPU, so it wrote job scripts and this ran them natively |

## Runs, in order

| Job | Script | What it did | Essay section |
|---|---|---|---|
| 001 | `src/lab.py smoke` | Set up Python and MLX, downloaded the model, checked the hooks | Method notes |
| 002 | `src/lab.py calib2` | Compared three ways of building concept vectors on a steering test | What ran on the Mac mini |
| 003 | `src/lab.py all` (`config_main.json`), then `inject` (`config_medium.json`) | Concept vectors; the main run (720 injected trials, 60 controls, 24 random vectors); think / don't-think test; steering examples; medium reasoning effort | Asked directly, it says no; Thinking about a word on command; More room to think |
| 004 | `src/lab.py inject` (`config_permission.json`) | Main setup plus a permission message | Telling it that it's allowed |
| 005 | `src/judge.py --model gemma4:26b` | Blind reader: Gemma 4 26B guesses the injected concept from each answer | Checking it blind |
| 005-gemma-calib | — | Queued, then cancelled; never ran | — |
| 006 to 008 | — | Closed the watcher | — |
| 009 | `src/extra.py random2`, `yesbias` | Random vectors at every setting; unrelated yes/no questions | Is it the concept, or just the push? |
| 010, 010b | `src/extra2.py` (`baseline`, `logits`, `latent`, `null2`) | Vector switched off before the answer; meaningless and mundane-word vectors | Does it notice the disturbance itself? |
| 011 | `src/extra3.py yn` | Yes/no probability readout against three factual questions | Does it notice the disturbance itself? |
| 012, 012b, 012c | `src/extra4.py yn2` | The same readout over 23 questions (uncertainty fit); 012b failed, 012c completed it | Does it notice the disturbance itself? |
| 013 | `src/judge2.py`; `src/extra5.py fc`, `matrix` | Blind matcher for the naming grades; 36-way word pick; 36×36 match matrix | Is the information there at all? |
| 014 | `src/extra5.py fc` (`config_fc2.json`) | Word pick with the neutral "pick a word" framing | Is the information there at all? |

The job scripts in `mailbox/` show the exact commands and environment variables for each run.

## Rerunning

Needs a Mac with Apple Silicon and at least 24 GB of memory. The blind-reader steps also need [Ollama](https://ollama.com) with `gemma4:26b`.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export LAB="$PWD"                       # the scripts default to ~/agent/lab
export HF_HOME="$LAB/.cache/huggingface"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/gpt-oss-20b-MXFP4-Q8')"   # about 12 GB
.venv/bin/python src/lab.py smoke       # checks the hooks against a plain forward pass
.venv/bin/python src/lab.py all --config config_main.json
```

Then follow the table above. Every script writes to `$LAB/results`. Software used: Python 3.12, MLX 0.32.4, mlx-lm 0.32.2, macOS 27.2, Apple M6 with 24 GB; `requirements.txt` pins the full environment.

## Rebuilding the page from the results

```sh
.venv/bin/python analysis/build_data.py results analysis/grades_all.json analysis/data.json
python3 report/prose.py report/prose.json
python3 report/assemble.py analysis/data.json report/prose.json report/report.html
python3 report/to_markdown.py report/prose.json analysis/data.json report/REPORT.md
```

These reproduce the published page byte for byte. `index.html` is `report/report.html` with a standard `<!doctype html>`, charset and viewport header added.

## Next runs

Listed in order at the end of the essay. First up is gpt-oss-20b with its refusal direction removed, following the approach in [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396).

## License

Code: MIT ([`LICENSE`](LICENSE)). Essay text, figures and data: CC BY 4.0 ([`LICENSE-CC-BY-4.0.md`](LICENSE-CC-BY-4.0.md)).
