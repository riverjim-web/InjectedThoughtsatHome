# Where each margin note comes from

| Term | Source |
|---|---|
| activations, forward pass, internal state, residual stream, direction, layers, token | Model config: 24 layers, hidden size 2,880 (`mlx-community/gpt-oss-20b-MXFP4-Q8`, method notes); hook code in `src/lab.py` |
| steered | The steering test in `results/calib2.jsonl` and the story figure |
| point the same way (cosine similarity) | Standard definition; the page's existing claim that the two recipes "barely point the same way" |
| controls, random vector | Main run design, `config_main.json`, `src/lab.py` |
| temperature 1, greedy decoding | Sampling settings in `config_main.json` and the calibration run |
| chain of thought / reasoning effort | OpenAI, [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/) (low, medium, high effort; full chain of thought); analysis/final channels as shown in every trial |
| quantized, mixture-of-experts | OpenAI, [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/): MXFP4, 21B total / 3.6B active, 32 experts with 4 active per token, runs in 16 GB |
| Gemma, Ollama | `src/judge.py`, `src/judge2.py` (`gemma4:26b` through Ollama) |
| shuffled labels | Matcher baseline in `src/judge2.py` |
| much bigger models (Lederman and Mahowald) | [arXiv 2603.05414](https://arxiv.org/abs/2603.05414), abstract: models notice an anomaly without identifying the concept |
| read the model's probabilities directly | `first_token_probs` in `src/extra2.py` |
| natural-log odds scale, 95% interval, r | Standard definitions; computed in `analysis/build_data.py` |
| priming | Standard definition (psychology); the neutral forced-choice result in `results/forced_choice.jsonl` |
| Claude Fable 5.2 | The review that led to the access runs |
| refusal direction | [arXiv 2603.21396](https://arxiv.org/abs/2603.21396), already cited on the page |
