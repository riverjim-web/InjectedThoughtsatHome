# Frozen before any edit (12 Nov 2026, 10:35 ET)

Pass line for the revised page: a reader with only the page scores at least 8 of 10 and higher
than on the current page (version 7). Scoring per question: 1 = every key point present,
0.5 = the main point present but a key point missing or muddled, 0 = wrong, missing or
"not explained". The reader is told to use only the page; outside knowledge doesn't count.

## Questions and key points

1. What physically is an "injected thought" here: what gets added, to what, and where did it come from?
   - Key: a vector (list of numbers) added to the model's internal state, the residual stream, at one layer, partway
     through; built from the model's own internal state while it talked about the concept, minus the average for
     baseline words.
2. The model sometimes said "yes, I detect something". Why doesn't that count as evidence it noticed the concept?
   - Key: meaningless vectors (random or split) of the same size made it say yes about as often (29 vs 30 of 396),
     so the claims respond to any push, not to the concept.
3. What does "switching the vector off before the answer" test, and what happened?
   - Key: whether naming needs the vector to keep steering the words while it writes; with it off, it still claimed
     a detection sometimes (14 of 216) but never named the concept.
4. In the yes/no readout, what exactly is measured, and why is the rise in "Yes" on the detection question not
   evidence of detection?
   - Key: the probability the first word of the answer is Yes vs No (odds); injection shifts answers to questions the
     model is unsure about more, across 23 questions, and the detection question shifts about as much as its
     uncertainty predicts (nothing left over).
5. What is the word-pick (forced-choice) test, and why does the neutral version matter?
   - Key: the model is shown the 36 words and picks one; it picks the injected one well above chance (3%); with a
     neutral "pick a word" prompt it picks it even more often (38% vs 30%), so it's priming, not self-report.
6. What does "the matching concept ranked 17.9 of 36 on average" mean, and what would a model that could check
   its own state have shown?
   - Key: for each word, asking "is the injected thought about X?" under all 36 injections and ranking where the
     truly injected one falls by its chance of Yes; chance is 18.5; introspection would put it near 1.
7. What did the blind reader (Gemma) find, and what does it say about whether the concept got into the model?
   - Key: reading only gpt-oss's answers, it named the injected concept about 12 times more often than luck
     (including in denials), so the concept got in and shaped the words even when the model said nothing was there.
8. What happened when the model was told it was allowed to report unusual internal states?
   - Key: it backfired: more policy talk and refusals, detection claims fell (18 to 2), the correct/related names
     went to zero.
9. How does this compare with Lindsey's result for Claude Opus 4.1?
   - Key: Opus said yes and named the right concept about 20% of the time at the best settings, with almost no false
     claims on controls; gpt-oss showed nothing beyond what controls explain.
10. Give two reasons this result doesn't show that language models in general can't introspect.
    - Key (any two): one small model (20B); different vector recipe from Lindsey's; quantized mixture-of-experts;
      the prompt is read as role-play / trained denial policy; Lindsey found it mainly in the most capable models.

## Terms of art that must be explained at or before first use

activations · forward pass · residual stream · layer · token · concept vector · baseline words · steering ·
strength (fraction of the residual stream's size) · control trial · random vector · split vector · baseword vector ·
reasoning / analysis channel · reasoning effort · temperature · greedy decoding · prefill · probability of the
first token · log-odds · false positive · chance level · shuffled-label baseline · blind reader · priming ·
forced choice · cosine similarity · mixture-of-experts · quantized · 95% interval · refusal direction
