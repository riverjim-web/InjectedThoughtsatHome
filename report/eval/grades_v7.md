# Baseline: blind reader agent on the current page (version 7)

Weaker check: the reader is a language model told to use only the page. It already knows how
to read technical prose, so it sits near the ceiling. Graded against the key points frozen in
frozen.md.

| Q | Score | Note |
|---|---|---|
| 1 | 1 | All key points; says layers and the residual stream aren't really explained |
| 2 | 1 | 29 vs 30 of 396 |
| 3 | 1 | 14 of 216, never named |
| 4 | 1 | Uncertainty explanation present; "first word" implicit |
| 5 | 1 | 30% vs 38%, priming |
| 6 | 1 | 17.9 vs 18.5, near 1 for introspection |
| 7 | 1 | 12x luck, including denials |
| 8 | 1 | Backfired, 18 to 2, hits to zero |
| 9 | 0.5 | Gives Opus's 20% and gpt-oss's 2.5% but not that controls explain what gpt-oss did |
| 10 | 1 | Small model, different vectors, role-play, MoE |
| **Total** | **9.5 / 10** | |

Consequence for the frozen pass line ("at least 8 and higher than the current page"): with this
reader only a 10 can pass. The confusion list (answers.md, CONFUSING PASSAGES) is the more useful
output of this check, and the human cold read remains the real test.
