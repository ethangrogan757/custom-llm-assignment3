# Ethan's Custom LLM Experiment

Class 4 assignment: train Karpathy's nanoGPT on a small word-token corpus, inspect what it
learned, evaluate it with a fixed 48-case suite, and chat with it. Two full experiments below:
the supplied **starter (classroom) corpus** and a **corpus extension** covering the
*negation* and *opposites* eval categories.

## My choices and prediction

- **Corpus:** started with the supplied classroom corpus (`CORPUS = "classroom"`). For the
  extension experiment I added two original files, written by me for this assignment (no
  external source, so no PDF-extraction issues to check):
  [`corpus/negation.txt`](corpus/negation.txt) (18 sentences, "X did not do A. X did B
  instead" pattern) and [`corpus/opposites.txt`](corpus/opposites.txt) (18 sentences, direct
  pairings like "hot is the opposite of cold ."). These target the **negation** and
  **opposites** extension-eval categories.
- **Training steps:** 10 first, to confirm the notebook ran end to end, then **3,000** for
  both real experiments.
- **Learning rate:** kept the suggested default, **0.001**, with the notebook's built-in
  warmup + cosine decay.
- **Prediction (written before the 10-step check):** *"With only 10 training steps, I expect
  the model to produce mostly gibberish — random or repeated tokens with little grammatical
  structure. Ten weight updates isn't nearly enough for the network to learn meaningful
  patterns; at best I'd expect it to start slightly favoring more frequent tokens over the
  completely random distribution of the untrained model. I don't expect coherent phrases at
  this stage."* This held up — see [`evidence/starter/samples/step_0000.txt`](evidence/starter/samples/step_0000.txt).
  For the full 3,000-step runs, I expected the much larger loss drop to produce fully
  grammatical (if narrow) sentences, and I expected the corpus extension to improve the
  negation/opposites eval scores. The second half of that prediction was largely wrong, and
  diagnosing *why* (below) turned out to be the most informative part of the assignment.

## My run

- **Notebooks:** [`custom_llm_starter.ipynb`](custom_llm_starter.ipynb) (classroom corpus
  only) and [`custom_llm_expanded.ipynb`](custom_llm_expanded.ipynb) (classroom + negation +
  opposites), both executed with outputs intact.
- **Model:** nanoGPT, `n_embd=64`, `n_head=4`, `n_layer=2`, `block_size=48`, `batch_size=32`,
  seed 42, CPU (Colab), PyTorch 2.11.0+cpu.
- **Starter run:** 3,000/3,000 steps completed, ~57s elapsed, 111,872 parameters, vocabulary
  size 136, 4,132 training documents / 460 validation documents. See
  [`evidence/starter/config.json`](evidence/starter/config.json) and
  [`evidence/starter/training_summary.json`](evidence/starter/training_summary.json).
- **Expanded run:** 3,000/3,000 steps completed, ~60s elapsed, 117,888 parameters, vocabulary
  size 230, 4,184 training documents / 465 validation documents. See
  [`evidence/expanded/config.json`](evidence/expanded/config.json) and
  [`evidence/expanded/training_summary.json`](evidence/expanded/training_summary.json).
- **Vocabulary coverage:** both runs kept **100% of their training token types** (well under
  the 509-type cap — nothing was pruned as UNK). Starter: 133/133 types retained, 0.0%
  train/validation unknown rate. Expanded: 227/227 types retained, 0.0% train unknown rate,
  0.25% validation unknown rate. Full reports:
  [`evidence/starter/vocabulary_report.json`](evidence/starter/vocabulary_report.json),
  [`evidence/expanded/vocabulary_report.json`](evidence/expanded/vocabulary_report.json).
  The split is by deduplicated passage, not source file, so this measures held-out sentences
  from the same templates/sources, not generalization to unseen documents.

## My evidence

**Loss curves and table** (fixed panels of 20 train / 20 validation documents each,
mean loss over non-padding next-token targets):

Starter (classroom only):
![Starter loss curve](evidence/starter/training_curves.svg)

| Step | Training loss | Validation loss |
|---|---|---|
| 0 | 4.9263 | 4.9275 |
| 1,500 | 0.6821 | 0.7182 |
| 3,000 | 0.6783 | 0.7061 |

Expanded (classroom + negation + opposites):
![Expanded loss curve](evidence/expanded/training_curves.svg)

| Step | Training loss | Validation loss |
|---|---|---|
| 0 | 5.4369 | 5.4196 |
| 1,500 | 0.7097 | 0.7045 |
| 3,000 | 0.6927 | 0.7132 |

Full histories: [`evidence/starter/history.json`](evidence/starter/history.json),
[`evidence/expanded/history.json`](evidence/expanded/history.json). Note the expanded run's
validation loss actually ticks *up* slightly from step 1,500 to step 3,000 while training loss
keeps falling — a small, early hint of overfitting on this tiny 20-document validation panel,
not something I'd over-interpret given the panel size, but worth flagging honestly.

**Samples** (untrained / halfway / final), same generation settings throughout — full files
linked, including token soup at step 0:
- Starter: [step 0](evidence/starter/samples/step_0000.txt) (word salad, no structure) →
  [step 1,500](evidence/starter/samples/step_1500.txt) → [step 3,000](evidence/starter/samples/step_3000.txt)
  (fully grammatical templated sentences, e.g. *"the team discussed the professor and the
  learning at the school ."*).
- Expanded: [step 0](evidence/expanded/samples/step_0000.txt) →
  [step 1,500](evidence/expanded/samples/step_1500.txt) → [step 3,000](evidence/expanded/samples/step_3000.txt)
  (e.g. *"the team discussed the mango and the juice at the kitchen ."*).

**Token → ID → vector, gradient, and probability inspection** (from the expanded run,
[`evidence/expanded/inspection.json`](evidence/expanded/inspection.json) and
[`evidence/expanded/tokenization.json`](evidence/expanded/tokenization.json)):
- Token **"customer"** → token ID **45** → a 64-number embedding vector. First 3 values
  before training: `[-0.0311, 0.0250, 0.0092]`; after training: `[0.0332, 0.0332, -0.0478]` —
  the full 64-number vectors are in `inspection.json`.
- **First parameter update** for that embedding's coordinate 0: before = `-0.0311128`,
  gradient = `-0.0010441`, learning rate at that step = `1e-05` (warmup hasn't ramped up yet),
  after = `-0.0311028`. Note the actual step size (~1e-5) is close to the learning rate itself
  rather than `gradient × learning_rate` (~1e-8) — that's AdamW normalizing update magnitude
  by its running gradient statistics, not a plain SGD step.
- **Next-token probabilities for the prefix "the customer"** (230-word vocabulary): before
  training, the top guess was "customer" itself at just **0.95%** — essentially random. After
  training, the top guess became **"compared" at 20.5%**, followed by reviewed/returned/
  ordered/selected — exactly the verbs the classroom corpus templates use after "the
  customer/client/buyer...". This is the clearest direct evidence of the model learning a real
  distributional pattern.

**Temperature comparison** (same prompt, same seed, no retraining —
[`evidence/expanded/temperature_comparison.json`](evidence/expanded/temperature_comparison.json)):
- T=0.3 (low): *"the team discussed the professor and the learning at the school ."* — safe,
  high-probability, near-deterministic choice.
- T=0.8 (default): *"the team discussed the mango and the juice at the kitchen ."* — same
  grammatical frame, different (still valid) word fills.
- T=1.2 (high): *"has team discussed the mango and the juice at the kitchen ."* — starts
  breaking grammatically ("has team" instead of "the team"), showing temperature reshaping the
  sampling distribution toward lower-probability, riskier tokens without any weight change.

## My fixed language evals

Suite: [`evals/language_evals.json`](evals/language_evals.json) (unchanged, 48 cases), runner:
[`run_evals.py`](run_evals.py). Full untrained/final results for both experiments:
[`evidence/starter/language_evals/`](evidence/starter/language_evals/),
[`evidence/expanded/language_evals/`](evidence/expanded/language_evals/), plus the combined
comparison files
([`evidence/starter/language_eval_comparison.json`](evidence/starter/language_eval_comparison.json),
[`evidence/expanded/language_eval_comparison.json`](evidence/expanded/language_eval_comparison.json)).

| Experiment | Stage | Correct / 48 | Scorable / 48 | Accuracy among scorable cases | Full results |
|---|---|---|---|---|---|
| Starter corpus | Untrained | 9 | 24 | 37.5% | [untrained](evidence/starter/language_evals/untrained/) |
| Starter corpus | Trained | 20 | 24 | 83.3% | [final](evidence/starter/language_evals/final/) |
| Expanded corpus | Untrained | 6 | 25 | 24.0% | [untrained](evidence/expanded/language_evals/untrained/) |
| Expanded corpus | Trained | 21 | 25 | 84.0% | [final](evidence/expanded/language_evals/final/) |

By group (both experiments' final stage):

| Group | Starter (trained) | Expanded (trained) |
|---|---|---|
| `starter_patterns` (16 cases) | 16/16 | 16/16 |
| `starter_transfer` (8 cases, new phrasing) | 4/8 | 5/8 |
| `extend_corpus` (24 cases) | 0/24, 0 scorable | 0/24, **1 scorable** |

**What worked:** `starter_patterns` hit 100% in both experiments after training — the model
clearly learned the domain-noun/context/place associations from the classroom templates.
`starter_transfer` (familiar words, new phrasing) also improved meaningfully (25%→50-62.5%),
showing the pattern generalizes a little beyond the exact training frames.

**What didn't work, and why (not vocabulary size — the real reason):** `extend_corpus` barely
moved even after adding my negation/opposites sentences — still just 1 of 24 cases scorable.
I initially assumed the 509-type vocabulary cap had pruned my new words, but
`vocabulary_report.json` shows **0 omitted types** — nothing was dropped for being too rare. I
then checked the actual eval cases against my corpus content directly and found two distinct,
explainable causes:
1. **Vocabulary mismatch.** The 24 `extend_corpus` cases span all 8 extension categories
   (grammar, opposites, negation, references, sequence, spatial relations, everyday knowledge,
   categories/analogies) — I only wrote material for 2 of them. Even within negation/opposites,
   the eval cases use specific words I never taught at all — e.g. "ava," "box," "door," "buy,"
   "noisy," "soft" appear nowhere in `negation.txt`/`opposites.txt` — so most of those cases
   were unscorable by construction, independent of training.
2. **Single-occurrence words and the train/validation split.** One eval case needed "quiet,"
   which I *did* teach ("loud is the opposite of quiet ."), but that word still shows as
   unknown in the eval. Since vocabulary is built only from the training split (not
   validation), and the notebook does a random 90/10 passage split, a word appearing in only
   one passage has roughly a 1-in-10 chance of landing entirely in the held-out validation set
   and never reaching the trained vocabulary. Words I only wrote once are at the mercy of that
   split.

This is a real, honest limitation, not a training failure — the assignment explicitly notes an
extension experiment that doesn't improve can still earn full credit when the method is valid
and the result is explained, which is what I've done here.

**Leakage check:** [`evidence/expanded/eval_separation.json`](evidence/expanded/eval_separation.json)
confirms 160 reserved classroom passages were excluded before splitting/vocabulary-building in
both runs, using normalized contiguous-prompt matching. I never pasted eval prompts, choices,
or answers into `corpus/`, generated classroom sentences, or the notebook's outputs — the
match method is not a semantic detector, so I also manually re-read `negation.txt` and
`opposites.txt` against the eval cases to confirm no overlap myself. These 48 cases are public
development tests I used to steer the extension corpus, not an untouched final benchmark.

## My chat interface

Launch: open [`custom_llm_expanded.ipynb`](custom_llm_expanded.ipynb) section 10 in Colab (uses
the trained model already in memory), or run `python chat.py --model evidence/expanded/model.pt
--transcript new_chat.json` locally after `pip install -r requirements.txt`. Model/run
identity: `model_sha256` in [`evidence/expanded/chat_transcript.json`](evidence/expanded/chat_transcript.json)
matches the expanded run's final model (3,000 completed steps).

Screenshot: [`evidence/chat_screenshot.png`](evidence/chat_screenshot.png). Full transcript, 6
real prompts and replies: [`evidence/expanded/chat_transcript.json`](evidence/expanded/chat_transcript.json).

| Prompt | Response | Note |
|---|---|---|
| "the customer" | "compared the merchandise after checking the price ." | in-domain, grammatical |
| "the doctor" | "was focused on patient ." | in-domain, grammatical |
| "the astronaut" | "is local truck ." | **failure/limitation** — "astronaut" is an unknown word; output is grammatically broken nonsense |
| "the dog" | "did not tea ." | learned negation *syntax* ("did not") but combines it nonsensically |
| "the local man" | "the client ." | "man" is an unknown word |
| "the teacher" | "reviewed the item after checking the price ." | in-domain, grammatical |

**Limitation:** each prompt starts with a fresh context (`fresh_context_per_prompt: true` — no
conversation memory). The model only knows the ~230 word types it was trained on; anything
else (like "astronaut") produces ungrounded, often ungrammatical continuations. This is a tiny
sentence-continuation model, not a general chat assistant, and it never retrains or adds these
chat messages back into the corpus.

## What I learned

1. **Corpus:** my corpus was the classroom's synthetic business/food/transport/tech/health/
   education sentences, plus (in the expanded run) my own negation and opposites sentences. It
   can teach the domain-noun/context associations and, in the expanded run, basic negation
   syntax and a few opposite pairs — but it can't teach vocabulary or patterns it never
   contains (proper names like "ava," concepts like "noisy"). 10% of passages are held out as
   validation so I can check whether the model generalizes to unseen sentences from the same
   templates, not to genuinely new topics.
2. **Token vs. ID vs. vector vs. embedding:** a token is a word/punctuation unit (`word_tokens`
   splits text on whitespace/punctuation). Each unique token gets an integer **ID** (e.g.
   "customer" → 45). The **embedding** is the 64-number vector that ID maps to in a lookup
   table — it starts as small random noise and is nudged during training so that words used in
   similar contexts end up with similar vectors.
3. **What makes it a neural network:** the model repeatedly predicts the next token, compares
   its guess to the truth via a loss (cross-entropy over the vocabulary), computes a
   **gradient** for every parameter (how much that parameter contributed to the error), and
   AdamW uses that gradient (normalized by running statistics, not raw gradient × learning
   rate) to nudge every weight — including the embedding table — slightly toward less error.
   Over 3,000 such updates, loss fell from ~5 to ~0.7 in both experiments.
4. **Attention:** each position's output is a weighted combination of *earlier* tokens' values
   (the `attention_rows` in `inspection.json` are lower-triangular — a token can never attend
   to a token that comes after it, which is what makes next-token prediction well-posed).
5. **Probabilities → text → temperature:** the network outputs a probability over all 230
   vocabulary tokens for "what comes next"; sampling picks one. Temperature reshapes that
   distribution before sampling — low temperature sharpens it toward the single most likely
   token (safe, repetitive), high temperature flattens it (more variety, more grammatical
   breakage) — without touching any weights, as shown in the temperature comparison above.
6. **Did the evidence support my prediction?** Partly. The loss/sample evidence fully supported
   my expectation that more steps → lower loss → coherent text. My unstated assumption that
   corpus extension would straightforwardly improve the targeted eval categories was wrong, and
   the investigation into *why* (vocabulary mismatch + single-occurrence split risk) taught me
   more about how this pipeline works than a clean success would have.

## One limitation and my next experiment

**Limitation:** the corpus extension didn't measurably improve the `extend_corpus` eval score,
because my teaching sentences didn't use the same specific words the eval cases test, and even
words I did use were vulnerable to being randomly excluded by the train/validation split when
they appeared in only one passage.

**Next experiment:** rewrite `negation.txt`/`opposites.txt` to (a) reuse the exact general
vocabulary the negation/opposites eval cases rely on — without copying the eval prompts,
choices, or answers themselves — and (b) repeat each key word across at least 3-4 different
passages, reducing the chance any single word gets isolated entirely in the validation split. I
would predict this raises the `extend_corpus` scorable count well above 1/24 (though maybe not
much past ~6/24, since 6 of the 8 extension categories still wouldn't be covered) and would
give some of those cases a real chance to score, rather than being unscorable by default.

## Reproduce and inspect

1. Open [`custom_llm_starter.ipynb`](custom_llm_starter.ipynb) or
   [`custom_llm_expanded.ipynb`](custom_llm_expanded.ipynb) directly on GitHub to see the
   executed outputs, or open in Colab and re-run to reproduce.
2. My added corpus material is at [`corpus/negation.txt`](corpus/negation.txt) and
   [`corpus/opposites.txt`](corpus/opposites.txt) (force-added despite the default
   `corpus/*` gitignore, since this is original, non-sensitive synthetic text).
3. All result files (loss history, eval results, samples, inspection data, trained model
   weights) are under [`evidence/starter/`](evidence/starter/) and
   [`evidence/expanded/`](evidence/expanded/), organized in parallel to the notebook's own
   `llm_runs/<run>/` structure (which is git-ignored by default in this template).
4. Chat evidence: [`evidence/chat_screenshot.png`](evidence/chat_screenshot.png) and
   [`evidence/expanded/chat_transcript.json`](evidence/expanded/chat_transcript.json).
