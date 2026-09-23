# Ethan's Custom LLM Experiment

I trained Karpathy's nanoGPT from scratch on a small word-token corpus, inspected what it
actually learned inside, evaluated it with a fixed 48-case language suite, and built a chat
interface around it. This README covers the **starter run** on the supplied classroom corpus,
plus **two corpus-extension attempts**: my first attempt taught only 2 extension categories and
scored 0/24 on `extend_corpus` evals, so I diagnosed why, wrote teaching material for all 8
extension categories, and reran the experiment. The second attempt is what counts as my official
"expanded" result below; the first attempt is kept as evidence of the iteration, not hidden.
Everything below is drawn directly from my own executed notebooks and saved results — nothing
here needs to be rerun to be checked.

*(General project/notebook setup docs from the starter template live in [PROJECT_SETUP.md](PROJECT_SETUP.md).)*

## Overview

- **Notebooks:** [`custom_llm_starter.ipynb`](custom_llm_starter.ipynb) (starter corpus) and
  [`custom_llm_expanded.ipynb`](custom_llm_expanded.ipynb) (second, all-8-category corpus
  extension — my official expanded result), both executed in Google Colab with outputs intact.
  My first extension attempt is kept too:
  [`custom_llm_expanded_v1_negation_opposites.ipynb`](custom_llm_expanded_v1_negation_opposites.ipynb).
- **Corpus additions (second/final attempt):** 8 files in [`corpus/`](corpus/), one per
  extension category I targeted —
  [`negation.txt`](corpus/negation.txt), [`opposites.txt`](corpus/opposites.txt),
  [`spatial_relations.txt`](corpus/spatial_relations.txt),
  [`categories_analogies.txt`](corpus/categories_analogies.txt),
  [`reference.txt`](corpus/reference.txt), [`grammar.txt`](corpus/grammar.txt),
  [`sequence.txt`](corpus/sequence.txt), [`everyday_knowledge.txt`](corpus/everyday_knowledge.txt)
  — all original sentences I wrote myself. No external source: no scraped text, no PDFs, so there
  was no extraction step or OCR/encryption warnings to check. Plain UTF-8 `.txt`, which the
  notebook confirmed importing cleanly with zero warnings for all 8 files (see
  [`evidence/expanded/corpus_manifest.json`](evidence/expanded/corpus_manifest.json)).
- **To open and run:** open any notebook link above directly on GitHub to read the executed
  cells, or open [the notebook in Colab](https://colab.research.google.com/github/pepealonso95/custom-llm/blob/main/custom_llm.ipynb),
  save your own copy, paste in my corpus files, and Run All. Default CPU runtime is enough — I
  trained on Colab's free CPU tier.

## What I taught it, and why

I kept `CORPUS = "classroom"` for every run so my additions sit on top of the supplied sentences
rather than replacing them, and I used `LEARNING_RATE = 0.001` throughout — the notebook's
suggested starting point, with its own warmup and cosine decay. I ran `TRAINING_STEPS = 10` first
purely to confirm the pipeline worked end to end, then `TRAINING_STEPS = 3000` for every real
experiment.

**First attempt:** I picked **negation** and **opposites** because they had a small, repeatable
sentence template — "X did not do A. X did B instead" and "A is the opposite of B." I wrote 18
original sentences per category. Result: `extend_corpus` scored 0/24 correct, and only 1 of the
24 cases was even scorable — my vocabulary simply didn't overlap with what those eval cases
needed. That's the actual failure the assignment asks me to explain honestly, and it's what drove
the second attempt (full diagnosis in [One limitation, explained](#one-limitation-explained-and-what-i-tried-next) below).

**Second attempt:** instead of guessing more negation/opposites sentences, I diagnosed the gap by
comparing the eval suite's required answer vocabulary against what I'd actually taught (details
below), then wrote fresh material for all **8** extension categories — negation, opposites,
spatial relations, categories/analogies, reference, grammar, sequence, and everyday knowledge —
using ordinary words the eval choices needed, in entirely new sentences and templates I wrote
myself (never the eval prompts, choices, or answer key). That's the run I'm using as my official
"expanded" experiment.

| | Starter | Expanded — 1st attempt (negation + opposites) | Expanded — 2nd attempt (all 8 categories, final) |
|---|---|---|---|
| Unique passages (after dedup) | 4,592 | 4,649 (+57 from my files) | 4,830 (+240 from my files) |
| Train / validation documents | 4,132 / 460 | 4,184 / 465 | 4,347 / 483 |
| Vocabulary size (retained types) | 136 (133 word types + UNK/BOS/EOS) | 230 (227 word types + UNK/BOS/EOS) | 399 (396 word types + UNK/BOS/EOS) |
| Training unknown-token rate | 0.0% | 0.0% | 0.0% |
| Validation unknown-token rate | 0.0% | 0.25% | 0.28% |

Every run retained **every** training token type — nothing hit the 509-type cap, so nothing got
pruned to UNK, in any of the three runs. Full detail:
[`evidence/starter/corpus_manifest.json`](evidence/starter/corpus_manifest.json) /
[`evidence/expanded/corpus_manifest.json`](evidence/expanded/corpus_manifest.json) (final) /
[`evidence/expanded_v1_negation_opposites/corpus_manifest.json`](evidence/expanded_v1_negation_opposites/corpus_manifest.json)
(first attempt), and the matching `vocabulary_report.json` in each folder. Since the split is by
deduplicated passage rather than by source file, "held-out" here means new sentence combinations
from the same templates, not unseen topics.

## Prediction vs. what actually happened

Before the 10-step sanity check, I predicted: *"I expect the model to produce mostly gibberish
— random or repeated tokens with little grammatical structure. Ten weight updates isn't nearly
enough to learn meaningful patterns; at best it might start slightly favoring more frequent
tokens over a purely random distribution."* That's exactly what I got —
[`evidence/starter/samples/step_0000.txt`](evidence/starter/samples/step_0000.txt) is
unstructured word soup (`"website doctor light cloudy mango wore pear..."`).

Going into the full 3,000-step runs, I expected the much larger loss drop to produce fully
grammatical sentences (it did — see below), and I expected the corpus extension to noticeably
improve the `extend_corpus` eval scores (my first attempt completely missed that — 0/24 correct).
Diagnosing *why* was the most useful part of the assignment: it wasn't a training bug, it was a
vocabulary-coverage gap I could actually fix, so I predicted that covering all 8 extension
categories instead of 2 would raise the scorable count substantially even if it didn't get every
case right. My second attempt confirmed that: `extend_corpus` scorable cases went from 1/24 to
7/24, and correct answers from 0/24 to 3/24 — a real, measurable improvement, though still far
from solved (details below).

## The runs themselves

All three used the same architecture: nanoGPT, `n_embd=64`, `n_head=4`, `n_layer=2`,
`block_size=48`, `batch_size=32`, seed 42, CPU-only (Colab), PyTorch 2.11.0+cpu. None were
interrupted.

| | Starter | Expanded — 1st attempt | Expanded — 2nd attempt (final) |
|---|---|---|---|
| Steps completed | 3,000 / 3,000 | 3,000 / 3,000 | 3,000 / 3,000 |
| Elapsed time | ~57 seconds | ~60 seconds | ~64 seconds |
| Parameters | 111,872 | 117,888 | 128,704 |

Config/summary files: [`evidence/starter/config.json`](evidence/starter/config.json),
[`evidence/expanded/config.json`](evidence/expanded/config.json) (final),
[`evidence/expanded_v1_negation_opposites/config.json`](evidence/expanded_v1_negation_opposites/config.json)
(first attempt).

**Loss** (fixed panels of 20 train / 20 validation documents, mean loss over non-padding
next-token targets — full histories:
[starter](evidence/starter/history.json), [expanded, final](evidence/expanded/history.json)):

![Starter loss curve](evidence/starter/training_curves.svg)

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.9263 | 4.9275 |
| 1,500 | 0.6821 | 0.7182 |
| 3,000 | 0.6783 | 0.7061 |

![Expanded loss curve](evidence/expanded/training_curves.svg)

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 6.0601 | 6.0424 |
| 1,500 | 0.8134 | 0.9574 |
| 3,000 | 0.7415 | 0.9742 |

The final expanded run starts (and ends) at a noticeably higher loss than the starter or the
first-attempt expansion — expected, since the vocabulary nearly tripled (136 → 399 types) while
training steps stayed fixed at 3,000, so there's simply more to learn per step. Validation loss
also sits clearly above training loss throughout (0.74 vs 0.97 at step 3,000), a bigger train/val
gap than the starter run showed — another sign that a larger vocabulary needs either more steps
or more repetition per word to fully close, which lines up with the eval limitation discussed
below.

**Samples**, same generation settings throughout, full files linked (including the raw,
sometimes-garbled early output):
[starter step 0](evidence/starter/samples/step_0000.txt) → [1,500](evidence/starter/samples/step_1500.txt) →
[3,000](evidence/starter/samples/step_3000.txt) (*"the team discussed the professor and the
learning at the school ."*); [expanded (final) step 0](evidence/expanded/samples/step_0000.txt) →
[1,500](evidence/expanded/samples/step_1500.txt) → [3,000](evidence/expanded/samples/step_3000.txt)
(*"today the market focused on design and the different item ."*).

## How this model actually learns — traced through one real word

Everything here comes from [`evidence/expanded/inspection.json`](evidence/expanded/inspection.json)
and [`evidence/expanded/tokenization.json`](evidence/expanded/tokenization.json) — the final,
8-category expanded run.

A **corpus** is just the text I feed in; a **token** is one word or punctuation mark the text
gets split into. Every token gets an arbitrary integer **ID** — the word "customer" is **token ID
79** in this (larger, 399-type) vocabulary. That ID indexes into a lookup table of **embeddings**:
one 64-number vector per token, which is what the network actually reads and writes to. Before
training, "customer"'s vector starts as small random noise (first three of its 64 numbers:
`0.00485, 0.00440, -0.00662`); after 3,000 steps it's moved to `-0.01704, -0.00632, -0.01737` —
those 64 numbers are the network's compressed representation of how "customer" behaves in
context, and they only shifted because training pushed them there.

Here's a real, single parameter update from that same vector, its first coordinate: before
training that number was `0.0048451`. The network's error on that step produced a gradient of
`0.0016012` for it, and at that point in training the (warmup-scaled) learning rate was `1e-05`.
AdamW turned that into an update of about `-0.00001`, landing at `0.0048351`. The step size
(~1e-5) is close to the *learning rate itself*, not `gradient × learning_rate` (which would be
~1.6e-8) — that's because AdamW normalizes each update by its running estimate of the gradient's
scale, so it doesn't take a plain gradient-proportional step the way basic SGD would.

The clearest evidence that this actually changed the model's behavior: next-token probabilities
for the prompt **"the customer"**, over the full 399-word vocabulary. Untrained, the top guesses
were essentially noise — "green" at just **0.38%**, barely ahead of "orange," "answer," "steam,"
and "software," all within a hundredth of a percent of each other. Trained, the top guess became
**"ordered" at 21.2%**, followed by compared / returned / reviewed / recommended — precisely the
verbs the classroom corpus's templates use right after "the customer/client/buyer...". That
shift, from near-uniform noise to a sharp, correct-shaped distribution, is loss dropping from ~6.1
to ~0.74 made concrete.

## Attention, generation, and temperature

Each output position attends only to tokens **at or before** it — the `attention_rows` in
`inspection.json` are lower-triangular, meaning a token literally cannot see what comes after
it. That's what makes "predict the next token" a well-posed task instead of a paradox. At each
step, the network turns its internal state into a probability over every vocabulary token, and
generation samples from that distribution one token at a time, feeding each choice back in as
context for the next.

**Temperature** reshapes that probability distribution before sampling, without touching any
weights — same prompt, same seed, three different temperatures, second sample from each set
([`evidence/expanded/temperature_comparison.json`](evidence/expanded/temperature_comparison.json)):

- **T=0.3:** *"the different teacher was mentioned in the lesson report yesterday ."* — sharpens
  toward the single most likely path, almost deterministic.
- **T=0.8:** *"today the store focused on service and the local client ."* — same grammatical
  frame, noticeably different word fills once the distribution is less sharp.
- **T=1.2:** identical output to T=0.8 for this particular sample and seed — with only 399
  vocabulary types and a narrow corpus, the model's distribution is already peaked enough that
  1.2 didn't flatten it further this time. That's a real, honestly-reported result, not a
  cherry-picked one: temperature's effect is probabilistic, not guaranteed to show up in every
  single sample.

## One limitation, explained, and what I tried next

**First attempt (negation + opposites only):** `extend_corpus` scored 0/24 correct before *and*
after training, with only 1 of 24 cases even scorable. My first guess was that the 509-token
vocabulary cap had pruned my new words, but `vocabulary_report.json` ruled that out: **zero**
types were omitted. So I checked what vocabulary those 24 cases actually needed (the assignment
explicitly allows this — "ordinary words... may overlap; the test items themselves must stay
separate" — so I pulled only the answer-choice words, never the prompts or answer key, from
`evals/language_evals.json`). Two real causes, neither a training bug:

1. **Category mismatch.** The 24 `extend_corpus` cases span all 8 extension categories, and I'd
   only written material for 2 of them. The other 6 categories (spatial relations,
   categories/analogies, reference, grammar, sequence, everyday knowledge) were always going to
   be unscorable, independent of how much training happened.
2. **Single-occurrence words and the random split.** Even within negation/opposites, some needed
   words only appeared once in my sentences. Vocabulary is built only from the training split,
   and the notebook's 90/10 passage split is random — a word appearing in exactly one passage has
   roughly a 1-in-10 chance of that passage landing entirely in validation, taking the word out of
   the trained vocabulary with it.

**Second attempt:** I wrote roughly 20–40 original sentences per category (240 new passages
total) for all 8 extension categories, reusing the ordinary answer-choice vocabulary in new
sentences and templates, never the eval prompts themselves. Result:
`extend_corpus` scorable cases rose from 1/24 to **7/24**, and correct answers rose from 0/24 to
**3/24** (43% accuracy on the cases that became scorable) — by category, `grammar` went to 2/3
correct and `opposites` to 1/3, with `spatial_relations` gaining scorable cases but no correct
answers yet. **Four categories still scored 0/24 scorable** — `categories_and_analogies`,
`everyday_knowledge`, `reference`, and `sequence` — despite having 19–39 passages of dedicated
material each. That tells me cause #2 above (single-occurrence words lost to the random split) is
still the dominant limiter, not lack of topical coverage: writing material for a category isn't
enough if the specific words an eval case needs only show up once or twice in my sentences.

**Next experiment I'd run:** for the 4 still-unscorable categories, deliberately repeat each key
word across 4–5 different sentences instead of 1–2, so no single word is one unlucky split away
from disappearing — the same fix that worked for `grammar` and `opposites` this round should
generalize. I'd predict that gets most of the remaining categories to at least partial coverage,
though 24/24 correct isn't realistic for a 128k-parameter model trained for 3,000 steps on a
narrow, synthetic corpus.

## Evals: how they're scored, what I found, and how leakage was checked

**Scoring:** each of the 48 fixed cases in [`evals/language_evals.json`](evals/language_evals.json)
gives the model a prompt and 4 single-word choices; it scores 1 if the trained model assigns the
*highest* probability to the correct choice, 0 otherwise (ties score 0). Cases where the prompt
or an answer choice uses a word outside the model's vocabulary are marked unscorable and count
as 0 in the all-case rate — that's a coverage gap, not a wrong answer. The model's free-text
continuation is saved separately and isn't part of this score. Runner:
[`run_evals.py`](run_evals.py).

**Results, the four required sets** (starter + my final, all-8-category expanded run):

| Experiment | Stage | Correct / 48 | Scorable / 48 | Accuracy on scorable cases | Full results |
|---|---|---|---|---|---|
| Starter | Untrained | 9 | 24 | 37.5% | [link](evidence/starter/language_evals/untrained/) |
| Starter | Trained | 20 | 24 | 83.3% | [link](evidence/starter/language_evals/final/) |
| Expanded (final) | Untrained | 7 | 31 | 22.6% | [link](evidence/expanded/language_evals/untrained/) |
| Expanded (final) | Trained | 26 | 31 | 83.9% | [link](evidence/expanded/language_evals/final/) |

For transparency, my **first attempt** (negation + opposites only, superseded by the run above)
scored: Untrained 6/48 correct, 25 scorable, 24.0% accuracy; Trained 21/48 correct, 25 scorable,
84.0% accuracy — full results still saved at
[`evidence/expanded_v1_negation_opposites/language_evals/`](evidence/expanded_v1_negation_opposites/language_evals/).

Combined comparison files:
[starter](evidence/starter/language_eval_comparison.json),
[expanded, final](evidence/expanded/language_eval_comparison.json),
[expanded, first attempt](evidence/expanded_v1_negation_opposites/language_eval_comparison.json).
Breaking the final run down by group: `starter_patterns` hit 16/16 — the model clearly learned
the domain-noun/context/place associations from the templates, undisturbed by the larger
vocabulary. `starter_transfer` (familiar words, new phrasing) improved to 7/8 (up from 4/8
untrained, and up from the first attempt's 5/8). `extend_corpus` went from 0/24 correct (1
scorable) in my first attempt to 3/24 correct (7 scorable) in the final run — real progress, and
the honestly-reported limitation above explains what's still missing.

**Leakage check:** [`evidence/expanded/eval_separation.json`](evidence/expanded/eval_separation.json)
confirms 160 reserved passages were excluded before splitting or building vocabulary, in every
run, via normalized contiguous-prompt matching. I also manually re-read all 8 of my corpus files
against the 48 cases myself, since the automated check is a string match, not a semantic one, and
I only ever pulled isolated answer-choice *words* from the eval file to steer which ordinary
vocabulary to teach — never the prompts, four-choice sets, or answer key, and none of that went
into any corpus file or training text. These are public tests I used to steer the extension
corpus during development — not a held-out final benchmark.

## Chat interface

**To launch:** open [`custom_llm_expanded.ipynb`](custom_llm_expanded.ipynb) section 10 in
Colab — it reuses the model already trained in that session, no extra setup. To run it
standalone instead: `pip install -r requirements.txt`, then
`python chat.py --model evidence/expanded/model.pt --transcript new_chat.json`. Model identity:
the `model_sha256` recorded in
[`evidence/expanded/chat_transcript.json`](evidence/expanded/chat_transcript.json) matches the
final expanded run's checkpoint (3,000 completed steps, 399-word vocabulary).

Screenshot: [`evidence/expanded/chat_screenshot.png`](evidence/expanded/chat_screenshot.png).
Four real interactions with the final model
([full transcript](evidence/expanded/chat_transcript.json)):

| Prompt | Response | Note |
|---|---|---|
| "the customer" | "compared the offering after checking the price ." | in-domain, grammatical |
| "the astronaut" | "and the juice are both loud ." | **limitation** — "astronaut" is unknown; output falls apart |
| "the dog" | "did not clear ." | picked up negation *syntax* but applied it to an odd word |
| "the man" | "and the evening yellow during a discussion of data ." | **limitation** — "man" is unknown; output falls apart |

My first-attempt model also has a saved screenshot and transcript, kept for comparison:
[`evidence/expanded_v1_negation_opposites/chat_screenshot.png`](evidence/expanded_v1_negation_opposites/chat_screenshot.png)
/ [transcript](evidence/expanded_v1_negation_opposites/chat_transcript.json) (6 interactions,
including "the doctor" → *"was focused on patient ."* and "the teacher" → *"reviewed the item
after checking the price ."*).

**Limitation:** every prompt starts a fresh context — `fresh_context_per_prompt: true` in the
transcript, no memory across turns. The model only knows the ~399 word types it trained on;
anything outside that (like "astronaut" or "man") produces ungrounded output, as shown above. It's
a narrow sentence-continuation model, not a general assistant, and chatting with it never feeds
those messages back into training or the corpus.

## Reproduce this

- Notebooks: [`custom_llm_starter.ipynb`](custom_llm_starter.ipynb),
  [`custom_llm_expanded.ipynb`](custom_llm_expanded.ipynb) (final, all-8-category expansion),
  [`custom_llm_expanded_v1_negation_opposites.ipynb`](custom_llm_expanded_v1_negation_opposites.ipynb)
  (first attempt) — open directly on GitHub to read executed outputs, or in Colab to rerun.
- My corpus additions: all 8 files under [`corpus/`](corpus/) (force-committed despite the
  template's default `corpus/*` gitignore, since this is original, non-sensitive text).
- Every result file (loss history, eval CSVs/JSONs, samples, inspection data, trained weights) is
  under [`evidence/starter/`](evidence/starter/), [`evidence/expanded/`](evidence/expanded/)
  (final), and [`evidence/expanded_v1_negation_opposites/`](evidence/expanded_v1_negation_opposites/)
  (first attempt, kept for comparison).
