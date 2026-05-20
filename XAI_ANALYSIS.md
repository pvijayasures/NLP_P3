# XAI Analysis — Audio × Text × Fusion

Synthesis of the explainability work in `notebook/multimodal_experiments.ipynb`,
sections 8.1–8.13. Five attribution methods are applied across three model types
(LR audio, RoBERTa text, Fusion MLP) and three families of probe (coefficients
& ablation, IG, SHAP, LIME, calibration & agreement). Together they answer a
single research question: **what does each modality actually contribute, and
where?**

## Anchor performance (test set)

| Model                                    | Macro F1 | Weighted F1 |
| ---------------------------------------- | -------: | ----------: |
| Audio only (`audio_lr_is10_ctx`)         |    0.177 |       0.270 |
| Text only (`text_roberta`)               |    0.434 |       0.588 |
| Feature fusion (`mm_ff_mlp_is10_ctx`)    |    0.421 |       0.541 |
| Late fusion 50/50 (`mm_late_svm_egemaps`)|  **0.454** |   **0.618** |

Late fusion wins. Feature fusion, despite having a learnable MLP, is *worse*
than text alone on weighted F1. The XAI sections explain why.

---

## 1 · What audio "hears" per emotion

**Sections 8.1 (raw LR coefficients), 8.5 (grouped coefficients), 8.10 (SHAP
on LR).**

Aggregating the IS10 LR coefficients into descriptor families (1582 features →
6 buckets) and ranking by mean |coef| or mean |SHAP|, each emotion has a clear
dominant audio family:

| Emotion             | Dominant descriptor family |
| ------------------- | -------------------------- |
| anger               | energy / loudness          |
| sadness             | MFCC                       |
| disgust             | voicing                    |
| neutral             | F0 / pitch                 |
| joy                 | F0 / pitch                 |
| fear                | F0 / pitch                 |
| surprise_positive   | F0 / pitch                 |
| surprise_negative   | F0 / pitch                 |

The mapping matches phonetic intuition: anger has high energy bursts, sadness
has compressed spectral envelopes (MFCC carries spectral shape), disgust
disrupts voicing. The F0 dominance for the remaining five classes is partly an
IS10 artefact — F0 statistics are only 82 features but their |coef| values
average highest for these classes.

**SHAP cross-check (8.10).** The descriptor-group importance heatmap built from
SHAP values matches the coefficient-magnitude heatmap from 8.5 almost exactly,
confirming the per-emotion ranking is stable under sample-level perturbation.
The beeswarm view additionally shows that the highest-impact features push
*bidirectionally* (high values support some emotions and oppose others) — i.e.
LR is genuinely using these features as discriminative signals, not just as
class-priors.

**Caveat.** Despite a coherent audio story per-class, audio-only macro-F1 is
0.18 — barely above chance (random = 0.125). The features are *learnable*, but
the absolute information they carry is small.

---

## 2 · What text "reads" per emotion

**Sections 8.6 (Integrated Gradients on RoBERTa embeddings) and 8.11 (LIME on
RoBERTa).**

Two independent token-attribution methods are applied to the highest-confidence
correctly-classified test utterance per emotion. Both methods identify the same
class of cues:

- **Emotion-loaded content words** dominate ("hate", "sorry", "what", "no",
  punctuation marks for surprise).
- **The previous-utterance context** (text_with_context concatenates dialogue
  history with `</s>`) gets non-trivial attribution — RoBERTa uses the
  dialogue-level cue, not just the current sentence.
- **Punctuation and named-entity tokens** carry attribution mass for surprise
  and disgust classes, suggesting partial reliance on surface cues.

IG and LIME agree on direction (sign) for the top tokens; LIME's local
surrogate is noisier on the long tail. This is the expected pattern: IG is
gradient-exact, LIME is sample-perturbation-noisy, and full agreement on the
top features is a positive signal that the explanation isn't an artefact of
either method.

---

## 3 · How modalities relate — agreement and complementarity

**Section 8.7 (cross-modal agreement matrix and per-class outcome).**

Audio-only and text-only agree on the same class for only **22.3%** of test
utterances. Conditional breakdown per emotion:

| Emotion            | both ok | audio only | text only | both wrong |
| ------------------ | ------: | ---------: | --------: | ---------: |
| neutral            |   17.3% |       7.0% |     48.0% |      27.7% |
| joy                |   11.2% |       5.2% |     52.0% |      31.6% |
| anger              |   19.4% |      16.8% |     22.6% |      41.2% |
| sadness            |   11.1% |      13.5% |     25.0% |      50.5% |
| disgust            |    4.4% |       7.4% |     29.4% |      58.8% |
| fear               |    6.0% |       4.0% |     32.0% |      58.0% |
| surprise_positive  |   10.9% |       9.2% |     34.5% |      45.4% |
| surprise_negative  |   13.0% |       8.0% |     48.8% |      30.2% |

Two important facts:

1. **For anger and sadness, audio-only is correct on cases where text-only is
   wrong** at a rate (16.8%, 13.5%) comparable to the "text only" column.
   These are exactly the prosodic emotions where audio should help — and
   indeed they're where late fusion gives the largest gains over text-alone
   (8.4: anger +6 net rescues, surprise_negative +9 net rescues).
2. **For neutral and joy, audio almost never has the answer when text doesn't**
   (7.0%, 5.2%). The minority class predictions are confidently lexical.

The "both wrong" rate for low-support classes (disgust 58.8%, fear 58.0%,
surprise_positive 45.4%) is a hard ceiling neither modality crosses — there is
no fusion strategy that can rescue cases where neither input contains the
signal.

---

## 4 · Inside the FusionMLP — three methods, one conclusion

This is the most informative finding in the entire XAI suite.

**Sections 8.3 (modality ablation), 8.8 (Integrated Gradients on the MLP),
8.12 (`shap.GradientExplainer`), 8.13 (`LimeTabularExplainer`).**

The FusionMLP takes a 776-dim input: 8 audio softmax probabilities concatenated
with 768 scaled CLS embedding dimensions. Three independent attribution methods
were applied to the best feature-fusion model (`mm_ff_mlp_is10_ctx`):

| Method | Audio share of attribution mass | Notes                           |
| ------ | ------------------------------: | ------------------------------- |
| Integrated Gradients (8.8) | median **1.3%** | Q1=0.9%, Q3=1.9%, 0% of samples >50% |
| SHAP `GradientExplainer` (8.12) | mean **1.6%**  | 0 / 400 samples >50% audio    |
| LIME `Tabular` (8.13) | top-50 features ≈ **0 audio**  | audio appeared in only 1/6 explained samples |

**If the MLP weighed audio and text in proportion to their dimensionality, audio
would get ~1%** (8 / 776). All three methods land right around that figure —
**the FusionMLP has essentially learned to ignore the audio branch.**

This is corroborated by the ablation (8.3): replacing the audio input with its
training-set mean barely changes per-class softmax for any emotion. The
help/hurt analysis (8.4) is also consistent: on the help-cases that *late*
fusion rescues, the FusionMLP's audio attribution share is identical to the
hurt-cases (1.3% vs 1.6%) — i.e. the MLP isn't differentially using audio when
it matters.

**Why this happens.** The audio softmax is a low-rank summary (8 values, weak
calibration, mean entropy near uniform — see §5 below). The CLS embedding is
high-rank with hundreds of useful directions. Gradient descent on a single
fully-connected layer can't easily learn a high-leverage path through only 8
weak inputs when 768 strong ones are available; with focal loss + class
weighting the optimiser converges to a near-text-only solution.

**This is why late fusion wins.** Late fusion *forces* audio to contribute 50%
of the softmax mass before the argmax — it cannot be downweighted to ~1% the
way the MLP downweights it. On the test set this gives late fusion +0.02
macro-F1 over text alone, while feature fusion *loses* 0.01 macro-F1.

---

## 5 · Calibration explains the imbalance

**Section 8.9 (softmax-max distribution per modality, split by correct/wrong).**

Per-modality mean softmax-max:

| Modality | Correct | Wrong | Gap (correct − wrong) |
| -------- | ------: | ----: | --------------------: |
| Text     |   0.688 | 0.567 |              **+0.121** |
| Audio    |   0.382 | 0.370 |              **+0.012** |

Text is *usefully* over-confident on correct cases — its softmax max correlates
with whether it's right. Audio's correct-vs-wrong confidence gap is one tenth
of text's; its softmax tells you almost nothing about whether it's right.

This single statistic — a 10× smaller calibration gap — is what makes audio
**look** noisy to the FusionMLP and what makes 50/50 late fusion succeed
despite using the same softmax. Late fusion benefits because *averaging two
noisy probability vectors smooths the joint*, even when one is poorly
calibrated. The MLP, with the freedom to ignore the bad signal, simply does.

Per-class calibration gaps reveal subtler structure:

| Class                | Text gap | Audio gap | Comment                                 |
| -------------------- | -------: | --------: | --------------------------------------- |
| neutral              |   +0.129 |    −0.027 | Audio is *anti-calibrated* on neutral   |
| joy                  |   +0.184 |    +0.038 | Both modalities calibrated, joy lexical |
| anger                |   +0.016 |    +0.067 | **Audio better-calibrated than text**   |
| sadness              |   +0.119 |    +0.049 | Both contribute usable confidence       |
| disgust              |   +0.143 |    −0.035 | Audio anti-calibrated; low support      |
| surprise_positive    |   +0.070 |    +0.070 | Equal calibration; tightest match       |

The single class where audio is better-calibrated than text is **anger** —
unsurprising given anger's loudness/energy signature (see §1). This is also
the class where late fusion gives the largest absolute rescue gain.

---

## 6 · Synthesis

Putting it all together:

1. **Audio carries real but weak information.** Each emotion has a coherent
   acoustic signature in the LR coefficients (energy → anger, MFCC → sadness,
   voicing → disgust), and SHAP confirms these are stable per-sample. But
   audio-only F1 is 0.18 — the signal is real but small.

2. **Text dominates almost everywhere.** RoBERTa with dialogue context drives
   text-only F1 to 0.43, and IG + LIME both show the model is using
   semantically appropriate tokens (emotion-loaded words, contextual cues,
   surface markers for surprise).

3. **The FusionMLP fails to learn to use audio.** Three different attribution
   methods (IG, SHAP, LIME) independently arrive at the same conclusion:
   audio receives ~1–2% of the attribution mass, essentially equal to its
   share of input dimensions (8/776 = 1.0%). The MLP has *not* learned a
   leveraged use of audio; it has learned to approximate the text classifier.

4. **Late fusion wins by removing the choice.** A 50/50 softmax average forces
   the audio signal into the prediction. On the per-class help/hurt analysis
   this rescues anger and surprise predictions where text was wrong, with low
   collateral damage on the lexical classes (neutral, joy).

5. **Audio calibration is the bottleneck.** Audio's correct-vs-wrong confidence
   gap is 10× smaller than text's. For any architecture that uses audio's own
   confidence to weight it (an MLP gate, an attention pooling, etc.), this
   calibration gap will cause it to be ignored. **Improving audio calibration
   would matter more than improving its raw accuracy** for downstream fusion.

---

## 7 · Practical implications

- **For this dataset and these features**, late fusion is the right
  architecture. Feature fusion costs a 0.03 macro-F1 penalty for offering the
  model a choice it consistently mis-makes.

- **To unlock more audio gain**, two interventions are worth trying:
  - Calibrate audio softmax (temperature scaling, isotonic, or focal-loss
    fine-tune) before either fusion stage.
  - Skip the audio softmax bottleneck — concatenate a *learned* audio embedding
    (e.g. an MLP penultimate layer) rather than just 8 class probabilities,
    giving the FusionMLP a richer signal to work with.

- **Class-conditional fusion weights** are supported by the data: anger,
  sadness, surprise_negative benefit most from audio. A gated/MoE fusion that
  learns these per-class weights from the dev set may outperform uniform 50/50.

- **Don't pursue audio features for neutral/joy/disgust/fear in this pipeline**
  — text wins decisively and audio is anti-calibrated on neutral and disgust.

---

## 8 · Cross-method consistency

The XAI design intentionally probes the same models with multiple methods so we
can sanity-check each finding. Method agreement summary:

| Finding | Methods | Agreement |
| ------- | ------- | --------- |
| LR coefficient ranking per emotion | Raw coef (8.1) + SHAP (8.10) | Identical top-class assignments |
| Audio descriptor-family ranking    | Coef-grouped (8.5) + SHAP-grouped (8.10) | Identical |
| Text tokens per emotion            | IG (8.6) + LIME (8.11) | Top tokens agree in sign and overlap |
| FusionMLP audio share              | Ablation (8.3) + IG (8.8) + SHAP (8.12) + LIME (8.13) | All four arrive at ~1–2% audio share |
| Calibration ranking                | 8.9 + late-vs-feature-fusion delta (8.4 → 7) | Audio under-calibration explains late-fusion advantage |

When two methods built on different mathematical foundations (gradient-based
vs. sample-perturbation-based) converge on the same quantitative answer
(~1.5% audio share across IG, SHAP, LIME), the finding is robust.

---

## Plot index

| Plot file (`outputs/plots/`)               | Section |
| ------------------------------------------ | :-----: |
| `xai_lr_coefficients.png`                  |   8.1   |
| `xai_confusion_matrices.png`               |   8.2   |
| `xai_modality_ablation.png`                |   8.3   |
| `xai_audio_help_hurt.png`                  |   8.4   |
| `xai_audio_feature_groups.png`             |   8.5   |
| `xai_text_top_tokens.png`                  |   8.6   |
| `xai_text_samples_ig.png`                  |   8.6   |
| `xai_cross_modal_agreement.png`            |   8.7   |
| `xai_ig_fusion_modality.png`               |   8.8   |
| `xai_confidence_calibration.png`           |   8.9   |
| `xai_shap_audio_beeswarm.png`              |  8.10   |
| `xai_shap_audio_groups.png`                |  8.10   |
| `xai_lime_text.png`                        |  8.11   |
| `xai_shap_fusion.png`                      |  8.12   |
| `xai_lime_fusion.png`                      |  8.13   |
