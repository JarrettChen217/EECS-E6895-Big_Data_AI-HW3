# EECS 6895 Assignment 3 — Code Explanation

This document explains each TODO implemented in `EECS6895_Assignment3_Student.ipynb`.

---

## Problem 1: CRIT Socratic Reasoning

### TODO 1: `build_single_shot_prompt(passage)`

Builds a **single-step prompt** that asks the model to analyze a passage all at once. The prompt embeds the passage text and instructs the model to return a single JSON object containing:
- `main_claim`: the central conclusion
- `supporting_reasons`: 2–3 reasons backing the claim
- `evidence`: one piece of evidence per reason
- `counterargument`: one opposing argument

The prompt explicitly specifies the JSON schema and instructs the model to output valid JSON only, which ensures compatibility with `extract_first_json_block`.

---

### TODO 2: `build_definition_prompt(passage)`

Implements the **Definition stage** of the CRIT pipeline. This is the first structured step: the model reads the passage and identifies only the main claim and its supporting reasons, without yet assessing evidence quality or generating counterarguments. Separating this step from the full analysis makes the claim identification more focused and reliable.

---

### TODO 3: `build_elenchus_prompt(passage, reason)`

Implements the **Elenchus (cross-examination) stage**. Given a single supporting reason extracted during Definition, this prompt asks the model to:
- Find the evidence in the passage that supports it (`evidence_text`)
- Classify the evidence type (empirical, example, expert_opinion, conceptual_reasoning, none)
- Score the evidence's `validity_score` (logical soundness, 1–5) and `credibility_score` (source trustworthiness, 1–5)

This is called once per reason, so each reason gets its own evidence assessment.

---

### TODO 4: `build_dialectic_prompt(passage, reason)`

Implements the **Dialectic stage**. Given a reason, the model generates a counterargument that directly challenges it and rates how strong that counterargument is (`counter_strength`, 1–5). This simulates adversarial reasoning to stress-test each supporting reason.

---

### TODO 5: `compute_final_score(elenchus_outputs, dialectic_outputs)`

**Deterministic score aggregation** across all elenchus and dialectic outputs for a passage. The formula:

1. Average `validity_score` and `credibility_score` across all reasons → `evidence_quality`
2. Average `counter_strength` across all dialectic outputs → `avg_counter`
3. Final score = `evidence_quality − 0.3 × (avg_counter − 3)`, clamped to [1, 5]

This penalizes arguments whose reasons face strong counterarguments, while rewarding arguments with high-quality, credible evidence. The penalty factor (0.3) is tuned so that a maximally strong counter (5) reduces the score by 0.6, a meaningful but not overwhelming deduction.

---

### TODO 6: `build_reasoning_comparison_table(passages)`

Runs the **full single-shot and staged CRIT pipelines** on both passages and assembles a comparison DataFrame. For each passage it:
1. Runs `run_single_shot` → captures the single-shot claim
2. Runs `run_definition` → captures the staged claim and the list of reasons
3. Runs `run_elenchus` and `run_dialectic` on each reason
4. Calls `compute_final_score` to produce the aggregate quality score

Columns: `title`, `single_shot_claim`, `staged_claim`, `num_reasons`, `final_score`.

---

### TODO 7: `build_breadth_prompt(topic_text)`

Generates **4 broad exploratory questions** covering different dimensions of the topic. This is the first half of the breadth-to-depth Socratic dialogue: starting wide before narrowing down. The prompt instructs the model to produce exactly 4 questions in the `broad_questions` array.

---

### TODO 8: `build_depth_prompt(topic_text, selected_question)`

Given one broad question selected by `FOCUS_INDEX`, generates **3 deeper follow-up questions** that drill further into that specific thread, plus a `focused_answer` of 1–3 sentences. This models how a Socratic dialogue naturally narrows from exploration to focused inquiry.

---

### TODO 9: `build_contentiousness_prompt(topic_text, contentiousness_level)`

Generates a response to the topic with a **controlled tone**:
- `"low"` contentiousness: calm, collaborative, neutral, diplomatic
- `"high"` contentiousness: bold, assertive, challenging, emphatic

The level-specific style instruction is injected into the prompt so the model modulates its language accordingly. The output JSON contains `contentiousness` (echoing the level) and `response` (the generated text), enabling direct comparison of tone, emphasis, and language between the two outputs.

---

## Problem 2: Big Five Personality Classification

### TODO 10: `summarize_split(name, split_df)`

Returns a statistics dictionary for a dataset split with columns:
- `split`: name of the split (train/val/test)
- `num_examples`: number of rows
- `avg_chars`: average character length of essays
- `O/C/E/A/N_positive_rate`: fraction of examples labeled positive for each Big Five trait

This gives a quick view of class balance (important for interpreting classifier performance) and data volume across splits.

---

### TODO 11: `build_tfidf_features(train_texts, val_texts, test_texts)`

Builds **TF-IDF bag-of-words features** from raw essay text using `TfidfVectorizer`:
- `max_features=10000`: keeps the 10,000 most frequent unigrams and bigrams
- `sublinear_tf=True`: log-scales term frequencies to reduce the dominance of very common words
- `min_df=2`: ignores terms appearing in fewer than 2 documents (noise reduction)
- `ngram_range=(1, 2)`: includes both single words and two-word phrases

The vectorizer is fit on training data only; val and test are transformed with the learned vocabulary to prevent data leakage.

---

### TODO 12: `build_empath_features(texts)`

Builds **psycholinguistic category features** using the Empath lexicon. For each essay, `lexicon.analyze(text, categories=EMPATH_CATEGORIES, normalize=True)` returns normalized scores (word count in category / total words) for the 10 predefined categories:

`achievement`, `social_media`, `friends`, `work`, `positive_emotion`, `negative_emotion`, `help`, `communication`, `reading`, `thinking`

The scores are assembled into a numpy array of shape `(n_samples, 10)` and then z-score normalized using a `StandardScaler` fit on training data.

---

### TODO 13: `fit_multioutput_logistic(X_train, y_train)`

Trains a **multi-output logistic regression** classifier using `MultiOutputClassifier`, which wraps a separate `LogisticRegression` for each of the 5 Big Five traits (O, C, E, A, N). Each binary classifier is trained independently. Settings:
- `max_iter=1000`: ensures convergence
- `C=1.0`: standard L2 regularization
- `solver="lbfgs"`: efficient for dense and moderately sparse data
- `n_jobs=-1`: trains the 5 classifiers in parallel

This function is called three times to train the TF-IDF-only, Empath-only, and Combined models.

---

### TODO 14: `evaluate_model(model, X_test, y_test, model_name)`

Evaluates a trained multi-output model and returns a DataFrame with one row per trait plus a `mean` row. Metrics reported:
- **Accuracy**: fraction of correct binary predictions per trait
- **F1-score**: harmonic mean of precision and recall (handles class imbalance better than accuracy)
- **ROC-AUC**: area under the ROC curve using predicted probabilities; `safe_auc` guards against single-class splits returning 0.5

Probabilities are extracted from each sub-estimator via `predict_proba`.

---

### TODO 15: Ablation Study

Performs a **leave-one-out ablation** over the 10 Empath categories. For each category, a new Empath-only model is trained with that category's column removed, then evaluated on the test set. The mean Accuracy, F1, and ROC-AUC across all 5 traits are reported for each ablated model.

Comparing the ablated scores to the baseline Empath model reveals which category contributes most to predictive power: if dropping a category causes a large drop in performance, that feature is important; if scores stay the same or improve, the category was redundant or noisy.
