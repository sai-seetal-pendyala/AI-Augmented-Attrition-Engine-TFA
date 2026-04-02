# Methodology

## Analytical approach

The core question -- what predicts applicant withdrawal -- could be framed as a single classification problem. We deliberately split it into three parallel approaches instead. Behavioral-only, academic-only, and combined models let us isolate which dimension of an applicant's profile actually carries predictive signal, rather than throwing everything into one model and hoping the algorithm sorts it out.

Each approach runs through the same five algorithms (KNN, Naive Bayes, C5.0 Decision Trees, ANN, SVM with radial kernel) with the same preprocessing, the same class balancing method, and the same 80/20 stratified train-test split. The only variable that changes across approaches is the input feature set. This controlled comparison is what makes the results interpretable: when Combined SVM outperforms Behavioral SVM, we can attribute that to the additional academic features, not to a different split or different tuning strategy.

## Model selection rationale

| Model | Why included | Strengths | Weaknesses |
|:------|:-------------|:----------|:-----------|
| KNN | Simple distance-based baseline | No assumptions about data distribution | Struggles with high-dimensional dummy-encoded features |
| Naive Bayes | Probabilistic baseline | Fast, interpretable probabilities | Independence assumption rarely holds in practice |
| C5.0 Decision Tree | Rule-based, interpretable | Produces human-readable splits, handles boosting | Prone to overfitting without pruning |
| ANN (nnet) | Captures nonlinear patterns | Flexible decision boundaries | Harder to interpret, sensitive to architecture |
| SVM (radial) | Strong margin-based classifier | Handles nonlinear separation well | Computationally expensive, less interpretable |

The Combined SVM emerged as the champion not because it had the highest raw accuracy (that was Decision Trees at 67.9%) but because it provided the most balanced sensitivity-specificity tradeoff. In a real TFA deployment, a model that catches withdrawals (specificity) matters more than one that confirms completers (sensitivity). Decision Trees achieved 74.1% sensitivity but only 41.6% specificity -- it would tell TFA "this person will complete" correctly most of the time but miss the majority of actual dropouts.

## Assumptions and limitations

**Class imbalance.** The original dataset is 80.7% completers vs 19.3% withdrawals. We used ROSE hybrid sampling (oversample minority + undersample majority) to reach 50/50 on training data only. Synthetic sampling creates observations that may not represent real applicant behavior, and the resulting models may overestimate the data's separability.

**Missing context.** The dataset captures what applicants do on paper (dates, essays, GPA) but not why they do it. Personal circumstances, financial constraints, competing job offers, family obligations -- none of these are represented. A recruiter talking to an applicant for five minutes would know things no model can infer from timestamps.

**Feature engineering limits.** Timing variables (Days to Start, Days to Submit, Deadline Gap) capture pace but not intent. An applicant who starts late because they're deliberate looks identical to one who starts late because they're losing interest.

**No cost-weighted evaluation.** All errors are treated equally in our metrics. In practice, missing a high-risk applicant (false negative) costs TFA more than flagging a low-risk one (false positive). A cost-sensitive evaluation would likely shift the optimal model threshold.

**No temporal validation.** The dataset was not split by application year, so we cannot confirm that patterns from one cycle generalize to the next. TFA's applicant pool composition shifted year to year as the macro economy changed.

## With more time and data

Three extensions would materially improve this work:

**Recruiter interaction data.** Number of touchpoints, response times to emails, engagement with TFA content -- these behavioral signals are richer than application timestamps and would likely improve withdrawal prediction substantially.

**Temporal holdout validation.** Train on 2013-2015, validate on 2016. This would test whether the model generalizes across admission cycles, not just across a random split within one cycle.

**Cost-sensitive modeling.** Assign asymmetric misclassification costs (false negatives weighted higher than false positives) and retune the threshold. This would produce models optimized for TFA's actual operational priorities rather than balanced accuracy.
