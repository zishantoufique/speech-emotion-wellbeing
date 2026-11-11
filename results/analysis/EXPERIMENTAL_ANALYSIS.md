# Phase 4: Generalisation and Experimental Analysis

## Evaluation design

The primary comparison is speaker-independent. Every model was evaluated on exactly 240 RAVDESS recordings from actors 01, 09, 17, and 19. These actors were absent from the 16-actor training partition and four-actor validation partition. This guards against speaker leakage.

Ordinary random record-level evaluation was intentionally not run: it can put recordings from the same actor into training and test partitions and is therefore not a valid estimate of unseen-speaker performance.

## Held-out comparison

| model | accuracy | precision_weighted | recall_weighted | f1_weighted | test_recordings | unseen_test_actors |
| --- | --- | --- | --- | --- | --- | --- |
| SVM | 0.3583 | 0.4314 | 0.3583 | 0.3387 | 240 | 1, 9, 17, 19 |
| Random Forest | 0.3458 | 0.3449 | 0.3458 | 0.3131 | 240 | 1, 9, 17, 19 |
| KNN | 0.3167 | 0.3251 | 0.3167 | 0.3095 | 240 | 1, 9, 17, 19 |
| CNN | 0.1667 | 0.0757 | 0.1667 | 0.0749 | 240 | 1, 9, 17, 19 |

By weighted F1, **SVM** was best (0.3387) and **CNN** was weakest (0.0749) on unseen speakers.

The CNN was trained from scratch on log-Mel spectrograms for six CPU epochs; its result should be interpreted as a compact baseline rather than an optimized deep-learning benchmark. The classical models use fixed-size summaries of acoustic features. All results are about acted RAVDESS labels, not wellbeing or clinical states.

## Emotion-specific errors

- **SVM:** calm → sad (12); happy → sad (9); angry → disgust (8).
- **Random Forest:** sad → calm (14); neutral → calm (12); surprised → calm (10).
- **KNN:** calm → sad (15); neutral → calm (10); disgust → sad (9).
- **CNN:** sad → calm (32); surprised → calm (31); happy → calm (27).

Per-emotion precision, recall, and F1 are stored in `emotion_level_metrics.csv`; confusion matrices should be used together with those scores because small class supports, especially neutral, can make a single metric unstable.

## Grouped cross-validation

A six-fold GroupKFold evaluation is provided for the three classical models. Each fold holds out whole actors. It is supplementary: hyperparameters remain the configurations selected during Phase 2, so the fixed 4-actor test comparison above remains the primary four-model comparison.

## Cross-corpus generalisation

**Status: not performed.** No separately labelled compatible external emotion-speech corpus is present in data/external; the project contains only RAVDESS.

A valid cross-corpus experiment requires a separately licensed corpus with compatible emotion labels and speaker identifiers. It must be kept external to RAVDESS model selection and reported as a distinct evaluation; no cross-corpus result is claimed here.

## Phase 5 preparation

Use the saved figures, tables, methodology statement, and ethical limitations to assemble the final report and Streamlit presentation. Discuss cultural and language scope (acted North American English), actor/sample limits, label ambiguity, and the unsuitability of these models for hiring, education, monitoring, or diagnosis.