# Speech Emotion & Wellbeing Signal Recognition System

An introductory AI university project that classifies **acted speech emotion labels** from short recordings. It compares engineered acoustic-feature models with a compact CNN trained from scratch on log-Mel spectrograms using speaker-independent evaluation.

> Educational scope: this is **not** a mental-health, wellbeing, or internal-emotion detector. It must not be used for recruitment, education assessment, workplace monitoring, surveillance, diagnosis, or decisions about people.

## Original assignment objectives

The project identifies emotion labels from acoustic speech features—MFCCs, pitch, energy, zero-crossing rate, and spectral properties—then compares classical classifiers with a spectrogram CNN. It evaluates generalisation to unseen speakers, analyses errors, and documents cultural, language, and high-impact-use risks.

## Dataset

The project uses the audio-only **speech** portion of [RAVDESS on Zenodo](https://zenodo.org/records/1188976), under its CC BY-NC-SA 4.0 terms. The official speech archive contains 1,440 WAV recordings from 24 professional actors with eight acted labels: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

RAVDESS filenames use `03-01-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR.wav`. For example, `03-01-06-01-02-01-12.wav` means audio-only speech, fearful, normal intensity, statement 02, repetition 01, actor 12.

## Project architecture

```text
speech-emotion-wellbeing/
├── app.py                         # Streamlit inference and results dashboard
├── requirements.txt               # Phase 1/2/4 + Streamlit environment
├── requirements-cnn.txt           # Separate TensorFlow CPU environment
├── src/
│   ├── download_dataset.py         # Official RAVDESS download/extraction
│   ├── run_preprocessing.py        # Validation and acoustic features
│   ├── train_classical_models.py   # SVM, Random Forest, KNN
│   ├── train_cnn.py                # Log-Mel CNN training
│   ├── predictors.py               # Saved-model inference used by the app
│   ├── run_grouped_speaker_cv.py   # Grouped speaker CV
│   └── run_experimental_analysis.py# Phase 4 tables, figures, report
├── tests/                          # Unit, integration, app, and artifact checks
├── data/                           # Local only; excluded from Git
├── models/                         # Local trained artifacts; excluded from Git
└── results/                        # Local generated artifacts; excluded from Git
```

## Installation (Windows 11 / Python 3.13)

The project deliberately uses two isolated environments. The normal environment supports preprocessing, classical ML, analysis, and Streamlit. The CNN environment adds TensorFlow and Streamlit without altering the normal environment.

```powershell
cd "speech-emotion-wellbeing"
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CNN training or CNN inference in Streamlit:

```powershell
py -3.13 -m venv .venvcnn
.\.venvcnn\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cnn.txt
```

TensorFlow 2.21 supports Python 3.13 on Windows CPU. Native Windows TensorFlow does not use GPU acceleration in current releases; use WSL2 only if GPU training is required.

## Data preparation

```powershell
.\.venv313\Scripts\Activate.ps1
python -m src.download_dataset
python -m src.run_preprocessing
```

If automatic download fails, download `Audio_Speech_Actors_01-24.zip` from the official Zenodo page, place it in `data/raw/`, then rerun `python -m src.download_dataset`.

Raw WAV files remain untouched. Validation rejects malformed names, unreadable audio, empty files, and non-speech records. Feature outputs are stored as one compressed `.npz` file per recording plus metadata and a dataset summary.

## Acoustic features

- **MFCCs (13):** compact spectral-envelope descriptors commonly used for speech.
- **Pitch:** voiced fundamental-frequency estimate.
- **RMS energy:** loudness proxy per frame.
- **Zero-crossing rate:** signal sign-change rate, related to noisiness/high-frequency content.
- **Spectral centroid:** brightness centre of mass.
- **Spectral rolloff:** frequency below which most spectral energy lies.
- **Spectral bandwidth:** spread around the spectral centroid.

For classical models, every frame-level feature is summarised with mean, standard deviation, minimum, and maximum (plus voiced-pitch fraction), producing 77 fixed features per recording.

## Classical ML methodology

```powershell
.\.venv313\Scripts\Activate.ps1
python -m src.train_classical_models --seed 42
```

The split is actor-disjoint: 16 actors / 960 recordings for training, 4 actors / 240 recordings for validation, and 4 actors / 240 recordings for the untouched test set. SVM and KNN use `StandardScaler` inside their pipelines, fitted only on training data. Random Forest does not require scaling. Small validation-only parameter grids select model settings before the final refit on training plus validation actors.

## CNN methodology

```powershell
.\.venvcnn\Scripts\Activate.ps1
python -m src.train_cnn --epochs 20 --batch-size 16 --seed 42
```

The CNN is trained from scratch—no pretrained emotion model or external API. Audio is converted to mono at 22,050 Hz, silence-trimmed, fixed to four seconds, converted to a 64-band log-Mel spectrogram, then per-recording z-score normalized. Its simple architecture has three convolution blocks, max pooling, global average pooling, a dense ReLU layer, dropout, and softmax output. Early stopping and best-checkpoint saving are enabled.

If the CPU session is bounded, resume from the saved checkpoint:

```powershell
python -m src.train_cnn --resume --epochs 1 --batch-size 16 --seed 42
```

The CNN uses the same actor assignments as the classical models. Preprocessing configuration is saved with the model so inference matches training.

## Training and speaker-independent testing

The common held-out test set contains 240 recordings from unseen actors **01, 09, 17, and 19**. No recording from those actors occurs in train or validation. This is the primary result and is more meaningful than ordinary random record-level splitting, which can leak speaker identity and overstate generalisation.

Actual held-out results:

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.3583 | 0.4314 | 0.3583 | 0.3387 |
| Random Forest | 0.3458 | 0.3449 | 0.3458 | 0.3131 |
| KNN | 0.3167 | 0.3251 | 0.3167 | 0.3095 |
| CNN | 0.1667 | 0.0757 | 0.1667 | 0.0749 |

The compact CNN was trained from scratch for six CPU epochs and is a baseline, not a fully optimized deep model. SVM is the best model here by weighted F1; CNN is the weakest. Six-fold `GroupKFold` CV supplies an additional speaker-independent estimate for classical models: SVM F1 0.4017 ± 0.0553, Random Forest 0.3985 ± 0.0668, KNN 0.3372 ± 0.0232.

Generate all Phase 4 artifacts with:

```powershell
.\.venv313\Scripts\Activate.ps1
python -m src.run_grouped_speaker_cv
python -m src.run_experimental_analysis
```

See `results/analysis/EXPERIMENTAL_ANALYSIS.md` for confusion patterns. Examples: SVM often confuses calm with sad; Random Forest often confuses sad with calm; the small CNN often predicts calm for sad, surprised, and happy recordings.

## Cross-corpus generalisation

**Not performed.** No separately licensed, labelled, compatible external emotion-speech corpus is included in this project. The repository makes no cross-corpus claim. A valid future study should reserve a separate corpus solely for external testing rather than merging it into RAVDESS training or tuning.

## Run the Streamlit application

Use the CNN environment so all four model choices are available:

```powershell
.\.venvcnn\Scripts\Activate.ps1
streamlit run app.py
```

The application loads saved models only. Upload a short WAV file, listen to it, view its waveform, select SVM/Random Forest/KNN/CNN, obtain a dataset-label prediction and confidence where supported, inspect classical features, and review comparison tables and confusion matrices. Uploaded files are temporary and deleted after inference.

## Limitations

- **Acted versus natural speech:** RAVDESS expressions are performed; spontaneous conversation and mixed affect can sound very different.
- **Speaker and accent variation:** pitch, prosody, vocal anatomy, age, accent, and speaking style can dominate acoustic patterns.
- **Culture and language:** the corpus is acted North American English. Emotion display and interpretation vary across communities and languages; results do not transfer automatically.
- **Audio conditions:** background noise, microphones, compression, distance, reverberation, and clipping can change all extracted features.
- **Dataset scale and balance:** 24 actors and scripted statements cannot represent population diversity or real communication contexts.
- **Label ambiguity:** emotions overlap acoustically; the observed calm/sad and sad/calm confusions demonstrate this.
- **Vocal expression is not internal state:** a vocal pattern is not reliable evidence of what someone feels, intends, or experiences.

## Ethical considerations

This project must remain an educational classification experiment.

- **Recruitment and hiring:** using inferred emotion could discriminate against accents, disability-related speech patterns, neurodivergence, culture, or communication style.
- **Education and workplaces:** monitoring students or employees pressures people to perform emotion, is error-prone, and can create unfair consequences.
- **Privacy and consent:** voice is personal data. Recording, feature extraction, storage, and model use require meaningful consent, minimisation, and secure handling.
- **Bias and discrimination:** corpus demographics and acted labels cannot establish fairness for broader populations.
- **Misuse:** probabilities communicate model uncertainty, not truth. They must never be treated as evidence of wellbeing, honesty, competence, risk, or mental-health condition.

## Testing

Base environment checks:

```powershell
.\.venv313\Scripts\Activate.ps1
python -m pip check
python -m unittest discover -s tests -v
python -m compileall -q src app.py
```

CNN model loading and inference check:

```powershell
.\.venvcnn\Scripts\Activate.ps1
python -m pip check
python -m unittest tests.test_trained_cnn tests.test_predictors -v
```

## GitHub preparation

`.gitignore` excludes environments, raw/processed data, uploads, trained artifacts, generated results, logs, secrets, and temporary files. The scripts, requirements files, tests, and README are intended for version control. Do not commit the RAVDESS archive or recordings; follow its licence and link users to the official source instead. No repository has been published or pushed by this project.
