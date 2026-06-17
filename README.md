# Titanic Survival Classification Model

**Author:** Sy Savane Usman
**Date:** June 2026
**Tools:** Python, Pandas, scikit-learn
**Dataset:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

---

## Project Overview

The sinking of the Titanic on 15 April 1912 resulted in the deaths of over 1,500 people — one of the deadliest maritime disasters in history. Survival was not random. Factors including passenger class, sex, age, and family size all influenced who made it onto a lifeboat.

This project builds a **logistic regression classifier** to predict whether a given passenger survived, based on those features. The goal is not only to achieve strong predictive accuracy but to extract **actionable insights** from the model's coefficients — understanding *which factors mattered most* and *by how much*.

---

## Dataset

- **Source:** Kaggle Titanic Competition (`train.csv`)
- **Size:** 891 passengers, 12 original columns
- **Target variable:** `Survived` (0 = did not survive, 1 = survived)

### Features Used After Cleaning

| Feature | Description |
|---|---|
| Pclass | Passenger class (1 = First, 2 = Second, 3 = Third) |
| Age | Passenger age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket price paid |
| Sex_male | Binary encoding: 1 = Male, 0 = Female |
| Embarked_Q | Boarded at Queenstown, Ireland |
| Embarked_S | Boarded at Southampton, England |

**Dropped columns:** `Name`, `PassengerId`, `Ticket`, `Cabin` — either non-informative identifiers or too sparse to use.

---

## Methodology

### 1. Data Cleaning

Two columns had missing values:

- **Age** (177 missing): filled with the mean age (~29 years)
- **Embarked** (2 missing): filled with the mode (Southampton)

### 2. Feature Encoding

Categorical columns cannot be passed directly into a logistic regression model. Two columns required encoding:

- **Sex** — binary category, encoded as `Sex_male` (1 = male, 0 = female)
- **Embarked** — three categories (S, C, Q), encoded using one-hot encoding via `pd.get_dummies()` with `drop_first=True`

`drop_first=True` drops one category per variable to avoid multicollinearity. The dropped (reference) categories are **Female** and **Cherbourg** — their effect is captured by the model's intercept.

### 3. Train/Test Split

The dataset was split 80/20 into training (712 passengers) and test (179 passengers) sets using `random_state=42` for reproducibility.

### 4. Model Training

A logistic regression model was trained using scikit-learn's `LogisticRegression` with `max_iter=1000`. The model learns by minimising log-loss via gradient descent — adjusting feature coefficients iteratively until predictions stabilise.

### 5. Threshold Tuning

The default classification threshold of 0.5 was evaluated alongside a tuned threshold of 0.4. Lowering the threshold makes the model more aggressive about predicting survival — trading some precision for higher recall.

---

## Results

### Model Coefficients

| Feature | Coefficient | Interpretation |
|---|---|---|
| Sex_male | -2.59 | Strongest predictor — being male sharply reduced survival odds |
| Pclass | -0.94 | Higher class number (lower class) reduced survival odds |
| SibSp | -0.30 | Larger sibling/spouse groups slightly reduced survival odds |
| Parch | -0.11 | Parent/child travel had a minor negative effect |
| Embarked_S | -0.41 | Southampton passengers less likely to survive than Cherbourg |
| Embarked_Q | -0.09 | Queenstown passengers slightly less likely to survive than Cherbourg |
| Age | -0.03 | Older passengers slightly less likely to survive |
| Fare | +0.003 | Higher fare marginally increased survival odds |

### Default Threshold (0.5)

```
Confusion Matrix:
[[90  15]
 [19  55]]

              precision    recall  f1-score   support
           0       0.83      0.86      0.84       105
           1       0.79      0.74      0.76        74

    accuracy                           0.81       179
```

- **Accuracy: 81%** — the model correctly classifies 8 in 10 passengers
- **Precision (survivors): 0.79** — when the model predicts survival, it is right 79% of the time
- **Recall (survivors): 0.74** — the model identifies 74% of all actual survivors
- **False negatives: 19** — 19 actual survivors were incorrectly predicted as dead

### Tuned Threshold (0.4)

```
Confusion Matrix:
[[85  20]
 [15  59]]

              precision    recall  f1-score   support
           0       0.85      0.81      0.83       105
           1       0.75      0.80      0.77        74

    accuracy                           0.80       179
```

- **Recall (survivors): 0.80** — the model now catches 80% of actual survivors (up from 74%)
- **False negatives: 15** — 4 fewer survivors missed compared to the default threshold
- **Trade-off:** Precision on survivors drops from 0.79 to 0.75 — more false alarms

### Threshold Comparison

| Metric | Threshold 0.5 | Threshold 0.4 |
|---|---|---|
| Accuracy | 81% | 80% |
| Survivor Precision | 0.79 | 0.75 |
| Survivor Recall | 0.74 | 0.80 |
| False Negatives | 19 | 15 |
| False Positives | 15 | 20 |

---

## Business Insights

**1. Gender was the dominant survival factor.**
The coefficient for `Sex_male` (-2.59) is more than twice the magnitude of any other feature. The "women and children first" evacuation protocol is clearly reflected in the data. In any rescue prioritisation model, gender must be treated as the primary signal.

**2. Passenger class was a strong structural predictor.**
First-class passengers had significantly better survival odds than third-class passengers. This likely reflects both physical proximity to lifeboats (first-class cabins were on upper decks) and social dynamics during evacuation. Class-based inequalities were not incidental — they were embedded in the ship's architecture.

**3. Embarkation port carried indirect survival signal.**
Cherbourg passengers survived at higher rates than Southampton or Queenstown passengers. This is not because the port itself mattered — it is because Cherbourg had a higher proportion of first-class passengers. Embarkation is a proxy for wealth and class, not a direct cause of survival.

**4. The right threshold depends on the objective.**
At 0.5, the model optimises overall accuracy. At 0.4, it prioritises recall — catching more survivors at the cost of some false positives. In a real rescue scenario where missing a survivor has serious consequences, the lower threshold is the more appropriate choice. Threshold selection is a business decision, not a technical one.

---

## Limitations & Next Steps

**Current limitations:**
- Age imputation with the mean introduces noise — a more sophisticated approach would use median imputation by passenger class and sex
- The model does not capture interaction effects (e.g. being female *and* first class may compound survival odds beyond what either feature captures independently)
- `Cabin` was dropped due to sparsity, but deck level likely carried genuine survival signal given its correlation with lifeboat proximity

**Next steps:**
- Evaluate model performance using ROC curves and AUC score for systematic threshold selection
- Test tree-based models (Decision Tree, Random Forest) to capture non-linear relationships
- Engineer new features: family size (SibSp + Parch + 1), title extracted from Name, deck extracted from Cabin
- Submit predictions to the Kaggle competition leaderboard

---

## Repository Structure

```
titanic-survival-classifier/
│
├── titanic_classifier.py   # Full model pipeline
├── train.csv               # Source dataset (Kaggle)
└── README.md               # This report
```
