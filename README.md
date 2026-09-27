# Financial Statement Analyzer

I'm building a model to see how well it can learn about company financial
health from historical statements without relying on hand-picked ratios.

## What this project is

Many financial ML models start with hand-built ratios like debt-to-equity,
current ratio, or return on assets. I'm trying a different approach: give
the model the balance sheet, income statement, and cashflow data directly
and see which relationships it can learn.

I'm starting with simple baselines. After that, I want to try an LSTM and
eventually a Transformer that learns embeddings for financial accounts.
The idea is similar to how BERT learns word embeddings, but applied to
balance sheet line items.

I plan to build the LSTM and later neural models in PyTorch. I'm also
working on a custom tensor and autograd library in C in a separate repo
called AutoGrad. I hope to use it as a lower-level backend later.

## Data

I use SimFin's free US annual filings from 2020 through 2025. The balance
sheet, income statement, and cashflow tables are joined on SimFinId and
Fiscal Year:
- Balance sheet: 13 features after the 15% missing-value filter
- Income statement: 8 features
- Cashflow statement: 7 features

The merged data has about 16,000 company-year rows and 36 features.

I fill missing values in this order:
1. Company mean across years, so each company's scale is kept
2. Industry median, to account for differences between sectors
3. Global median, for anything still missing

For sequence models, I plan to use forward fill or interpolation so the
data keeps more of its year-to-year movement.

## Prediction target

The target is binary: did a company's Total Liabilities increase the
following year?
- 1 = debt increased
- 0 = debt stayed flat or decreased

I sort rows by SimFinId and Fiscal Year, then use `shift(-1)` to compare
each year's Total Liabilities with the next one. If a company has no
following filing, its target stays missing and the row is dropped.

## Baseline result

My current logistic regression uses `class_weight="balanced"`. It gets
0.53 recall for debt increases on the training years (2020–2023) and
0.46 on the held-out 2024 rows. That seven-point gap suggests some
overfitting, but the model still generalizes reasonably for a linear
model using one year's data at a time.

Test precision for debt increases is 0.70. There are 99 positive labels
in the 160 test rows. Since the model sees one snapshot at a time, it
can't learn how a company's accounts change from year to year.

This is a small test set, so the result could change with more labeled
years. One reason I want to try sequence models is to see how a
company's financial history adds useful signal.

## Time-split and target-label limitation

My statements cover 2020 through 2025. I train on labeled rows from
2020-2023. The test split starts in 2024, but it currently has 160
labeled rows from 2024 and none from 2025.

The target needs a following-year filing. Rows without one keep a
missing target and are excluded. That leaves a much smaller test set
than training set, so I want to be careful when reading the result.

An LSTM could use a company's financial history, but I'll still need
enough later filings to make a useful test set.

## Roadmap

- [x] Data pipeline (SimFin, merge, imputation, target derivation)
- [x] Logistic regression baseline
- [ ] XGBoost baseline
- [ ] Simple feedforward neural network baseline
- [ ] LSTM sequence model (multiple years per company as input)
- [ ] Transformer with learned account embeddings
- [ ] Self-supervised pretraining (masked financial value reconstruction)
- [ ] Port to custom C autograd engine once tensor implementation is stable

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your SimFin API key to .env
python src/pipeline.py
```

Requires a free SimFin API key from simfin.com.

## Project layout

```
src/pipeline.py: pulls and filters SimFin data, then saves the CSVs
notebooks/01_baseline: prepares features, trains the baseline, and evaluates it
data/: generated CSV files, ignored by Git
```
