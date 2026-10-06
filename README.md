# FinFormer

I use raw balance sheet, income statement, and cashflow data to test whether models can learn financial signals without hand-made ratios. The notebook compares single-year baselines with three-year LSTM and XGBoost models.

## Why I built this

I wanted to see whether a model could use the statements directly, and whether keeping several years together helps predict a change in liabilities.

## Data

The pipeline loads annual US statements from SimFin. The notebook has 12,345 labeled rows from 2020 to 2024 and uses 34 statement features in its sequence windows. It joins tables by `SimFinId` and `Fiscal Year`, drops statement columns at or above the 15% missing-to-observed cutoff, then fills missing values with company means, industry medians, and global medians, in that order.

## Prediction target

Class 1 means Total Liabilities increase in the next reported year. I sort by company and year, shift liabilities within each company, preserve missing next-year values, and drop those rows.

## How I evaluate

I split by time, training on years before 2023 and testing on 2023 onward. I focus on recall for class 1 to track how many increases the model catches. Precision and accuracy are also reported. In the sequence test windows, 55% of labels are 1, so always predicting 1 gives 0.55 accuracy.

## Stage 1: single-year baselines

The saved logistic regression report has 0.52 training recall and 0.45 test recall for increases, with 0.54 test accuracy. XGBoost has 0.90 training recall and 0.71 test recall, with 0.59 test accuracy. These are row-level results and are separate from the sequence comparison below.

## Stage 2: three-year windows

Each window has three consecutive years and 34 features per year. Its label asks whether liabilities increase after the window's final year. The notebook makes 2,280 training windows and 2,280 held-out windows. It applies a signed log transform, then standardizes with training-window statistics.

## Stage 3: LSTM and XGBoost results

Both LSTM configs use 16 hidden units and 40 epochs across five runs. The regularized config adds 0.3 dropout and 0.001 weight decay. XGBoost uses the same windows flattened to 102 features and is run once.

| Model | Train accuracy | Test accuracy | Test precision | Test recall |
|---|---:|---:|---:|---:|
| LSTM, old config, five runs | 0.707 | 0.568 +/- 0.005 | 0.628 +/- 0.006 | 0.528 +/- 0.024 |
| LSTM, regularized config, five runs | 0.641 | 0.559 +/- 0.003 | 0.626 +/- 0.006 | 0.495 +/- 0.013 |
| XGBoost, one run | not reported | 0.58 | 0.62 | 0.62 |
| Always predict 1 | not applicable | 0.55 | 0.55 | 1.00 |

## What I found

The LSTM scores are close to the 0.55 always-positive accuracy baseline. XGBoost has similar accuracy and higher class 1 recall in its single run. The regularized LSTM has lower training accuracy, but its test scores do not improve in these runs.

## Mistakes I made and fixed

A boolean comparison with a missing next-year value can label that row false. I now preserve the missing target and drop the row before converting labels to integers. I also set each seed before creating the LSTM so it controls the initial weights.

## Limitations

The test covers one later period, and neighboring windows overlap. Mean and median imputation uses information across years.

## Upcoming

- New Data
- Transformer implementation. Once, new data is in.

## Setup

Install `requirements.txt`, copy `.env.example` to `.env`, add a SimFin API key as `SIM_FIN`, then run `python src/pipeline.py`. The baseline and sequence experiments are in `notebooks/01_baseline.ipynb`.

## Project layout

- `src/pipeline.py`: loads and combines SimFin tables
- `notebooks/01_baseline.ipynb`: data preparation, baselines, sequence models, and evaluation
- `data/`: generated CSV files
