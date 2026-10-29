# House Price Prediction

A machine-learning project that predicts the **median house value** of a neighbourhood in California from its location, housing and income data, using a **Random Forest** regression pipeline. It was built as a Month 2 task of an AI/ML internship.

## Results

The model was trained on 80% of the data (16,512 rows) and tested on the other 20% (4,128 rows):

| Metric | Value |
|---|---|
| R² | **0.817** |
| Mean absolute error (MAE) | $31,668 |
| Root mean squared error (RMSE) | $48,962 |

In plain words: the model explains about 82% of the differences in house value between neighbourhoods, and a typical prediction is off by roughly $32,000 (larger mistakes weigh more in the RMSE).

**What drives the prediction** (Random Forest feature importance):

| Feature | Importance |
|---|---|
| `median_income` | 49.1% |
| Inland location (`ocean_proximity = INLAND`) | 14.1% |
| `longitude` | 10.6% |
| `latitude` | 10.2% |
| `housing_median_age` | 5.2% |

Income of the neighbourhood is by far the strongest signal, and location matters next (inland areas are cheaper).

## The data

`housing.csv` is the well-known **California Housing** dataset: 20,640 rows, each describing one census block group (a small neighbourhood).

| Column | Meaning |
|---|---|
| `longitude`, `latitude` | Where the block group is |
| `housing_median_age` | Median age of the houses |
| `total_rooms`, `total_bedrooms` | Rooms and bedrooms in the block group (207 values of `total_bedrooms` are missing) |
| `population`, `households` | People and households |
| `median_income` | Median household income (in tens of thousands of dollars) |
| `ocean_proximity` | `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY` or `ISLAND` |
| `median_house_value` | The value to predict (US dollars) |

## What the notebook does

1. Loads and explores the data (`info`, `describe`, missing values).
2. Fills the 207 missing `total_bedrooms` values with the column mean.
3. Builds one scikit-learn `Pipeline`: median imputation for the numeric columns, one-hot encoding for `ocean_proximity`, then a `RandomForestRegressor` (100 trees, default depth).
4. Splits 80/20 with a fixed random seed, trains the pipeline and predicts the test set.
5. Reports MAE, MSE, RMSE and R², prints the feature importances and plots actual against predicted prices.
6. Predicts the price of one example neighbourhood (near the bay, median income 8.33) as about $431,942. That neighbourhood is the first row of the dataset, whose recorded value is $452,600; it is only a demonstration, not a test on unseen data.

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook House_Price_Prediction.ipynb
```

Keep `housing.csv` in the same folder as the notebook.

## Limitations

- **Prices are capped.** The dataset stops at $500,001, and 965 rows (4.7%) sit exactly at that value, so the model cannot predict expensive areas correctly.
- **Old data.** The values come from the 1990 census, so the dollar amounts are not today's prices.
- **One model, no tuning.** Only a default Random Forest was trained: no hyperparameter search, no cross-validation and no comparison with other models.
- **Possible optimism in the score.** The missing bedrooms were filled with the mean of the whole dataset before the split, and neighbouring block groups are similar to each other, so a random split can flatter the result a little.
- The `ISLAND` category has only 5 rows, so the model has almost nothing to learn from for it.
