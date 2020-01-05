# house_prices_RF
Random Forest practice using house price data from kaggle

Utilizing [Fastai](https://www.fast.ai/) as a guide and for accessory functions.
Utilizing scikit-learn RandomForestRegressor.

Things considered and implemented:    
**Preprocessing data**:
  - converting categorical string variables to "categories" (which encode the numeric information necessary for machine learning)
  - performing feature extractions if there are dates for example
  - reordering any ordinal variable categories to make more sense ("high", "medium", "low")
  - taking care of any missing data, which we cannot pass directly to a Random Forest

fastai function **train_cats** to convert strings to pandas categories.   
Check for missing values.   
fastai function **proc_df** to handle missing continuous data (replacing missing values with the median).

split dataset into training and validation sets. Validation set is 25% of total dataset.   
Consider **OOB score**.

**Attempt to reduce overfitting**   
**Subsampling**: fastai function **set_rf_samples** to give each tree a random sample of n random rows (default is to use all rows with replacement)   
Grow trees less deeply: adjust the min_samples_leaf parameter of RandomForestRegressor   
Increase variation among trees: randomly sample columns for each split by adjusting the **max_features** parameter of RandomForestRegressor.
