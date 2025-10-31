# Exploration plots and profile

This folder contains the visual artifacts and a short summary produced by the data explorer run.

Generated plots
- `exploration_distributions_numerical.png` — distributions for numerical features
- `exploration_distributions_target.png` — target distribution plots
- `exploration_correlations.png` — correlation matrix heatmap
- `exploration_outliers.png` — outlier analysis visuals

Key dataset summary (from `exploration_data_profile.json`):

- Rows: 284,807
- Columns: 31
- Memory usage: 67.36 MB
- Numerical columns: 31
- Categorical columns: 0
- Class distribution: ~99.8% class 0, ~0.2% class 1 (strong class imbalance)

Recommendations (automatically suggested):
- Remove duplicate rows (1,081 found)
- Handle outliers for several features (V6,V8,V20,V27,V28,Amount etc.)
- Consider log/box-cox transforms for highly skewed features (Amount, V8, V28, ...)
- Consider feature engineering (interactions, polynomials)

Files in this folder:

```
ls -la
```

If you'd like these files committed to the repository for review, I can add them to a branch and open a PR.
