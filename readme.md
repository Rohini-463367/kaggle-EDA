# EDA Project

This project gives you a simple, reusable exploratory data analysis setup for a Kaggle CSV dataset.

## Files

- `eda.ipynb` - notebook for step-by-step analysis
- `eda.py` - helper functions and script version of the same EDA flow
- `requirements.txt` - required Python packages
- `data/` - place your Kaggle CSV file here
- `output/` - saved charts will appear here

## How To Use

1. Put your Kaggle CSV file in the `data` folder.
2. Open `eda.ipynb` in Jupyter Notebook or VS Code.
3. Run the cells from top to bottom.

You can also run the script version:

```bash
python eda.py
```

The script automatically uses the first CSV file it finds in `data`.

## What The Analysis Covers

- dataset shape and memory usage
- data types and unique values
- missing values and duplicate rows
- numeric summary statistics
- categorical summary statistics
- date column detection
- outlier detection with the IQR method
- histograms, boxplots, category count plots, and a correlation heatmap
- grouped averages for low-cardinality categorical columns
