# Futures-Price-Prediction

1. Collect data

Soybean historical data is from Kaggle https://www.kaggle.com/motorcity/soybean-price-factor-data-19622018.
This dataset includes not only the soybean future daily price (1962-2018) but also the associated commodities price as well, like gold, oil, soyoil, meat, etc. The size of dataset contains 20K rows and 47 columns.

2. Feature engineering
Based on the original features, I use the lag features and rolling average on past prices for new features to help soybean_settle price prediction.

3. Modeling
With two different approaches, the first part is using regression model and the other one is adopting time-series model. 

4. Performance 
