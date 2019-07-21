# Market Close Predictor
A prediction model built in Python to estimate the closing price and volume of stocks based on historical data.

Created as part of APAC Hackathon 2018 organised by Credit Suisse.

## Working
The script reads the datasets (available [here](https://drive.google.com/drive/folders/1k743itLNnNY5O6POGV-zEIU7OaaBTh7j)) and fits a  Random Forest regression model on the data.

This is then used to predict the volume and price direction (more, lesser or same) when the market closes.
