import pandas as pd
from sklearn.ensemble import RandomForestRegressor

print("Reading training data...")
train = pd.read_csv("trainingData1.csv", usecols=lambda col: col not in ["date", "stock", "binStartTime", "binEndTime", "auctionIndicator"])
train = train[train["binNum"] > 1]
train = train[train["binNum"] < 68]
train = train.dropna(axis=1, how="all")
train = train.dropna(axis=0, how="any")
train_columns = train.columns.tolist()
train_columns = [c for c in train_columns if c not in ["output_remainingVolume", "output_closeAuctionVolume", "output_closePriceDirection"]]
print("Completed!")

print("\nReading test volume data...")
test_volume = pd.read_csv("testingVolume1.csv")
test_volume = test_volume[test_volume["binNum"] == 62]
test_volume = test_volume.dropna(axis=1, how="all")
test_volume = test_volume.fillna(0)
test_volume_columns = test_volume.columns.tolist()
test_volume_columns = [c for c in test_volume_columns if c not in ["date", "stock", "binStartTime", "binEndTime", "auctionIndicator"]]
print("Completed!")

print("\nReading test price data...")
test_price = pd.read_csv("testingPrice1.csv")
test_price = test_price[test_price["binNum"] == 67]
test_price = test_price.dropna(axis=1, how="all")
test_price = test_price.fillna(0)
test_price_columns = test_price.columns.tolist()
test_price_columns = [c for c in test_price_columns if c not in ["date", "stock", "binStartTime", "binEndTime", "auctionIndicator"]]
print("Completed!")

target_one = "output_remainingVolume"
target_two = "output_closeAuctionVolume"
target_three = "output_closePriceDirection"

model = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, n_jobs=-1, random_state=1)

# Predict Remaining Volume
print("\nTraining model for predicting remainingVolume...")
model.fit(train[train_columns], train[target_one])
print("Completed!")
print("\nPredicting remainingVolume...")
predictions_one = model.predict(test_volume[test_volume_columns])
print("Completed!")

# Predict Close Auction Volume
print("\nTraining model for predicting closeAuctionVolume...")
model.fit(train[train_columns], train[target_two])
print("Completed!")
print("\nPredicting closeAuctionVolume...")
predictions_two = model.predict(test_volume[test_volume_columns])
print("Completed!")

print("\nWriting output to outputVolume.csv...")
test_volume["output_remainingVolume"] = predictions_one
test_volume["output_closeAuctionVolume"] = predictions_two
test_volume.drop(["volume", "binStartPrice", "binEndPrice", "day_openPrice", "day_closePrice", "day_lowPrice", "day_highPrice"], axis=1, inplace=True)
test_volume.to_csv("./outputVolume.csv")
print("Completed!")

# Predict Close Price Direction
print("\nTraining model for predicting closePriceDirection...")
train_columns = [c for c in train_columns if c not in ["day_closePrice"]]
model.fit(train[train_columns], train[target_three])
print("Completed!")
print("\nPredicting closePriceDirection...")
predictions_three = model.predict(test_price[test_price_columns])
close_price_direction = []
for p in predictions_three:
	if p < -0.1:
		close_price_direction.append(-1)
	elif p < 0.1:
		close_price_direction.append(0)
	else:
		close_price_direction.append(1)
print("Completed!")

print("\nWriting output to outputPrice.csv...")
test_price["output_closePriceDirection"] = close_price_direction
test_price.drop(["volume", "binStartPrice", "binEndPrice", "day_openPrice", "day_lowPrice", "day_highPrice"], axis=1, inplace=True)
test_price.to_csv("./outputPrice.csv")
print("Completed!")