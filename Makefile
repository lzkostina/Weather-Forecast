.PHONY: download_data clean_data process_data predictions

# Target to clean up the original and processed data directories
clean_data:
	@echo "Cleaning up data/processed directories..."
	rm -rf data/processed
	@echo "Cleaning complete."

# Target to download data from Kaggle and NOAA
download_data: clean_data
	@echo "Downloading NOAA datasets..."
	python3 analysis/download_noaa.py || exit 1
	@echo "Downloading OpenWeather datasets..."
	python3 analysis/download_openweather.py || exit 1
	@echo "Download complete."

# Target to convert data from Kaggle and NOAA to dataframes
process_data:
	@echo "Processing NOAA datasets..."
	python3 analysis/process_noaa.py || exit 1
	@echo "Processing OpenWeather datasets..."
	python3 analysis/process_openweather.py || exit 1
	@echo "Restructuring NOAA datasets..."
	python3 analysis/restructure_noaa.py || exit 1
	@echo "Combining NOAA and OpenWeather..."
	python3 analysis/combine_noaa_hourly.py || exit 1
	@echo "Processing complete."
	python3 analysis/create_regression_dataset.py || exit 1
	@echo "Regression datasets created."

# Target to train and save all models
train_models:
	@echo "Training models..."
	PYTHONPATH=$(CURDIR) python3 predictor/train_models.py|| exit 1
	@echo "Models trained and saved."

# Target to run the prediction script
predictions:
	@echo "Downloading new data..."
	python3 analysis/download_openweather.py || exit 1
	@echo "Combining new data to NOAA dataset..."
	python3 analysis/combine_noaa_hourly.py || exit 1
	@echo "Running predictions..."
	python3 main.py || exit 1
	@echo "Predictions complete."
