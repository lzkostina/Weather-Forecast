.PHONY: download_data process_data predictions clean rawdata train_models_full train_models

# Target to clean up the original and processed data directories
clean:
	@echo "Cleaning up data/processed directories..."
	rm -rf data/processed
	@echo "Cleaning complete."

## Target to delete all except code and raw data
#clean:
#	@echo "Cleaning project directory..."
#	# Remove everything except .py, .md files, raw data directory, and .git directory
#	find . -type f ! -name '*.py' ! -name '*.md' ! -path './data/raw/*' ! -path './.git/*' -delete
#	# Remove all non-empty directories except for the raw data directory and .git directory
#	find . -type d ! -path './data/raw' ! -path './.git' -empty -delete
#	@echo "Project cleaned except for code, raw data, and git repository."

# Target to delete and re-download raw data
rawdata: clean_data
	@echo "Deleting contents of raw data directory..."
	rm -rf data/raw/* data/raw/.* || true
	@echo "Re-downloading raw data..."
	make download_data
	@echo "Raw data re-downloaded."

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

train_models_full:
	@echo "Training models on full dataset..."
	PYTHONPATH=$(CURDIR) python3 predictor/train_all_full_data.py || exit 1
	@echo "Models trained and saved on full dataset."

train_models:
	@echo "Training models on partial dataset..."
	PYTHONPATH=$(CURDIR) python3 predictor/train_all_partial.py || exit 1
	@echo "Models trained and saved on partial dataset."


# Target to run the prediction script
predictions:

	@echo "Activating virtual environment and downloading new data..."
	. /Weather-Forecast/venv/bin/activate && python3 analysis/download_openweather.py || exit 1
	@echo "Combining new data to NOAA dataset..."
	. /Weather-Forecast/venv/bin/activate && python3 analysis/combine_noaa_hourly.py || exit 1
	@echo "Processing NOAA datasets..."
	. /Weather-Forecast/venv/bin/activate && python3 analysis/process_noaa.py || exit 1
	@echo "Processing OpenWeather datasets..."
	. /Weather-Forecast/venv/bin/activate && python3 analysis/process_openweather.py || exit 1
	@echo "Restructuring NOAA datasets..."
	. /Weather-Forecast/venv/bin/activate && python3 analysis/restructure_noaa.py || exit 1
	@echo "Combining NOAA and OpenWeather..."
	. /Weather-Forecast/venv/bin/activate && python3 analysis/combine_noaa_hourly.py || exit 1
	@echo "Processing complete."
	. /Weather-Forecast/venv/bin/activate && python3 analysis/create_regression_dataset.py || exit 1
	@echo "Regression datasets created."
	@echo "Running predictions..."
	. /Weather-Forecast/venv/bin/activate && python3 main.py || exit 1
	@echo "Predictions complete."
