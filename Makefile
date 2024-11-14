.PHONY: download_data original_data predictions

# Target to clean up the original data directory
original_data:
	@echo "Cleaning up data/original directory..."
	rm -rf data/original

# Target to download data from Kaggle and NOAA
download_data: original_data
	@echo "Downloading Kaggle dataset..."
	python3 analysis/download_kaggle.py || exit 1
	@echo "Downloading NOAA dataset..."
	python3 analysis/download_noaa.py || exit 1
	@echo "Download complete."

# Target to run the prediction script
predictions:
	@echo "Running predictions..."
	python3 predictor/main.py || exit 1
	@echo "Predictions complete."

    