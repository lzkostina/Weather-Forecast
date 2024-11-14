.PHONY: download_data clean_data process_data predictions

# Target to clean up the original and processed data directories
clean_data:
	@echo "Cleaning up data/original and data/processed directories..."
	rm -rf data/original
	rm -rf data/processed
	@echo "Cleaning complete."

# Target to download data from Kaggle and NOAA
download_data: clean_data
	@echo "Downloading Kaggle dataset..."
	python3 analysis/download_kaggle.py || exit 1
	@echo "Downloading NOAA datasets..."
	python3 analysis/download_noaa.py || exit 1
	@echo "Download complete."

# Target to convert data from Kaggle and NOAA to dataframes
process_data: 
	@echo "Processing Kaggle dataset..."
	python3 analysis/process_kaggle.py || exit 1
	@echo "Processing NOAA datasets..."
	python3 analysis/process_noaa.py || exit 1
	@echo "Processing complete."

# Target to run the prediction script
predictions:
	@echo "Running predictions..."
	python3 predictor/main.py || exit 1
	@echo "Predictions complete."


    