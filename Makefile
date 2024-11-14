.PHONY: rawdata prdictions

rawdata:
	rm -rf data/kaggle
	rm -rf data/noaa


predictions:
	python predictor/main.py
    