# Weather-Forecast

which commands works now?
git pull -to copy everything from git repo 
make clean_data -to clean data folder
make download_data -to download kaggle and noaa data in their original format
make process_data -to convert original data into csv formats and put these files in data/processed folder
docker build -t weather-forecast . -to create a docker image 
docker run --rm weather-forecast  -to see the prediction (now it returns zeroes because we haven't models except the test one
)
