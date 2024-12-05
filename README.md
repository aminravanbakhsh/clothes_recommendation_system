## clothes_recommendation_system


## create a new conda environment
conda create -n proactive python=3.8 -y


# activate the environment
conda activate proactive


# install the requirements
pip install -r requirements.txt


# set user info
create a file named user_info.py and set the following variables:

# data

create an empty folder named data. By default, it will be ignored by git.
download the data from [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and put it in the data folder.


In order to download the data, you need to create an account on kaggle and accept the competition rules.


the structure of the data folder should be the following:


```
data
   - images
   - articles.csv
```

# RUN

Streamlit run app.py


# free streamlit ports
ps aux | grep streamlit | grep $(whoami) | awk '{print $2}' | xargs kill -9


# pytest
# Running Tests

You can test the modules via the `tests` folder. To run the tests, use the following command:

```
pytest tests/test_search_engine.py
pytest tests/test_app.py
```




# tests

I am looking for a long white shirt.

# docker

docker build -t clothes_recommendation_system .
docker run -p 8510:8510 clothes_recommendation_system

docker run -p 8510:8510 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/user_info.py:/app/user_info.py \
  clothes_recommendation_system