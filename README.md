## clothes_recommendation_system


## create a new conda environment
conda create -n proactive python=3.8 -y


# activate the environment
conda activate proactive


# install the requirements
pip install -r requirements.txt






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


# free streamlit ports
ps aux | grep streamlit | grep $(whoami) | awk '{print $2}' | xargs kill -9


# pytest
# Running Tests

You can test the modules via the `tests` folder. To run the tests, use the following command:

```
pytest tests/test_search_engine.py
```
