import os, sys


########################################################
# add the data directory to the system path
########################################################


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)


sys.path.append(os.path.join(root_dir, "data"))


########################################################


from search_engine import SearchEngine
import pandas as pd
import numpy as np
import pdb


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


data_dir                = os.path.join(root_dir, "data")
# init_vector_database    = True
init_vector_database    = False


SE = SearchEngine(data_dir=data_dir, init_vector_database=init_vector_database)


query = "I am looking for a new blue dress"


results = SE.embedding_search(query, k_top=3).matches


for result in results:
   image_path = SE.get_image_path(result["id"])
   img = mpimg.imread(image_path)
   plt.imshow(img)
   plt.title(f"ID: {result['id']}")
   plt.axis('off')
   plt.show()






pdb.set_trace()