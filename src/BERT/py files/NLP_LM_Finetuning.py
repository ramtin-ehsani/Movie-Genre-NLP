#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install pytorch-transformers')


# In[ ]:





# In[ ]:


import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import  tqdm_notebook


# In[ ]:


directory_path = '/content/drive/My Drive/nlp'


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


hotel_df = pd.read_csv(os.path.join(directory_path,'data_for_bert.csv'))


# In[ ]:


hotel_df.shape


# In[ ]:


changed_text=hotel_df.preprocessed_plot.apply(lambda x:x+"\n"+"\n")


# In[ ]:


changed_text


# In[ ]:


open(os.path.join(directory_path,'data_lm_plots.txt'), "w").write(''.join(changed_text))


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/My Drive/nlp')
import os
os.listdir(os.getcwd())


# In[ ]:


get_ipython().system('python3 pregenerate_training_data.py --train_corpus data_lm_plots.txt --bert_model bert-base-uncased --do_lower_case --output_dir training/ --epochs_to_generate 2 --max_seq_len 128')


# In[ ]:


os.listdir(os.getcwd())


# In[ ]:


get_ipython().system('python3 finetune_on_pregenerated.py --pregenerated_data training/ --bert_model bert-base-uncased --do_lower_case --train_batch_size 16  --output_dir finetuned_lm/ --epochs 2')


# In[ ]:




