import json
import fasttext
import pandas as pd
import numpy as np
import lightgbm

path_train = '/home/ubuntu/train.json'
path_dev = '/home/ubuntu/dev.json'
path_test = '/home/ubuntu/test.json'

with open(path_train, "r") as f:
    train = json.load(f)

with open(path_dev, "r") as f:
    dev = json.load(f)

with open(path_test, "r") as f:
    test = json.load(f)

model = fasttext.load_model("model_all_perseus_dgl_patr_texts")
#model = fasttext.load_model("/home/ubuntu/model_skipgram_p_d")

### 3 before 3 after
for n, t in enumerate(train):
  t['ending'] = t['form'][-2:]
  if train[n -1]["sentid"] == t["sentid"]:
    pre1 = model.get_word_vector(train[n-1]['form'])
    t['ending_pre1'] = train[n-1]['form'][-2:]
  else:
    pre1 = np.zeros(300)
    t['ending_pre1'] = "_"
  if train[n -2]["sentid"] == t["sentid"]:
    pre2 = model.get_word_vector(train[n-2]['form'])
    t['ending_pre2'] = train[n-2]['form'][-2:]
  else:
    pre2 = np.zeros(300)
    t['ending_pre2'] = "_"
  if train[n -3]["sentid"] == t["sentid"]:
    pre3 = model.get_word_vector(train[n-3]['form'])
    t['ending_pre3'] = train[n-3]['form'][-2:]
  else:
    pre3 = np.zeros(300)
    t['ending_pre3'] = "_"
  try:
   if train[n + 1]["sentid"] == t["sentid"]:
    fol1 = model.get_word_vector(train[n+1]['form'])
    t['ending_fol1'] = train[n+1]['form'][-2:]
   else:
    fol1 = np.zeros(300)
    t['ending_fol1'] = "_"
  except:
    fol1 = np.zeros(300)
    t['ending_fol1'] = "_"
  try:
   if train[n + 2]["sentid"] == t["sentid"]:
    fol2 = model.get_word_vector(train[n+2]['form'])
    t['ending_fol2'] = train[n+2]['form'][-2:]
   else:
    fol2 = np.zeros(300)
    t['ending_fol2']= "_"
  except:
    fol2 = np.zeros(300)
    t['ending_fol2']= "_"
  try:
   if train[n + 3]["sentid"] == t["sentid"]:
    fol3 = model.get_word_vector(train[n+3]['form'])
    t['ending_fol3'] = train[n+3]['form'][-2:]
   else:
    fol3 = np.zeros(300)
    t['ending_fol3']= "_"
  except:
    fol3 = np.zeros(300)
    t['ending_fol3']= "_"
  a1 = np.append(pre3, pre2)
  b = np.append(a1, pre1)
  a2 = np.append(b, model.get_word_vector(t['form']))
  a3 = np.append(a2, fol1)
  a4 = np.append(a3, fol2)
  a5 = np.append(a4, fol3)
  t['word_embed'] = a5

for n, t in enumerate(dev):
  t['ending'] = t['form'][-2:]
  if dev[n -1]["sentid"] == t["sentid"]:
     pre1 = model.get_word_vector(dev[n-1]['form'])
     t['ending_pre1'] = dev[n-1]['form'][-2:]
  else:
     pre1 = np.zeros(300)
     t['ending_pre1'] = "_"
  if dev[n -2]["sentid"] == t["sentid"]:
     pre2 = model.get_word_vector(dev[n-2]['form'])
     t['ending_pre2'] = dev[n-2]['form'][-2:]
  else:
     pre2 = np.zeros(300)
     t['ending_pre2'] = "_"
  if dev[n -3]["sentid"] == t["sentid"]:
     pre3 = model.get_word_vector(dev[n-3]['form'])
     t['ending_pre3'] = dev[n-3]['form'][-2:]
  else:
     pre3 = np.zeros(300)
     t['ending_pre3'] = "_"
  try:
   if dev[n + 1]["sentid"] == t["sentid"]:
     fol1 = model.get_word_vector(dev[n+1]['form'])
     t['ending_fol1'] = dev[n+1]['form'][-2:]
   else:
     fol1 = np.zeros(300)
     t['ending_fol1'] = "_"
  except:
     fol1 = np.zeros(300)
     t['ending_fol1'] = "_"
  try:
   if dev[n + 2]["sentid"] == t["sentid"]:
     fol2 = model.get_word_vector(dev[n+2]['form'])
     t['ending_fol2'] = dev[n+2]['form'][-2:]
   else:
     fol2 = np.zeros(300)
     t['ending_fol2'] = "_"
  except:
     fol2 = np.zeros(300)
     t['ending_fol2'] = "_"
  try:
   if dev[n + 3]["sentid"] == t["sentid"]:
     fol3 = model.get_word_vector(dev[n+3]['form'])
     t['ending_fol3'] = dev[n+3]['form'][-2:]
   else:
     fol3 = np.zeros(300)
     t['ending_fol3'] = "_"
  except:
     fol3 = np.zeros(300)
     t['ending_fol3'] = "_"
  a1 = np.append(pre3, pre2)
  b = np.append(a1, pre1)
  a2 = np.append(b, model.get_word_vector(t['form']))
  a3 = np.append(a2, fol1)
  a4 = np.append(a3, fol2)
  a5 = np.append(a4, fol3)
  t['word_embed'] = a5

for n, t in enumerate(test):
  t['ending'] = t['form'][-2:]
  if test[n -1]["sentid"] == t["sentid"]:
     pre1 = model.get_word_vector(test[n-1]['form'])
     t['ending_pre1'] = test[n-1]['form'][-2:]
  else:
     pre1 = np.zeros(300)
     t['ending_pre1'] = "_"
  if test[n -2]["sentid"] == t["sentid"]:
     pre2 = model.get_word_vector(test[n-2]['form'])
     t['ending_pre2'] = test[n-2]['form'][-2:]
  else:
     pre2 = np.zeros(300)
     t['ending_pre2'] = "_"
  if test[n -3]["sentid"] == t["sentid"]:
     pre3 = model.get_word_vector(test[n-3]['form'])
     t['ending_pre3'] = test[n-3]['form'][-2:]
  else:
     pre3 = np.zeros(300)
     t['ending_pre3'] = "_"
  try:
   if test[n + 1]["sentid"] == t["sentid"]:
     fol1 = model.get_word_vector(test[n+1]['form'])
     t['ending_fol1'] = test[n+1]['form'][-2:]
   else:
     fol1 = np.zeros(300)
     t['ending_fol1'] = "_"
  except:
     fol1 = np.zeros(300)
     t['ending_fol1'] = "_"
  try:
   if test[n + 2]["sentid"] == t["sentid"]:
     fol2 = model.get_word_vector(test[n+2]['form'])
     t['ending_fol2'] = test[n+2]['form'][-2:]
   else:
     fol2 = np.zeros(300)
     t['ending_fol2'] = "_"
  except:
     fol2 = np.zeros(300)
     t['ending_fol2'] = "_"
  try:
   if test[n + 3]["sentid"] == t["sentid"]:
     fol3 = model.get_word_vector(test[n+3]['form'])
     t['ending_fol3'] = test[n+3]['form'][-2:]
   else:
     fol3 = np.zeros(300)
     t['ending_fol3'] = "_"
  except:
     fol3 = np.zeros(300)
     t['ending_fol3'] = "_"
  a1 = np.append(pre3, pre2)
  b = np.append(a1, pre1)
  a2 = np.append(b, model.get_word_vector(t['form']))
  a3 = np.append(a2, fol1)
  a4 = np.append(a3, fol2)
  a5 = np.append(a4, fol3)
  t['word_embed'] = a5
###

split = []
for u in train, dev, test:
    df = pd.DataFrame.from_dict(u)
    split.append(df)

# word embeddings as column in a dataframe
train_embed = split[0]['word_embed'].apply(pd.Series)
train_embed['ending'] = split[0]['ending'].astype("category")
train_embed['ending_pre1'] = split[0]['ending_pre1'].astype("category")
train_embed['ending_pre2'] = split[0]['ending_pre2'].astype("category")
train_embed['ending_pre3'] = split[0]['ending_pre3'].astype("category")
train_embed['ending_fol1'] = split[0]['ending_fol1'].astype("category")
train_embed['ending_fol2'] = split[0]['ending_fol2'].astype("category")
train_embed['ending_fol3'] = split[0]['ending_fol3'].astype("category")

# word embeddings as column in a dataframe
dev_embed = split[1]['word_embed'].apply(pd.Series)
dev_embed['ending'] = split[1]['ending'].astype("category")
dev_embed['ending_pre1'] = split[1]['ending_pre1'].astype("category")
dev_embed['ending_pre2'] = split[1]['ending_pre2'].astype("category")
dev_embed['ending_pre3'] = split[1]['ending_pre3'].astype("category")
dev_embed['ending_fol1'] = split[1]['ending_fol1'].astype("category")
dev_embed['ending_fol2'] = split[1]['ending_fol2'].astype("category")
dev_embed['ending_fol3'] = split[1]['ending_fol3'].astype("category")


test_embed = split[2]['word_embed'].apply(pd.Series)
test_embed['ending'] = split[2]['ending'].astype("category")
test_embed['ending_pre1'] = split[2]['ending_pre1'].astype("category")
test_embed['ending_pre2'] = split[2]['ending_pre2'].astype("category")
test_embed['ending_pre3'] = split[2]['ending_pre3'].astype("category")
test_embed['ending_fol1'] = split[2]['ending_fol1'].astype("category")
test_embed['ending_fol2'] = split[2]['ending_fol2'].astype("category")
test_embed['ending_fol3'] = split[2]['ending_fol3'].astype("category")


myGBM = lightgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=50, max_depth=-1, learning_rate=0.03, n_estimators=47946, subsample_for_bin=100000, objective='multiclass', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=1, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0, reg_lambda=0.001, random_state=1, n_jobs=-1, silent=True, importance_type='split', max_bin=500)
myfit = myGBM.fit(train_embed, split[0]['postag'],eval_metric="multi_error", eval_set=[(dev_embed, split[1]['postag'])])
