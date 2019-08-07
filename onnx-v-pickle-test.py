#%%
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnxruntime as rt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from joblib import dump, load

#%% Get Our Sample Data

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

#%% Create our model pipeline

x = newsgroups_train.data
y = newsgroups_train.target

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression()),
])

pipeline.fit(x, y)

#%% Create onnx and pickle (joblib) of the pipeline
model_onnx = convert_sklearn(pipeline, "pipe",
                             initial_types=[("input", StringTensorType([1, 1]))]
                            )

with open("onnx-test.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

dump(pipeline, 'pickle-test.joblib')

#%% Run ONNX And PICKLE on test set

# First convert test data to np array
x=np.asarray(newsgroups_test.data)

sess = rt.InferenceSession("onnx-test.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: x})[0]

# Print results
print('ONNX!!!')
print(pred_onx)
print(len(pred_onx))

clf = load('pickle-test.joblib')
pred_pkl = clf.predict(x)

# Print results
print('PICKLE!!!')
print(pred_pkl)
print(len(pred_pkl))

# Print results before persisting
pred_pre = pipeline.predict(x)
print('Prior to Saving')
print(pred_pre)
print(len(pred_pre))

# Note to self - need to compare JM results v Radek results (new machine)
# Note to self - flag file sizes

#%%
