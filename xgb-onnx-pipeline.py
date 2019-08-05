# Attempt to use SK Learn Pipeline converter with XGB SK Estimator
# http://onnx.ai/sklearn-onnx/auto_examples/plot_tfidfvectorizer.html#sphx-glr-auto-examples-plot-tfidfvectorizer-py

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


#%%

df = pd.DataFrame(columns=['data', 'label'])

df['data'] = pd.util.testing.rands_array(10, 100)
df['label'] = 0

indicies = [1,2,20,11,99,77,45,23]
df.loc[indicies,'label'] = 1

#%%

x = df['data']
y = df['label']

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression()),
])

pipeline.fit(x, y)

#%%
model_onnx = convert_sklearn(pipeline, "pipe",
                             initial_types=[("input", StringTensorType([1, 1]))]
                            )

with open("test-pipeline.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

#%%
sess = rt.InferenceSession("test-pipeline.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: x.values})[0]
pred_onx
