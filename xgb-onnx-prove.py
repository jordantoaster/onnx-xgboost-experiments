# 'Notebook' Code

#%% 
import pandas as pd
import xgboost as xgb
import numpy as np
import onnx
import onnxruntime as rt
import onnxmltools
import onnxmltools.convert.common.data_types

#%% 
df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
df['label'] = 0

# Change some labels
indicies = [1,2,20,11,99,77,45,23]
df.loc[indicies,'label'] = 1

#%%
x = df[['A', 'B', 'C', 'D']]
y = df['label']


#%% 
# Save / convert
num_features = 4
initial_type = [('feature_input', FloatTensorType([1, num_features]))]
onx = onnxmltools.convert.convert_xgboost(gbm, initial_types=initial_type)

with open("test.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# setup runtime
sess = rt.InferenceSession("test.onnx")

# get model metadata
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(input_name, label_name)

# geta  result
pred_onx = sess.run([label_name], {input_name: x.values.astype(np.float32)})[0]
print(pred_onx)

#%%
