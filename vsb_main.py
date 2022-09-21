
import os
import numpy as np
from tqdm import tqdm

from vsb_data import VSB_Dataset
from vsb_model import VSBNet
from vsb_metrics import *


data = VSB_Dataset(data_type="train", batch_size=16)

model = VSBNet()
print(model)

model.train_model(data, epochs=30, lr=1e-3)

y_pred_train, y_pred_val = model.evaluate_model(data)

np.save(os.path.join("submissions","y_train.npy"),data.y_train)
np.save(os.path.join("submissions","y_val.npy"),data.y_val)
np.save(os.path.join("submissions","y_pred_train.npy"),y_pred_train)
np.save(os.path.join("submissions","y_pred_val.npy"),y_pred_val)

print("pred sizes")
print(np.shape(y_pred_val))  # torch.Size([436, 1])
print(np.shape(y_pred_train))  # torch.Size([8275, 1])
print(np.shape(data.y_val))  # (436, 1)
print(np.shape(data.y_train))  # (8275, 1)

c_train = confusion(y_pred_train > 0, data.y_train)
print(c_train)  # (212, 118, 7658, 287)
print(mcc(*c_train))  # 0.49838358624446083

c_val = confusion(y_pred_val > 0, data.y_val)
print(c_val)   # (16, 3, 407, 10)
print(mcc(*c_val))  # 0.7053190215817398

