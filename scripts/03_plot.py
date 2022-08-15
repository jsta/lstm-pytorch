import torch
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# run from scripts folder

data = pickle.load(open("../data/data.ardata", "rb"))
y_train = torch.from_numpy(data.y_train).type(torch.Tensor).view(-1)
y_preds = torch.load("../data/y_preds.pytorch")
hist = torch.load("../data/hist.pytorch")


dt = pd.DataFrame(
    {
        "y_train": y_train.detach().numpy(),
        "y_pred": y_preds[len(y_preds) - 1].detach().numpy(),
    }
)

plt.plot(hist, label="Training loss")
plt.legend()
plt.savefig("../loss.png")
plt.close()

plt.plot(dt.y_pred, label="Preds")
plt.plot(dt.y_train, label="Data")
plt.legend()
plt.savefig("../predict.png")
