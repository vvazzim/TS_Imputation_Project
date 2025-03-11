import pandas as pd
import numpy as np
import torch

from sklearn.impute import KNNImputer

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) GPU")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU")

# Google colab
# DEFAULT_PATH = "/content"
# Local
DEFAULT_PATH = "."

INPUT_TRAIN = DEFAULT_PATH + "/data/train.csv"
INPUT_TEST = DEFAULT_PATH + "/data/test.csv"

TEMPLATE_SUBMISSION = DEFAULT_PATH + "/data/step1_submission_example.csv"
OUTPUT_SUBMISSION = "submission.csv"


df_train = pd.read_csv(INPUT_TRAIN,index_col=0)
data_train = df_train.values

df_test = pd.read_csv(INPUT_TEST,index_col=0)
data_test = df_test.values

# data_train_tensor = torch.tensor(data_train, dtype=torch.float32, device=device)
data_test_tensor = torch.tensor(data_test, dtype=torch.float32, device=device)

data_test_np = data_test_tensor.cpu().numpy()



imputer = KNNImputer(n_neighbors=5)


# Compute the values for the missing positions
# data_test_imputed_tensor = impute_interpolate(data_test_tensor)
# data_test_imputed_tensor = impute_knn(data_test_tensor)
data_test_imputed = imputer.fit_transform(data_test_np)


# Export the submission
submission_example = pd.read_csv(TEMPLATE_SUBMISSION)
missing_positions = submission_example["ID"].values

flattened_test_imputed = data_test_imputed.flatten()
predicted_values = flattened_test_imputed[missing_positions]

submission = pd.DataFrame({"ID": missing_positions, "TARGET": predicted_values})
submission.to_csv(OUTPUT_SUBMISSION, index=False)
print("Successfully exported to", OUTPUT_SUBMISSION)