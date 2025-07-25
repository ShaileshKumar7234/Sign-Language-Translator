import numpy as np

label_map = np.load('models/label_map.npy', allow_pickle=True).item()
print("Label Map Contents:")
for k, v in label_map.items():
    print(f"{k}: {v}")
