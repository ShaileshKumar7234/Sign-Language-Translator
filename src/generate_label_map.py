import numpy as np
import string

# Create a mapping: 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'
label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

# Save it
np.save('./models/label_map.npy', label_map)

print("[✅] label_map.npy with A-Z saved.")
