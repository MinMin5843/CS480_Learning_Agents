import numpy as np

RAW_PATH = "data/raw/optdigits-orig.txt"

def load_bitmap_file(path):
    """
    Loads the original Optdigits bitmap dataset and converts it from a 32-by-32 character image into a 
    flattened 1024-dimensional fearture vector.

    Args:
        path: the path to the raw optdigits file containing the data.
    
    Yields:
        The input and target files for neural network training.
        
    """
    inputs = []
    targets = []

    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines) and not (set(lines[i]) <= {"0", "1"} and len(lines[i]) == 32):
        i += 1

    while i < len(lines):
        bitmap = lines[i:i+32]
        if len(bitmap) < 32:
            break
        i += 32

        flat = []
        for row in bitmap:
            flat.extend([int(c) for c in row])

        if i >= len(lines):
            break
        label_line = lines[i].strip()
        i += 1

        if label_line == "":
            continue

        label = int(label_line)

        inputs.append(flat)
        targets.append(label)

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.int64)

X, y = load_bitmap_file(RAW_PATH)

np.save("data/inputs.npy", X)
np.save("data/targets.npy", y)

print("Task 1 complete:", X.shape, y.shape)

