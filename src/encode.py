import numpy as np

RAW_PATH = "data/raw/optdigits-orig.txt"

def load_bitmap_file(path):
    inputs = []
    targets = []

    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # Skip header lines until we reach the first bitmap row
    i = 0
    while i < len(lines) and not (set(lines[i]) <= {"0", "1"} and len(lines[i]) == 32):
        i += 1

    # Now parse the dataset
    while i < len(lines):
        # Read 32 bitmap rows
        bitmap = lines[i:i+32]
        if len(bitmap) < 32:
            break
        i += 32

        # Flatten bitmap into 1024-length vector
        flat = []
        for row in bitmap:
            flat.extend([int(c) for c in row])

        # Read label line
        if i >= len(lines):
            break
        label_line = lines[i].strip()
        i += 1

        # Skip empty lines between samples
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

