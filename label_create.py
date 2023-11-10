import random 
import numpy as np

train_text_file = "train_list.txt"

train = []
with open(train_text_file, "r") as f:
    for line in f:
        train.append(line[:-1])

random.shuffle(train)
print(len(train))

label_idx = np.random.randint(low=0, high=len(train), size=7000, dtype=int)

labeled_train = []
for idx in label_idx:
	labeled_train.append(train[idx])

with open("labeled_train_list.txt", 'w') as f:
    for item in labeled_train:
        f.write("%s\n" % item)

print(len(labeled_train))