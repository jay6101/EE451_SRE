def label_unlabeled_idxs(train_text_file,labelled_train_text_file):

	labelled_train = []
	labeled_idxs = []
	unlabeled_idxs = []

	with open(labelled_train_text_file, "r") as f:
	    for line in f:
	        labelled_train.append((line[:-1].split())[0])

	i = 0
	with open(train_text_file, "r") as f:
	    for line in f:
	        if (line[:-1].split())[0] not in labelled_train:
	            unlabeled_idxs.append(i)
	        else:
	            labeled_idxs.append(i)
	        i += 1

	return labeled_idxs, unlabeled_idxs

