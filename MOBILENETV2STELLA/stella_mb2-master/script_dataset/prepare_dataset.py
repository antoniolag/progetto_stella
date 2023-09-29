import os



train_folder = "../data/dataset/"

if os.path.exists(train_folder):
	print('directory già presente')
else:
	os.system("mkdir -p " + train_folder)
os.system("cp  ../stella_dataset/labels_map.pbtxt " + train_folder)

train_folder = "../data/dataset/train"
if os.path.exists(train_folder):
	print('directory già presente')
else:
	os.system("mkdir -p " + train_folder)


with open("../stella_dataset/train.txt") as f:
	lines = f.readlines()
	for line in lines:
		image = line.split()[0] + ".jpg"
		label = line.split()[0] + ".xml"
		os.system("cp ../stella_dataset/images/" + image + " " + train_folder)
		os.system("cp ../stella_dataset/annotations/" + label + " " + train_folder)


val_folder = "../data/dataset/val"

if os.path.exists(val_folder):
	print('directory già presente')
else:
	os.system("mkdir -p " + val_folder)


val_folder = "../data/dataset/val"
if os.path.exists(val_folder):
	print('directory già presente')
else:
	os.system("mkdir -p " + val_folder)

with open("../stella_dataset/val.txt") as f:
	lines = f.readlines()
	for line in lines:
		image = line.split()[0] + ".jpg"
		label = line.split()[0] + ".xml"
		os.system("cp ../stella_dataset/images/" + image + " " + val_folder)
		os.system("cp ../stella_dataset/annotations/" + label + " " + val_folder)



test_folder = "../data/dataset/test"
if os.path.exists(test_folder):
	print('directory già presente')
else:
	os.system("mkdir -p " + test_folder)

with open("../stella_dataset/test.txt") as f:
	lines = f.readlines()
	for line in lines:
		image = line.split()[0] + ".jpg"
		label = line.split()[0] + ".xml"
		os.system("cp ../stella_dataset/images/" + image + " " + test_folder)
		os.system("cp ../stella_dataset/annotations/" + label + " " + test_folder)



