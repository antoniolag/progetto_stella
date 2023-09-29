import os
classes = []
with open("../stella_dataset/labels.txt","r") as file:
	lines = file.readlines()
	for line in lines:
		line = line.replace(",","")
		classes.append(line.split()[0])

print("Classes: " )
print(classes)
id = 1
with open("../stella_dataset/labels_map.pbtxt","w") as f:
	for label in classes:
		f.write('item { \n')
		f.write('\tname:\'{}\'\n'.format(label))
		f.write('\tid:{}\n'.format(id))
		f.write('}\n')
		id = id+1;