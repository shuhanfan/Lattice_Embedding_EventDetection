import numpy as np
import json
lines=open("dataset/data_middle.json","r").readlines()
f=open("dataset/data_small.json","w")
cut_length=10
label={}
# label=["a","b","c"]
# new_label = np.random.randint(3, size=cut_length)
new_lines = []
for index,line in enumerate(lines):
	flag=0
	new_line = json.loads(line)
	for item in new_line["detection_label"]:
		if item!="NEGATIVE":
			flag=1
			break
	if flag==1:
		f.write(json.dumps(new_line)+"\n")

f.close()