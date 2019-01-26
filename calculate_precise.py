import numpy as np
lines=open("exp_data_small.txt").readlines()
pred_right=0
pred_total=0
gold_total=0
for line in lines:
	# print(line)
	if "Epoch:" in line:
		print(line)
		precision=0.0 if pred_total==0 else float(pred_right)/float(pred_total)
		recall=0.0 if gold_total==0 else float(pred_right)/float(gold_total)
		F = 0.0 if precision + recall==0 else 2 * precision * recall / (precision + recall)
		print("precision,recall,F:",precision,recall,F,"\n")
		pred_right=0
		pred_total=0
		gold_total=0
	#Pred_total: 20, Pred_right: 0, Gold_total: 1
	elif "Pred_total" in line:
		line=line.split(",")
		pred_total += int(line[0].split(":")[1][1:])
		pred_right += int(line[1].split(":")[1][1:])
		gold_total += int(line[2].split(":")[1][1:])
		# print(pred_total,pred_right,gold_total)

