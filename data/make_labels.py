# num_class = 360

# labels = {}
# for i in range(num_class):
#     labels[i] = str(i)
labels = {}
for i in [0,45,90,135,180,225,270,315]:
    labels[i] = str(i)
import json
with open('labels.json', 'w') as fp:
    json.dump(labels, fp, indent=4)