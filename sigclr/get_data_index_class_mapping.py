from torchsig.datasets.sig53 import Sig53
import os
import csv



root_qa = os.getenv("ROOT_VAL","/project/def-msteve/torchsig/sig53_qa/")

qa_dataset = Sig53(
    root=root_qa,
    train=False,
    impaired=False,
)

data_index_to_class_map = []
count = 0
for x, y in qa_dataset:
    data_index_to_class_map.append((count,y[0],Sig53.convert_idx_to_name(y[0])))
    count += 1
    print(count, len(qa_dataset))
    if count == len(qa_dataset):
        break

with open('data_index_to_class_map.csv', mode='w', newline='') as file:
    writer=csv.writer(file)
    writer.writerows(data_index_to_class_map)
