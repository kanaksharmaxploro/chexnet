
import sys
import torch
import re
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from DensenetModels import DenseNet121
from DatasetGenerator import DatasetGenerator

chunk_file = sys.argv[1]
output_file = sys.argv[2]

pathDirData = "/kaggle/working/all_images"
pathModel = "./m-04072026-172922.pth.tar"
nnClassCount = 14
trBatchSize = 8
transResize = 256
transCrop = 224

model = DenseNet121(nnClassCount, True).cuda()
model = torch.nn.DataParallel(model).cuda()

modelCheckpoint = torch.load(pathModel)
pattern = re.compile(r"^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$")
state_dict = modelCheckpoint["state_dict"]
for key in list(state_dict.keys()):
    res = pattern.match(key)
    if res:
        new_key = res.group(1) + res.group(2)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
model.load_state_dict(state_dict)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = [transforms.Resize(transResize), transforms.CenterCrop(transCrop), transforms.ToTensor(), normalize]
transformSequence = transforms.Compose(transformList)

datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=chunk_file, transform=transformSequence)
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=0, shuffle=False, pin_memory=True)

outGT_list = []
outPRED_list = []

model.eval()

for i, (input, target) in enumerate(dataLoaderTest):
    if i % 100 == 0:
        print(f"Batch {i}", flush=True)
    outGT_list.append(target.cpu())
    with torch.no_grad():
        varInput = input.cuda(non_blocking=True)
        out = model(varInput)
        outPRED_list.append(out.data.cpu())

outGT = torch.cat(outGT_list, 0)
outPRED = torch.cat(outPRED_list, 0)

with open(output_file, "wb") as f:
    pickle.dump({"outGT": outGT, "outPRED": outPRED}, f)

print("Chunk done. Saved to", output_file)
