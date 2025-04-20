import argparse
import torch
import datetime
import json
import yaml
import os
import setproctitle
import pandas as pd
import ast
import numpy as np
import torch.nn as nn
import pickle

from main_model import CSDI_Shanghai
from dataset_shanghai import get_dataloader
from utils import train, evaluate

# setproctitle.setproctitle("")
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser(description="DiffuseApp")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="shanghai_fold0_20240307_123843")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/shanghai_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)


location_emb = pd.read_csv('data/shanghai/location_emb.txt', sep='\t', names=['idx', 'emb'])
# loc_emb_dic = {location_emb.iloc[i]['idx']: ast.literal_eval(location_emb.iloc[i]['emb'])
#                for i in range(len(location_emb))}
loc_emb_dic = [ast.literal_eval(location_emb.iloc[i]['emb']) for i in range(len(location_emb))]
loc_emb_dic.append([0.0 for _ in range(32)])
loc_emb_dic = torch.tensor(np.array(loc_emb_dic)).to(args.device).float()

with open("data/shanghai/appemb.pk", "rb") as f:
    _, vectors = pickle.load(f)
app_embed = torch.tensor(vectors).to(args.device).float()

model = CSDI_Shanghai(config, args.device, app_embed, loc_emb_dic).to(args.device)


if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)