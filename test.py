import torch
from model import PretrainPrompt
from trainer import MutitaskTrainer
from dataload import *
import torch
import argparse
from torch.optim import AdamW
import os

import copy
import numpy as np
from modeling_cpt import CPTForConditionalGeneration
import torch
from torch.utils.data import DataLoader
# from dataload import TrainDataLoader, get_dataloaders, num_datasets, tokenizer, Dataset_list
from dataload import *
from tqdm import tqdm
import os
import fastNLP
import time
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli



parser = argparse.ArgumentParser()
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--save_every", default=10000, type=int)
parser.add_argument("--batch_size", default=6, type=int)
# parser.add_argument("--n_steps", default=2000000, type=int)
parser.add_argument("--n_epochs", default=2000, type=int)
parser.add_argument("--print_every", default=1, type=int)
parser.add_argument("--eval_every", default=1, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--n_prompts", default=6, type=int)
parser.add_argument("--seed", default=41, type=int)
parser.add_argument("--lr_router", default=1e-4, type=float)
parser.add_argument("--lr_prompt", default=5e-5, type=float)
parser.add_argument("--anneal_rate", default=None, type=float)
parser.add_argument("--anneal_min", default=.1, type=float)
parser.add_argument("--init_temperature", default=1., type=float)
parser.add_argument("--step_size1", default=None, type=int)
parser.add_argument("--step_size2", default=10000, type=int)
parser.add_argument("--gamma1", default=.1, type=float)
parser.add_argument("--gamma2", default=.1, type=float)
parser.add_argument("--is_downstream", default=True, type=bool)
args = parser.parse_args()



save_path = f'./results/PromptTokens{args.n_prompt_tokens}_BatchSize{args.batch_size}_NPrompts{args.n_prompts}_LrRouter{args.lr_router}_LrPrompt{args.lr_prompt}_AnnealParams{args.init_temperature};{args.anneal_rate};{args.anneal_min}'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
args.save_path = save_path
torch.manual_seed(args.seed)

test_data = Cmrc2018Dataset().get_dataset(split='test') 
testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,collate_fn=BasicDataset.collate)


model = PretrainPrompt(args.intrinsic_dim, args.n_prompt_tokens, num_datasets, args.n_prompts, args.init_temperature, args.is_downstream)
path = os.path.join(save_path, "best.th")
state = torch.load(path)
model.prompt_embed_model.load_state_dict(state['skilled_prompts'])
model.model.qa_outputs.weight = state['lmhead']

# test_loss, test_acc = args._test_epoch()


# args.logger.info("Evaluating...")
model.model.eval()
model.prompt_embed_model.eval()
model.to(args.device)
with torch.no_grad():
    total_loss, total_acc,n_batchs = 0., 0.,0
    for i,iter in tqdm(enumerate(testloader)):
        for k, v in iter.items():
            if iter[k] is not None:
                iter[k] = v.to(args.device)
        iter['is_train'] = False
        loss, acc = model(**iter)
        total_loss += loss.item()
        total_acc += acc
        n_batchs += 1
    total_loss /= n_batchs
    total_acc /= n_batchs



test_str = f"test loss {total_loss}, acc {total_acc}"
print(test_str)
