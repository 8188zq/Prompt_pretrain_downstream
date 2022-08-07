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


class MutitaskTrainer(object):
    def __init__(self, args, model, optimizer, scheduler=None):
        """
        :param model: 模型
        :param optimizer: 优化器
        :param save_path: 模型存储位置
        """
        self.logger = fastNLP.logger
        self.save_path = args.save_path
        self.optim = optimizer
        # self.n_steps = args.n_steps
        self.n_epochs = args.n_epochs
        self.scheduler = scheduler
        self.eval_every = args.eval_every
        self.batch_size = args.batch_size
        self.save_every = args.save_every
        self.print_every = args.print_every
        self.anneal_rate = args.anneal_rate
        self.anneal_min = args.anneal_min
        self.n_prompt_tokens = args.n_prompt_tokens
        self.total_loss = 0.
        self.epochs = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.seed = args.seed
        self.model = model
        self.device = args.device
        # self.train_loader = TrainDataLoader(self.batch_size)
        # self.dev_loaders = get_dataloaders(batch_size=self.batch_size, split='validation')
        # data = OcnliDataset().get_dataset(split='downstream', batchszie=self.batch_size, seed=self.seed)
        data = CoteBdDataset().get_dataset(split='downstream', seed=self.seed)
        train_data =  data['train']
        eval_data = data['dev']
        test_data = CoteBdDataset().get_dataset(split='test') 
        self.trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,collate_fn=BasicDataset.collate) 
        self.evalloader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=True,collate_fn=BasicDataset.collate)
        self.testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True,collate_fn=BasicDataset.collate)
        self.logger.info(
            '-------------Trainer info-------------\n'
            f'Save path {self.save_path}\n'
            f'Number of epochs {self.n_epochs}\n'
            f'Batch size {self.batch_size}\n'
            f'Scheduler {self.scheduler}\n'
            f'Saves every {self.save_every} steps\n'
            '---------End of Trainer info----------\n'
        )

    def _write_summary(self, *args):
        with open(os.path.join(self.save_path, 'logs.txt'), 'a+') as f:
            print(*args, file=f)

    def _write_router(self):
        with open(os.path.join(self.save_path, 'router.txt'), 'a+') as f:
            print(f' - epoch {self.epochs}: {self.model.prompt_embed_model.prompt_logits2train}', file=f)

    def _preview_datasets(self):
        for i in range(num_datasets):
            batch, task_id = next(self.train_loader)
            info_str = ('-----------------------------------------\n'
                        f'Dataset [{Dataset_list[task_id].__name__}] with task id [{task_id.item()}].\n'
                        f'An example: [{tokenizer.decode(batch["input_ids"][0][self.n_prompt_tokens + 1:])}]\n'
                        f'Its label is [{tokenizer.decode(batch["input_ids"][0][batch["start_positions"][0]: batch["end_positions"][0] + 1])}]\n'
                        '-----------------------------------------\n')
            self.logger.info(info_str)
            self._write_summary(info_str)

    def train(self):
        # self._preview_datasets()
        for param in self.model.model.model.parameters():
            param.requires_grad = False
        for param in self.model.prompt_embed_model.parameters():
            param.requires_grad = True
        self.model.model.qa_outputs.weight.requires_grad = False # False
        self.model.to(self.device)
        total_time = time.time()
        self.logger.info("Start training...")
        for i_epoch in tqdm(range(self.n_epochs)):
            print("epoch{}:".format(i_epoch+1))
            self.total_loss = 0. 
            n_batchs = 0 
            for i,iter in enumerate(self.trainloader):
                for k, v in iter.items():
                    if iter[k] is not None:
                        iter[k] = v.to(self.device)
                self.model.prompt_embed_model.train()
                # self.model.model.model.eval()
                # self.model.model.qa_outputs.eval()
                self.model.model.eval()
                self.model.zero_grad()
                loss, acc = self.model(**iter)
                self.total_loss += loss.item()
                # self.steps += 1
                n_batchs += 1
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
            self.epochs+=1
            self._write_summary("train_loss", self.total_loss / n_batchs, i_epoch+1)
            self._write_router()
            self.logger.info(f" - Step {i_epoch+1}: router {self.model.prompt_embed_model.prompt_logits2train}")
            self.logger.info(f" - Step {i_epoch+1}: loss {self.total_loss / n_batchs}")
            self.logger.info(f" - Step {i_epoch+1}: temperature {self.model.prompt_embed_model.temperature}")
            if self.anneal_rate is not None and self.anneal_min is not None:
                self._anneal(i_epoch)
            if i_epoch % self.eval_every == self.eval_every - 1:
                dev_loss, dev_acc = self._eval_epoch()
                self._dump_model_state(f"{i_epoch}.th")
                eval_str = f"loss {dev_loss}, acc {dev_acc}"
                self.logger.info(eval_str)
                self._write_summary(eval_str)

                if dev_acc > self.best_acc:
                    self.best_acc = dev_acc
                    self.best_epoch = i_epoch
                    self.logger.info("Updating best model...")
                    self._save_model()
                    self.logger.info("Model saved.")
                    self.logger.info(f"Current best acc [{self.best_acc}] occurred at step [{self.best_epoch}].")
            if i_epoch == self.n_epochs-1 : print(f"Current best acc [{self.best_acc}] occurred at step [{self.best_epoch}].")
        path = os.path.join(self.save_path, "best.th")
        state = torch.load(path)
        self.model.prompt_embed_model.load_state_dict(state['skilled_prompts'])
        test_loss, test_acc = self._test_epoch()
        test_str = f"test loss {test_loss}, acc {test_acc}"
        self.logger.info(test_str)
        self._write_summary(test_str)
        self.logger.info("Training finished. Elapse {:.4f} hours.".format((time.time() - total_time) / 3600))
    def _eval_epoch(self):
        self.logger.info("Evaluating...")
        dev_losses = []
        dev_accs = []
        self.model.model.eval()
        self.model.prompt_embed_model.eval()
        with torch.no_grad():
            total_loss, total_acc,n_batchs = 0., 0.,0
            for i,iter in tqdm(enumerate(self.evalloader)):
                for k, v in iter.items():
                    if iter[k] is not None:
                        iter[k] = v.to(self.device)
                iter['is_train'] = False
                loss, acc = self.model(**iter)
                total_loss += loss.item()
                total_acc += acc
                n_batchs += 1
            total_loss /= n_batchs
            total_acc /= n_batchs
        return total_loss, total_acc

    def _test_epoch(self):
        self.logger.info("Evaluating...")
        self.model.model.eval()
        self.model.prompt_embed_model.eval()
        with torch.no_grad():
            total_loss, total_acc,n_batchs = 0., 0.,0
            for i,iter in tqdm(enumerate(self.testloader)):
                for k, v in iter.items():
                    if iter[k] is not None:
                        iter[k] = v.to(self.device)
                iter['is_train'] = False
                loss, acc = self.model(**iter)
                total_loss += loss.item()
                total_acc += acc
                n_batchs += 1
            total_loss /= n_batchs
            total_acc /= n_batchs
        return total_loss, total_acc


    def _save_model(self):
        save_path = os.path.join(self.save_path, "best.th")
        torch.save({
            'skilled_prompts': self.model.prompt_embed_model.state_dict(),
            'lmhead': self.model.model.qa_outputs.weight,
            'optimizer': self.optim.state_dict()
        }, save_path)

    def _dump_model_state(self, name):
        save_path = os.path.join(self.save_path, "models", name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'skilled_prompts': self.model.prompt_embed_model.state_dict(),
            'lmhead': self.model.model.qa_outputs.weight,
            'optimizer': self.optim.state_dict()
        }, save_path)

    def _anneal(self, i_step):
        self.model.prompt_embed_model.temperature = max(self.anneal_min,
                                                        self.model.prompt_embed_model.temperature * np.exp(
                                                            -self.anneal_rate * i_step))
