import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm
import optuna
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples

#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, cfg, log):
        self.lr = cfg['learning_rate']
        self.num_epochs = cfg['n_epoch']
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        print(f'split: {split}')
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, cfg_optim , cfg_lr_scheduler, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optimizer = cfg_optim(model.parameters(), lr=self.lr)
        #scheduler
        if cfg_lr_scheduler == 'no':
            pass
        elif cfg_lr_scheduler == 'lambda':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:0.5 ** epoch)
        elif cfg_lr_scheduler == 'multiplicative':
            scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda epoch:0.5 ** epoch)
        elif cfg_lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma= 0.5)
        elif cfg_lr_scheduler == 'cosineannealing':
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=2, T_mult=1, eta_max=0.1,  T_up=5, gamma=0.5)
        
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optimizer.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
                if cfg_lr_scheduler == 'no':
                    pass
                else:
                    scheduler.step()
        return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def objective(trial):
    # setting hyperparameter for AutoML
    cfg = { 
        'learning_rate': trial.suggest_loguniform('learning_rate', 2e-5, 2e-4),
        'optimizer': trial.suggest_categorical('optimizer', [optim.RMSprop, optim.Adam, optim.AdamW]),
        'batch_size': trial.suggest_categorical('batch_size',[16,32,64]),
        'n_epoch': trial.suggest_int('n_epoch',2,5,step=1),
        'freeze' : trial.suggest_categorical('freeze',['no','last','last_transformer']),
        'lr_scheduler' : trial.suggest_categorical('lr_scheduler',['lambda', 'multiplicative', 'step', 'cosineannealing', 'no'])
    }
    #print edited configuration
    print(f"edited configuration: \n{cfg}")
    # define parser and arguments
    args = get_train_test_args()
    #setting args with cfg
    args.lr = cfg['learning_rate']
    args.num_epochs = int(cfg['n_epoch'])
    args.batch_size = int(cfg['batch_size'])
    args.optimizer = str(cfg['optimizer'])
    args.freeze = str(cfg['freeze'])
    args.lr_scheduler = str(cfg['lr_scheduler'])
    
    util.set_seed(args.seed)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if cfg['freeze']=='last':
        for param in model.distilbert.parameters():
            param.requires_grad = False
    elif cfg['freeze']=='last_transformer':
        for param in model.distilbert.parameters():
            param.requires_grad = False
        for param in model.distilbert.transformer.layer[5].parameters():
            param.requires_grad = True
            
    #Train
    print("TRAIN START...")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args.device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    trainer = Trainer(args, cfg, log)
    train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
    log.info("Preparing Validation Data...")
    val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg['batch_size'],
                            sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg['batch_size'],
                            sampler=SequentialSampler(val_dataset))
    best_scores = trainer.train(model, cfg['optimizer'], cfg['lr_scheduler'], train_loader, val_loader, val_dict)

    #Evaluate
    print("EVALUATE START...")
    args.device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    split_name = 'test' if 'test' in args.eval_dir else 'validation'
    log = util.get_logger(args.save_dir, f'log_{split_name}')
    trainer = Trainer(args, cfg, log)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
    model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    model.to(args.device)

    eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=cfg['batch_size'],
                             sampler=SequentialSampler(eval_dataset))
    eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                               eval_dict, return_preds=True,
                                               split=split_name)
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
    log.info(f'Eval {results_str}')
    # Write submission file
    sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(eval_preds):
            csv_writer.writerow([uuid, eval_preds[uuid]])

    score = 0
    for _, v in eval_scores.items():
        score += v
    return score
        

if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    import joblib

    study_result = './save/'
    pkl_path = os.path.join(study_result, 'result_optuna.pkl')
    joblib.dump(study, pkl_path)