"""Train the model."""

import argparse
import os
from collections import Counter

import pandas as pd
import os
import torch
import yaml
from torch import nn, optim
from torch.nn import init
from torch.nn.utils import clip_grad_norm_
from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from utils.preprocessing import preprocess_sent
from models import BiRNNTextClassifier
from konlpy.tag import Mecab
from gensim.models.keyedvectors import KeyedVectors
from data import DataFrameDataset, SeriesExample

from models.cnn_based import CNNClassifier


def train(args):
    # Data preprocessing & construct torchtext dataloader
    train_df = pd.read_csv(os.path.join(args.train_data), encoding='utf-8')
    test_df = pd.read_csv(os.path.join(args.test_data), encoding='utf-8')

    train_df['News'] = train_df['News'].apply(preprocess_sent)
    test_df['News'] = test_df['News'].apply(preprocess_sent)

    if args.tokenizer == 'mecab':
        tokenizer = Mecab()
        tokenize = tokenizer.morphs
    else:
        tokenize = None

    TEXT = data.Field(use_vocab=True, tokenize=tokenize, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=True, is_target=True, unk_token=None)

    fields = { 'Topic' : LABEL, 'News' : TEXT }
    train_dataset = DataFrameDataset(train_df, fields)
    test_dataset = DataFrameDataset(test_df, fields)

    vectors = Vectors(name=args.word_embedding)
    TEXT.build_vocab(train_dataset, min_freq=10, vectors=vectors) 
    LABEL.build_vocab(train_dataset)

    train_loader = data.BucketIterator(
        dataset=train_dataset, batch_size=args.batch_size, device=args.gpu, sort_within_batch=True,
        train=True, repeat=False)
    test_loader = data.BucketIterator(
        dataset=test_dataset, batch_size=args.batch_size, device=args.gpu,
        train=False, repeat=False)

    # word embedding model
    embeddings = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=True)

    # model & loss
    # ADDED
    if args.flag:
        filter_sizes = []
        for i in range(args.filter_count):
            filter_sizes.append(int(input('Enter filter size : ')))
        print('filter_sizes =', filter_sizes)
        model = CNNClassifier(
            num_classes = len(LABEL.vocab), 
            word_dim = args.word_dim, 
            hidden_dim = args.hidden_dim, 
            clf_dim = args.clf_dim, 
            n_filters = args.n_filters, 
            filter_sizes = filter_sizes, 
            dropout_prob=0)
    else:
        model = BiRNNTextClassifier(
            rnn_type=args.rnn_type, num_classes=len(LABEL.vocab), word_dim=args.word_dim,
            hidden_dim=args.hidden_dim, clf_dim=args.clf_dim,
            dropout_prob=args.dropout)
    print(model)

    optimizer = optim.Adam(list(embeddings.parameters()) + list(model.parameters()))
    
    loss_weight = None
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    # scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma)

    if args.gpu:
        embeddings.cuda(args.gpu)
        model.cuda(args.gpu)
        criterion.cuda(args.gpu)

    def run_iter(batch):
        inputs, length = batch.News
        inputs = embeddings(inputs)
        logit = model(inputs=inputs, length=list(length))

        label = batch.Topic
        loss = criterion(input=logit, target=label)
        accuracy = torch.eq(logit.max(1)[1], label).float().mean()
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
        return loss.data.item(), accuracy.data.item()

    def validate():
        loss_sum = accuracy_sum = 0
        num_batches = len(test_loader)
        embeddings.eval()
        model.eval()
        for valid_batch in test_loader:
            loss, accuracy = run_iter(valid_batch)
            loss_sum += loss
            accuracy_sum += accuracy
        return loss_sum / num_batches, accuracy_sum / num_batches

    iter_count = 0
    best_valid_accuracy = -1
    for cur_epoch in range(args.max_epoch):
        if cur_epoch == args.finetune_embedding:
            embeddings.weight.requires_grad = True
        for train_batch in tqdm(train_loader, desc=f'Epoch {cur_epoch:2d}'):
            if not model.training:
                embeddings.train()
                model.train()
            train_loss, train_accuracy = run_iter(train_batch)
            iter_count += 1

        # scheduler.step()
        print(f'* Epoch {cur_epoch} finished')
        valid_loss, valid_accuracy = validate()
        print(f'  - Valid Loss = {valid_loss:.4f}')
        print(f'  - Valid Accuracy = {valid_accuracy:.4f}')

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            model_filename = (f'model-{cur_epoch}-{valid_accuracy:.4f}.pt')
            model_path = os.path.join(args.save_dir, model_filename)
            state_dict = {'embeddings': embeddings.state_dict(), 'model':  model.state_dict()}
            torch.save(state_dict, model_path)
            torch.save(state_dict, os.path.join(args.save_dir,'final.pt'))
            print(f'  - Saved the new best model to {model_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', default='data/BalancedNewsCorpus_train.csv',
                        help='The path of training data csv.')
    parser.add_argument('--test-data', default='data/BalancedNewsCorpus_test.csv',
                        help='The path of test data csv.')
    parser.add_argument('--rnn-type', default='lstm')
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--word-embedding', default='word_embeddings/Word2Vec_300D_token.model',
                        help='The path of word embedding file.')
    parser.add_argument('--finetune-embedding', default=-1, type=int,  
                        help='start epoch of finetuning word embedding (default: do not finetune)')
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--save-dir', default='trained_models/baseline')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--max-epoch', default=100, type=int)
    parser.add_argument('--gpu', default='cuda:0')
    parser.add_argument('--allow-overwrite', default=True,
                        action='store_false',
                        help="If set, it doesn't check whether the save "
                             "directory already exists or not.")

    # ADDED
    ###################
    ##   CNN MODEL   ##
    ###################
    parser.add_argument('--flag', default=True, type=bool, help='If True, use my model(CNN)')
    parser.add_argument('--n_filters', default='250', type=int)
    parser.add_argument('--filter_count', default='3', type=int)

    ##### RESULTS #####
    # 50 epoch , filter size 250, kernel size [6, 6, 6] > 78.5%
    # 50 epoch , filter size 250, kernel size [10, 10, 10] > 76.5%
    # 50 epoch , filter size 250, kernel size [6, 7, 8] > 77.3%
    # 50 epoch , filter size 250, kernel size [6, 6, 6, 6, 6] > 77.3%
    # 50 epoch , filter size 150, kernel size [4, 5, 6] > 77.8%
    # 50 epoch , filter size 400, kernel size [4, 5, 6] > 77.8%
    ###################
            
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=args.allow_overwrite)

    config = {
        'word_embedding': {'word_dim': args.word_dim, 
                           'word_embedding': args.word_embedding},
        'model': {'word_dim': args.word_dim,
                  'hidden_dim': args.hidden_dim,
                  'clf_dim': args.clf_dim,
                  'dropout_prob': args.dropout,
                  'rnn_type': args.rnn_type},
        'train': {'batch_size': args.batch_size,
                  'finetune_embedding': args.finetune_embedding}
    }
    with open(os.path.join(args.save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    train(args)


if __name__ == '__main__':
    main()


