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


def train(args):
    # Data preprocessing & construct torchtext dataloader
    train_df = pd.read_csv(os.path.join(args.train_data), encoding='utf-8')
    test_df = pd.read_csv(os.path.join(args.test_data), encoding='utf-8')

    train_df['News'] = train_df['News'].apply(preprocess_sent)
    test_df['News'] = test_df['News'].apply(preprocess_sent)

    tokenizer = Mecab()
    TEXT = data.Field(use_vocab=True, tokenize=tokenizer.morphs, include_lengths=True)
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
    # use Word2Vec 300D token for now
    word_embeddings = KeyedVectors.load_word2vec_format(args.word_embedding, binary=False, encoding='utf-8')
    # if word not in pre-trained embedding, random initialize
    # embeddings = nn.Embedding(num_embeddings=len(TEXT.vocab), embedding_dim=word_embeddings.vector_size)
    # nn.init.uniform_(embeddings.weight.data)
    # for i, w in enumerate(TEXT.vocab.itos):
    #     if w in word_embeddings:
    #         embeddings.weight.data[i] = torch.FloatTensor(word_embeddings[w])
    embeddings = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)

    # model & loss
    model = BiRNNTextClassifier(
        rnn_type=args.rnn_type, num_classes=len(LABEL.vocab), word_dim=args.word_dim,
        hidden_dim=args.hidden_dim, clf_dim=args.clf_dim,
        dropout_prob=args.dropout)
    print(model)

    if args.train_embedding:
        optimizer = optim.Adam(list(embeddings.parameters()) + list(model.parameters()))
    else:
        optimizer = optim.Adam(list(model.parameters()))
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
    parser.add_argument('--train-embedding', default=False, action='store_true', 
                        help='True if finetuning the word embedding')
    parser.add_argument('--save-dir', default='trained_models/baseline')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--max-epoch', default=50, type=int)
    parser.add_argument('--gpu', default='cuda:0')
    parser.add_argument('--allow-overwrite', default=True,
                        action='store_false',
                        help="If set, it doesn't check whether the save "
                             "directory already exists or not.")
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
                  'train_embedding': args.train_embedding}
    }
    with open(os.path.join(args.save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    train(args)


if __name__ == '__main__':
    main()
