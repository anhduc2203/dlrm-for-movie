# encoding: utf-8
# created by ducva on 2019, Nov 03

from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import pickle as pkl
import math
import copy


def create_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('userId')['rating'].count()

    # get unique index of user
    unique_users = ratings.userId.unique()
    # re-index for user
    user_to_index = {oldId:newId for newId, oldId in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)

    # get unique index of item
    unique_items = ratings.itemId.unique()
    # re-index for item
    item_to_index = {oldId:newId for newId, oldId in enumerate(unique_items)}
    new_items = ratings.itemId.map(item_to_index)

    n_users = unique_users.shape[0]
    n_items = unique_items.shape[0]

    user_item_ = pd.DataFrame({'userId':new_users, 'itemId':new_items})
    ratings_ = ratings['rating'].astype(np.float32)
    return (n_users, n_items), (user_item_, ratings_), (user_to_index, item_to_index)


class ReviewsIterator:
    """
    iteration through the dataset one batch after another
    """
    def __init__(self, user_item_, ratings_, batch_size=32, shuffle=True):
        user_item_, ratings_ = np.asarray(user_item_), np.asarray(ratings_)
        if shuffle:
            index = np.random.permutation(user_item_.shape[0])
            user_item_, ratings_ = user_item_[index], ratings_[index]

        self.user_item_ = user_item_
        self.ratings_ = ratings_
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(user_item_.shape[0]//batch_size))
        self._current_batch = 0


    def __iter__(self):
        return self


    def __next__(self):
        return self.next()


    def next(self):
        if self._current_batch >= self.n_batches:
            raise StopIteration()
        curr_batch = self._current_batch
        self._current_batch += 1
        batch_size_ = self.batch_size
        return self.user_item_[curr_batch*batch_size_:(curr_batch+1)*batch_size_], self.ratings_[curr_batch*batch_size_:(curr_batch+1)*batch_size_]


def batches(user_item_, ratings_, batch_size=32, shuffle=True):
    """
    split dataset into smaller chunks during training or validation process
    """
    for user_item_batch, ratings_batch in ReviewsIterator(user_item_, ratings_, batch_size, shuffle):
        user_item_batch = torch.LongTensor(user_item_batch)
        ratings_batch = torch.LongTensor(ratings_batch)
        yield user_item_batch, ratings_batch.view(-1, 1)


def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError("Layer configuration should be a single number or a list of numbers")


class DLRMNet(nn.Module):
    """
    create a dense network with embedding layers
    """
    def __init__(self, n_users, n_items, n_factors=50, embedding_dropout=0.1, hidden=10, dropouts=0.1):
        super(DLRMNet, self).__init__()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)
        output = hidden[-1]

        def gen_layers(n_in):
            # capture hidden and dropouts from outer function
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)

            for n_out, rate in zip(hidden, dropouts):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0:
                    yield nn.Dropout(rate)
                n_in = n_out

        self.user = nn.Embedding(n_users, n_factors)
        self.item = nn.Embedding(n_items, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(*list(gen_layers(n_factors*2)))
        self.fc = nn.Linear(output, 1)
        self.initialize()


    def forward(self, users, items, minmax=None):
        #print("Training for user: {} - item: {}".format(users, items))
        features = torch.cat([self.user(users), self.item(items)], dim=1)
        tmp = self.drop(features)
        tmp = self.hidden(tmp)
        pred = torch.sigmoid(self.fc(tmp))
        if minmax is not None:
            min_rating, max_rating = minmax
            pred = pred*(max_rating-min_rating+1) + min_rating - 0.5
        return pred


    def initialize(self):
        """initial values for embeddings and hidden layers"""
        def init(layer):
            if type(layer)==nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
        self.user.weight.data.uniform_(-0.05, 0.05)
        self.item.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)


def train_model(user_item_, ratings_, n_users, n_items):
    lr = 0.001
    weight_decay = 0.00001
    batch_size = 256
    epochs = 2
    patience = 10
    no_improvements = 0
    best_loss = np.inf
    best_weights = None
    history = []
    lr_history = []

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    cuda = torch.device(device)

    ui_train, ui_validation, r_train, r_validation = train_test_split(
        user_item_,
        ratings_,
        test_size=0.2,
        random_state=42
    )
    datasets = {'train':(ui_train, r_train), 'val':(ui_validation, r_validation)}
    dataset_sizes = {'train':len(ui_train), 'val':len(r_validation)}
    minmax = ratings_.rating.min().astype(float), ratings_.rating.max().astype(float)
    print(minmax)

    net = DLRMNet(
        n_users=n_users,
        n_items=n_items,
    )
    #print(net)

    net.to(cuda)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    iteractions_per_epoch = int(math.ceil(dataset_sizes['train']//batch_size))

    info = "Epoch {}, train: {}, val: {}"
    net.train()
    for epoch in range(epochs):
        stats = {'epoch':epoch+1, 'total':epochs}
        for phase in ('train', 'val'):
            training = phase == 'train'
            running_loss = 0.0
            n_batches = 0
            iterator = batches(*datasets[phase], shuffle=training, batch_size=batch_size)

            for batch in iterator:
                user_item, rating = [b.to(cuda) for b in batch] # type(batch) = tensor
                optimizer.zero_grad()

                #print("user: {}".format(user))
                #print("item: {}".format(item))
                print("user-item: {}".format(user_item[:, 0]))


                with torch.set_grad_enabled(training):
                    preds = net(user_item[:, 0], user_item[:, 1], minmax)

                    loss = criterion(preds, rating)
                    if training:
                        optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss/dataset_sizes[phase]
            stats[phase] = epoch_loss
            if phase == 'val':
                if epoch_loss<best_loss:
                    print("Loss improvement on epoch {}".format(epoch+1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(net.state_dict())
                    no_improvements += 0
                else:
                    no_improvements += 1

    history.append(stats)
    print(info.format(**stats))


if __name__=="__main__":
    path = "/home/ducva/Documents/data/ml-100k/"
    files = os.listdir(path)

    ratings = pd.read_csv(filepath_or_buffer=path + 'u.data', delim_whitespace=True,
                          names=['userId', 'itemId', 'rating', 'timestamp'])
    #print(ratings.info())
    #print(ratings.head(1))

    items_info = ['item id', 'item title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
                   'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv(filepath_or_buffer=path + 'u.item', delimiter='|', names=items_info, encoding="ISO-8859-1")
    # print(items.info())
    # print(items.head(1))
    #tabular_preview(ratings, items)
    (n_users, n_items), (user_item_, ratings_), _ = create_dataset(ratings)
    print("Embedding: {} users and {} items".format(n_users, n_items))
    print("Shape of dataset: {}".format(user_item_.shape))
    print("Shape of ratings: {}".format(ratings_.shape))

    #for user_item_batch, ratings_batch in batches(user_item_, ratings_, batch_size=4):
    #    print("user_item batch: {}".format(user_item_batch))
    #    print("ratings batch: {}".format(ratings_batch))
    #    break
    #print(DLRMNet(n_users, n_items))in
    train_model(user_item_, ratings, n_users, n_items)