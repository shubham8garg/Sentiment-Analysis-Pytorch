import os
from string import punctuation
from collections import Counter
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model import SentimentLSTM
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

def read_files_in_location(location):
    files = os.listdir(location)
    read_list = []
    for filename in files:
        with open(location+'/'+filename, 'r') as f:
            read_list.append(f.read().lower())
    return read_list

def makeLabels():
    labels = []
    for i in range(12500):
        labels.append('negative')
    for i in range(12500):
        labels.append('positive')
    return labels   

def cleanLine(line):
    for x in line:
        if x in punctuation:
            line = line.replace(x,'')
    return line

def tokenizeReviews(reviews):
    all_text = ' '.join(reviews)
    words = all_text.split()
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    return vocab_to_int

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)    
    for i, review in enumerate(reviews_int):
        review_len = len(review)        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = review+zeroes
        elif review_len > seq_length:
            new = review[0:seq_length]        
        features[i,:] = np.array(new)    
    return features

def main():
    negative_reviews = read_files_in_location('aclImdb/train/neg')
    positive_reviews = read_files_in_location('aclImdb/train/pos')
    reviews = negative_reviews + positive_reviews
    labels = makeLabels()
    reviews_clean = [cleanLine(c) for c in reviews]
    vocab_to_int = tokenizeReviews(reviews_clean)
    reviews_int = []
    for review in reviews_clean:
        r = [vocab_to_int[w] for w in review.split()]
        reviews_int.append(r)
    encoded_labels = [1 if label =='positive' else 0 for label in labels]
    encoded_labels = np.array(encoded_labels)
    features = pad_features(reviews_int, seq_length=500)
    features, encoded_labels = shuffle(features, encoded_labels, random_state=8)
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.1, random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=8)
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # dataloaders
    batch_size = 50
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

    print(net)
    
    # loss and optimization functions
    lr=0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    # training params

    epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip=5 # gradient clipping

    train_on_gpu = torch.cuda.is_available()
    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = inputs.type(torch.LongTensor)
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))

if __name__ == "__main__":
    main()