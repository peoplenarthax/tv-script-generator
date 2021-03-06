
# TV Script Generation

In this project,I generate my own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs.  We use the publicly available [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons.  The Neural Network will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.

## Get the Data

The data is already provided for you in `./data/Seinfeld_Scripts.txt` and you're encouraged to open that file and look at the text. 
* As a first step, we'll load in this data and look at some samples. 
* Then, you'll be tasked with defining and training an RNN to generate a new script!


```python
# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
```

## Explore the Data
This is how a sample of the dataset looks like:


```python
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

view_line_range = (0, 20)
print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 46367
    Number of lines: 109233
    Average number of words in each line: 5.544240293684143
    
    The lines 0 to 20:
    jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go. 
    
    jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. 
    
    george: are you through? 
    
    jerry: you do of course try on, when you buy? 
    
    george: yes, it was purple, i liked it, i dont actually recall considering the buttons. 
    
    jerry: oh, you dont recall? 
    
    george: (on an imaginary microphone) uh, no, not at this time. 
    
    jerry: well, senator, id just like to know, what you knew and when you knew it. 
    
    claire: mr. seinfeld. mr. costanza. 
    
    george: are, are you sure this is decaf? wheres the orange indicator? 
    

## Implement Pre-processing Functions

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following **tuple** `(vocab_to_int, int_to_vocab)`


```python
import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    ## Order by most common word
    vocab = Counter(text).most_common()
    ## Start index at 1 in case we need to padd results later
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab
```


### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.

This dictionary will be used to tokenize the symbols and add the delimiter (space) around it.  This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    ## Would be great if text.punctuation had a name dict
    punctuation_dict = {
        '.': '||Period||', 
        ',': '||Comma||',
        '"': '||Quotation_Mark||', 
        ';': '||Semicolon||',
        '!': '||Exclamation_Mark||', 
        '?': '||Question_Mark||',
        '(': '||Left_Parentheses||', 
        ')': '||Rigth_Paranthesis||',
        '-': '||Dash||', 
        '\n': '||Return||',
    }
    
    return punctuation_dict
```


## Pre-process all the data and save it

We used some external helpers to do some preprocessing on the data, you can check them [here](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-tv-script-generation/helper.py)
```python
# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point

```python
import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network

### Check Access to GPU

```python
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
```

## Input
Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

You can create data with TensorDataset by passing in feature and target tensors. Then create a DataLoader as usual.

### Batching
Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.

>You can batch words using the DataLoader, but it will be up to you to create `feature_tensors` and `target_tensors` of the correct size and content for a given `sequence_length`.

For example, say we have these as input:
```
words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4
```

Your first `feature_tensor` should contain the values:
```
[1, 2, 3, 4]
```
And the corresponding `target_tensor` should just be the next "word"/tokenized word value:
```
5
```
This should continue with the second `feature_tensor`, `target_tensor` being:
```
[2, 3, 4, 5]  # features
6             # target
```


```python
from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_batches = len(words)//batch_size
    # Like in the Sentient analysis, we just want full batches
    words = words[:n_batches*batch_size]
    target_len = len(words) - sequence_length
    features, target = [], []
    
    # The labels is the last word of the sentence
    for sequence_start in range(0, target_len):
        sequence_end = sequence_length + sequence_start
        features_batch = words[sequence_start:sequence_end]
        features.append(features_batch)
        target_batch =  words[sequence_end]  
        target.append(target_batch)

    # Create DataSet
    data = TensorDataset(torch.from_numpy(np.asarray(features)), torch.from_numpy(np.asarray(target)))

    dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)

    return dataloader
```

### Test the dataloader 
Below, we're generating some test text data and defining a dataloader using the function you defined, above. Then, we are getting some sample batch of inputs `sample_x` and targets `sample_y` from our dataloader.

#### Sizes
Your sample_x should be of size `(batch_size, sequence_length)` or (10, 5) in this case and sample_y should just have one dimension: batch_size (10). 

#### Values

You should also notice that the targets, sample_y, are the *next* value in the ordered test_text data. So, for an input sequence `[ 28,  29,  30,  31,  32]` that ends with the value `32`, the corresponding output should be `33`.


```python
import numpy as np
numerical_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

numerical_batched = batch_data(numerical_sequence, 2, 5)
# Printing batches
dataiter = iter(numerical_batched)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

```

    Sample input size:  torch.Size([5, 2])
    Sample input: 
     tensor([[ 7,  8],
            [ 1,  2],
            [ 2,  3],
            [ 3,  4],
            [ 5,  6]])
    
    Sample label size:  torch.Size([5])
    Sample label: 
     tensor([ 9,  3,  4,  5,  7])


---
## Build the Neural Network
I am using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module). I implemented both GRU and LSTM.

**The output of this model should be the *last* batch of word scores** after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.

```python
import torch.nn as nn

## I started by using a GRU as it is a newer algorithm and it removes part of the complexity (it has less gates)
class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        print("USING GRU")
        
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Even a non-trained word embedding can help us, usually loss will decay faster in a pre-trained word embedding, but for this exercise we can skip it
        # https://towardsdatascience.com/pre-trained-word-embeddings-or-embedding-layer-a-dilemma-8406959fd76c 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # The GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        
        # Helping us with regularisation
        self.dropout = nn.Dropout(dropout)
        
        # Output 
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        batch_size = nn_input.size(0)
        nn_input = nn_input.long()
        # embeddings and gru output
        embeds = self.embedding(nn_input)
        gru_out, hidden = self.gru(embeds, hidden)

        # stack up gru outputs
        gru_out = gru_out.contiguous().view(-1, self.hidden_dim)

        # dropout, fully-connected layer and sigmoid to output
        out = self.dropout(gru_out)
        out = self.fc(out)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        # get ONLY the last batch of labels
        out = out[:, -1] 

        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        weight = next(self.parameters()).data
        # initialize hidden state with zero weights, and move to GPU if available
        # Return only hidden layer, in LSTM we return hidden state and cell state 
        # https://discuss.pytorch.org/t/gru-cant-deal-with-self-hidden-attributeerror-tuple-object-has-no-attribute-size/17283
        if (train_on_gpu):
            hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        else:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

        return hidden
    def getType(self):
        return "GRU"
```


```python
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        print("Using LSTM")
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Even a non-trained word embedding can help us, usually loss will decay faster in a pre-trained word embedding, but we can skip it
        # https://towardsdatascience.com/pre-trained-word-embeddings-or-embedding-layer-a-dilemma-8406959fd76c 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # The LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        
        # Output 
        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        batch_size = nn_input.size(0)
        nn_input = nn_input.long()
        # embeddings and gru output
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up gru outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout, fully-connected layer and sigmoid to output
        out = self.fc(out)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        # get ONLY the last batch of labels
        out = out[:, -1] 

        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        #self.batch_size = batch_size

        weight = next(self.parameters()).data

        # initialize hidden state with zero weights, and move to GPU if available
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

    def getType(self):
        return "LSTM"
```


### Define forward and backpropagation

It should return the average loss over a batch and the hidden state returned by a call to `RNN(inp, hidden)`. Recall that you can get this loss by computing it, as usual, and calling `loss.item()`.

```python
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    if(train_on_gpu and "cuda" not in type(inp).__name__):
        inp, target = inp.cuda(), target.cuda()
        rnn.cuda()
    # Without separated hidden state we will do back prop thorugh all the RNN |history 
    if (rnn.getType() == 'GRU'):
        hidden_state = hidden.data
    else:
        hidden_state = tuple([each.data for each in hidden])

    # zero accumulated gradients
    rnn.zero_grad()

    # get the output from the model
    output, hidden_state = rnn(inp, hidden_state)

    # calculate the loss and perform backprop
    loss = criterion(output, target)
    loss.backward()
    # prevent exploding gradient problem in RNNs with `clip_grad_norm
    clip=5
    nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden_state
```

## Neural Network Training

### Train Loop

The training loop is implemented for you in the `train_decoder` function. This function will train the network over all the batches for the number of epochs given. The model progress will be shown every number of batches. This number is set with the `show_every_n_batches` parameter. You'll set this parameter along with other parameters in the next section.


```python
def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100, epoch_history=[]):
    batch_losses = []
    
    rnn.train()
    
    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                epoch_history.append({epoch_i, n_epochs, np.average(batch_losses)})
                state = {'epoch': 4, 'state_dict': rnn.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch_history': epoch_history }
                torch.save(state, rnnType + str(batch_size) + str(embedding_dim) + str(hidden_dim) + ".pth.tar")
                batch_losses = []

    # returns a trained rnn
    return rnn
```

### Hyperparameters

Set and train the neural network with the following parameters:
- Set `sequence_length` to the length of a sequence.
- Set `batch_size` to the batch size.
- Set `num_epochs` to the number of epochs to train for.
- Set `learning_rate` to the learning rate for an Adam optimizer.
- Set `vocab_size` to the number of uniqe tokens in our vocabulary.
- Set `output_size` to the desired size of the output.
- Set `embedding_dim` to the embedding dimension; smaller than the vocab_size.
- Set `hidden_dim` to the hidden dimension of your RNN.
- Set `n_layers` to the number of layers/cells in your RNN.
- Set `show_every_n_batches` to the number of batches at which the neural network should print progress.


```python
# Data params
# Sequence Length 
# https://stats.stackexchange.com/questions/158834/what-is-a-feasible-sequence-length-for-an-rnn-to-model
sequence_length = 8  # of words in a sequence
# Batch Size
# https://twitter.com/ylecun/status/989610208497360896?lang=en
batch_size = 128

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)
```


```python
# Training parameters
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int) + 1 # We started with index 1
# Output size
output_size = vocab_size # we want to generate a text :D
# Embedding Dimension
embedding_dim = 256 # Slightly smaller than in the previous projects since our vocabulary is smaller
# Hidden Dimension
hidden_dim = 256
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 1000
```

**My choice of hyperparameters:** 
So first I tried between GRU and LSTM, and both demonstrated to be fairly similar for this scenario. (Had some problems with the loss since I was inspired by our sentient analysis RNN and I used sigma in the output layer.

For the sequence_length I played with long ones and shorter ones, shorter ones (10 - 20) had a better convergence in the end. Probably because it is closer to the kind of output we want to generate. This last one has 8 words as closer to the average per sentence in the original data.
For batch sizes, based on Yan LeCun, we should never use something bigger than 32 for minibatches, but then our model became very very slow, I increased the size to 128 and it seemed to be still effective. Probably in the long run 32 minibatches allow micro optimizations on this.
For hidden and embedding I tried different coombinations based on the lectures of Udacity, embedding needs to be slightly over 200 hundred to create good relationship patterns between words. For hidden dimensions I did not have a clear output, just made it bigger than the embedding size.

One of the changes that helped this converge faster was to get rid of the dropout layer for the LSTM implementation, since the LSTM and GRU RNN already come with Dropout and you may fall in the pit of vanishing gradient.

Learned a bit more about LSTM vs GRU implementations in https://blog.floydhub.com/gru-with-pytorch/

### Train

You should also experiment with different sequence lengths, which determine the size of the long range dependencies that a model can learn.

This function loads back teh state of the model and the optimizer to can resume it.

One important step to avoid problems and wonder where did you fail is to transform all the values of the state into GPU (well, if you are using GPU)


```python
""" 
LOAD A CHECKPOINT
"""
def load_checkpoint(model, optimizer, epoch_history, filename='checkpoint'):
    start_epoch = 0

    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename + '.pth.tar')

    # This is just because I wanted to extend the epoch size, otherwise I should substract it from the epoch configuration
    start_epoch = 20 - checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Transforming the recover state to GPU only if they are saved as tensors
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Recovering the loss history from the previous epochs            
    epoch_history = checkpoint['epoch_history']
    print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))

    return model, optimizer, start_epoch, epoch_history
```


```python
# This line helps with debugging if something went wrong with CUDA
CUDA_LAUNCH_BLOCKING=1

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
rnnType = rnn.getType()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

epoch_history = []
```

This piece of code recovers the RNN models and optimizer models for a given configuration

```python
"""
CHECKPOINT RECOVER
"""
# Only run if we want to start from a checkpoint
rnn, optimizer, num_epochs, epoch_history = load_checkpoint(rnn, optimizer, epoch_history, rnnType + str(batch_size) + str(embedding_dim) + str(hidden_dim))
```


```python
# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches, epoch_history)


# I modified the output manually afterwards to show only the progress between epochs
```

    Training for 10 epoch(s)...
    Epoch:    1/10    Loss: 5.16559678864479
    
    Epoch:    2/10    Loss: 4.042971907107811
    
    Epoch:    3/10    Loss: 3.814186099825836
    
    Epoch:    4/10    Loss: 3.6761786748723284
    
    Epoch:    5/10    Loss: 3.5914850631138173
    
    Epoch:    6/10    Loss: 3.509746023067614
    
    Epoch:    7/10    Loss: 3.4542041876451757
    
    Epoch:    8/10    Loss: 3.4100723473764045
    
    Epoch:    9/10    Loss: 3.3670633022135834
    
    Epoch:   10/10    Loss: 3.3316375527197755


### Checkpoint Save

To succesfully save the model state while running (so you can actually stop your epochs and continue with them later), you will need to save both the state of the model and in the case of Adam optimizer, since it is an adaptative algorithm and varies with time the steps that it makes.

In my case I decided to save the model with a name that does reference to the batch size, embedding dimensions and hidden dimensions together with the RNN type. So I can play with different parameters and resume the training of different ones when needed.

```python
state = {'epoch': 4, 'state_dict': rnn.state_dict(),
             'optimizer': optimizer.state_dict(), 'epoch_history': epoch_history }
torch.save(state, rnnType + str(batch_size) + str(embedding_dim) + str(hidden_dim) + ".pth.tar")
```

---
# Checkpoint Continue

In case you want to recover your model you can run the next lines


```python
import torch
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')
```

### Generate Text
To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. The `generate` function takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Also note that it uses [topk](https://arxiv.org/pdf/1903.06059.pdf) sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores!


```python
import torch.nn.functional as F

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences
```

### Generate a New Script
It's time to generate the text. Set `gen_length` to the length of TV script and we set `prime_word` to one of the following to start the prediction:
- "jerry"
- "elaine"
- "george"
- "kramer"

### The TV Script is Not Perfect
It's ok if the TV script doesn't make perfect sense. It should look like alternating lines of dialogue, here is one such example of a few generated lines.

This is totally normal, the amount of data is quite small and the epochs that we have run are not enough to generate something with more context. Models like [GPT-2](http://jalammar.github.io/illustrated-gpt2/)


```python
gen_length = 400
prime_word = 'jerry' # name for starting the script

pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)
```

    >jerry:
    >
    >jerry: no.
    >
    >george: i know.
    >
    >jerry:(still looking at his father) you know, i think it's my favorite business.
    >
    >jerry: i don't know if i could be able to do this.
    >
    >morty: well, what do you think?
    >
    >kramer:(looking towards the door) i don't think so.
    >
    >kramer:(to kramer) what happened?
    >
    >jerry: well, i'm sorry. i'm going to be able to get some coffee.
    >
    >kramer:(still looking in his breath) oh, i can't believe it!
    >
    >kramer: hey, i got to tell you, i don't have to get a little too much of this.
    >
    >jerry: well, i just came in with a woman.
    >
    >elaine: i don't want to see you. i mean, if i don't have it, i'll call you at the office, and you don't want to see it.
    >
    >jerry:(on the phone) yeah. i got it, i'm going to have to get it.(to george) what are you doing? i don't know, i just remembered i have to get a cab.
    >
    >george:(to jerry) i can't believe you guys. i was wondering if i should do it. i can't believe this.(to jerry) hey, hey!
    >
    >kramer: hey. hey!(jerry shakes his head)
    >
    >elaine: well, it's all over to minsk.(jerry walks off)
    >
    >jerry:(to jerry) what is it?
    >
    >george:(still in the direction of the car)
    >
    >kramer: oh...
    >
    >jerry:(to elaine) what about you?!
    >
    >jerry: no.
    >
    >george:(looking at his watch) you know, it's not that easy, and i just don't have the big salad.
    >
    >elaine: oh, yeah.
    >
    >george:(to george) hey, what is this?
    >
    >george: what, are you sure?
    >
    >jerry: i




