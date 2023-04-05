import numpy as np
import torch
from torch import nn

#  Reading in text to be trained/tested
sentences = []

with open("tiny-shakespeare.txt", 'r') as file:
  for item in file:
    if item.strip() != "":
        line = item.strip()
        sentences.append(line)

#  Step 1, extract all characters
characters = set(''.join(sentences))
print(characters)

#  Step 2, set up the vocabulary
intChar = dict(enumerate(characters))
print(intChar)

charInt = {character: index for index, character in intChar.items()}
print(charInt)

#  We're noto going to pad, because I'm not going to use batches here. Leave that to the students.
#  We need to offset our input and output sentences
input_sequence = []
target_sequence = []
for i in range(len(sentences)):
    # Remove the last character from the input sequence
    input_sequence.append(sentences[i][:-1])
    # Remove the first element from target sequences
    target_sequence.append(sentences[i][1:])

#  Next, construct the one hots! First step, replace all characters with integer
for i in range(len(sentences)):
    input_sequence[i] = [charInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charInt[character] for character in target_sequence[i]]

#  Converting target_sequence into a tensor. Apparently for loss, you just need the int output.
#  TENSORS!!!
#  Tensor is essentially a list, usually 3 dimensions
#  Stores sequence of operations (or calculations) done on the elements of the tensor

#  Need vocab size to make the one-hots
vocab_size = len(charInt)


def create_one_hot(sequence, vocab_size):
    #  Tensor is of the form (batch size, sequence length, one-hot length)
    encoding = np.zeros((1, len(sequence), vocab_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
    return encoding


#  Don't forget to convert to tensors using torch.from_numpy
create_one_hot(input_sequence[0], vocab_size)


# Create the neural network model!
class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #  Define the network!
        #  Batch first defines where the batch parameter is in the tensor
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_state = self.init_hidden()
        output, hidden_state = self.rnn(x, hidden_state)
        # Shouldn't need to resize if using batches, this eliminates the first dimension
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)

        return output, hidden_state

    def init_hidden(self):
        #  Hey,this is our hidden state. Hopefully if we don't have a batch it won't yell at us
        #  Also a note, pytorch, by default, wants the batch index to be the middle dimension here.
        #  So it looks like (row, BATCH, column)
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden


model = RNNModel(vocab_size, vocab_size, 100, 1)

#  Define Loss
loss = nn.CrossEntropyLoss()

#  Use Adam again
optimizer = torch.optim.Adam(model.parameters())
sum = 0
averageLoss = 0
for epoch in range(150):
    print("Epoch: " + str(epoch))
    for i in range(len(input_sequence)):
        optimizer.zero_grad()
        x = torch.from_numpy(create_one_hot(input_sequence[i], vocab_size))
        y = torch.Tensor(target_sequence[i])
        output, hidden = model(x)

        lossValue = loss(output, y.view(-1).long())
        #  Calculates gradient
        lossValue.backward()
        #  Updates weights
        optimizer.step()

        if epoch == 149:
            sum = sum + lossValue.item()
    if epoch == 149:
        averageLoss = sum / len(input_sequence)
print("Average Loss: " + str(averageLoss))

            #print("Loss: {:.4f}".format(lossValue.item()))


#  Okay, let's output some stuff.
#  This makes a pretty big assumption
#  which is that I'm going to pass in longer and longer sequences, which is fine I guess
def predict(model, character):
    characterInput = np.array([charInt[c] for c in character])
    characterInput = create_one_hot(characterInput, vocab_size)
    characterInput = torch.from_numpy(characterInput)
    out, hidden = model(characterInput)

    #  Get output probabilities

    prob = nn.functional.softmax(out[-1], dim=0).data
    character_index = torch.max(prob, dim=0)[1].item()

    return intChar[character_index], hidden


def sample(model, out_len, start='QUEEN:'):
    characters = [ch for ch in start]
    currentSize = out_len - len(characters)
    for i in range(currentSize):
        character, hidden_state = predict(model, characters)
        characters.append(character)
    return ''.join(characters)


print(sample(model, 100))
