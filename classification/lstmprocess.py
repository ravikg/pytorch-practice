import datautils
from lstmclassifier import LSTMClassifier
import torch.nn as nn
import torch.optim as optim

train_data = datautils.read_data("data/sample-data-train.csv")
test_data = datautils.read_data("data/sample-data-test.csv")

word_to_ix = {}

for sms, _ in train_data + test_data:
    for word in sms:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
label_to_ix = {"NONE": 0, "DEBIT": 1, "CREDIT" : 2}
NUM_LABELS = len(label_to_ix)

EMBEDDING_DIM = 16
HIDDEN_DIM = 6

model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LABELS)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = datautils.prepare_sequence(train_data[0][0], word_to_ix)
tag_scores = model(inputs)
#print(tag_scores)

for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch: "+ str(epoch))
    for sentence, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = datautils.prepare_sequence(sentence, word_to_ix)
        #print(sentence)
        targets = datautils.make_target(tags, label_to_ix)
        #print("--123==321--")
        #print(targets)


        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        #print("--54123==321--")
        #print(tag_scores.size())

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores[-1], targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = datautils.prepare_sequence(test_data[7][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
print(datautils.tag_name(tag_scores[-1].view(1, -1), test_data[7][1]))