#import csv
import datautils
import torch
from bowclassifier import BoWClassifier
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

train_data = datautils.read_data("data/sample-data-train.csv")
test_data = datautils.read_data("data/sample-data-test.csv")


#print(train_data)

word_to_ix = {}

for sms, _ in train_data + test_data:
    for word in sms:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

#print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
label_to_ix = {"NONE": 0, "DEBIT": 1, "CREDIT" : 2}
NUM_LABELS = 3

def make_bow_vector(sms, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sms:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

#for param in model.parameters():
#    print(param)

# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
sample = train_data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
#print(log_probs)

# Run on test data before we train, just to see a before-and-after
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    #print(log_probs)

# Print the matrix column corresponding to "creo"
#print(next(model.parameters())[:, word_to_ix["Transaction"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for instance, label in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Variable as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

smax = nn.Softmax()
correct = 0
linenum = 0
pred_out_resp = []
for instance, label in test_data:
    linenum += 1
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    pred_out = smax(log_probs)[0].data
    #print(pred_out)
    if(pred_out[0] > pred_out[1] and pred_out[0] > pred_out[2]):
        pred_out_resp.append((linenum, label, "NONE", pred_out[0]))
        if label == "NONE":
            correct += 1
            #print(linenum)
    elif(pred_out[1] > pred_out[0] and pred_out[1] > pred_out[2]):
        pred_out_resp.append((linenum, label, "DEBIT", pred_out[1]))
        if label == "DEBIT":
            #print(linenum)
            correct += 1
    elif pred_out[2] > pred_out[0] and pred_out[2] > pred_out[1]:
        pred_out_resp.append((linenum, label, "CREDIT", pred_out[2]))
        if label == "CREDIT":
            #print(linenum)
            correct += 1

print(correct)
print(len(test_data))
print(pred_out_resp)

# Index corresponding to Spanish goes up, English goes down!
#print(next(model.parameters())[:, word_to_ix["Transaction"]])