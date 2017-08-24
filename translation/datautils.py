import csv
import torch
import torch.autograd as autograd
import torch.nn as nn


def read_data(filename):
    data = []
    with open(filename) as csvfile:
        csvdata = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in csvdata:
            data.append((row[1].split(), row[0]))
    return data

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    #print(idxs)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def make_target(label, label_to_ix):
    #print([label_to_ix[label]])
    return autograd.Variable(torch.LongTensor([label_to_ix[label]]))

def tag_name(log_probs, label):
    smax = nn.Softmax()
    pred_out = smax(log_probs)[0].data
    # print(pred_out)
    out = False
    p_out = 0
    p_label = label
    if (pred_out[0] > pred_out[1] and pred_out[0] > pred_out[2]):
        p_out = pred_out[0]
        p_label = "NONE"
        if label == "NONE":
            out = True
    elif (pred_out[1] > pred_out[0] and pred_out[1] > pred_out[2]):
        p_out = pred_out[1]
        p_label = "DEBIT"
        if label == "DEBIT":
            out = True
    elif pred_out[2] > pred_out[0] and pred_out[2] > pred_out[1]:
        p_out = pred_out[2]
        p_label = "CREDIT"
        if label == "CREDIT":
            out = True
    return (out, p_out, p_label)


def readLangData(lang):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/alice.%s' % (lang), encoding='utf-8'). \
        read().strip().split('\n')

    return lines


