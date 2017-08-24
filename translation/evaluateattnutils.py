import random
import torch
from torch.autograd import Variable
from langclass import SOS_token, EOS_token

def evaluateAttn(encoder, decoder, input_variable, encoder_output_max_length, output_lang):
    #input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(encoder_output_max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    #decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(encoder_output_max_length, encoder_output_max_length)

    for di in range(encoder_output_max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        #decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomlyAttn2(encoder, decoder, pairs, output_lang, max_length, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[2])
        print('=', pair[3])
        output_words, attentions = evaluateAttn(encoder, decoder, pair[0], max_length, output_lang)
        join_by = ''
        if output_lang.wc_type == 'WORD':
            join_by = ' '
        output_sentence = join_by.join(output_words)
        print('<', output_sentence)
        print('')