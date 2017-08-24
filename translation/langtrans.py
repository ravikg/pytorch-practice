from datautils import readLangData
from langclass import Lang
from encoderrnn import EncoderRNN
from decoderrnn import DecoderRNN
from decoderrnnattn import AttnDecoderRNN
from trainutils import trainIters
from evaluateutils import evaluateRandomly
from trainattnutils import trainItersAttn
from evaluateattnutils import evaluateRandomlyAttn


# read the data from file
lang_src = "x"
lang_tar = "en"
wc_type = "CHAR" # or "WORD"

# source language
input_data = readLangData(lang_src)
input_lang = Lang(lang_src, wc_type)
for sentence in input_data:
    input_lang.addSentence(sentence)

input_vocab_size = len(input_lang.wc2index)
input_embedding_dim = 16

# target language
output_data = readLangData(lang_tar)
output_lang = Lang(lang_tar, wc_type)
for sentence in output_data:
    output_lang.addSentence(sentence)

output_vocab_size = len(output_lang.wc2index)
output_embedding_dim = 16  # can be different then input
output_size = output_vocab_size  # as each timestep will produce probability
# across the whole vocab of output lang and the best or beam search can help select it

# hidden dim of both the encoder and decoder need to be same as
# final hidden output of encoder is initial input of decoder
hidden_dim = 256

# lstm or gru
lstm_gru = 'GRU' # or 'LSTM'
# encoder
encoder_model = EncoderRNN(input_vocab_size, input_embedding_dim, hidden_dim, lstm_gru)

# decoder w/o attn
#decoder_model = DecoderRNN(output_vocab_size, output_embedding_dim, hidden_dim, output_size, lstm_gru)

# decode w/ attn
max_length =  input_lang.max_length # max attn length
print("max_length: %s" % (max_length))
decoder_attn_model = AttnDecoderRNN(hidden_dim, output_vocab_size, max_length)

hidden_size = 256

## input & output sentence variable pair
pairs = []
for i in range(len(input_data)):
    pairs.append((
        input_lang.variableFromSentence(input_data[i]),
        output_lang.variableFromSentence(output_data[i]),
        input_data[i],
        output_data[i]
    ))

# train params
n_iters = 100
plot_every=10
print_every=10
learning_rate=0.005

# decoder rnn w/o attn
#trainIters(encoder_model, decoder_model, pairs[:2000], 10, plot_every=1, print_every=1)
#evaluateRandomly(encoder_model, decoder_model, pairs[2000:], output_lang)

# decoder rnn w/ attn
trainItersAttn(encoder_model, decoder_attn_model, pairs[:2000], n_iters, max_length, print_every, plot_every, learning_rate)
evaluateRandomlyAttn(encoder_model, decoder_attn_model, pairs[2000:], output_lang, max_length)





