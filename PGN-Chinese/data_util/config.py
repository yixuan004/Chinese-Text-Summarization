train_data_path = "data/chunked/train/train_*"
valid_data_path = "data/chunked/valid/valid_*"
test_data_path = "data/chunked/test/test_*"
vocab_path = "data/vocab.txt"
demo_vocab_path = "data/vocab.txt"
demo_vocab_size = 40000

# Hyperparameters
hidden_dim = 512
emb_dim = 256
batch_size = 32 # while valadting, change it to 32
max_enc_steps = 100  #99% of the articles are within length 55
max_dec_steps = 20  #99% of the titles are within length 15
beam_size = 4
min_dec_steps = 3
vocab_size = 40000
require_improvement = 10000 # if there isn't some improvement in every 10000 step, stop the training.

lr = 0.005
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
accumulation_steps = 4

eps = 1e-12
max_iterations = 5000000

save_model_path = "data/saved_models"
demo_model_path = "data/saved_models"

intra_encoder = False
intra_decoder = False

cuda = True
