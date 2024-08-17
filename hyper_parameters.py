batch_size = 32
num_epochs = 1
total_timesteps = 32
norm_groups = 8
learning_rate = 2e-4

img_size = 32
img_channels = 3
clip_min = -1.0
clip_max = 1.0

BATCHSIZE = 32
EPOCHS = 1
DATAPATH = '/media/jan-wolski/Dysk plików/rescaled/'
IMAGE_SIZE = (32, 64)

first_conv_channels = 12
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 1 
