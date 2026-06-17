# Constants for categorizing appearances count
FEW_APPEARANCES_THRESHOLD = 100 # players with less than indicated appearances will have few appearance value of 1
AVG_APPEARANCES_THRESHOLD = 150 # players with less than indicated but more than few will have avg_appearances value of 1

# Default hyperparameter values
beta_m = 0.9
beta_r = 0.999
alpha = 0.01
batch_size = 256
lambd = 0.1