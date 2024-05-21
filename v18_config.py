# Dictionary storing network parameters.
params = {
    'batch_size': 64,# Batch size.
    'num_epochs': 500,# Number of epochs to train for.
    'learning_rate': 2e-4,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 25,# After how many epochs to save checkpoints and generate test output.
    'dataset'  : 'Cloud18',# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!
    'vza_list1': [0,0],
    'vza_list2': [15,30],
    'sza_list1': [4.0,4.0],
    'sza_list2': [20.0, 40.0]
}
