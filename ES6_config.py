# Dictionary storing network parameters.
params = {
    'batch_size': 64,# Batch size.
    'num_epochs': 1,# Number of epochs to train for.
    'learning_rate': 1e-3,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 25,# After how many epochs to save checkpoints and generate test output.
    'dataset'  : 'ES5',# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!
}
