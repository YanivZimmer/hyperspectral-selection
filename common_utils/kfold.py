# Define the K-fold Cross Validator
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm

K_FOLDS = 10

dataset = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossValidator:
    def __init__(self, display, k_folds = K_FOLDS):
        self.k_folds = k_folds
        self.display = display

    def cross_validate(self, model_creator: Callable,
                       dataset: Dataset,num_of_epochs: int, batch_size=256):
        kfold = KFold(n_splits=self.k_folds, shuffle=False)
        results = {}
        num_gates_positive_prob = {}
        num_gates_prob_one = {}
        gates_idx = {}
        # Start print
        print('--------------------------------')

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size, sampler=test_subsampler)
            network, optimizer, loss_function, _ = model_creator()

            #train
            self.train(network, optimizer, loss_function, trainloader, num_of_epochs,
                       display_iter=100, device=device, display=self.display)
            #test
            #to hard choose best k: network.test = True
            (results[fold], gates_idx[fold],
             num_gates_prob_one[fold], num_gates_positive_prob[fold])\
                = self.test(network, fold, testloader)

        print("gates_idx", gates_idx)
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {self.k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum / len(results.items())} %')
        print("num_gates_prob_one", num_gates_prob_one)
        print("num_gates_positive_prob", num_gates_positive_prob)


    def train_naive(self, network, optimizer,loss_function, trainloader, num_epochs):
        for epoch in range(0, num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                inputs, targets = data

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0
        # Process is complete.
        print('Training process has finished. Not saving trained model.')

    def test(self, network, fold, testloader):
        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(network.state_dict(), save_path)

        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                #
                # Generate outputs
                outputs = network(inputs.to(device))

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                #TODO verify warning
                correct += (predicted == targets.to(device)).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            gates, gates_prob_one, gates_positive_prob = self.get_non_zero_bands(network)

            #return results of test
            return 100.0 * (correct / total), gates, gates_prob_one, gates_positive_prob

    def get_non_zero_bands(self,model):
        if not hasattr(model, "get_gates"):
            return None
        gates = model.get_gates(mode="prob")
        if gates is None:
            return None, 0, 0
        return gates, sum(gates == 1), sum(gates > 0)

    def train(self, net, optimizer, criterion, data_loader, epoch,
              display_iter=100, device=torch.device('cuda'), display=None,
              val_loader=None, supervision='full'):
        """
        Training loop to optimize a network for several epochs and a specified loss

        Args:
            net: a PyTorch model
            optimizer: a PyTorch optimizer
            data_loader: a PyTorch dataset loader
            epoch: int specifying the number of training epochs
            criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
            device (optional): torch device to use (defaults to CPU)
            display_iter (optional): number of iterations before refreshing the
            display (False/None to switch off).
            scheduler (optional): PyTorch scheduler
            val_loader (optional): validation dataset
            supervision (optional): 'full' or 'semi'
        """

        if criterion is None:
            raise Exception("Missing criterion. You must specify a loss function.")

        if hasattr(net, "set_fs_device"):
            net.set_fs_device(device=device)

        net.to(device)

        save_epoch = epoch // 20 if epoch > 20 else 1

        losses = np.zeros(1000000)
        regs = np.zeros(1000000)
        mean_losses = np.zeros(100000000)
        mean_regs = np.zeros(100000000)
        iter_ = 1
        loss_win, val_win = None, None
        val_accuracies = []

        for e in tqdm(range(1, epoch + 1), desc="Training the network"):

            # Set the network to training mode
            net.train()
            avg_loss = 0.

            # Run the training loop for one epoch
            for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                reg = 0
                if supervision == 'full':
                    output = net(data)
                    # target = target - 1
                    loss = criterion(output, target)
                    reg = net.regularization()
                    # TODO add rego
                    # if hasattr(net, "regularization"):
                    #    reg = net.regularization()
                    #    print("reg",reg.item(),"pure loss",loss)
                    #print("reg", reg.item(), "loss", loss.item())
                    loss = loss + reg
                elif supervision == 'semi':
                    outs = net(data)
                    output, rec = outs
                    # target = target - 1
                    loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
                else:
                    raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                losses[iter_] = loss.item()
                mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])
                regs[iter_] = reg
                mean_regs[iter_] = np.mean(regs[max(0, iter_ - 100):iter_ + 1])

                if display_iter and iter_ % display_iter == 0:
                    string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f} Regu: {:.6f}'
                    string = string.format(
                        e, epoch, batch_idx *
                                  len(data), len(data) * len(data_loader),
                                  100. * batch_idx / len(data_loader), mean_losses[iter_], mean_regs[iter_])
                    update = None if loss_win is None else 'append'
                    loss_win = display.line(
                        X=np.arange(iter_ - display_iter, iter_),
                        Y=mean_losses[iter_ - display_iter:iter_],
                        win=loss_win,
                        update=update,
                        opts={'title': "Training loss",
                              'xlabel': "Iterations",
                              'ylabel': "Loss"
                              }
                    )
                    tqdm.write(string)
                    if hasattr(net, "feature_selector"):
                        print(net.feature_selector.get_gates('prob'))

                    if len(val_accuracies) > 0:
                        val_win = display.line(Y=np.array(val_accuracies),
                                               X=np.arange(len(val_accuracies)),
                                               win=val_win,
                                               opts={'title': "Validation accuracy",
                                                     'xlabel': "Epochs",
                                                     'ylabel': "Accuracy"
                                                     })
                iter_ += 1
                del (data, target, loss, output)

            # Update the scheduler
            avg_loss /= len(data_loader)
            metric = avg_loss
