# Define the K-fold Cross Validator
from typing import Callable
from common_utils.visualizer import Visualizer

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import uuid
uuid.uuid4()
K_FOLDS = 10
N_BANDS = 103
dataset = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from dog import LDoG
from dog import PolynomialDecayAverager

class CrossValidator:
    def __init__(self, display, dataset_name, k_folds = K_FOLDS):
        self.k_folds = k_folds
        self.display = display
        self.dataset_name = dataset_name
        self.visualizer = Visualizer()

    def cross_validate(self, model_creator: Callable,
                       dataset: Dataset,num_of_epochs: int,lam,algo_name,batch_size=256):
        #ATTENTION- shuffle changed to true
        kfold = KFold(n_splits=self.k_folds, shuffle=True)
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
            #if fold%3!=2:
            #    continue
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
            self.train(network, optimizer, loss_function, trainloader, num_of_epochs,fold=fold,lam=lam,
                       display_iter=100, device=device, display=self.display,val_loader=testloader,algo_name=algo_name)
            #test
            #to hard choose best k: network.test = True
            (results[fold], gates_idx[fold],
            num_gates_prob_one[fold], num_gates_positive_prob[fold])\
                = self.test(network, fold, testloader)
            #TODO- this should not stay, just a temp for running only one fold
            #break
        print("gates_idx", gates_idx)
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {self.k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        str_base=""
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            str_base += f'Fold {key}: {value} %\n'
            sum += value
        print(f'Average: {sum / len(results.items())} %')
        str_base += f'Average: {sum / len(results.items())} %\n'
        print("num_gates_prob_one", num_gates_prob_one)
        str_base += f"num_gates_prob_one {num_gates_prob_one}\n"
        print("num_gates_positive_prob", num_gates_positive_prob)
        str_base += f"num_gates_positive_prob {num_gates_positive_prob}\n"
        return str_base


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
            self.visualizer.write_last_gates(self.dataset_name,gates,gates_prob_one)
            #return results of test
            return 100.0 * (correct / total), gates, gates_prob_one, gates_positive_prob

    def get_non_zero_bands(self,model):
        if not hasattr(model, "get_gates"):
            return None, 0, 0
        gates = model.get_gates(mode="prob")
        if gates is None:
            return None, 0, 0
        return gates, sum(gates == 1), sum(gates > 0)

    def train(self, net, optimizer, criterion, data_loader, epoch,
              fold=None,lam=0,display_iter=100,device=torch.device('cuda'), display=None,
              val_loader=None,algo_name=None, supervision='full',):
        regu_early_start = 1
        regu_early_step = 0
        regu_weird=False
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
        optimizer = LDoG(net.parameters())
        averager = PolynomialDecayAverager(net)

        save_gates_progression = True
        gates_progression = np.empty((N_BANDS,))
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
        train_accuracies = []
        for e in tqdm(range(1, epoch + 1), desc="Training the network"):
            if regu_weird:
                regu_early_start = min(regu_early_start + regu_early_step, 1)
                print("Discount factor=", regu_early_start)
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
                    try:
                        reg = net.regularization() if not regu_weird else regu_early_start*net.regularization()
                    #TODO -specific error
                    except:
                        reg = 0
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
                if not averager is None:
                    averager.step()
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
            if save_gates_progression and hasattr(net, "feature_selector"):
                curr_gates=net.feature_selector.get_gates('prob')
                gates_progression=np.vstack((gates_progression, curr_gates))
                print(curr_gates.shape)
                print(gates_progression.shape)
            # Update the scheduler
            avg_loss /= len(data_loader)
            metric = avg_loss
            if val_loader is not None:
                val_acc = self.val(net, val_loader, device=device, supervision=supervision)
                val_accuracies.append(val_acc)
                train_acc = self.val(net, data_loader, device=device, supervision=supervision)
                train_accuracies.append(train_acc)
        if save_gates_progression:
            print("Saving the gates progression image...")
            gates, gates_prob_one, gates_positive_prob = self.get_non_zero_bands(net)
            self.visualizer.gates_progression_image(self.dataset_name,gates_progression,fold,lam,gates_prob_one)
        if val_loader is not None:
            self.visualizer.save_acc_plot(self.dataset_name,train=train_accuracies,val=val_accuracies,algo_name=algo_name,fold=fold,lam=lam,optimizer=optimizer)

    def val(self,net, data_loader, device=torch.device('cuda'), supervision='full'):
        # TODO : fix me using metrics()
        net.eval()
        accuracy, total = 0., 0.
        ignored_labels = data_loader.dataset.ignored_labels
        for batch_idx, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                if supervision == 'full':
                    output = net(data)
                elif supervision == 'semi':
                    outs = net(data)
                    output, rec = outs
                _, output = torch.max(output, dim=1)
                # target = target - 1
                for pred, out in zip(output.view(-1), target.view(-1)):
                    if out.item() in ignored_labels:
                        continue
                    else:
                        accuracy += out.item() == pred.item()
                        total += 1
        return accuracy / total