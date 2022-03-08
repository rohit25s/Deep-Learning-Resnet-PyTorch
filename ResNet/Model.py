import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
		self.criterion = nn.CrossEntropyLoss()
        self.learning_rate=0.001
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate,weight_decay=0.0001, momentum=0.9)
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if(epoch%100==0):
                self.learning_rate=self.learning_rate/10
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                current_batch = curr_x_train[i*self.config.batch_size:(i+1)*self.config.batch_size, :]         
                labels=curr_y_train[i*self.config.batch_size:(i+1)*self.config.batch_size]
                parsed=[]
                for image in current_batch:
                    parsed.append(parse_record(image,True))
                outputs = self.network(torch.tensor(np.array(parsed)).cuda())
				labels_to_tensor = torch.tensor(labels,dtype=torch.long).cuda()
                loss = self.criterion(outputs, labels_to_tensor)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            #print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            num_samples=x.shape[0]
            num_batches = num_samples // self.config.batch_size
            predictions = []
            for i in tqdm(range(num_batches)):
                ### YOUR CODE HERE
                x_train_batch = x[i*self.config.batch_size:(i+1)*self.config.batch_size, :]         
                labels=y[i*self.config.batch_size:(i+1)*self.config.batch_size]
                softmax=torch.nn.Softmax()
                for image in x_train_batch:
                    x_in=parse_record(image,False)
                    prediction = self.network(torch.tensor(np.array([x_in],dtype=float)).cuda())
                    index=int(torch.argmax(prediction))
                    predictions.append(index)
            for i in tqdm(range(num_batches*self.config.batch_size,num_samples)):
                x_in=parse_record(x[i],False)
                prediction = self.network(torch.tensor(np.array([x_in],dtype=float)).cuda())
                index=int(torch.argmax(prediction))
                predictions.append(index)
				### YOUR CODE HERE
            y = torch.tensor(y[:])
            predictions= torch.tensor(predictions)
			#print("model_v1")
            print('Test accuracy: {:.4f}'.format(torch.sum(predictions==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))