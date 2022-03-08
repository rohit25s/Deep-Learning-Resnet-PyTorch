import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1
        self.conv1 = nn.Conv2d(3, self.first_num_filters, kernel_size=7, stride=1, padding=3, bias=False)
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
		self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.relu = nn.ReLU()
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        return self.relu(self.batch_norm(inputs))
        ### YOUR CODE HERE

class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        if projection_shortcut is not None:
            self.projection_shortcuts = projection_shortcut
        else:
           self.projection_shortcuts=None  
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        self.conv1 = nn.Conv2d(int(filters/strides), filters, kernel_size=3, padding=1, stride=strides, bias=False)
      
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(filters)


        self.stride = strides
        self.relu = nn.ReLU()
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        identity = inputs.clone()

        x = self.relu(self.batch_norm2(self.conv1(inputs)))
        x = self.batch_norm2(self.conv2(x))

        if self.projection_shortcuts is not None:
            identity = self.projection_shortcuts(identity)
        #print(x.shape)
        #print(identity.shape)
        x += identity
        x = self.relu(x)
        return x
        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        if projection_shortcut is not None:
            self.projection_shortcuts = projection_shortcut
            self.bottle_batch_norm1 = nn.BatchNorm2d(int(filters/2))
            self.bottle_conv1 = nn.Conv2d(int(filters/2), int(filters/4), kernel_size=1, padding=0, stride=strides, bias=False)
        else:
           self.projection_shortcuts=None
           self.bottle_batch_norm1 = nn.BatchNorm2d(filters)
           self.bottle_conv1 = nn.Conv2d(filters, int(filters/4), kernel_size=1, padding=0, stride=strides, bias=False)

        self.bottle_batch_norm2 = nn.BatchNorm2d(int(filters/4))
        self.bottle_conv2 = nn.Conv2d(int(filters/4), int(filters/4), kernel_size=3, padding=1, stride=1, bias=False)
        
        self.bottle_batch_norm3 = nn.BatchNorm2d(int(filters/4))
        self.bottle_conv3 = nn.Conv2d(int(filters/4), filters, kernel_size=1, padding=0, stride=1, bias=False)
        
        self.bottle_batch_relu = nn.ReLU()
        ### YOUR CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
		identity = inputs.clone()
        x = self.bottle_conv1(self.bottle_batch_relu(self.bottle_batch_norm1(inputs)))
        x = self.bottle_conv2(self.bottle_batch_relu(self.bottle_batch_norm2(x)))
        x = self.bottle_conv3(self.bottle_batch_relu(self.bottle_batch_norm3(x)))

        if self.projection_shortcuts is not None:
            identity = self.projection_shortcuts(identity)

		#print(identity.shape)
        #print(x.shape)
		x += identity
        return x
        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        self.projection_shortcut = nn.Conv2d(
                in_channels = fin_channels,
                out_channels = filters_out,
                kernel_size = 1,
                stride = 2,
                padding = 0,
            )
        self.size1=resnet_size
        self.standard_blocks = nn.ModuleList()
        for i in range(resnet_size):
            if i==0:
                if filters!=first_num_filters:
                    self.standard_blocks.append(block_fn(filters, self.projection_shortcut, strides, first_num_filters))
                else:
                    self.standard_blocks.append(block_fn(filters, None, 1, first_num_filters))
            else:
                self.standard_blocks.append(block_fn(filters, None, 1, first_num_filters))
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        for i in range(self.size1):
            inputs=self.standard_blocks[i](inputs)
        #print(inputs)
        return inputs
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        
        ### END CODE HERE
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if resnet_version==2:
          self.fc = nn.Linear(filters, num_classes)
        else:
          self.fc = nn.Linear(filters, num_classes)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        x = self.avgpool(inputs)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        ### END CODE HERE