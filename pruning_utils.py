import torch
import math
from copy import deepcopy


def residual_pruning(model, genes, filters):
    model = decode_first_backbone_conv(model, genes[0], filters[0])
    model = decode_middle_layer_pruning(model, genes[0], filters[0])
    model = decode_middle_layer_pruning(model, genes[1], filters[1], which_conv='conv2')

    model = decode_residual_block_pruning(model, genes[2], filters[2])

    return model


def decode_first_backbone_conv(model, gene_middle_layers, middle_filters):
    start_idx = 0
    # Residual Nets always begin with a Convolutional Layer
    end_idx = middle_filters['backbone.conv1']

    current_layer_object = getattr(model, 'backbone')
    current_layer_object = getattr(current_layer_object, 'conv1')
    mask_list = gene_middle_layers[start_idx: end_idx]

    if not mask_list.any():  # Test if the mask contains only zeros
        mask_list[0] = 1

    mask = torch.from_numpy(mask_list)

    new_weight_current_layer = current_layer_object.weight.data[mask]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.out_channels = new_weight_current_layer.shape[0]

    # Batchnormalization
    bn_object = getattr(model, 'backbone')
    bn_object = getattr(bn_object, 'bn1')

    new_bn_running_mean = bn_object.running_mean.data[mask]
    bn_object.running_mean.data = new_bn_running_mean

    bn_object.num_features = sum(mask)

    new_bn_running_var = bn_object.running_var.data[mask]
    bn_object.running_var.data = new_bn_running_var

    new_bn_weight = bn_object.weight.data[mask]
    bn_object.weight.data = new_bn_weight

    new_bn_bias = bn_object.bias.data[mask]
    bn_object.bias.data = new_bn_bias

    # Inputs for the backbone.layer1.0.conv1 layer
    current_layer_object = getattr(model, 'backbone')
    current_layer_object = getattr(current_layer_object, 'layer1')
    current_layer_object = current_layer_object[0].conv1

    new_weight_current_layer = current_layer_object.weight.data[:, mask]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    # Inputs for the backbone.layer1.0.downsample.0 layer
    current_layer_object = getattr(model, 'backbone')
    current_layer_object = getattr(current_layer_object, 'layer1')
    current_layer_object = current_layer_object[0].downsample
    current_layer_object = current_layer_object[0]

    new_weight_current_layer = current_layer_object.weight.data[:, mask]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    return model


def decode_middle_layer_pruning(model, gene_middle_layers, middle_filters, which_conv='conv1'):
    start_idx = 0
    end_idx = 0

    list_layers = list(middle_filters.keys())

    for i in range(1, len(list_layers)):
        end_idx += middle_filters[list_layers[i]]

        layer_name = list_layers[i].split(".")

        # Modify the output feature maps of the current layer
        current_layer_object = getattr(model, layer_name[0])
        current_layer_object = getattr(current_layer_object, layer_name[1])
        current_layer_object = current_layer_object[int(layer_name[2])]
        current_layer_object = getattr(current_layer_object, layer_name[3])

        mask = gene_middle_layers[start_idx : end_idx]
        
        if not mask.any():  # Test if the mask contains only zeros
            mask[0] = 1
        
        mask = torch.from_numpy(mask)

        new_weight_current_layer = current_layer_object.weight.data[mask]

        current_layer_object.weight.data = new_weight_current_layer
        current_layer_object.out_channels = new_weight_current_layer.shape[0]

        # Modify the shape of the Batch Normalization running mean
        bn_name = layer_name
        if which_conv == 'conv1':
            bn_name[-1] = 'bn1'
        else:
            bn_name[-1] = 'bn2'

        bn_object = getattr(model, bn_name[0])
        bn_object = getattr(bn_object, bn_name[1])
        bn_object = bn_object[int(bn_name[2])]
        bn_object = getattr(bn_object, bn_name[3])

        new_bn_running_mean = bn_object.running_mean.data[ mask ]
        bn_object.running_mean.data = new_bn_running_mean

        # Modify number of features in Batch Normalization layer
        bn_object.num_features = sum(mask)

        # Modify the shape of the Batch Normalization running var
        new_bn_running_var = bn_object.running_var.data[ mask ]
        bn_object.running_var.data = new_bn_running_var

        # Modify the shape of the Batch Normalization weight
        new_bn_weight = bn_object.weight.data[ mask ]
        bn_object.weight.data = new_bn_weight

        # Modify the shape of the Batch Normalization bias
        new_bn_bias = bn_object.bias.data[ mask ]
        bn_object.bias.data = new_bn_bias

        # Modify the input feature maps of the next layer
        next_layer_name = layer_name

        if which_conv == 'conv1':
            next_layer_name[-1] = 'conv2' # Conv2
            #next_layer_name += '.weight'
        else:
            next_layer_name[-1] = 'conv3'  # Conv3

        next_layer_object = getattr(model, next_layer_name[0])
        next_layer_object = getattr(next_layer_object, next_layer_name[1])
        next_layer_object = next_layer_object[int(next_layer_name[2])]
        next_layer_object = getattr(next_layer_object, next_layer_name[3])

        new_weight_next_layer = next_layer_object.weight.data[ :, mask ]
        
        next_layer_object.weight.data = new_weight_next_layer
        next_layer_object.in_channels = new_weight_next_layer.shape[1]

        start_idx = end_idx

    return model


def decode_residual_block_pruning(model, gene_blocks, block_filters):
    start_idx = 0
    end_idx = 0
    num_features_maps = 0

    list_layers = list(block_filters.keys())

    mask = []

    for i in range(len(list_layers)):
        end_idx += block_filters[list_layers[i]]

        layer_name = list_layers[i]
        current_mask = gene_blocks[start_idx : end_idx]
        
        if not current_mask.any():
            current_mask[0] = 1

        mask.append(torch.from_numpy(current_mask))

        # Get current residual block object
        layer_name = layer_name.split('.')
        current_block_object = getattr(model, layer_name[0])
        current_block_object = getattr(current_block_object, layer_name[1])

        for j in range(len(current_block_object)):
            ######### Modify layer_i[j].conv3 #######################################
            current_layer_object = current_block_object[j].conv3

            ###### Only modifies the output channels ###############################   
            num_features_maps += sum(gene_blocks[start_idx : end_idx])

            new_weight_current_layer = current_layer_object.weight.data[mask[-1]]
            current_layer_object.weight.data = new_weight_current_layer
            current_layer_object.out_channels = new_weight_current_layer.shape[0]

            ###### Modify layer_i[j].bn2 ############################################

            bn_object = current_block_object[j].bn3

            new_bn_running_mean = bn_object.running_mean.data[mask[-1]]
            bn_object.running_mean.data = new_bn_running_mean

            # Modify number of features in Batch Normalization layer
            bn_object.num_features = sum(mask[-1])

            # Modify the shape of the Batch Normalization running var
            new_bn_running_var = bn_object.running_var.data[mask[-1]]
            bn_object.running_var.data = new_bn_running_var

            # Modify the shape of the Batch Normalization weight
            new_bn_weight = bn_object.weight.data[mask[-1]]
            bn_object.weight.data = new_bn_weight

            # Modify the shape of the Batch Normalization bias
            new_bn_bias = bn_object.bias.data[mask[-1]]
            bn_object.bias.data = new_bn_bias
            #########################################################################


            ########## Modify layer_i[j + 1].conv1 ######################################
            ######## Only modifies the input channels ##############################
            if j <= (len(current_block_object) - 2):
                current_layer_object = current_block_object[j + 1].conv1

                new_weight_current_layer = current_layer_object.weight.data[:, mask[-1]]
                current_layer_object.weight.data = new_weight_current_layer
                current_layer_object.in_channels = new_weight_current_layer.shape[1]
            
            if (j == len(current_block_object) - 1) and (i < (len(list_layers) - 1)):
                ########## Modify layer_(i+1)[0].conv1 #################################
                next_block_object = getattr(model, list_layers[i + 1].split('.')[0])
                next_block_object = getattr(next_block_object, list_layers[i+1].split('.')[1])

                next_layer_object = next_block_object[0].conv1

                new_weight_next_layer = next_layer_object.weight.data[:, mask[-1]]
                next_layer_object.weight.data = new_weight_next_layer
                next_layer_object.in_channels = new_weight_next_layer.shape[1]

            ########################################################################

        ######## Deals with the downsampling skip connection

        ### Output feature maps
        num_features_maps += sum(gene_blocks[start_idx : end_idx])
        current_layer_object = current_block_object[0].downsample[0]

        new_weight_current_layer = current_layer_object.weight.data[mask[-1]]
        current_layer_object.weight.data = new_weight_current_layer
        current_layer_object.out_channels = new_weight_current_layer.shape[0]

        bn_object = current_block_object[0].downsample[1]

        new_bn_running_mean = bn_object.running_mean.data[mask[-1]]
        bn_object.running_mean.data = new_bn_running_mean

        bn_object.num_features = sum(mask[-1])

        new_bn_running_var = bn_object.running_var.data[mask[-1]]
        bn_object.running_var.data = new_bn_running_var

        new_bn_weight = bn_object.weight.data[mask[-1]]
        bn_object.weight.data = new_bn_weight

        new_bn_bias = bn_object.bias.data[mask[-1]]
        bn_object.bias.data = new_bn_bias

        if layer_name[-1] != 'layer1':
            # Input feature maps
            new_weight_current_layer = current_layer_object.weight.data[:, mask[-2]]
            current_layer_object.weight.data = new_weight_current_layer
            current_layer_object.in_channels = new_weight_current_layer.shape[1]

        start_idx = end_idx

    # Change ASPP layer
    # model.classifier[0].convs[0][0]
    # Input feature maps
    current_layer_object = model.classifier[0].convs[0][0]
    new_weight_current_layer = current_layer_object.weight.data[:, mask[-1]]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    current_layer_object = model.classifier[0].convs[1][0]
    new_weight_current_layer = current_layer_object.weight.data[:, mask[-1]]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    current_layer_object = model.classifier[0].convs[2][0]
    new_weight_current_layer = current_layer_object.weight.data[:, mask[-1]]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    current_layer_object = model.classifier[0].convs[3][0]
    new_weight_current_layer = current_layer_object.weight.data[:, mask[-1]]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    current_layer_object = model.classifier[0].convs[4][1]
    new_weight_current_layer = current_layer_object.weight.data[:, mask[-1]]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    # Change aux_classifier layer
    # model.aux_classifier[0]
    # Input feature maps
    current_layer_object = model.aux_classifier[0]
    new_weight_current_layer = current_layer_object.weight.data[:, mask[-2]]
    current_layer_object.weight.data = new_weight_current_layer
    current_layer_object.in_channels = new_weight_current_layer.shape[1]

    return model
