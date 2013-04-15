#Functions for constructing lattice-based switchboards
#A specialized localized/sparse receptive field is also supported
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 10 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy

#Computes the coordinates of the lattice points that lie within the image
def compute_lattice_matrix(v1, v2, mask, x_in_channels, y_in_channels, in_channel_dim=1, n0_1 = 0, n0_2 = 0, wrap_x= False, wrap_y= False, input_dim = None, dtype = None, ignore_cover = True, allow_nonrectangular_lattice=False):
    if v1[1] != 0 | v1[0] <= 0 | v2[0] < 0 | v2[1] <= 0:
        err = "v1 must be horizontal: v1[0] > 0, v1[1] = 0, v2[0] >= 0, v2[1] > 0"
        raise Exception(err)  

    if in_channel_dim != 1:
        err = "only single channel inputs supported now"
        raise Exception(err)  

#assume no wrapping 
    image = numpy.array(range(0, x_in_channels * y_in_channels))
    image.reshape((y_in_channels, x_in_channels))
    sub_image = numpy.array(range(0, mask.shape[0] * mask.shape[1]))
    sub_image.reshape((mask.shape[0], mask.shape[1]))
    mask_i = mask.astype("int")
    mask_height, mask_width = mask.shape
    out_channel_dim = mask_i.sum()
#    print "Mask shape is ", mask.shape
    
    mat_height = (y_in_channels - mask.shape[0])/v2[1] + 1
    mat_width = (x_in_channels-mask.shape[1])/v1[0] + 1
    
    mat = numpy.ones((mat_height, mat_width, 2)) * -1
#Create Index Matrix, -1 entries equal empty cell
#    print "Mat shape is ", mat.shape
    ind_y = 0
    for iy in range(0, mat_height):
            #x,y are real subimage positions
            #ix, iy are the coefficients of x,y in base v1 and v2
            #ind_y, ind_x are the indices in the matrix mat that contains the centers (upper-left corners) of each subimage
        y = iy * v2[1]
        min_ix = -1 * numpy.int(iy * v2[0] / v1[0])
        max_ix = numpy.floor( (x_in_channels - mask.shape[1] - iy *v2[0]) * 1.0 /  v1[0])
        max_ix = numpy.int(max_ix)
        ind_x = 0
        for ix in range(min_ix, max_ix + 1):
            x = iy *v2[0] + ix * v1[0]
 #           print "value of ind_x, ind_y = ", (ind_x, ind_y)
 #           print "Adding Point (", x, ", ", y, ")"
            mat[ind_y, ind_x] = (x, y)
            ind_x = ind_x + 1
        ind_y = ind_y + 1

    if not allow_nonrectangular_lattice:
        if mat_width > 1:
            if (-1, -1) in mat[:,mat_width-1]:
                mat = mat[:,:mat_width-2]
        else:
            print "Warning, mat_width <= 1 !!!"
    return mat



def compute_lattice_matrix_connections_with_input_dim(v1, v2, preserve_mask, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=False):
    print "shape of preserve_mask is: ", preserve_mask.shape
    print "x_in_channels = ", x_in_channels
    print "y_in_channels = ", y_in_channels

    if  in_channel_dim > 1:
        if in_channel_dim is not preserve_mask.shape[2]:
            err = "preserve_mask.shape[2] and in_channel_dim do not agree!!! "
            raise Exception(err)
#        
        preserve_mask = preserve_mask.flatten().reshape(preserve_mask.shape[0],  in_channel_dim *  preserve_mask.shape[1])
        v1 = list(v1)
        v2 = list(v2)
#remember, vectors have coordinates x, y 
        v1[0] = v1[0] * in_channel_dim
        v2[0] = v2[0] * in_channel_dim
#    
        x_in_channels = x_in_channels * in_channel_dim
        y_in_channels = y_in_channels
        in_channel_dim = 1
    
#    lat_mat = compute_lattice_matrix(v1, v2, preserve_mask, x_in_channels, y_in_channels, in_channel_dim, allow_nonrectangular_lattice=allow_nonrectangular_lattice)             
    return compute_lattice_matrix_connections(v1, v2, preserve_mask, x_in_channels, y_in_channels, allow_nonrectangular_lattice=allow_nonrectangular_lattice)
#
#    print "lat_mat =", lat_mat
#    image_positions = numpy.array(range(0, y_in_channels * x_in_channels))
#    image_positions = image_positions.reshape(y_in_channels, x_in_channels)
##
##
#    mask_indices = image_positions[0:preserve_mask.shape[0], 0:preserve_mask.shape[1]][preserve_mask].flatten()
##
#    connections = None
#    for ind_y in range(lat_mat.shape[0]):
#        for ind_x in range(lat_mat.shape[1]):
#            if(lat_mat[ind_y, ind_x][0] != -1):
#                if connections is None:
#                    connections = numpy.array(mask_indices + (lat_mat[ind_y, ind_x][0] + lat_mat[ind_y, ind_x][1]*x_in_channels))
#                else:
#                    connections = numpy.concatenate((connections, mask_indices + (lat_mat[ind_y, ind_x][0] + lat_mat[ind_y, ind_x][1]*x_in_channels) ))
#            else:
#                print "Void entry in lattice_matrix skipped (to avoid asymmetry)"
##
##
#    print "Connections are: ", connections.astype('int')
#    return (connections.astype('int'), lat_mat)


def compute_lattice_matrix_connections(v1, v2, preserve_mask, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=False, verbose=0):
        if in_channel_dim > 1:
            err = "Error, feature not supported in_channel_dim > 1"
            raise Exception(err)
#
        lat_mat = compute_lattice_matrix(v1, v2, preserve_mask, x_in_channels, y_in_channels, allow_nonrectangular_lattice=allow_nonrectangular_lattice)             
#
        if verbose:
            print "lat_mat =", lat_mat
            
        image_positions = numpy.array(range(0, x_in_channels * y_in_channels))
        image_positions = image_positions.reshape(y_in_channels, x_in_channels)
#
#
        mask_indices = image_positions[0:preserve_mask.shape[0], 0:preserve_mask.shape[1]][preserve_mask].flatten()
#
        connections = None
        for ind_y in range(lat_mat.shape[0]):
            for ind_x in range(lat_mat.shape[1]):
                if(lat_mat[ind_y, ind_x][0] != -1):
                    if connections is None:
                        connections = numpy.array(mask_indices + (lat_mat[ind_y, ind_x][0] + lat_mat[ind_y, ind_x][1]*x_in_channels))
                    else:
                        connections = numpy.concatenate((connections, mask_indices + (lat_mat[ind_y, ind_x][0] + lat_mat[ind_y, ind_x][1]*x_in_channels) ))
                else:
                    print "Void entry in lattice_matrix skipped"
#
#
        if verbose:
            print "Connections are: ", connections.astype('int')
        return (connections.astype('int'), lat_mat)


#base_size either 2 or 3, but other base sizes might also work
def compute_lsrf_n_values(xy_in_channels, base_size, increment):
    n_values = []
    current_size = base_size
#
    n_values.append(base_size)
    prev_n = base_size
    while 1:
        next_n = prev_n * 2 + increment
        if next_n >= xy_in_channels:
            break
        n_values.append(next_n)

        prev_n = next_n
        current_size = current_size * 2
    return n_values[::-1]

#Improvement> let nx_value and ny_value become vectors y_
def compute_lsrf_preserve_masks(x_field_channels, y_field_channels, nx_value, ny_value, in_channel_dim):  
    if in_channel_dim > 1:
        preserve_mask_local = numpy.ones((y_field_channels, x_field_channels, in_channel_dim)) > 0.5
    else:
        preserve_mask_local = numpy.ones((y_field_channels, x_field_channels)) > 0.5
    
    if nx_value > 0:
        h_vector_sparse = numpy.ones((1, nx_value+1)) > 0.5
#        h_vector_sparse[0][x_field_channels:nx_value] = False
        h_vector_sparse[0][x_field_channels:nx_value] = False
    else:
        h_vector_sparse = numpy.ones((1, x_field_channels)) > 0.5

    if ny_value > 0:
        v_vector_sparse = numpy.ones((ny_value+1,1)) > 0.5
        v_vector_sparse[y_field_channels:ny_value,0] = False
    else:
        v_vector_sparse = numpy.ones((y_field_channels, 1)) > 0.5
      
    if in_channel_dim > 1:
        vector_in_channel_dim = numpy.ones((1,1,in_channel_dim)) > 0.5
        preserve_mask_sparse = (v_vector_sparse * h_vector_sparse)[:,:,numpy.newaxis] * vector_in_channel_dim            
    else:
        preserve_mask_sparse = (v_vector_sparse * h_vector_sparse) 

#    print v_vector_sparse, h_vector_sparse, vector_in_channel_dim 
 
        
    return preserve_mask_local, preserve_mask_sparse

#Wrapper to support in_channel > 1
def compute_lsrf_matrix_connections_with_input_dim(v1, v2, preserve_mask_local, preserve_mask_sparse, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=False, verbose=False):
    if verbose:
        print "shape of preserve_mask_local is: ", preserve_mask_local.shape
        if preserve_mask_sparse != None:
            print "shape of preserve_mask_sparse is: ", preserve_mask_sparse.shape
        print "x_in_channels = ", x_in_channels
        print "y_in_channels = ", y_in_channels

    if  in_channel_dim > 1:
        if in_channel_dim != preserve_mask_sparse.shape[2]:
            err = "preserve_mask_sparse.shape[2] and in_channel_dim do not agree!!! "
            raise Exception(err)
        elif in_channel_dim != preserve_mask_local.shape[2]:
            err = "preserve_mask_local.shape[2] and in_channel_dim do not agree!!! "
            raise Exception(err)
#        
        preserve_mask_local = preserve_mask_local.flatten().reshape(preserve_mask_local.shape[0],  in_channel_dim *  preserve_mask_local.shape[1])
        if preserve_mask_sparse != None:
            preserve_mask_sparse = preserve_mask_sparse.flatten().reshape(preserve_mask_sparse.shape[0],  in_channel_dim *  preserve_mask_sparse.shape[1])
        v1 = list(v1)
        v2 = list(v2)
#remember, vectors have coordinates x, y 
        v1[0] = v1[0] * in_channel_dim
        v2[0] = v2[0] * in_channel_dim
#    
        x_in_channels = x_in_channels * in_channel_dim
        y_in_channels = y_in_channels
        in_channel_dim = 1
    
#    lat_mat = compute_lattice_matrix(v1, v2, preserve_mask, x_in_channels, y_in_channels, in_channel_dim, allow_nonrectangular_lattice=allow_nonrectangular_lattice)             
    if preserve_mask_sparse != None:
        return compute_lsrf_matrix_connections(v1, v2, preserve_mask_local, preserve_mask_sparse, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=allow_nonrectangular_lattice)
    else:
        return compute_lattice_matrix_connections(v1, v2, preserve_mask_local, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=allow_nonrectangular_lattice)
    
# Implementation of (LSRF) Localized/Sparse receptive field
# This code should be backwards compatible!!!, only the preserve masks should have changed!!!
# For the lsrf, typically: v1=(2,0), v2=(0,1), preserve_mask = [1M 1M 0M 0M 1M] for suitable square matrices 1M and 0M. 
# Add checking for too small matrix compared to masks
def compute_lsrf_matrix_connections(v1, v2, preserve_mask_local, preserve_mask_sparse, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=False, verbose=False):
        if preserve_mask_sparse == None:
            print "Defaulting to compute_lattice_matrix_connections"
            return compute_lattice_matrix_connections(v1, v2, preserve_mask_local, x_in_channels, y_in_channels, in_channel_dim=1, allow_nonrectangular_lattice=False, verbose=False)
        
        if in_channel_dim > 1:
            err = "Error, feature not supported in_channel_dim > 1"
            raise Exception(err)
#
        lat_mat = compute_lattice_matrix(v1, v2, preserve_mask_local, x_in_channels, y_in_channels, allow_nonrectangular_lattice=allow_nonrectangular_lattice)             
#
        if verbose:
            print "lat_mat =", lat_mat
            
        image_positions = numpy.array(range(0, x_in_channels * y_in_channels))
        image_positions = image_positions.reshape(y_in_channels, x_in_channels)
#
#
        if verbose:
            print image_positions
            print preserve_mask_sparse
        
        mask_indices = image_positions[0:preserve_mask_sparse.shape[0], 0:preserve_mask_sparse.shape[1]][preserve_mask_sparse].flatten()
#
        mask_x_coordinates = mask_indices % x_in_channels
        mask_y_coordinates = mask_indices / x_in_channels

        connections = None
        for ind_y in range(lat_mat.shape[0]):
            for ind_x in range(lat_mat.shape[1]):
                if(lat_mat[ind_y, ind_x][0] != -1):
#                    print "ind_y, ind_x, mask_x_coordinates, mask_x_coordinates = ", ind_y, ind_x, mask_x_coordinates, mask_x_coordinates
#                    print "lat_mat[ind_y, ind_x] = ", lat_mat[ind_y, ind_x]
                    new_x_coordinates = (mask_x_coordinates + lat_mat[ind_y, ind_x][0]) % x_in_channels
                    new_y_coordinates = (mask_y_coordinates + lat_mat[ind_y, ind_x][1]) % y_in_channels

#                    new_connections = numpy.array(mask_indices + (lat_mat[ind_y, ind_x][0] + lat_mat[ind_y, ind_x][1]*x_in_channels))
                    new_connections = new_y_coordinates * x_in_channels + new_x_coordinates
                                                  
                    if connections is None:
                        connections = new_connections
                    else:
                        connections = numpy.concatenate((connections, new_connections))
                else:
                    print "Void entry in lattice_matrix skipped"
#
#
        if verbose:
            print "Connections are: ", connections.astype('int')
        return (connections.astype('int'), lat_mat)

#TODO: The following is test code, should me moved into a test module or submodule
#x_in_channels = 32
#y_in_channels = 1
#x_field_channels = 2
#y_field_channels = 1
#nx_value = 30
#ny_value = None
#base = 2
#increment = 2
#in_channel_dim=2
#print compute_lsrf_n_values(x_in_channels, base, increment)
#mask_local, mask_sparse = compute_lsrf_preserve_masks(x_field_channels, y_field_channels, 30, None, in_channel_dim)
##print mask_sparse, mask_sparse.shape
##print mask_local, mask_local.shape
#connections, lat_mat = compute_lsrf_matrix_connections_with_input_dim(v1=(2,0), v2=(0,1), preserve_mask_local=mask_local, preserve_mask_sparse = mask_sparse, x_in_channels=x_in_channels, y_in_channels=y_in_channels, in_channel_dim=in_channel_dim, allow_nonrectangular_lattice=False, verbose=0)
##print "Connections=", connections

