import mdp
import more_nodes
import patch_mdp
import image_loader
import numpy
import inspect

# Attention: The parameters explicitly named in this file are the only ones used for caching
# (the function __values__() returns the elements that define the hash of the object)

TOP_LEFT_CORNER = 0


class ParamsNetwork(object):
    """This class contains a high-level representation of a hierarchical network (to be constructed).

    The attributes L0 to L10 represent the first 10 layers, and are provided for convenience only. These attributes
    are ignored, and only the abstract layers contained in 'layers' are used. 'node_list' contains a list of mdp nodes
    after the network has been constructed using network_builder.construct_network().
    """
    def __init__(self):
        self.name = "test hierarchical network"
        self.L0 = None
        self.L1 = None
        self.L2 = None
        self.L3 = None
        self.L4 = None
        self.L5 = None
        self.L6 = None
        self.L7 = None
        self.L8 = None
        self.L9 = None
        self.L10 = None
        self.layers = []
        self.node_list = None

    def __values__(self):
        return self.L0, self.L1, self.L2, self.L3, self.L4, self.L5, self.L6, self.L7, self.L8, self.L9, self.L10


class ParamsSFASuperNode(object):
    """High-level description of a non-hierarchical node.
     
    A non-hierarchical node is composed of at most six mdp-nodes, referred to as pca_node, ord_node, gen_exp, 
    red_node, clip_node, and sfa_node. One can specify the class type and parameters of most of these nodes, except
    for the gen_exp node, which is always of class GeneralExpansionNode, and clip_node, which is always of
    class PointwiseFunctionNode.
    """
    def __init__(self):
        self.name = "SFA Supernode"

        self.in_channel_dim = 1
        self.pca_node_class = None
        self.pca_out_dim = None
        self.pca_args = {}

        self.ord_node_class = None
        self.ord_args = {}

        self.exp_funcs = None
        self.inv_use_hint = True
        self.inv_max_steady_factor = 0.35
        self.inv_delta_factor = 0.6
        self.inv_min_delta = 0.0001

        self.red_node_class = None
        self.red_out_dim = None  # 0.99999
        self.red_args = {}

        self.clip_func = None
        self.clip_inv_func = None

        self.sfa_node_class = mdp.nodes.SFANode
        self.sfa_out_dim = 15
        self.sfa_args = {}

        self.node_list = None


# SFALayer: PInvSwitchboard, pca_node, ord_node, gen_exp, red_node, clip_node, sfa_node
class ParamsSFALayer(ParamsSFASuperNode):
    """High-level description of a hierarchical node.
     
     
    A hierarchical node is composed of the same elements that compose a non-hierarchcal node, with the addition of
    a switchboard (of class PInvSwitchboard).
    """
    def __init__(self):
        super(ParamsSFALayer, self).__init__()
        self.name = "SFA Layer"

        self.x_field_channels = 3
        self.y_field_channels = 3
        self.x_field_spacing = 3
        self.y_field_spacing = 3
        self.nx_value = None
        self.ny_value = None
        self.cloneLayer = True
        self.layer_number = None

        # self.pca_out_dim = 0.99999
        # self.pca_args = {"block_size": 1}
        # self.red_out_dim = 0.99999
        # self.red_args = {"block_size": 1, "cutoff": 4}


class ParamsSystem(object):
    """Describes all the parameters of particular experiment.
    
    Three datasets are considered in this description. A training dataset, used to train a (supervised) dimensionality
    reduction algorithm, a seenid dataset, used to train a supervised step on top of the extracted features, and
    a test dataset (Newid). Each of these datasets is described abstractly (by an object of type ParamsInput, e.g., 
    iTraining) and concretely (by an object of type ParamsDataLoading, e.g., sTraining). The most relevant part 
    is the function 'load_data' of the ParamsDataLoading objects, which loads or creates the actual data, represented 
    as an ndarray.
    """
    def __init__(self):
        self.name = "test system"
        self.network = None
        self.iTraining = None
        self.sTraining = None
        self.iSeenid = None
        self.sSeenid = None
        self.iNewid = None
        self.sNewid = None
        self.analysis = None
        self.block_size = None
        self.train_mode = None
        # Warning: just guessing common values here
        self.enable_reduced_image_sizes = True
        self.reduction_factor = 2.0
        self.hack_image_size = 64
        self.enable_hack_image_size = True
        self.patch_network_for_RGB = False

    def __values__(self):
        return self.network, self.iTraining, self.sTraining, self.iSeenid, self.sSeenid, self.iNewid, self.sNewid

    def create(self):
        return


class ParamsInput(object):
    """Describes the (abstract) parameters of a particular dataset, includying the ground truth information."""
    def __init__(self):
        self.name = "test input"
        self.data_base_dir = None
        self.ids = None
        self.ages = [999]
        self.MIN_GENDER = -3
        self.MAX_GENDER = 3
        self.GENDER_STEP = 0.1
        self.genders = [0]
        self.racetweens = [999]
        self.expressions = [0]
        self.morphs = [0]
        self.poses = [0]
        self.lightings = [0]
        self.slow_signal = 0
        self.step = 1
        self.offset = 0
        self.correct_labels = None
        self.correct_classes = None
        self.include_latest = False
        self.block_size = 1
        self.train_mode = None

class ParamsDataLoading(object):
    """Describes the concrete parameters of a particular dataset, e.g., filenames, image sizes, distortions."""
    def __init__(self):
        self.name = "test input data"
        self.input_files = []
        self.images_array = None
        self.num_images = 0
        self.image_width = 256
        self.image_height = 192
        self.subimage_width = 135
        self.subimage_height = 135
        self.pre_mirror_flags = False
        self.pixelsampling_x = 1
        self.pixelsampling_y = 1
        self.subimage_pixelsampling = 2
        self.subimage_first_row = 0
        self.subimage_first_column = 0
        self.subimage_reference_point = TOP_LEFT_CORNER
        self.add_noise_L0 = True
        self.convert_format = "L"
        self.background_type = None
        self.translation = 0
        self.translations_x = None
        self.translations_y = None
        self.trans_sampled = True
        self.rotation = None
        self.contrast_enhance = None
        self.load_data = None
        self.block_size = 1
        self.train_mode = None
        self.include_latest = False
        self.node_weights = None
        self.edge_weights = None
        self.filter = None
        self.obj_avgs = None
        self.obj_stds = None


# This code is way preliminary
class ExperimentResult(object):
    """Stores all relevant variables and results generated by an excecution of Cuicuilco.
    
    These variables include how many features are used in the supervised step, the parameters used to create the
    datasets, the classification rates, regression accuracy, and feature slowness. 
    """
    def __init__(self):
        self.name = "Simulation Results"
        self.network_name = None
        self.layers_name = None

        self.reg_num_signals = None

        self.iTrain = None
        self.sTrain = None
        self.typical_delta_train = None
        self.typical_eta_train = None
        self.brute_delta_train = None
        self.brute_eta_train = None
        self.class_rate_train = None
        self.mse_train = None
        self.msegauss_train = None

        self.iSeenid = None
        self.sSeenid = None
        self.typical_delta_seenid = None
        self.typical_eta_seenid = None
        self.brute_delta_seenid = None
        self.brute_eta_seenid = None
        self.class_rate_seenid = None
        self.mse_seenid = None
        self.msegauss_seenid = None

        self.iNewid = None
        self.sNewid = None
        self.typical_delta_newid = None
        self.typical_eta_newid = None
        self.brute_delta_newid = None
        self.brute_eta_newid = None
        self.class_rate_newid = None
        self.mse_newid = None
        self.msegauss_newid = None


# class NetworkOutputs(object):
#     def __init__(self):
#         self.num_samples = 0
#         self.sl = []
#         self.correct_classes = []
#         self.correct_labels = []
#         self.classes = []
#         self.labels = []
#         self.block_size = []
#         self.eta_values = []
#         self.delta_values = []
#         self.class_rate = 0
#         self.gauss_class_rate = 0
#         self.reg_mse = 0
#         self.gauss_reg_mse = 0


def test_object_contents(an_object):
    """Displays the None elements of an object."""
    a_dict = an_object.__dict__
    list_none_elements = []
    for w in a_dict.keys():
        if a_dict[w] is None:
            list_none_elements.append(str(w))
    if len(list_none_elements) > 0:
        print "Warning!!! object %s contains 'None' fields: " % (str(an_object)), list_none_elements


# apply element-wise in case of lists
def scale_sSeq(sSeq, reduction_factor):
    """Scales a concrete data description.
    
    The reductio_factor argument indicates the reduction in the size of the images. If reduction factor is two, the
    resulting images are 50% smaller. The pixel_sampling are adjusted accordingly so that the same input area is
    considered. The translations are also adjusted (if necessary)."""
    sSeq.subimage_width = sSeq.subimage_width / reduction_factor
    sSeq.subimage_height = sSeq.subimage_height / reduction_factor
    sSeq.pixelsampling_x = sSeq.pixelsampling_x * reduction_factor
    sSeq.pixelsampling_y = sSeq.pixelsampling_y * reduction_factor
    # TODO: Review the logic here
    if sSeq.trans_sampled is True:
        print "sSeq.translations_x=", sSeq.translations_x
        sSeq.translations_x = sSeq.translations_x / reduction_factor
        sSeq.translations_y = sSeq.translations_y / reduction_factor
    print sSeq.subimage_width, "SCALE!"


def take_first_02D(obj_list):
    """returns the element obj_list[0][0] if obj_list is a nested list of depth 2, or obj_list otherwise."""
    if isinstance(obj_list, list):
        if isinstance(obj_list[0], list):
            return obj_list[0][0]
        else:
            er = "obj_list is a list but not a 2D list"
            raise Exception(er)
    else:
        return obj_list


def take_0_k_th_from_2D_list(obj_list, k=0):
    """returns the element obj_list[0][k] if obj_list is a nested list of depth 2, or obj_list otherwise."""
    if isinstance(obj_list, list):
        print "obj_list is:", obj_list
        if isinstance(obj_list[0], list):
            return obj_list[0][k]
        else:
            er = "obj_list is a list but not a 2D list"
            raise Exception(er)
    else:
        return obj_list


def sSeq_force_image_size(sSeq, forced_subimage_width, forced_subimage_height):
    """Modifies a given object of type ParamsDataLoading (or a nested list of depth 2 of such objects). The
    resulting object(s) will then be able to load images of width given by forced_subimage_width, and height given by 
    forced_subimage_height.
    """
    if isinstance(sSeq, list):
        for sSeq_vect in sSeq:
            if sSeq_vect is not None:
                for sSeq_entry in sSeq_vect:
                    sSeq_force_image_size(sSeq_entry, forced_subimage_width, forced_subimage_height)
    else:
        if sSeq is not None:
            sSeq.subimage_width = forced_subimage_width
            sSeq.subimage_height = forced_subimage_height


# TODO:Verify that all elements in sSeq have the same format
def sSeq_getinfo_format(sSeq):
    """Guesses the values of max_clip, signals_per_image, and in_channel dim according to the convert_format value
     
    Understood values for convert_format are "RGB", "L", "HOG2".
    """
    if isinstance(sSeq, list):
        sSeq = sSeq[0][0]

    if sSeq.convert_format == "RGB":
        subimage_shape = (sSeq.subimage_height, sSeq.subimage_width, 3)
        max_clip = 1.0
        signals_per_image = sSeq.subimage_height * sSeq.subimage_width * 3
        in_channel_dim = 3
    elif sSeq.convert_format == "L":
        subimage_shape = (sSeq.subimage_height, sSeq.subimage_width)
        max_clip = 255
        signals_per_image = sSeq.subimage_height * sSeq.subimage_width
        in_channel_dim = 1
    elif sSeq.convert_format == "HOG02":
        subimage_shape = (sSeq.subimage_height, sSeq.subimage_width)
        max_clip = 1.0
        signals_per_image = sSeq.subimage_height * sSeq.subimage_width
        in_channel_dim = 8
    else:
        # Binary format, seems to be in range (0,1)
        subimage_shape = (sSeq.subimage_height, sSeq.subimage_width)
        max_clip = 1.0
        signals_per_image = sSeq.subimage_height * sSeq.subimage_width
        in_channel_dim = 1
    return subimage_shape, max_clip, signals_per_image, in_channel_dim


# Perhaps iSeq should be included and block size, train_mode, etc.. included from
# Notice the recursive nature of this function, only for
# Takes an sSeq structure: either [[sSeq1, ...], ... [sSeqN,...]] or
def convert_sSeq_to_funcs_params_sets(sSeq, verbose=True):
    """From an object of type ParamsDataLoading (or a nested list of depth 2 of such objects) this function returns
    an argument-less function that extracts the final ndarray, as well as a list of parameters that are typically
    needed during training.
    """
    print "conversion of sSeq:", sSeq
    if isinstance(sSeq, list):
        funcs_sets = []
        params_sets = []
        for sSeq_vect in sSeq:
            if sSeq_vect is not None:
                funcs_sets.append([])
                params_sets.append([])
                for sSeq_entry in sSeq_vect:
                    data_func, params = convert_sSeq_to_funcs_params_sets(sSeq_entry, verbose)
                    funcs_sets[-1].append(data_func)
                    params_sets[-1].append(params)
            else:
                funcs_sets.append(None)
                params_sets.append(None)

        print "sSeq (list):", sSeq
        if verbose:
            print "funcs_sets=", funcs_sets
            print "params_sets=", params_sets
        return funcs_sets, params_sets
    else:
        if sSeq is not None:
            if sSeq.load_data is not None:
                print "BB0"

                def data_func():
                    return sSeq.load_data(sSeq)

                print "BB1"
            else:
                print "CC0"

                def data_func():
                    return load_data_from_sSeq(sSeq)

                print "CC1"
            params = {"block_size": sSeq.block_size, "train_mode": sSeq.train_mode,
                      "include_latest": sSeq.include_latest, "node_weights": sSeq.node_weights,
                      "edge_weights": sSeq.edge_weights}
        else:
            data_func = None
            params = None
        return data_func, params


# Here is the actual data loading from hard drive performed
# TODO: Remove this. This function should be particularly specified by each experiment
def load_data_from_sSeq(self):
    """This is a default function that extracts the ndarray data described by a ParamsDataLoading object."""
    seq = self
    if seq.input_files == "LoadBinaryData00":
        data = image_loader.load_natural_data(seq.data_base_dir, seq.base_filename, seq.samples, verbose=False)
    elif seq.input_files == "LoadRawData":
        data = image_loader.load_raw_data(seq.data_base_dir, seq.base_filename, input_dim=seq.input_dim, dtype=seq.dtype,
                                         select_samples=seq.samples, verbose=False)
    else:
        data = image_loader.load_image_data(seq.input_files, seq.images_array, seq.image_width, seq.image_height,
                                           seq.subimage_width,
                                           seq.subimage_height, seq.pre_mirror_flags, seq.pixelsampling_x,
                                           seq.pixelsampling_y,
                                           seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0,
                                           seq.convert_format, seq.translations_x, seq.translations_y,
                                           seq.trans_sampled, seq.rotation, seq.contrast_enhance, seq.obj_avgs,
                                           seq.obj_stds, background_type=seq.background_type,
                                           color_background_filter=seq.filter, verbose=False)
    return data


def append_dataset_arrays_or_functions(dataset1, dataset2):
    """Concatenates the features of two ndarrays (or functions returning ndarrays)."""
    if inspect.isfunction(dataset1):
        print "d1 executed"
        d1 = dataset1()
    else:
        print "d1 array"
        d1 = dataset1
    if inspect.isfunction(dataset2):
        print "d2 executed"
        d2 = dataset2()
    else:
        print "d2 array"
        d2 = dataset2
    n1 = d1.shape[0]
    n2 = d2.shape[0]
    if n1 != n2:
        er = "incompatible number of samples: ", n1, " and ", n2
        raise Exception(er)
    return numpy.concatenate((d1, d2), axis=1)


def expand_dataset_with_additional_features(train_data_sets, additional_features_training):
    """Extends the data in a dataset (ndarray or function returning ndarray) (or nested list of depth 2 of such
    objects) with additional features. Useful to extend the training data with another feature representation. 
    """
    if isinstance(train_data_sets, list):
        if isinstance(train_data_sets[0], list):
            if len(train_data_sets[0]) == 1:
                training_data = train_data_sets[0][0]

                def f():
                    return append_dataset_arrays_or_functions(training_data, additional_features_training)

                train_data_sets[0][0] = f
            else:
                er = "Not possible to add additional labels for these data"
                raise Exception(er)
    else:
        train_data_sets = append_dataset_arrays_or_functions(train_data_sets, additional_features_training)
    return train_data_sets
