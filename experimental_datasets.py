#####################################################################################################################
# experimental_datasets: This module contains various datasets that can be used for experiments. It is part         #
#                        of the Cuicuilco framework                                                                 #
# Main datasets: ParamsAgeFunc_128x128 (age, gender, and race estimation, MORPH-II),                                #
#                ParamsCIFAR10Func_32x32 (object classification, CIFAR-10),                                         #
#                ParamsGenderFunc (Gender estimation, artificial images),                                           #
#                ParamsMNISTFunc (Handwritten digit recognition, MNIST),                                            #
#                ParamsREyeTransXYFunc_64x64 (Eye detection, various databases)                                     #
#                ParamsRFaceCentering2Func (Face discrimination, various databases)                                 #
#                ParamsRGTSRBFunc (traffic sign recognition, GTSRB)                                                 #
#                ParamsRTransXYPAngScaleFunc_128x128 (face localization/pose estimation from frontal images)        #
#                ParamsRandomDataFunc_32x32 (a random artificial dataset)                                           #
#                ParamsRatlabFunc_320x40_training_40k (simulated view of a rat inside a box, RatLab)                #
#                                                                                                                   #
# Note: the user can define new datasets by means of the ImportDatasetsFromFile flag in cuicuilco_run               #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.rub.de                                                                #
# Ruhr-University-Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import copy
import os
import string
import PIL
from PIL import Image

from .import system_parameters
from .system_parameters import load_data_from_sSeq
from . import sfa_libs
from . import more_nodes
from . import image_loader
from .nonlinear_expansion import signed_expo

DAYS_IN_A_YEAR = 365.242

#############################################################################################
# ################# Read environment variables and setup directories ########################
#############################################################################################
experiment_seed = os.getenv("CUICUILCO_EXPERIMENT_SEED")  # 1112223339 #1112223339
if experiment_seed:
    experiment_seed = int(experiment_seed)
else:
    ex = """CUICUILCO_EXPERIMENT_SEED unset.
            Please use this environment variable to indicate the random seed of the experiments """
    raise Exception(ex)
print("experiment_seed=%d" % experiment_seed)
numpy.random.seed(experiment_seed)

experiment_basedir = os.getenv("CUICUILCO_EXPERIMENT_BASEDIR")  # 1112223339 #1112223339
if not experiment_basedir:
    ex = """CUICUILCO_EXPERIMENT_BASEDIR unset.
            Please use this environment variable to indicate the base directory of all databases"""
    raise Exception(ex)

# Setting default directories where the databases are located
user_base_dir = experiment_basedir  # "/local/tmp/escalafl/Alberto"
frgc_normalized_base_dir = experiment_basedir + "/FRGC_Normalized"
frgc_noface_base_dir = experiment_basedir + "/FRGC_NoFace"
alldb_noface_base_dir = experiment_basedir + "/faces/AllDB_NoFace"
alldbnormalized_base_dir = experiment_basedir + "/faces/AllDBNormalized"
frgc_eyeL_normalized_base_dir = experiment_basedir + "/FRGC_EyeL"
alldb_eyeLR_normalized_base_dir = experiment_basedir + "/faces/AllDB_EyeLR"


############################################################################################
# ####################   Experiment that loads CIFAR-10 data   #############################
############################################################################################
class ParamsCIFAR10Experiment(system_parameters.ParamsSystem):
    """ Experiment that loads the CIFAR-10 data (about 60k 32x32 images and 10 classes). """
    def __init__(self, experiment_seed, experiment_basedir, num_hold_out_images=0):
        super(ParamsCIFAR10Experiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.experiment_basedir = experiment_basedir
        self.num_hold_out_images = num_hold_out_images
        self.batches_meta = None
        # self.enable_reduced_image_sizes = False
        # self.reduction_factor = 1.0
        # self.hack_image_size = self.data_dimension[0]
        # self.enable_hack_image_size = False
        # self.patch_network_for_RGB = False

    # Python 2 version.
    @staticmethod
    def unpickle2(file):
        import cPickle
        with open(file, 'rb') as fo:
            a_dict = cPickle.load(fo)
        return a_dict

    # Python 2 version.
    @staticmethod
    def unpickle3(file):
        import pickle
        with open(file, 'rb') as fo:
            a_dict = pickle.load(fo, encoding='bytes')
        return a_dict

    def create(self):
        print("CIFR10: starting with experiment_seed= %d" % self.experiment_seed)
        numpy.random.seed(self.experiment_seed)

        self.batches_meta = self.unpickle2(self.experiment_basedir + '/batches.meta')
        # {'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
        # 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_cases_per_batch': 10000, 'num_vis': 3072}

        self.training_data = self.unpickle2(self.experiment_basedir + '/data_batch_1')
        # keys are: 'data', 'labels', 'batch_label', 'filenames'

        for batch_filename in ['data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
            batch = self.unpickle2(self.experiment_basedir + '/' + batch_filename)
            self.training_data['data'] = numpy.append(self.training_data['data'], batch['data'], axis=0)
            self.training_data['labels'] = self.training_data['labels'] + batch['labels']
            self.training_data['filenames'] = self.training_data['filenames'] + batch['filenames']
        self.training_data['data'] = self.training_data['data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("uint8").reshape(-1,3072)
        self.training_data['batch_label'] = "All batches together"
        self.training_data['labels'] = numpy.array(self.training_data['labels'])
        ordering = numpy.argsort(self.training_data['labels'])
        self.training_data['data'] = self.training_data['data'][ordering]
        self.training_data['labels'] = self.training_data['labels'][ordering]
        self.training_data['filenames'] = [self.training_data['filenames'][i] for i in ordering]

        self.test_data = self.unpickle2(self.experiment_basedir + '/test_batch')
        self.test_data['data'] = self.test_data['data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("uint8").reshape(-1,3072)
        self.test_data['labels'] = numpy.array(self.test_data['labels'])
        ordering = numpy.argsort(self.test_data['labels'])
        self.test_data['data'] = self.test_data['data'][ordering]
        self.test_data['labels'] = self.test_data['labels'][ordering]
        self.test_data['filenames'] = [self.test_data['filenames'][i] for i in ordering]

        self.training_data_size = self.training_data['data'].shape[0]
        self.test_data_size = self.test_data['data'].shape[0]

        if self.num_hold_out_images == 0:
            self.num_images_training = self.training_data_size
            # self.num_images_supervised = self.training_data_size
            self.num_images_test = self.test_data['data'].shape[0]
            self.indices_training = numpy.arange(self.training_data_size)
            # self.indices_supervised = all_subimages_training
            self.indices_test = numpy.arange(self.test_data_size)
            self.effective_test_data = self.test_data
        elif self.num_hold_out_images < self.training_data_size:
            all_subimages_training = numpy.arange(self.training_data_size)
            numpy.random.shuffle(all_subimages_training)
            self.num_images_training = self.training_data_size - self.num_hold_out_images
            # self.num_images_supervised = self.training_data_size - self.num_hold_out_images
            self.num_images_test = self.num_hold_out_images
            self.indices_training = all_subimages_training[0:self.num_images_training]
            # self.indices_supervised = all_subimages_training
            self.indices_test = all_subimages_training[self.num_images_training:]
            self.effective_test_data = self.training_data
        else:
            ex = "Invalid size of hold out set:" + str(self.num_hold_out_images)
            raise Exception(ex)

        self.iTrain = [[self.iSeqCreate(self.training_data, self.indices_training)]]
        self.sTrain = [[self.sSeqCreate(self.iTrain[0][0], seed=-1, use_RGB_images=True)]]

        self.iSeenid = self.iSeqCreate(self.training_data, self.indices_training)
        self.sSeenid = self.sSeqCreate(self.iSeenid, seed=-1, use_RGB_images=True)

        self.iNewid = [[self.iSeqCreate(self.effective_test_data, self.indices_test)]]
        self.sNewid = [[self.sSeqCreate(self.iNewid[0][0], seed=-1, use_RGB_images=True)]]

        print("len(self.iTrain[0][0].input_files)= %d" % len(self.iTrain[0][0].input_files))
        print("len(self.sTrain[0][0].input_files)= %d" % len(self.sTrain[0][0].input_files))

        self.name = "Function Based Data Creation for CIFAR10"

    @staticmethod
    def iSeqCreate(dataset, used_indices):
        print("***** Setting Information Parameters for CIFAR10 ******")
        iSeq = system_parameters.ParamsInput()
        iSeq.data_base_dir = ""
        iSeq.ids = used_indices
        iSeq.dataset = dataset

        iSeq.input_files = []
        iSeq.num_images = len(used_indices)
        iSeq.params = []

        iSeq.correct_classes = dataset['labels'][used_indices]
        iSeq.correct_labels = iSeq.correct_classes.astype('float')
        iSeq.block_size = 10
        iSeq.train_mode = ("classification", iSeq.correct_labels, 1.0)
        return iSeq

    @staticmethod
    def sSeqCreate(iSeq, seed=-1, use_RGB_images=True):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)

        print("******** Setting Data Parameters for CIFAR10  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.name = "CIFAR10"
        sSeq.num_images = iSeq.num_images
        sSeq.dataset = iSeq.dataset
        sSeq.ids = iSeq.ids
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        # sSeq.image_width = 32
        # sSeq.image_height = 32
        sSeq.subimage_width = 32
        sSeq.subimage_height = 32

        if use_RGB_images:
            sSeq.convert_format = "RGB"  # "RGB", "L"
        else:
            sSeq.convert_format = "L"

        def create_arrays(newSeq):
            return newSeq.dataset['data'][newSeq.ids] * 1.0 / 255.0

        sSeq.load_data = create_arrays
        return sSeq

ParamsCIFAR10Func_32x32 = ParamsCIFAR10Experiment(experiment_seed, experiment_basedir + '/cifar-10-batches-py',
                                                  num_hold_out_images=10000)

ParamsCIFAR10Func_32x32_test = ParamsCIFAR10Experiment(experiment_seed, experiment_basedir + '/cifar-10-batches-py',
                                                  num_hold_out_images=0)

############################################################################################
# ####################   Experiment that creates random data    #############################
############################################################################################
class ParamsRandomDataExperiment(system_parameters.ParamsSystem):
    """ Experiment for learning unsupervised or supervised features from artificial data. """
    def __init__(self, experiment_seed, num_latent_variables=10, regression=True, data_dimension=(32, 32),
                 num_images_training=20000, num_images_supervised=10000, num_images_test=10000):
        super(ParamsRandomDataExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.num_latent_variables = num_latent_variables
        self.regression = regression
        self.data_dimension = data_dimension
        self.flattened_data_dimension = numpy.array(self.data_dimension).prod()
        self.num_images_training = num_images_training
        self.num_images_supervised = num_images_supervised
        self.num_images_test = num_images_test
        self.enable_reduced_image_sizes = False
        self.reduction_factor = 1.0
        self.hack_image_size = self.data_dimension[0]
        self.enable_hack_image_size = False
        self.patch_network_for_RGB = False
        self.mixing_matrix = numpy.random.normal(size=(self.num_latent_variables, self.flattened_data_dimension))

    def create(self):
        print("RandomData: starting with experiment_seed= %d" % self.experiment_seed)
        numpy.random.seed(self.experiment_seed)

        self.iTrain = [[self.iSeqCreateRandomData(self.num_images_training)]]
        self.sTrain = [[self.sSeqCreateRandomData(self.iTrain[0][0], seed=-1, use_RGB_images=False)]]

        self.iSeenid = self.iSeqCreateRandomData(self.num_images_supervised)
        self.sSeenid = self.sSeqCreateRandomData(self.iSeenid, seed=-1, use_RGB_images=False)

        self.iNewid = [[self.iSeqCreateRandomData(self.num_images_training)]]
        self.sNewid = [[self.sSeqCreateRandomData(self.iNewid[0][0], seed=-1, use_RGB_images=False)]]

        print("len(self.iTrain[0][0].input_files)= %d" % len(self.iTrain[0][0].input_files))
        print("len(self.sTrain[0][0].input_files)= %d" % len(self.sTrain[0][0].input_files))

        self.name = "Function Based Data Creation for RandomData"
        self.network = "linearNetwork4L"  # Default Network, but ignored

    def iSeqCreateRandomData(self, num_images):
        print("***** Setting Information Parameters for RandomData ******")
        iSeq = system_parameters.ParamsInput()
        iSeq.data_base_dir = ""
        iSeq.ids = list(range(num_images))

        iSeq.input_files = []
        iSeq.num_images = num_images
        iSeq.params = []

        iSeq.block_size = 1
        iSeq.train_mode = "regular"
        iSeq.correct_labels = numpy.random.uniform(size=(num_images, self.num_latent_variables))
        iSeq.correct_labels[:-1,:] += iSeq.correct_labels[1:, :]
        # low-pass filtering is applied to the labels so that SFA can find them
        iSeq.correct_labels[:-1,:] += iSeq.correct_labels[1:, :]  # Again for extra slowness
        iSeq.correct_labels[:-1,:] += iSeq.correct_labels[1:, :]  # Again for extra slowness
        iSeq.correct_labels -= iSeq.correct_labels.min()
        iSeq.correct_labels /= iSeq.correct_labels.max()  # Normalize labels to range 0 to 1

        iSeq.correct_classes = (iSeq.correct_labels*10).clip(0, 9.9).astype(int)
        return iSeq

    def sSeqCreateRandomData(self, iSeq, seed=-1, use_RGB_images=False):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)

        print("******** Setting Data Parameters for RandomData  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.name = "RandomData"
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.image_width = self.data_dimension[1]
        sSeq.image_height = self.data_dimension[0]
        sSeq.subimage_width = sSeq.image_width
        sSeq.subimage_height = sSeq.image_height
        sSeq.correct_labels = iSeq.correct_labels
        sSeq.mixing_matrix = self.mixing_matrix

        if use_RGB_images:
            sSeq.convert_format = "RGB"  # "RGB", "L"
        else:
            sSeq.convert_format = "L"

        def create_arrays(newSeq):
            noise_data = 0.1 * numpy.random.normal(size=(newSeq.num_images, newSeq.image_width * newSeq.image_height))
            label_data = 0.1 * numpy.matmul(newSeq.correct_labels, newSeq.mixing_matrix)
            final_data = label_data + noise_data + 0.5
            return final_data.clip(0.0, 1.0) * 255.0

        sSeq.load_data = create_arrays
        return sSeq


ParamsRandomDataFunc_32x32 = ParamsRandomDataExperiment(experiment_seed, num_latent_variables=1, regression=True,
                                                        data_dimension=(32, 32), num_images_training=20000,
                                                        num_images_supervised=10000, num_images_test=10000)


def repeat_list_elements(a_list, rep):
    """ Creates a list where each element is repeated rep times. """
    return [element for _ in range(rep) for element in a_list]


def load_GT_labels(labels_filename, age_included=True, rAge_included=False, gender_included=True, race_included=False,
                   avgColor_included=False):
    """ This function reads an annotations file of space separated facial attributes.
    The attributes might include: integer age, real age, gender (M/F), race (B/O/A/H/W).
    """
    fd = open(labels_filename, 'r')

    c_age = 1
    if age_included:
        c_rAge = c_age + 1
    else:
        c_rAge = c_age
    if rAge_included:
        c_gender = c_rAge + 1
    else:
        c_gender = c_rAge
    if gender_included:
        c_race = c_gender + 1
    else:
        c_race = c_gender
    if race_included:
        c_avgColor = c_race + 1
    else:
        c_avgColor = c_race

    labels = {}
    for line in fd.readlines():
        all_labels = string.split(line, " ")
        filename = all_labels[0]
        this_labels = []
        if age_included:
            age = float(all_labels[c_age])
            this_labels.append(age)
        if rAge_included:
            rAge = float(all_labels[c_rAge])
            this_labels.append(rAge)
        if gender_included:
            gender = all_labels[c_gender][0]
            if gender == "M":
                gender = -1
            elif gender == "F":
                gender = 1
            else:
                er = "Gender *%s* not recognized" % str(gender)
                raise Exception(er)
            this_labels.append(gender)
        if race_included:
            race = all_labels[c_race][0]
            if race == "B":
                race = -2
            elif race == "O":
                race = -1
            elif race == "A":
                race = 0
            elif race == "H":
                race = 1
            elif race == "W":
                race = 2
            else:
                er = "Race *%s* not recognized" % str(race)
                raise Exception(er)
            this_labels.append(race)
        if avgColor_included:
            avgColor = float(all_labels[c_avgColor])
            this_labels.append(avgColor)
        labels[filename] = this_labels
    fd.close()
    return labels


####################################################################
# ##########        SYSTEM FOR GENDER ESTIMATION         ###########
####################################################################
class ParamsGenderExperiment(system_parameters.ParamsSystem):
    """ Experiment for gender estimation on artificial images. """
    def __init__(self, experiment_seed, experiment_basedir, learn_all_variables=True):
        super(ParamsGenderExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.analysis = None
        self.enable_reduced_image_sizes = True  # (False paper)
        self.reduction_factor = 1.0  # 2.0 # (1.0 paper)
        self.hack_image_size = 135  # 64   # (128 paper)
        self.enable_hack_image_size = True

    def create(self):
        numpy.random.seed(self.experiment_seed + 987987)
        contrast_enhance = "ContrastGenderMultiply"

        iTrainGender0 = self.iSeqCreateGender(first_id=0, num_ids=180, user_base_dir=user_base_dir,
                                              data_dir="RenderingsGender60x200", gender_continuous=True, seed=-1)
        sTrainGender0 = self.sSeqCreateGender(iTrainGender0, contrast_enhance, seed=-1)

        iTrainGender1 = self.iSeqCreateGender(first_id=0, num_ids=10, user_base_dir=user_base_dir,
                                              data_dir="RenderingsGender60x200", gender_continuous=True, seed=-1)
        sTrainGender1 = self.sSeqCreateGender(iTrainGender1, contrast_enhance, seed=-1)

        iSeenidGender = self.iSeqCreateGender(first_id=0, num_ids=180, user_base_dir=user_base_dir,
                                              data_dir="RenderingsGender60x200", gender_continuous=True, seed=-1)
        sSeenidGender = self.sSeqCreateGender(iSeenidGender, contrast_enhance, seed=-1)

        iNewidGender = self.iSeqCreateGender(first_id=180, num_ids=20, user_base_dir=user_base_dir,
                                             data_dir="RenderingsGender60x200", gender_continuous=True, seed=-1)
        sNewidGender = self.sSeqCreateGender(iNewidGender, contrast_enhance, seed=-1)

        # LABEL TRANSFORMATIONS FOR EXACT LABEL LEARNING
        random_labels = True and False
        discontinuous_labels = True and False
        if random_labels or discontinuous_labels:
            labels = numpy.sort(list(set(iTrainGender0.correct_labels)))

            if discontinuous_labels:
                random_labels = (numpy.arange(-3, 3, 0.1) + 3) % 3 - 6
            else:
                random_labels = numpy.random.normal(size=len(labels))
                random_labels = numpy.abs(numpy.random.normal(size=len(labels))).cumsum()

            print("labels =" + str(labels))
            print("random_labels = " + str(random_labels))
            for i, label_val in enumerate(labels):
                for correct_labels in (
                iTrainGender0.correct_labels, iTrainGender1.correct_labels, iSeenidGender.correct_labels,
                iNewidGender.correct_labels):
                    correct_labels[correct_labels == label_val] = random_labels[i]
        else:  # regular labels
            power = 1.0  # 2.0 # 3.0
            offset = 0.0  # 3.0 # 0.0
            if power == 2:
                iTrainGender0.correct_labels = iTrainGender0.correct_labels ** power
                iSeenidGender.correct_labels = iSeenidGender.correct_labels ** power
                iNewidGender.correct_labels = iNewidGender.correct_labels ** power
            elif power == 3 or power == 1:
                iTrainGender0.correct_labels = signed_expo(iTrainGender0.correct_labels + offset, power)
                iTrainGender1.correct_labels = signed_expo(iTrainGender1.correct_labels + offset, power)
                # iTrainGender2.correct_labels = signed_expo(iTrainGender2.correct_labels, 1.0)
                iSeenidGender.correct_labels = signed_expo(iSeenidGender.correct_labels + offset, power)
                iNewidGender.correct_labels = signed_expo(iNewidGender.correct_labels + offset, power)
            else:
                er = "Dont know how to handle power properly, power=", power
                raise Exception(er)

        add_skin_color_label_classes = True and False
        multiple_labels = add_skin_color_label_classes and False
        first_gender_label = True  # and False
        sampled_by_gender = True  # and False

        # Valid options for these boolean flags:
        # True, False, False, False => learn only skin color
        # True, True, True, True    => learn both gender and skin color, gender is first label and determines ordering
        # True, True, True, False   => learn both gender and skin color, gender is first label but
        #                              skin color determines ordering
        # False, False, True, True  => learn only gender
        if add_skin_color_label_classes:
            skin_color_labels_map = load_GT_labels(
                user_base_dir + "/RenderingsGender60x200" + "/GT_average_color_labels.txt", age_included=False,
                rAge_included=False, gender_included=False, race_included=False, avgColor_included=True)

            example_filename = "output133_a999_g362_rt999_e0_c0_p0_i0.tif"
            filename_len = len(example_filename)

            skin_color_labels_TrainGender0 = []
            for input_file in iTrainGender0.input_files:
                input_file_short = input_file[-filename_len:]
                skin_color_labels_TrainGender0.append(skin_color_labels_map[input_file_short])
            skin_color_labels_TrainGender0 = numpy.array(skin_color_labels_TrainGender0).reshape((-1, 1))

            skin_color_labels_SeenidGender = []
            for input_file in iSeenidGender.input_files:
                input_file_short = input_file[-filename_len:]
                skin_color_labels_SeenidGender.append(skin_color_labels_map[input_file_short])
            skin_color_labels_SeenidGender = numpy.array(skin_color_labels_SeenidGender).reshape((-1, 1))

            skin_color_labels_NewidGender = []
            for input_file in iNewidGender.input_files:
                input_file_short = input_file[-filename_len:]
                skin_color_labels_NewidGender.append(skin_color_labels_map[input_file_short])
            skin_color_labels_NewidGender = numpy.array(skin_color_labels_NewidGender).reshape((-1, 1))

            skin_color_labels = skin_color_labels_SeenidGender.flatten()
            sorting = numpy.argsort(skin_color_labels)
            num_classes = 60
            skin_color_classes_SeenidGender = numpy.zeros(iSeenidGender.num_images)
            skin_color_classes_SeenidGender[sorting] = numpy.arange(
                iSeenidGender.num_images) * num_classes // iSeenidGender.num_images
            avg_labels = more_nodes.compute_average_labels_for_each_class(skin_color_classes_SeenidGender,
                                                                          skin_color_labels)
            all_classes = numpy.unique(skin_color_classes_SeenidGender)
            print ("skin color avg_labels=" + str(avg_labels))
            skin_color_classes_TrainGender0 = more_nodes.map_labels_to_class_number(all_classes, avg_labels,
                                                                              skin_color_labels_TrainGender0.flatten())
            skin_color_classes_NewidGender = more_nodes.map_labels_to_class_number(all_classes, avg_labels,
                                                                              skin_color_labels_NewidGender.flatten())

            skin_color_classes_TrainGender0 = skin_color_classes_TrainGender0.reshape((-1, 1))
            skin_color_classes_SeenidGender = skin_color_classes_SeenidGender.reshape((-1, 1))
            skin_color_classes_NewidGender = skin_color_classes_NewidGender.reshape((-1, 1))

        if multiple_labels:
            if first_gender_label:
                print("Learning both skin color and gender, gender is first label")
                iTrainGender0.correct_labels = numpy.concatenate(
                    (iTrainGender0.correct_labels.reshape(-1, 1), skin_color_labels_TrainGender0), axis=1)
                iSeenidGender.correct_labels = numpy.concatenate(
                    (iSeenidGender.correct_labels.reshape(-1, 1), skin_color_labels_SeenidGender), axis=1)
                iNewidGender.correct_labels = numpy.concatenate(
                    (iNewidGender.correct_labels.reshape(-1, 1), skin_color_labels_NewidGender), axis=1)

                iTrainGender0.correct_classes = numpy.concatenate(
                    (iTrainGender0.correct_classes.reshape(-1, 1), skin_color_classes_TrainGender0), axis=1)
                iSeenidGender.correct_classes = numpy.concatenate(
                    (iSeenidGender.correct_classes.reshape(-1, 1), skin_color_classes_SeenidGender), axis=1)
                iNewidGender.correct_classes = numpy.concatenate(
                    (iNewidGender.correct_classes.reshape(-1, 1), skin_color_classes_NewidGender), axis=1)
            else:
                print ("Learning both skin color and gender, color is first label")
                iTrainGender0.correct_labels = numpy.concatenate(
                    (skin_color_labels_TrainGender0, iTrainGender0.correct_labels.reshape(-1, 1)), axis=1)
                iSeenidGender.correct_labels = numpy.concatenate(
                    (skin_color_labels_SeenidGender, iSeenidGender.correct_labels.reshape(-1, 1)), axis=1)
                iNewidGender.correct_labels = numpy.concatenate(
                    (skin_color_labels_NewidGender, iNewidGender.correct_labels.reshape(-1, 1)), axis=1)

                iTrainGender0.correct_classes = numpy.concatenate(
                    (skin_color_classes_TrainGender0, iTrainGender0.correct_classes.reshape(-1, 1)), axis=1)
                iSeenidGender.correct_classes = numpy.concatenate(
                    (skin_color_classes_SeenidGender, iSeenidGender.correct_classes.reshape(-1, 1)), axis=1)
                iNewidGender.correct_classes = numpy.concatenate(
                    (skin_color_classes_NewidGender, iNewidGender.correct_classes.reshape(-1, 1)), axis=1)
        else:
            if add_skin_color_label_classes:
                print ("Learning skin color instead of gender")
                iTrainGender0.correct_labels = skin_color_labels_TrainGender0.flatten()
                iSeenidGender.correct_labels = skin_color_labels_SeenidGender.flatten()
                iNewidGender.correct_labels = skin_color_labels_NewidGender.flatten()
                iTrainGender0.correct_classes = skin_color_classes_TrainGender0.flatten()
                iSeenidGender.correct_classes = skin_color_classes_SeenidGender.flatten()
                iNewidGender.correct_classes = skin_color_classes_NewidGender.flatten()
            else:
                print("Learning gender only")

        if not sampled_by_gender:
            print("reordering by increasing skin color label")
            for (iSeq, labels) in ((iTrainGender0, skin_color_labels_TrainGender0),
                                   (iSeenidGender, skin_color_labels_SeenidGender),
                                   (iNewidGender, skin_color_labels_NewidGender)):
                all_labels = labels.flatten()
                reordering = numpy.argsort(all_labels)
                if multiple_labels:
                    iSeq.correct_labels = iSeq.correct_labels[reordering, :]
                    iSeq.correct_classes = iSeq.correct_classes[reordering, :]
                else:
                    iSeq.correct_labels = iSeq.correct_labels[reordering]
                    iSeq.correct_classes = iSeq.correct_classes[reordering]
                reordered_files = []
                for i in range(len(iSeq.input_files)):
                    reordered_files.append(iSeq.input_files[reordering[i]])
                iSeq.input_files = reordered_files
                # print "reordered_files=", reordered_files
            print("labels/classes reordered, skin color is the first one")

        self.name = "Network that extracts gender information"
        self.network = None
        semi_supervised_learning = False
        if semi_supervised_learning:
            self.iTrain = [[iTrainGender0], [iTrainGender1, ]]
            self.sTrain = [[sTrainGender0], [sTrainGender1, ]]
        else:
            self.iTrain = [[iTrainGender0]]
            self.sTrain = [[sTrainGender0]]
        self.iSeenid = iSeenidGender
        self.sSeenid = sSeenidGender
        self.iNewid = [[iNewidGender]]
        self.sNewid = [[sNewidGender]]

    def iSeqCreateGender(self, first_id=0, num_ids=25, user_base_dir=user_base_dir, data_dir="RenderingsGender60x200",
                         gender_continuous=True, seed=None):
        print("******** Setting Training Information Parameters for Gender (simulated faces) **********")
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)
            print("Gender. Using seed %d" % seed)

        print("******** Setting Training Information Parameters for Gender **********")
        iSeq = system_parameters.ParamsInput()

        iSeq.name = "Gender60x200"

        iSeq.gender_continuous = gender_continuous
        iSeq.data_base_dir = user_base_dir + "/" + data_dir
        iSeq.ids = numpy.arange(first_id, first_id + num_ids)  # 160, but 180 for paper!
        iSeq.ages = [999]
        iSeq.MIN_GENDER = -3
        iSeq.MAX_GENDER = 3
        iSeq.GENDER_STEP = 0.10000  # 01. 0.20025 default. 0.4 fails, use 0.4005, 0.80075, 0.9005
        # iSeq.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
        iSeq.real_genders = numpy.arange(iSeq.MIN_GENDER, iSeq.MAX_GENDER, iSeq.GENDER_STEP)
        iSeq.genders = map(image_loader.code_gender, iSeq.real_genders)
        iSeq.racetweens = [999]
        iSeq.expressions = [0]
        iSeq.morphs = [0]
        iSeq.poses = [0]
        iSeq.lightings = [0]
        iSeq.slow_signal = 2
        iSeq.step = 1
        iSeq.offset = 0
        iSeq.input_files = image_loader.create_image_filenames2(iSeq.data_base_dir, iSeq.slow_signal, iSeq.ids,
                                                                iSeq.ages,
                                                                iSeq.genders, iSeq.racetweens, iSeq.expressions,
                                                                iSeq.morphs,
                                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset)
        iSeq.num_images = len(iSeq.input_files)
        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions,
                       iSeq.morphs, iSeq.poses, iSeq.lightings]

        if iSeq.gender_continuous:
            iSeq.block_size = iSeq.num_images // len(iSeq.params[iSeq.slow_signal])
            iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.real_genders, iSeq.block_size)
            iSeq.correct_classes = sfa_libs.wider_1Darray(iSeq.real_genders, iSeq.block_size)
        else:
            bs = iSeq.num_images // len(iSeq.params[iSeq.slow_signal])
            bs1 = len(iSeq.real_genders[iSeq.real_genders < 0]) * bs
            bs2 = len(iSeq.real_genders[iSeq.real_genders >= 0]) * bs

            iSeq.block_size = numpy.array([bs1, bs2])
            iSeq.correct_labels = numpy.array([-1] * bs1 + [1] * bs2)
            iSeq.correct_classes = numpy.array([-1] * bs1 + [1] * bs2)

        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateGender(self, iSeq, contrast_enhance, seed=-1):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)

        if iSeq is None:
            print ("Gender: iSeq was None, this might be an indication that the data is not available")
            sSeq = system_parameters.ParamsDataLoading()
            return sSeq

        print ("******** Setting Training Data Parameters for Gender  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.image_width = 256
        sSeq.image_height = 192
        sSeq.subimage_width = 135
        sSeq.subimage_height = 135
        sSeq.pixelsampling_x = 1
        sSeq.pixelsampling_y = 1
        sSeq.subimage_pixelsampling = 2
        # TODO: do these variables have to be integers?
        sSeq.subimage_first_row = sSeq.image_height / 2 - sSeq.subimage_height * sSeq.pixelsampling_y / 2
        sSeq.subimage_first_column = sSeq.image_width / 2 - sSeq.subimage_width * sSeq.pixelsampling_x / 2 + 5 * sSeq.pixelsampling_x
        sSeq.add_noise_L0 = True
        sSeq.convert_format = "L"
        sSeq.background_type = "black"
        sSeq.contrast_enhance = contrast_enhance
        sSeq.translation = 2
        sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
        sSeq.translations_y = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
        sSeq.trans_sampled = False
        if iSeq.gender_continuous:
            sSeq.train_mode = 'serial'  # ('mixed' for paper) regular, 'serial'
        else:
            sSeq.train_mode = 'clustered'
        sSeq.load_data = load_data_from_sSeq
        system_parameters.test_object_contents(sSeq)
        return sSeq

    def extract_gender_from_filename(self, filename):
        """ Checks wheather a file contains an M (male) or F (female). """
        if "M" in filename:
            return -1
        elif "F" in filename:
            return 1
        else:
            ex = "gender from filename %s not recognized" % filename
            raise Exception(ex)


ParamsGenderFunc = ParamsGenderExperiment(experiment_seed, experiment_basedir)


####################################################################
# ##########          SYSTEM FOR AGE ESTIMATION         ############
####################################################################

class ParamsAgeExperiment(system_parameters.ParamsSystem):
    """ Experiment for age estimation on artificial images. """
    def __init__(self, experiment_seed, experiment_basedir):
        super(ParamsAgeExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.train_mode = 'mixed'
        self.analysis = None
        self.enable_reduced_image_sizes = False
        self.reduction_factor = 1.0
        self.hack_image_size = 128
        self.enable_hack_image_size = True

    def create(self):
        numpy.random.seed(self.experiment_seed)

        iTrainAge = self.iSeqCreateAge(first_id=0, num_ids=180, user_base_dir=user_base_dir,
                                       data_dir="RendersAge200x23", seed=None)
        sTrainAge = self.sSeqCreateAge(iTrainAge, seed=-1)
        iSeenidAge = self.iSeqCreateAge(first_id=0, num_ids=180, user_base_dir=user_base_dir,
                                        data_dir="RendersAge200x23", seed=None)
        sSeenidAge = self.sSeqCreateAge(iSeenidAge, seed=-1)
        iNewidAge = self.iSeqCreateAge(first_id=180, num_ids=20, user_base_dir=user_base_dir,
                                       data_dir="RendersAge200x23", seed=None)
        sNewidAge = self.sSeqCreateAge(iNewidAge, seed=-1)

        self.name = "Network that extracts Age information"
        self.network = "linearNetwork4L"
        self.iTrain = [[iTrainAge]]
        self.sTrain = [[sTrainAge]]
        self.iSeenid = iSeenidAge
        self.sSeenid = sSeenidAge
        self.iNewid = [[iNewidAge]]
        self.sNewid = [[sNewidAge]]
        self.block_size = iTrainAge.block_size

    def iSeqCreateAge(self, first_id, num_ids, user_base_dir=user_base_dir, data_dir="RendersAge200x23", seed=None):
        print ("***** Setting Information Parameters for Age (simulated faces) ******")
        iSeqAge = system_parameters.ParamsInput()
        iSeqAge.name = "Age: 23 Ages x 200 identities"
        iSeqAge.data_base_dir = user_base_dir + "/" + data_dir
        iSeqAge.im_base_name = "age"
        iSeqAge.ids = numpy.arange(first_id, first_id + num_ids)  # 180, warning speeding up
        # Available ages: iSeqAge.ages = [15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65]
        iSeqAge.ages = numpy.array(
            [15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 42, 44, 45, 46, 48, 50, 55, 60, 65])
        # iSeqAge.ages = numpy.array([15, 20, 24, 30, 35, 40, 45, 50, 55, 60, 65])
        iSeqAge.genders = [None]
        iSeqAge.racetweens = [None]
        iSeqAge.expressions = [0]
        iSeqAge.morphs = [0]
        iSeqAge.poses = [0]
        iSeqAge.lightings = [0]
        iSeqAge.slow_signal = 1
        iSeqAge.step = 1
        iSeqAge.offset = 0
        iSeqAge.input_files = image_loader.create_image_filenames3(iSeqAge.data_base_dir, iSeqAge.im_base_name,
                                                                   iSeqAge.slow_signal, iSeqAge.ids, iSeqAge.ages, \
                                                                   iSeqAge.genders, iSeqAge.racetweens,
                                                                   iSeqAge.expressions, iSeqAge.morphs, \
                                                                   iSeqAge.poses, iSeqAge.lightings, iSeqAge.step,
                                                                   iSeqAge.offset, verbose=False)

        # print "Filenames = ", iSeqAge.input_files
        iSeqAge.num_images = len(iSeqAge.input_files)
        # print "Num Images = ", iSeqAge.num_images
        # iSeqAge.params = [ids, expressions, morphs, poses, lightings]
        iSeqAge.params = [iSeqAge.ids, iSeqAge.ages, iSeqAge.genders, iSeqAge.racetweens, iSeqAge.expressions, \
                          iSeqAge.morphs, iSeqAge.poses, iSeqAge.lightings]
        iSeqAge.block_size = iSeqAge.num_images // len(iSeqAge.ages)

        iSeqAge.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeqAge.ages)), iSeqAge.block_size)
        iSeqAge.correct_labels = sfa_libs.wider_1Darray(iSeqAge.ages, iSeqAge.block_size)

        system_parameters.test_object_contents(iSeqAge)
        return iSeqAge

    def sSeqCreateAge(self, iSeqAge, seed=-1):
        print ("******** Setting Data Parameters for Age  ****************")
        sSeqAge = system_parameters.ParamsDataLoading()
        sSeqAge.input_files = iSeqAge.input_files
        sSeqAge.num_images = iSeqAge.num_images
        sSeqAge.block_size = iSeqAge.block_size
        sSeqAge.image_width = 256
        sSeqAge.image_height = 192
        sSeqAge.subimage_width = 135
        sSeqAge.subimage_height = 135
        sSeqAge.pixelsampling_x = 1
        sSeqAge.pixelsampling_y = 1
        sSeqAge.subimage_pixelsampling = 2
        sSeqAge.subimage_first_row = sSeqAge.image_height / 2 - sSeqAge.subimage_height * sSeqAge.pixelsampling_y / 2
        sSeqAge.subimage_first_column = sSeqAge.image_width / 2 - sSeqAge.subimage_width * sSeqAge.pixelsampling_x / 2
        # sSeqAge.subimage_first_column = sSeqAge.image_width/2-sSeqAge.subimage_width*sSeqAge.pixelsampling_x/2+
        # 5*sSeqAge.pixelsampling_x
        sSeqAge.add_noise_L0 = True
        sSeqAge.convert_format = "L"
        sSeqAge.background_type = "blue"
        sSeqAge.translation = 1
        # sSeqAge.translations_x = numpy.random.random_integers(-sSeqAge.translation, sSeqAge.translation, sSeqAge.num_images)
        sSeqAge.translations_x = numpy.random.random_integers(-sSeqAge.translation, sSeqAge.translation,

                                                              sSeqAge.num_images)
        sSeqAge.translations_y = numpy.random.random_integers(-sSeqAge.translation, sSeqAge.translation,
                                                              sSeqAge.num_images)
        sSeqAge.trans_sampled = False
        sSeqAge.train_mode = 'mixed'
        sSeqAge.name = iSeqAge.name
        sSeqAge.load_data = load_data_from_sSeq
        system_parameters.test_object_contents(sSeqAge)
        return sSeqAge


ParamsAgeFunc_128x128 = ParamsAgeExperiment(experiment_seed, experiment_basedir)

# TODO: Delete ParamsNatural from cuicuilco_run.py
ParamsNatural = system_parameters.ParamsSystem()  # Define this empty object to prevent cuicuilco_run from crashing

##########################################################################
# #########        Function Defined Data Sources       ###################
##########################################################################


class ParamsREyePosXYExperiment(system_parameters.ParamsSystem):
    """ Experiment for the localization (x,y) of (screen-left) eyes in image patches (real photographs). """
    def __init__(self, experiment_seed, experiment_basedir, learn_all_variables=True, slow_var="PosX"):
        super(ParamsREyePosXYExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.learn_all_variables = learn_all_variables  # If True: combine two graphs for PosX and PosY
        self.slow_var = slow_var  # in {"PosX", and "PosY"}, the main slow parameter
        self.train_mode = 'serial'  # ignored
        self.analysis = None
        self.enable_reduced_image_sizes = True
        self.reduction_factor = 1.0
        self.hack_image_size = 64
        self.enable_hack_image_size = True

    def create(self):
        numpy.random.seed(self.experiment_seed)
        self.network = "linearNetwork4L"
        all_eyeLR_available_images = numpy.arange(7000, 45000)
        # 51434, Only use a fraction of the FRGC images because the first (non FRGC) images have more variations
        numpy.random.shuffle(all_eyeLR_available_images)

        eye_dx = 10.7  # 6 #12
        eye_dy = 10.7  # 6 #12
        eye_smax = (0.94875 + .01725) * 2.0
        eye_smin = (0.94875 - .01725) * 2.0
        eye_da = 10.0  # 22.5 #30.0 , 22.5
        obj_avg_std = 0.11  # 0.1 Final data in range (0, 1), centered at 0.5
        obj_std_min = 0.14  # 0.10
        obj_std_max = 0.16  # 0.20

        self.iTrain = [[self.iSeqCreateREyePosXY(30000, alldb_eyeLR_normalized_base_dir, all_eyeLR_available_images,
                                                 first_image=0, slow_var=self.slow_var, num_classes=50, eye_dx=eye_dx,
                                                 eye_dy=eye_dy, repetition_factor=4, seed=-1)]]  # 4
        self.sTrain = [[self.sSeqCreateREyePosXY(self.iTrain[0][0], eye_smax=eye_smax, eye_smin=eye_smin, eye_da=eye_da,
                                                 contrast_enhance=True, obj_avg_std=obj_avg_std,
                                                 obj_std_min=obj_std_min, obj_std_max=obj_std_max, seed=-1)]]
        self.iSeenid = self.iSeqCreateREyePosXY(4000 + 4000, alldb_eyeLR_normalized_base_dir,
                                                all_eyeLR_available_images, first_image=30000 - 4000,
                                                slow_var=self.slow_var, num_classes=50, eye_dx=eye_dx, eye_dy=eye_dy,
                                                repetition_factor=2, seed=-1)  # 2
        self.sSeenid = self.sSeqCreateREyePosXY(self.iSeenid, eye_smax=eye_smax, eye_smin=eye_smin, eye_da=eye_da,
                                                contrast_enhance=True, obj_avg_std=obj_avg_std, obj_std_min=obj_std_min,
                                                obj_std_max=obj_std_max, seed=-1)
        self.iNewid = [[self.iSeqCreateREyePosXY(4000, alldb_eyeLR_normalized_base_dir, all_eyeLR_available_images,
                                                 first_image=34000, slow_var=self.slow_var, num_classes=50,
                                                 eye_dx=eye_dx, eye_dy=eye_dy, repetition_factor=1, seed=-1)]]
        self.sNewid = [[self.sSeqCreateREyePosXY(self.iNewid[0][0], eye_smax=eye_smax, eye_smin=eye_smin, eye_da=eye_da,
                                                 contrast_enhance=True, obj_avg_std=0.0,
                                                 obj_std_min=0.5 * (obj_std_max + obj_std_min),
                                                 obj_std_max=0.5 * (obj_std_max + obj_std_min), seed=-1)]]
        self.name = self.sTrain[0][0].name
        self.block_size = self.iTrain[0][0].block_size

        if self.learn_all_variables:  # and False:
            iSeq = self.iTrain[0][0]
            sSeq = self.sTrain[0][0]
            sSeq.train_mode = [sSeq.train_mode, ("serial_regression50", iSeq.correct_labels[:, 1], 1.0)]

            # print ParamsREyeTransXYFunc.iTrain[0][0].correct_labels[:,0]
            # print ParamsREyeTransXYFunc.sTrain[0][0].translations_x

    # This dataset uses images in the closed intervals: [-eye_dx, eye_dx],
    # [-eye_dy, eye_dy], [eye_smin0, eye_smax0]
    def iSeqCreateREyePosXY(self, num_images, eyeLR_normalized_base_dir, all_eyeLR_available_images, first_image=0,
                            slow_var="PosX", num_classes=30, eye_dx=10, eye_dy=10, repetition_factor=8, seed=-1):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)

        print ("***** Setting information Parameters for REyePosXY******")
        iSeq = system_parameters.ParamsInput()
        iSeq.name = "Real EYE LOCATION (PosX / PosY)"

        iSeq.data_base_dir = eyeLR_normalized_base_dir
        iSeq.ids = all_eyeLR_available_images[numpy.arange(first_image, first_image + num_images)]

        iSeq.ages = [None]
        iSeq.genders = [None]
        iSeq.racetweens = [None]
        iSeq.expressions = [None]
        iSeq.morphs = [None]
        iSeq.poses = [None]
        iSeq.lightings = [None]
        iSeq.slow_signal = 0  # real slow signal is the translation in the x axis, added during image loading
        iSeq.step = 1
        iSeq.offset = 0
        iSeq.input_files = image_loader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids,
                                                                iSeq.ages,
                                                                iSeq.genders, iSeq.racetweens, iSeq.expressions,
                                                                iSeq.morphs,
                                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset,
                                                                len_ids=5, image_postfix=".jpg")
        iSeq.input_files = iSeq.input_files * repetition_factor  # warning!!! 8
        # To avoid grouping similar images next to one other
        numpy.random.shuffle(iSeq.input_files)

        iSeq.num_images = len(iSeq.input_files)

        iSeq.translations_x = numpy.random.uniform(-1 * eye_dx, eye_dx, iSeq.num_images)
        iSeq.translations_y = numpy.random.uniform(-1 * eye_dy, eye_dy, iSeq.num_images)
        iSeq.eye_dx = eye_dx
        iSeq.eye_dy = eye_dy

        if slow_var == "PosX":
            iSeq.translations_x.sort()
            print ("iSeq.translations_x=" + str(iSeq.translations_x))
        elif slow_var == "PosY":
            iSeq.translations_y.sort()
        else:
            er = "slow_var must be either PosX or PosY"
            raise Exception(er)

        # if len(iSeq.ids) != len(iSeq.translations_x):
        #    ex="Here the number of translations must be equal to the number of identities: ",
        # len(iSeq.translations_x),len(iSeq.ids)
        #    raise Exception(ex)

        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, iSeq.morphs, iSeq.poses,
                       iSeq.lightings]
        iSeq.block_size = iSeq.num_images // num_classes

        iSeq.train_mode = "serial"  # TODO:should learn both labels simultaneously
        all_classes = numpy.arange(iSeq.num_images, dtype="int") * num_classes // iSeq.num_images
        correct_classes_X = numpy.zeros(iSeq.num_images, dtype="int")
        correct_classes_Y = numpy.zeros(iSeq.num_images, dtype="int")
        correct_classes_X[numpy.argsort(iSeq.translations_x)] = all_classes.copy()
        correct_classes_Y[numpy.argsort(iSeq.translations_y)] = all_classes.copy()

        print ("correct_classes_X=" +  str(correct_classes_X))
        print ("correct_classes_Y=" +  str(correct_classes_Y))

        iSeq.correct_classes = numpy.stack((correct_classes_X, correct_classes_Y)).T
        iSeq.correct_labels = numpy.stack((iSeq.translations_x, iSeq.translations_y)).T

        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateREyePosXY(self, iSeq, eye_smax=0.84, eye_smin=0.81, eye_da=22.5, contrast_enhance=True,
                            obj_avg_std=0.0, obj_std_min=0.2, obj_std_max=0.2, seed=-1):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)
        # else seed <0 then, do not change seed
        print ("******** Setting Training Data Parameters for Real Eye TransX  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.image_width = 256
        sSeq.image_height = 192
        sSeq.subimage_width = 64
        sSeq.subimage_height = 64

        sSeq.trans_x_max = iSeq.eye_dx
        sSeq.trans_x_min = -iSeq.eye_dx
        sSeq.trans_y_max = iSeq.eye_dy
        sSeq.trans_y_min = -iSeq.eye_dy
        sSeq.min_sampling = eye_smin
        sSeq.max_sampling = eye_smax

        sSeq.translations_x = iSeq.translations_x
        sSeq.translations_y = iSeq.translations_y
        sSeq.rotations = numpy.random.uniform(low=-eye_da, high=eye_da, size=sSeq.num_images)
        sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
        sSeq.pixelsampling_y = sSeq.pixelsampling_x.copy()
        sSeq.subimage_first_row = sSeq.image_height / 2.0 - sSeq.subimage_height * sSeq.pixelsampling_y / 2.0
        sSeq.subimage_first_column = sSeq.image_width / 2.0 - sSeq.subimage_width * sSeq.pixelsampling_x / 2.0

        sSeq.contrast_enhance = "AgeContrastEnhancement_Avg_Std"  # "array_mean_std-127.5-45.0", None, "AgeContrastEnhancement_Avg_Std" (setup needed)
        if obj_avg_std > 0:
            sSeq.obj_avgs = numpy.random.normal(0.0, obj_avg_std,
                                                size=sSeq.num_images)  # mean intensity, centered at zero here
        else:
            sSeq.obj_avgs = numpy.zeros(sSeq.num_images)
        sSeq.obj_stds = numpy.random.uniform(obj_std_min, obj_std_max, sSeq.num_images)  # contrast

        sSeq.add_noise_L0 = True
        sSeq.convert_format = "L"
        sSeq.background_type = None
        sSeq.trans_sampled = False  # the translations are specified according to the original image, not according to the sampled patch
        sSeq.rotate_before_translation = True

        sSeq.name = "REyePosXY Dx in (%d, %d) Dy in (%d, %d), sampling in (%d, %d)" % (sSeq.trans_x_min,
                                                                                       sSeq.trans_x_max,
                                                                                       sSeq.trans_y_min,
                                                                                       sSeq.trans_y_max,
                                                                                       int(sSeq.min_sampling * 100),
                                                                                       int(sSeq.max_sampling * 100))
        iSeq.name = sSeq.name
        print (sSeq.name)

        sSeq.load_data = load_data_from_sSeq

        system_parameters.test_object_contents(sSeq)

        return sSeq


ParamsREyeTransXYFunc_64x64 = ParamsREyePosXYExperiment(experiment_seed, experiment_basedir, learn_all_variables=True,
                                                        slow_var="PosX")
ParamsREyeTransXYFunc_32x32 = copy.deepcopy(ParamsREyeTransXYFunc_64x64)
ParamsREyeTransXYFunc_32x32.reduction_factor = 2.0
ParamsREyeTransXYFunc_32x32.hack_image_size = 32


###############################################################################
# #######################    FACE CENTERING    ################################
###############################################################################
class ParamsRFaceCentering2Experiment(system_parameters.ParamsSystem):
    """ Experiment for detecting the presence and quality of centering of faces in image patches (real photographs).
    This is useful to discriminate face from no-face patches.
    """
    def __init__(self, experiment_seed, experiment_basedir, pipeline_fd_centering, iteration=0):
        super(ParamsRFaceCentering2Experiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.experiment_basedir = experiment_basedir
        self.pipeline_fd_centering = pipeline_fd_centering
        self.iteration = iteration

        self.train_mode = 'clustered'  # clustered improves final performance! mixed
        self.analysis = None
        self.enable_reduced_image_sizes = True
        self.reduction_factor = 2.0  # WARNING 2.0, 4.0, 8.0
        self.hack_image_size = 64  # WARNING 64, 32, 16
        self.enable_hack_image_size = True

    def create(self):
        if self.experiment_seed >= 0 or self.experiment_seed is None:
            numpy.random.seed(self.experiment_seed)

        alldbnormalized_available_images = numpy.arange(0, 55000)
        numpy.random.shuffle(alldbnormalized_available_images)
        av_fim = alldbnormalized_available_images
        alldb_noface_available_images = numpy.arange(0, 18000)  # = numpy.arange(0,12000) #???? image18552
        numpy.random.shuffle(alldb_noface_available_images)
        av_nfim = alldb_noface_available_images

        if self.iteration == 0:
            pipeline_factor = 1.2
        elif self.iteration == 1:
            pipeline_factor = 0.25
        else:
            er = "unexpected iteration value" + str(iteration)
            raise Exception(er)

        iSeq = iTrainRFaceCentering = [[self.iSeqCreateRFaceCentering(30000, av_fim, av_nfim, first_image=0,
                                                                      first_image_no_face=0, repetition_factor=4,
                                                                      seed=-1)]]
        sSeq = sTrainRFaceCentering = [[self.sSeqCreateRFaceCentering2(iSeq[0][0],
                                                                       pipeline_fd=self.pipeline_fd_centering,
                                                                       pipeline_factor=pipeline_factor, seed=-1)]]
        print (";)")

        # iSeq = iTrainRFaceCentering2 = \
        # [[iSeqCreateRFaceCentering(3000, av_fim, av_nfim, first_image=0, repetition_factor=2, seed=-1)], \
        # [iSeqCreateRFaceCentering(3000, av_fim, av_nfim, first_image=3000, repetition_factor=2, seed=-1),
        # iSeqCreateRFaceCentering(3000, av_fim, av_nfim, first_image=6000, repetition_factor=2, seed=-1)]]

        # sTrainRFaceCentering2 = [[sSeqCreateRFaceCentering(iSeq[0][0], seed=-1)], \
        #                        [sSeqCreateRFaceCentering(iSeq[1][0], seed=-1),
        # sSeqCreateRFaceCentering(iSeq[1][1], seed=-1)]]
        iSeq = iSeenidRFaceCentering2 = self.iSeqCreateRFaceCentering(15000, av_fim, av_nfim, first_image=30000,
                                                                      first_image_no_face=3000 * 4, repetition_factor=2,
                                                                      seed=-1)
        sSeenidRFaceCentering2 = self.sSeqCreateRFaceCentering2(iSeq, pipeline_fd=self.pipeline_fd_centering,
                                                                pipeline_factor=pipeline_factor, seed=-1)
        iSeq = iNewidRFaceCentering2 = [[self.iSeqCreateRFaceCentering(10000, av_fim, av_nfim, first_image=45000,
                                                                       first_image_no_face=3000 * 4 + 1500 * 2,
                                                                       repetition_factor=2, seed=-1)]]
        sNewidRFaceCentering2 = [[self.sSeqCreateRFaceCentering2(iSeq[0][0], pipeline_fd=self.pipeline_fd_centering,
                                                                 pipeline_factor=pipeline_factor, seed=-1)]]

        self.name = "Function Based Data Creation for RFaceCentering"
        self.network = "linearNetwork4L"  # Default Network, but ignored
        self.iTrain = iTrainRFaceCentering
        self.sTrain = sTrainRFaceCentering
        self.iSeenid = iSeenidRFaceCentering2
        self.sSeenid = sSeenidRFaceCentering2
        self.iNewid = iNewidRFaceCentering2
        self.sNewid = sNewidRFaceCentering2
        self.block_size = iTrainRFaceCentering[0][0].block_size

    def iSeqCreateRFaceCentering(self, num_images, alldbnormalized_available_images, alldb_noface_available_images,
                                 first_image=0, first_image_no_face=0, repetition_factor=1, seed=-1):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)
        # else seed <0 then, do not change seed

        print ("***** Setting information Parameters for RFaceCentering******")
        iSeq = system_parameters.ParamsInput()
        # (0.55+1.1)/2 = 0.825, 0.55/2 = 0.275, 0.55/4 = 0.1375, .825 + .1375 = .9625, .825 - .55/4 = .6875
        iSeq.name = "Real FACE DISCRIMINATION (Centered / Decentered)"

        if num_images > len(alldbnormalized_available_images):
            err = "Number of images to be used exceeds the number of available images"
            raise Exception(err)

        if num_images / 10 * repetition_factor > len(alldb_noface_available_images):
            err = "Number of no_face images to be used exceeds the number of available images"
            raise Exception(err)

        iSeq.data_base_dir = alldbnormalized_base_dir

        iSeq.ids = alldbnormalized_available_images[
                   first_image:first_image + num_images]  # 30000, numpy.arange(0,6000) # 6000
        iSeq.faces = numpy.arange(0, 10)  # 0=centered normalized face, 1=not centered normalized face
        block_size = len(iSeq.ids) // len(iSeq.faces)

        # iSeq.scales = numpy.linspace(pipeline_fd['smin0'], pipeline_fd['smax0'], 50) # (-50, 50, 2)
        if len(iSeq.ids) % len(iSeq.faces) != 0:
            ex = "Here the number of scales must be a divisor of the number of identities"
            raise Exception(ex)

        iSeq.ages = [None]
        iSeq.genders = [None]
        iSeq.racetweens = [None]
        iSeq.expressions = [None]
        iSeq.morphs = [None]
        iSeq.poses = [None]
        iSeq.lightings = [None]
        iSeq.slow_signal = 0  # real slow signal is whether there is the amount of face centering
        iSeq.step = 1
        iSeq.offset = 0
        iSeq.input_files = image_loader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids,
                                                                iSeq.ages,
                                                                iSeq.genders, iSeq.racetweens, iSeq.expressions,
                                                                iSeq.morphs,
                                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset,
                                                                len_ids=5, image_postfix=".jpg")
        iSeq.input_files = iSeq.input_files * repetition_factor  # 4 was overfitting non linear sfa slightly
        # To avoid grouping similar images next to one other or in the same block
        numpy.random.shuffle(iSeq.input_files)

        # Background images are not duplicated, instead more are taken
        iSeq.data2_base_dir = alldb_noface_base_dir
        iSeq.ids2 = alldb_noface_available_images[
                    first_image_no_face: first_image_no_face + block_size * repetition_factor]
        iSeq.input_files2 = image_loader.create_image_filenames3(iSeq.data2_base_dir, "image", iSeq.slow_signal,
                                                                 iSeq.ids2, iSeq.ages,
                                                                 iSeq.genders, iSeq.racetweens, iSeq.expressions,
                                                                 iSeq.morphs,
                                                                 iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset,
                                                                 len_ids=5, image_postfix=".jpg")
        iSeq.input_files2 = iSeq.input_files2
        numpy.random.shuffle(iSeq.input_files2)

        iSeq.input_files = iSeq.input_files[0:-block_size * repetition_factor] + iSeq.input_files2

        iSeq.num_images = len(iSeq.input_files)
        # iSeq.params = [ids, expressions, morphs, poses, lightings]
        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions,
                       iSeq.morphs, iSeq.poses, iSeq.lightings]
        iSeq.block_size = iSeq.num_images // len(iSeq.faces)
        iSeq.train_mode = "clustered"  # "serial" #"clustered"
        print ("BLOCK SIZE =" + str(iSeq.block_size))

        iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(iSeq.faces)), iSeq.block_size)
        iSeq.correct_labels = iSeq.correct_classes // (len(iSeq.faces) - 1)

        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateRFaceCentering(self, iSeq, pipeline_fd, pipeline_factor=2.0, seed=-1):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)
        # else seed <0 then, do not change seed

        print ("******** Setting Training Data Parameters for Real Face Centering****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.image_width = 256
        sSeq.image_height = 192
        sSeq.subimage_width = 135
        sSeq.subimage_height = 135

        sSeq.trans_x_max = pipeline_fd['dx0'] * 1.0 * pipeline_factor
        sSeq.trans_y_max = pipeline_fd['dy0'] * 1.0 * pipeline_factor  # * 0.998
        sSeq.pang_max = pipeline_fd['da0'] * pipeline_factor
        sSeq.min_sampling = pipeline_fd['smin0']  # - 0.1 #WARNING!!!
        sSeq.max_sampling = pipeline_fd['smax0']
        sSeq.avg_sampling = 0.825  # (sSeq.min_sampling + sSeq.max_sampling)/2

        sSeq.pixelsampling_x = numpy.zeros(sSeq.num_images)
        sSeq.pixelsampling_y = numpy.zeros(sSeq.num_images)
        sSeq.rotations = numpy.zeros(sSeq.num_images)
        sSeq.translations_x = numpy.zeros(sSeq.num_images)
        sSeq.translations_y = numpy.zeros(sSeq.num_images)

        num_blocks = sSeq.num_images // sSeq.block_size
        for block_nr in range(num_blocks):
            # For exterior box
            fraction = ((block_nr + 1.0) / (num_blocks - 1)) ** 0.25  # From 1/(N-1) to 1.0 # **0.333
            if fraction > 1:
                fraction = 1
            x_max = sSeq.trans_x_max * fraction
            y_max = sSeq.trans_y_max * fraction
            pang_max = sSeq.pang_max * fraction
            samp_max = sSeq.avg_sampling + (sSeq.max_sampling - sSeq.avg_sampling) * fraction * pipeline_factor
            samp_min = sSeq.avg_sampling + (sSeq.min_sampling - sSeq.avg_sampling) * fraction * pipeline_factor

            box_ext = [(-x_max, x_max), (-y_max, y_max), (-pang_max, pang_max), (samp_min, samp_max),
                       (samp_min, samp_max)]

            if block_nr >= 0:
                # For interior boxiSeq.ids = alldbnormalized_available_images[30000:45000]
                if block_nr < num_blocks - 1:
                    eff_block_nr = block_nr
                else:
                    eff_block_nr = block_nr - 1
                fraction2 = (eff_block_nr // (num_blocks - 1)) ** 0.25  # ** 0.333
                if fraction2 > 1:
                    fraction2 = 1
                x_max = sSeq.trans_x_max * fraction2
                y_max = sSeq.trans_y_max * fraction2
                pang_max = sSeq.pang_max * fraction2
                samp_max = sSeq.avg_sampling + (sSeq.max_sampling - sSeq.avg_sampling) * fraction2
                samp_min = sSeq.avg_sampling + (sSeq.min_sampling - sSeq.avg_sampling) * fraction2
                box_in = [(-x_max, x_max), (-y_max, y_max), (-pang_max, pang_max), (samp_min, samp_max),
                          (samp_min, samp_max)]

            samples = sub_box_sampling(box_in, box_ext, sSeq.block_size)
            sSeq.translations_x[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = samples[:, 0]
            sSeq.translations_y[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = samples[:, 1]
            sSeq.rotations[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = samples[:, 2]
            sSeq.pixelsampling_x[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = samples[:, 3]
            sSeq.pixelsampling_y[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = samples[:, 4]

        sSeq.subimage_first_row = sSeq.image_height / 2.0 - sSeq.subimage_height * sSeq.pixelsampling_y / 2.0
        sSeq.subimage_first_column = sSeq.image_width / 2.0 - sSeq.subimage_width * sSeq.pixelsampling_x / 2.0

        sSeq.add_noise_L0 = True
        sSeq.convert_format = "L"
        sSeq.background_type = None
        sSeq.trans_sampled = True

        sSeq.name = "Face Centering. Dx %d, Dy %d, Da %d, sampling in (%d, %d)" % (sSeq.trans_x_max, sSeq.trans_y_max,
                                                                                   int(sSeq.min_sampling * 1000),
                                                                                   int(sSeq.max_sampling * 1000),
                                                                                   int(sSeq.min_sampling * 1000),
                                                                                   int(sSeq.max_sampling * 1000))
        print (sSeq.name)
        iSeq.name = sSeq.name

        sSeq.load_data = load_data_from_sSeq  # lambda : load_data_from_sSeq(sSeq)

        system_parameters.test_object_contents(sSeq)
        return sSeq

    def sphere_sampling(self, radious, num_dims=1, num_vectors=1):
        output = numpy.random.uniform(-1, 1, (num_vectors, num_dims))
        norms = ((output ** 2).sum(axis=1)) ** 0.5
        new_radious = numpy.random.uniform(0, 1, num_vectors) ** (1.0 / num_dims) * radious
        factor = (1 / norms * new_radious).reshape((num_vectors, 1))
        norm_output = factor * output
        return norm_output

    def sub_sphere_sampling(self, radious_min, radious_max, num_dims=1, num_vectors=1):
        output = numpy.zeros((num_vectors, num_dims))
        incorrect = numpy.ones(num_vectors, dtype="bool")
        num_incorrect = num_vectors
        while num_incorrect > 0:
            print ("#", num_incorrect)
            # print "incorrect.sum()=",incorrect.sum()
            new_candidates = self.sphere_sampling(radious_max, num_dims, num_incorrect)
            output[incorrect] = new_candidates
            new_radious = (new_candidates ** 2).sum(axis=1) ** 0.5
            still_incorrect = (new_radious < radious_min)
            incorrect[incorrect == True] = still_incorrect
            num_incorrect -= (still_incorrect == False).sum()
        return output

    def sSeqCreateRFaceCentering2(self, iSeq, pipeline_fd, pipeline_factor=2.0, seed=-1):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)
        # else seed <0 then, do not change seed

        print ("******** Setting Training Data Parameters for Real Face Centering****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.image_width = 256
        sSeq.image_height = 192
        sSeq.subimage_width = 135
        sSeq.subimage_height = 135

        sSeq.trans_x_max = pipeline_fd['dx0'] * 1.0 * pipeline_factor
        sSeq.trans_y_max = pipeline_fd['dy0'] * 1.0 * pipeline_factor  # * 0.998
        sSeq.pang_max = pipeline_fd['da0'] * pipeline_factor
        sSeq.min_sampling = pipeline_fd['smin0']  # - 0.1 #WARNING!!!
        sSeq.max_sampling = pipeline_fd['smax0']
        sSeq.avg_sampling = 0.825  # (sSeq.min_sampling + sSeq.max_sampling)/2
        sSeq.delta_sampling = max(sSeq.max_sampling - sSeq.avg_sampling,
                                  sSeq.avg_sampling - sSeq.min_sampling) * pipeline_factor

        sSeq.pixelsampling_x = numpy.ones(sSeq.num_images)
        sSeq.pixelsampling_y = numpy.ones(sSeq.num_images)
        sSeq.rotations = numpy.zeros(sSeq.num_images)
        sSeq.translations_x = numpy.zeros(sSeq.num_images)
        sSeq.translations_y = numpy.zeros(sSeq.num_images)

        num_blocks = sSeq.num_images // sSeq.block_size
        for block_nr in range(num_blocks - 1):  # Last block (backgrounds) is not randomized
            radious_min = block_nr / (num_blocks - 1.0) * 4  #
            radious_max = (block_nr + 1.0) / (num_blocks - 1) * 4

            vectors = self.sub_sphere_sampling(radious_min, radious_max, num_dims=4, num_vectors=sSeq.block_size)
            sSeq.translations_x[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = \
                vectors[:, 0] * sSeq.trans_x_max
            sSeq.translations_y[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = \
                vectors[:, 1] * sSeq.trans_y_max
            sSeq.rotations[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = \
                vectors[:, 2] * sSeq.pang_max
            sSeq.pixelsampling_x[block_nr * sSeq.block_size: (block_nr + 1) * sSeq.block_size] = \
                sSeq.avg_sampling + vectors[:, 3] * sSeq.delta_sampling

        sSeq.pixelsampling_y = sSeq.pixelsampling_x.copy()

        sSeq.subimage_first_row = sSeq.image_height / 2.0 - sSeq.subimage_height * sSeq.pixelsampling_y / 2.0
        sSeq.subimage_first_column = sSeq.image_width / 2.0 - sSeq.subimage_width * sSeq.pixelsampling_x / 2.0

        sSeq.add_noise_L0 = True
        sSeq.convert_format = "L"
        sSeq.background_type = None
        sSeq.trans_sampled = True

        sSeq.name = "FaceCentering2. Dx %d, Dy %d, Da %d, Dsampling %d" % (
        sSeq.trans_x_max, sSeq.trans_y_max, int(sSeq.pang_max * 1000), int(sSeq.delta_sampling * 1000))
        print (sSeq.name)
        iSeq.name = sSeq.name

        sSeq.load_data = load_data_from_sSeq  # lambda : load_data_from_sSeq(sSeq)
        system_parameters.test_object_contents(sSeq)
        return sSeq


pipeline_fd_centering = dict(dx0=40, dy0=20, da0=22.5, smin0=0.694, smax0=0.981,
                             dx1=555, dy1=10, smin1=0.775, smax1=1.05)
fd_centering_iteration = 0  # valid values: 0 or 1
ParamsRFaceCentering2Func = ParamsRFaceCentering2Experiment(experiment_seed, experiment_basedir, pipeline_fd_centering,
                                                            iteration=fd_centering_iteration)


#####################################################################################################
# ################          German Traffic Sign Recognition Benchmark      ###########################
#####################################################################################################
# Base code originally taken from the competition website
class ParamsRGTSRBExperiment(system_parameters.ParamsSystem):
    """ Experiment for the recognition of traffic signs on German roads. """
    def __init__(self, experiment_seed, experiment_basedir):
        super(ParamsRGTSRBExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.experiment_basedir = experiment_basedir

        self.GTSRB_Images_dir_training = "/local/escalafl/Alberto/GTSRB/Final_Training/Images"
        self.GTSRB_HOG_dir_training = "/local/escalafl/Alberto/GTSRB/Final_Training/HOG"
        self.GTSRB_SFA_dir_training = "/local/escalafl/Alberto/GTSRB/Final_Training/HOG"

        self.GTSRB_Images_dir_test = "/local/escalafl/Alberto/GTSRB/Final_Test/Images"
        self.GTSRB_HOG_dir_test = "/local/escalafl/Alberto/GTSRB/Final_Test/HOG"
        self.GTSRB_SFA_dir_test = "/local/escalafl/Alberto/GTSRB/Final_Test/SFA"

        self.GTSRB_Images_dir_Online = "/local/escalafl/Alberto/GTSRB/Final_Test/Images"
        self.GTSRB_HOG_dir_Online = "/local/escalafl/Alberto/GTSRB/Final_Test/HOG"
        self.GTSRB_SFA_dir_Online = "/local/escalafl/Alberto/GTSRB/Final_Test/SFA"

        self.GTSRB_Images_dir_UnlabeledOnline = "/local/escalafl/Alberto/GTSRB/Online-Test/Images"
        self.GTSRB_HOG_dir_UnlabeledOnline = "/local/escalafl/Alberto/GTSRB/Online-Test/HOG"
        self.GTSRB_SFA_dir_UnlabeledOnline = "/local/escalafl/Alberto/GTSRB/Online-Test/SFA"

        self.analysis = None
        self.enable_reduced_image_sizes = True
        self.reduction_factor = 1.0  # WARNING 1.0, 2.0, 4.0, 8.0  / 1.0 2.0 ...
        self.hack_image_size = 128  # WARNING   128,  64,  32 , 16 / 160  80 ...
        self.enable_hack_image_size = True

    def create(self):
        numpy.random.seed(self.experiment_seed)
        # Annotations: filename, track, Width, Height, ROI.x1, ROI.y1, ROI.x2, ROI.y2, [label]
        # class 0 : 150   class 1 : 1500   class 2 : 1500   class 3 : 960   class 4 : 1320   class 5 : 1260
        # class 6 : 300   class 7 : 960   class 8 : 960
        # class 9 : 990   class 10 : 1350   class 11 : 900   class 12 : 1410   class 13 : 1440   class 14 : 540
        # class 15 : 420   class 16 : 300
        # class 17 : 750   class 18 : 810   class 19 : 150   class 20 : 240   class 21 : 240   class 22 : 270
        # class 23 : 360   class 24 : 180
        # class 25 : 1020   class 26 : 420   class 27 : 180   class 28 : 360   class 29 : 180   class 30 : 300
        # class 31 : 540   class 32 : 180
        # class 33 : 480   class 34 : 300   class 35 : 810   class 36 : 270   class 37 : 150   class 38 : 1380
        # class 39 : 210   class 40 : 240
        # class 41 : 180   class 42 : 180

        # repetition_factors = [10  1  1  1  1  1  5  1  1  1  1  1  1  1  2  3  5  2  1 10  6  6  5  4  8
        #  1  3  8  4  8  5  2  8  3  5  1  5 10  1  7  6  8  8]

        # repetition_factors = [5  1  1  1  1
        #                      1  3  1  1  1
        #                      1  1  1  1  2
        #                      2  3  2  1  5  4
        #                      4  3  4  8
        #  1  3  8  4  8  5  2  8  3  5  1  5 10  1  7  6  8  8]

        # For all tracks:
        # repetition_factorsTrain = [10,  1,  1,  1,  1,  1,  5,  1,  1,  1,  1,  1,  1,  1,  2,  3,  5,  2, 1, 10,
        # 6,  6,  5,  4,  8,  1,  3,  8,  4,  8,  5,  2,  8,  3,  5,  1,  5, 10,  1,  7,  6,  8,  8]

        # To support even/odd distribution
        # W
        # repetition_factors = numpy.array(repetition_factors)*2
        # repetition_factorsTrain = numpy.array(repetition_factorsTrain)*2

        # At most 15 tracks:
        # repetition_factors = [ 4, 1, 1, 1, 1, 1,
        #  2, 1, 1, 1, 1, 1,
        #  1, 1, 1, 2, 2, 1,
        #  1, 4, 3, 3, 2, 2,
        #  3, 1, 2, 3, 2, 3,
        #  2, 1, 3, 2, 2, 1,
        #  2, 4, 1, 3, 3, 3,
        #  3]

        GTSRB_present = os.path.lexists(
            "/local/escalafl/Alberto/GTSRB/")  # and os.path.lexists("/local/escalafl/on_lux10")
        if not GTSRB_present:
            er = "GTSRB: experiment data not found"
            raise Exception(er)

        GTSRBTrain_base_dir = "/local/escalafl/Alberto/GTSRB/Final_Training/Images"
        # "/local/escalafl/Alberto/GTSRB/Final_Training/Images"
        # GTSRBOnline_base_dir = "/local/escalafl/Alberto/GTSRB/Online-Test-sort/Images"
        # #"/local/escalafl/Alberto/GTSRB/Online-Test-sort/Images"
        # repetition_factorsTrain = None

        # TODO:Repetition factors should not be at this level, also randomization of scales does not belong here.
        # TODO:Update the databases used to the final system
        sign_selection = numpy.arange(43)

        sign_selection = list(set(sign_selection) - set([0, 19, 21, 24, 27, 29, 32, 37, 39, 41, 42]))

        GTSRB_annotationsTrain_Train = self.readTrafficSignsAnnotations(GTSRBTrain_base_dir, include_labels=True,
                                                                        sign_selection=sign_selection)
        # delta_rand_scale=0.07
        print ("Len GTSRB_annotationsTrain_Train= %d" % len(GTSRB_annotationsTrain_Train))
        GTSRB_annotationsTrain_Seenid = self.readTrafficSignsAnnotations(GTSRBTrain_base_dir, include_labels=True,
                                                                         sign_selection=sign_selection)
        # delta_rand_scale=0.07
        GTSRB_annotationsTrain_Newid = self.readTrafficSignsAnnotations(GTSRBTrain_base_dir, include_labels=True,
                                                                        sign_selection=sign_selection)

        #    GTSRB_annotationsOnline_Train = readTrafficSignsAnnotations(GTSRBOnline_base_dir, include_labels=True)
        #    GTSRB_annotationsOnline_Seenid = readTrafficSignsAnnotations(GTSRBOnline_base_dir, include_labels=True)
        #    GTSRB_annotationsOnline_Newid = readTrafficSignsAnnotations(GTSRBOnline_base_dir, include_labels=True)
        #    print "Len GTSRB_annotationsOnline_Newid=", len(GTSRB_annotationsOnline_Newid)
        #    quit()

        activate_HOG = True and False  # for direct image processing, true for SFA/HOG features
        # TODO: HOG set selection HOG02, SFA,
        switch_SFA_over_HOG = "HOG02"  # "SFA" # "HOG02", "SFA"

        print ("fixing max_samples_per_class")
        GTSRB_rep = 1  # 6
        GTSRB_constant_block_size = 360 * GTSRB_rep
        GTSRB_annotationsTrain_Train = self.enforce_max_samples_per_class(GTSRB_annotationsTrain_Train,
                                                                          GTSRB_constant_block_size,
                                                                          repetition=GTSRB_rep)
        GTSRB_annotationsTrain_Seenid = self.enforce_max_samples_per_class(GTSRB_annotationsTrain_Seenid,
                                                                           GTSRB_constant_block_size,
                                                                           repetition=GTSRB_rep)
        GTSRB_annotationsTrain_Newid = self.enforce_max_samples_per_class(GTSRB_annotationsTrain_Newid,
                                                                          GTSRB_constant_block_size)

        count = self.count_annotation_classes(GTSRB_annotationsTrain_Train)
        print ("number of samples per each class A (GTSRB_annotationsTrain_Train): ", count)

        # Correct class indices, so that only consecutive classes appear
        consequtive_classes = True
        if consequtive_classes:
            all_classes = []
            for row in GTSRB_annotationsTrain_Train:
                all_classes.append(row[8])
            unique, corrected_classes = numpy.unique(all_classes, return_inverse=True)
            correction_c = {}
            for i, c in enumerate(unique):
                correction_c[c] = i

            print("unique classes in GTSRB_annotationsTrain_Train:" + str(unique))

            for _, row in enumerate(GTSRB_annotationsTrain_Train):
                row[8] = correction_c[row[8]]
            for row in GTSRB_annotationsTrain_Seenid:
                row[8] = correction_c[row[8]]
            for row in GTSRB_annotationsTrain_Newid:
                row[8] = correction_c[row[8]]
            print("class consequtive correction mapping" + str(correction_c))

        count = self.count_annotation_classes(GTSRB_annotationsTrain_Train)
        print("number of samples per each class B (GTSRB_annotationsTrain_Train): " + str(count))

        shuffle_classes = True
        if shuffle_classes:
            all_classes = []
            for row in GTSRB_annotationsTrain_Train:
                all_classes.append(row[8])
            unique_c = numpy.unique(all_classes)
            print("unique_c=" + str(unique_c))

            shuffled_c = unique_c.copy()
            numpy.random.shuffle(shuffled_c)
            print("shuffled_c=" + str(shuffled_c))

            shuffling_c = {}
            for i, c in enumerate(unique_c):
                shuffling_c[c] = shuffled_c[i]
            print("shuffling_c=" + str(shuffling_c))

            for row in GTSRB_annotationsTrain_Train:
                row[8] = shuffling_c[row[8]]

            for row in GTSRB_annotationsTrain_Seenid:
                row[8] = shuffling_c[row[8]]

            for row in GTSRB_annotationsTrain_Newid:
                row[8] = shuffling_c[row[8]]

            print("class shuffling mapping" + str(shuffling_c))

        count = self.count_annotation_classes(GTSRB_annotationsTrain_Train)
        print("number of samples per each class C (GTSRB_annotationsTrain_Train): " + str(count))

        #    count = count_annotation_classes(GTSRB_annotationsTrain_Seenid)
        #    print "number of samples per each class (GTSRB_annotationsTrain_Seenid): ", count

        #    count = count_annotation_classes(GTSRB_annotationsTrain_Newid)
        #    print "number of samples per each class (GTSRB_annotationsTrain_Newid): ", count

        # print 500.0/count+0.99
        # print 700/count
        # print GTSRB_annotationsTrain[0]
        #
        # TODO:Add a first_sample parameter

        GTSRB_annotationsTrain_TrainOrig = GTSRB_annotationsTrain_Train
        GTSRB_annotationsTrain_Train = self.sample_annotations(GTSRB_annotationsTrain_TrainOrig, first_sample=0,
                                                               num_samples=360 * GTSRB_rep)  # W 0.6, 0.5, Odd, AllP,
        # Univ, 2/3
        GTSRB_annotationsTrain_Seenid = self.sample_annotations(GTSRB_annotationsTrain_TrainOrig,
                                                                first_sample=0 * GTSRB_rep,
                                                                num_samples=360 * GTSRB_rep)  # W 1.0, 0.3, Even, 1/3
        GTSRB_annotationsTrain_Newid = self.sample_annotations(GTSRB_annotationsTrain_TrainOrig,
                                                               first_sample=240 * GTSRB_rep,
                                                               num_samples=120 * GTSRB_rep)  # make testing faster for
        # now, 0.3 , 1.0

        #    GTSRB_annotationsTrain_Train = sample_annotations(GTSRB_annotationsTrain_TrainOrig,
        # first_sample = 0, num_samples=360*GTSRB_rep) #W 0.6, 0.5, Odd, AllP, Univ, 2/3
        #    GTSRB_annotationsTrain_Seenid = sample_annotations(GTSRB_annotationsTrain_TrainOrig,
        # first_sample = 0*GTSRB_rep, num_samples=360*GTSRB_rep ) #W 1.0, 0.3, Even, 1/3
        #    GTSRB_annotationsTrain_Newid = sample_annotations(GTSRB_annotationsTrain_TrainOrig,
        # first_sample = 240*GTSRB_rep, num_samples=10*GTSRB_rep ) #make testing faster for now, 0.3 , 1.0

        # GTSRB_annotationsOnline_Train = sample_annotations(GTSRB_annotationsOnline_Train, flag="Univ",
        # passing=0.25) #W 0.6, 0.5, Odd, AllP, Univ
        # GTSRB_annotationsOnline_Seenid = sample_annotations(GTSRB_annotationsOnline_Seenid, flag="Univ",
        # passing=0.25) #W 1.0, 0.3, Even
        # GTSRB_annotationsOnline_Newid = sample_annotations(GTSRB_annotationsOnline_Newid, flag="Univ",
        # passing=1.0) #make testing faster for now, 0.3 , 1.0

        # count = count_annotation_classes(GTSRB_annotationsTest)
        # print count
        # quit()
        #
        GTSRB_Unlabeled_base_dir = "/local/escalafl/Alberto/GTSRB/Final_Test/Images"
        # "/local/escalafl/Alberto/GTSRB/Online-Test/Images"
        GTSRB_Unlabeled_csvfile = "GT-final_test.csv"
        # "GT-final_test.csv" / "GT-final_test.test.csv" / "GT-online_test.test.csv"
        GTSRB_UnlabeledAnnotations = self.readTrafficSignsAnnotationsOnline(GTSRB_Unlabeled_base_dir,
                                                                            GTSRB_Unlabeled_csvfile, shrink_signs=True,
                                                                            correct_sizes=True, include_labels=True,
                                                                            sign_selection=sign_selection)
        # include_labels=False

        if consequtive_classes:
            for _, row in enumerate(GTSRB_UnlabeledAnnotations):
                if row[8] in correction_c.keys():
                    row[8] = correction_c[row[8]]

        if shuffle_classes:
            for row in GTSRB_UnlabeledAnnotations:
                if row[8] in shuffling_c.keys():
                    row[8] = shuffling_c[row[8]]

        GTSRB_UnlabeledAnnotations = self.sample_annotations(
            GTSRB_UnlabeledAnnotations)  # make testing faster for now, 0.3 , 1.0
        # print GTSRB_UnlabeledAnnotations[0], len(GTSRB_UnlabeledAnnotations)
        # quit()

        GTSRB_annotationsTrain = GTSRB_annotationsTrain_Train
        GTSRB_annotationsSeenid = GTSRB_annotationsTrain_Seenid
        GTSRB_annotationsTest = GTSRB_UnlabeledAnnotations  # GTSRB_UnlabeledAnnotations, GTSRB_annotationsOnline_Newid

        # # CASE [[F]]
        # WARNING!
        # #############################'''WAAAAAARNNNNIIINNNNNNGGGG TRTRAAAACKKKKSSSSSS 1111
        # W first track=1
        # #iSeq_set = iTrainRGTSRB_Labels = [[iSeqCreateRGTSRB_Labels(GTSRB_annotationsTrain, first_track = 0,
        # last_track=100, seed=-1),
        # #                                   iSeqCreateRGTSRB_Labels(GTSRB_annotationsTestData, first_track = 0,
        # last_track=4, labels_available=False, seed=-1)]]
        # #sSeq_set = sTrainRGTSRB = [[sSeqCreateRGTSRB(iSeq_set[0][0], enable_translation=True, enable_rotation=True,
        # contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG,
        # add_HOG_noise=True, seed=-1),
        # #                            sSeqCreateRGTSRB(iSeq_set[0][1], enable_translation=False, enable_rotation=False,
        #  contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG,
        #  add_HOG_noise=True,  seed=-1)]]
        # #
        # #iSeq_set = iSeenidRGTSRB_Labels = iSeqCreateRGTSRB_Labels(GTSRB_annotationsSeenid, first_track = 0,
        # last_track=100, seed=-1)
        # #sSeq_set = sSeenidRGTSRB = sSeqCreateRGTSRB(iSeq_set, enable_translation=True, enable_rotation=True,
        # contrast_enhance=True, activate_HOG=activate_HOG, switch_SFA_over_HOG= switch_SFA_over_HOG,
        # add_HOG_noise=True, seed=-1)
        # #    iSeq_set = iTrainRGTSRB_Labels = [[iSeqCreateRGTSRB_Labels(GTSRB_annotationsTrain, first_track = 0,
        # last_track=100, seed=-1),
        # #                                       iSeqCreateRGTSRB_Labels(GTSRB_annotationsTest, first_track = 0,
        # last_track=100, labels_available=False, seed=-1)]]
        # #    sSeq_set = sTrainRGTSRB = [[sSeqCreateRGTSRB(iSeq_set[0][0], enable_translation=True,
        # enable_rotation=True, contrast_enhance=True, activate_HOG=activate_HOG,
        # switch_SFA_over_HOG=switch_SFA_over_HOG, feature_noise=0.03, seed=-1),
        # #                                sSeqCreateRGTSRB(iSeq_set[0][1], enable_translation=True,
        # enable_rotation=True, contrast_enhance=True, activate_HOG=activate_HOG,
        # switch_SFA_over_HOG=switch_SFA_over_HOG, feature_noise=0.0,  seed=-1)]]

        # delta_translation=0.0, delta_scaling=0.1, delta_rotation=4.0
        contrast_enhance = "GTSRBContrastEnhancement"
        #    delta_translation_t= 1.25 #1.5 #1.25
        #    delta_scaling_t= 0.070 #0.075 #0.065
        #    delta_rotation_t = 3.5 #2.5 pointer
        #
        #    delta_translation_s= 2.0 #RETUNE! # 1.0 #1.5 #1.25
        #    delta_scaling_s= 0.110 # Retune! 0.056 #0.075 #0.065
        #    delta_rotation_s = 2.75 #2.8 #2.5 pointer
        # NOTE: TUNED PARAMS: 2.0, 0.11 and 2.75, however strange results,
        # so reverting to same parameters used for training

        delta_translation_t = 1.5
        delta_scaling_t = 0.090
        delta_rotation_t = 3.15

        delta_translation_s = 1.5
        delta_scaling_s = 0.090
        delta_rotation_s = 3.15

        iSeq_set = iTrainRGTSRB_Labels = [[self.iSeqCreateRGTSRB_Labels(GTSRB_annotationsTrain, repetition_factors=6,
                                                                        seed=-1)]]  # repetition_factors=4
        # ##last_track=100
        #    sSeq_set = sTrainRGTSRB = [[sSeqCreateRGTSRB(iSeq_set[0][0], delta_translation=0.0, delta_scaling=0.0,
        # delta_rotation=0.0, contrast_enhance=contrast_enhance, activate_HOG=activate_HOG,
        # switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.00, seed=-1) ]] #feature_noise=0.04
        sTrainRGTSRB = [[self.sSeqCreateRGTSRB(iSeq_set[0][0], delta_translation=delta_translation_t,
                                               delta_scaling=delta_scaling_t, delta_rotation=delta_rotation_t,
                                               contrast_enhance=contrast_enhance, activate_HOG=activate_HOG,
                                               switch_SFA_over_HOG=switch_SFA_over_HOG, feature_noise=0.00,
                                               seed=-1)]]  # feature_noise=0.04

        iSeq_set = iSeenidRGTSRB_Labels = self.iSeqCreateRGTSRB_Labels(GTSRB_annotationsSeenid, repetition_factors=6,
                                                                       seed=-1)  # repetition_factors=2
        #    sSeq_set = sSeenidRGTSRB = sSeqCreateRGTSRB(iSeq_set, delta_translation=0.0, delta_scaling=0.00,
        # delta_rotation=0.0, contrast_enhance=contrast_enhance, activate_HOG=activate_HOG,
        # switch_SFA_over_HOG= switch_SFA_over_HOG, feature_noise=0.00, seed=-1) #feature_noise=0.07
        sSeenidRGTSRB = self.sSeqCreateRGTSRB(iSeq_set, delta_translation=delta_translation_s,
                                              delta_scaling=delta_scaling_s, delta_rotation=delta_rotation_s,
                                              contrast_enhance=contrast_enhance, activate_HOG=activate_HOG,
                                              switch_SFA_over_HOG=switch_SFA_over_HOG, feature_noise=0.00,
                                              seed=-1)  # feature_noise=0.07

        # GTSRB_onlineAnnotations, GTSRB_annotationsTest
        iSeq_set = iNewidRGTSRB_Labels = [
            [self.iSeqCreateRGTSRB_Labels(GTSRB_annotationsTest, repetition_factors=1, seed=-1)]]
        sNewidRGTSRB = self.sSeqCreateRGTSRB(iSeq_set[0][0], delta_translation=0.0, delta_scaling=0.0,
                                             delta_rotation=0.0, contrast_enhance=contrast_enhance,
                                             activate_HOG=activate_HOG, switch_SFA_over_HOG=switch_SFA_over_HOG,
                                             feature_noise=0.0, seed=-1)
        # sSeq_set.add_noise_L0 = False
        # sSeq_set.rotations = 0.0

        #    print iTrainRGTSRB_Labels[0][0].input_files[0:5]
        #    print iSeenidRGTSRB_Labels.input_files[0:5]
        #    quit()
        print (iTrainRGTSRB_Labels[0][0].correct_labels)
        print (iSeenidRGTSRB_Labels.correct_labels)
        print (iNewidRGTSRB_Labels[0][0].correct_labels)
        # quit()

        self.name = "Function Based Data Creation for GTSRB"
        self.network = "linearNetwork4L"  # Default Network, but ignored
        self.iTrain = iTrainRGTSRB_Labels
        self.sTrain = sTrainRGTSRB

        self.iSeenid = iSeenidRGTSRB_Labels
        self.sSeenid = sSeenidRGTSRB

        self.iNewid = iNewidRGTSRB_Labels
        self.sNewid = sNewidRGTSRB

        self.block_size = iTrainRGTSRB_Labels[0][0].block_size
        self.train_mode = "ignored"  # clustered, #Identity recognition task
        self.analysis = None
        self.activate_HOG = activate_HOG

        if not activate_HOG:
            self.enable_reduced_image_sizes = True  # and False #Set to false if network is a cascade
            self.reduction_factor = 32.0 / 48  # 1.0 # WARNING   0.5, 1.0, 2.0, 4.0, 8.0
            self.hack_image_size = 48  # WARNING    64,  32,  16,   8
            self.enable_hack_image_size = True  # and False #Set to false if network is a cascade
        else:
            if switch_SFA_over_HOG == "HOG02":
                self.enable_reduced_image_sizes = False
                self.enable_hack_image_size = True
                self.hack_image_size = 16  # WARNING    32,  16,   8
            else:
                self.enable_reduced_image_sizes = False
                self.enable_hack_image_size = False

    # arguments: path to the traffic sign data, for example './GTSRB/Training'
    # returns: list of image filenames, list of all tracks, list of all image annotations, list of all labels,
    def readTrafficSignsAnnotations(self, rootpath, shrink_signs=True, shrink_factor=0.8, correct_sizes=True,
                                    include_labels=False, sign_selection=None):
        """Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels"""

        import csv
        #    images = [] # image filenames
        annotations = []  # annotations for each image
        #    labels = [] # corresponding labels
        #    tracks = []
        # loop over all 42 classes
        delta_rand_scale = 0.0
        repetition_factors = None

        if sign_selection is None:
            sign_selection = numpy.arange(43)

        if repetition_factors is None:
            repetition_factors = [1] * 43
        for c in sign_selection:
            prefix = rootpath + '/' + "%05d" % c + '/'  # subdirectory for class. format(c, '05d')
            gtFile = open(prefix + 'GT-' + "%05d" % c + '.csv')  # annotations file. format(c, '05d')
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
            gtReader.next()  # skip header
            # loop over all images in current annotations file
            for _, row in enumerate(gtReader):
                #            if ii%1000==0:
                #                print row
                image_filename = prefix + row[0]
                extended_row = [
                    image_filename]  # extended_row: filename, track number, im_width, im_height, x0, y0, x1, y1
                if correct_sizes:
                    im = Image.open(image_filename)
                    #            row[1:3] = map(int, row[1:3])
                    #            if row[1] != im.size[0] or row[2] != im.size[1]:
                    #                print "Image %s has incorrect size label"%image_filename, row[1:3], im.size[0:2]
                    row[1] = im.size[0]
                    row[2] = im.size[1]
                    del im
                extended_row.append(int(row[0][0:5]))  # Extract track number
                if shrink_signs:
                    sign_coordinates = map(float, row[3:7])
                    center_x = (sign_coordinates[0] + sign_coordinates[2]) / 2.0
                    center_y = (sign_coordinates[1] + sign_coordinates[3]) / 2.0
                    rand_scale_factor1 = 1.0 + numpy.random.uniform(-delta_rand_scale, delta_rand_scale)
                    rand_scale_factor2 = 1.0 + numpy.random.uniform(-delta_rand_scale, delta_rand_scale)
                    width = (sign_coordinates[2] - sign_coordinates[0] + 1) * shrink_factor * rand_scale_factor1
                    height = (sign_coordinates[3] - sign_coordinates[1] + 1) * shrink_factor * rand_scale_factor2
                    row[3] = center_x - width // 2
                    row[5] = center_x + width // 2
                    row[4] = center_y - height // 2
                    row[6] = center_y + height // 2
                extended_row = extended_row + map(float, row[1:7])
                if include_labels:
                    extended_row.append(int(row[7]))  # the 8th column is the label
                for i in range(repetition_factors[c]):
                    annotations.append(extended_row)
            gtFile.close()

            #     #Correct class indices, so that only consecutive classes appear
            #     if correct_classes:
            #         all_classes=[]
            #         for row in annotations:
            #             all_classes.append(row[8])
            #         unique, corrected_classes = numpy.unique(all_classes, return_inverse=True)
            #         print "unique classes:", unique
            #         for ii, row in enumerate(annotations):
            #             row[8] = corrected_classes[ii]

        return annotations

    def readTrafficSignsAnnotationsOnline(self, prefix, csv_file, shrink_signs=True, correct_sizes=True,
                                          include_labels=False, sign_selection=None):
        """Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels"""
        import csv
        if sign_selection is None:
            sign_selection = numpy.arange(43)

        # images = [] # image filenames
        annotations = []  # annotations for each image
        labels = []  # corresponding labels
        # tracks = []
        # loop over all 42 classes
        gtFile = open(prefix + '/' + csv_file)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.next()  # skip header
        # loop over all images in current annotations file
        for _, row in enumerate(gtReader):
            #            if ii%1000==0:
            #                print row
            image_filename = prefix + "/" + row[0]
            extended_row = [image_filename]  # extended_row: filename, track number, im_width, im_height, x0, y0, x1, y1
            if correct_sizes:
                im = Image.open(image_filename)
                #                row[1:3] = map(int, row[1:3])
                #                if row[1] != im.size[0] or row[2] != im.size[1]:
                #                    print "Image %s has incorrect size label"%image_filename, row[1:3], im.size[0:2]
                row[1] = im.size[0]
                row[2] = im.size[1]
                del im
            extended_row.append(0)  # Fictuous track number
            if shrink_signs:
                shrink_factor = 0.8
                sign_coordinates = map(float, row[3:7])
                center_x = (sign_coordinates[0] + sign_coordinates[2]) / 2.0
                center_y = (sign_coordinates[1] + sign_coordinates[3]) / 2.0
                width = (sign_coordinates[2] - sign_coordinates[0] + 1) * shrink_factor
                height = (sign_coordinates[3] - sign_coordinates[1] + 1) * shrink_factor
                row[3] = center_x - width // 2
                row[5] = center_x + width // 2
                row[4] = center_y - height // 2
                row[6] = center_y + height // 2
            extended_row = extended_row + map(float, row[1:7])
            if include_labels:
                label = int(row[7])
                if label in sign_selection:
                    extended_row.append(label)  # the 8th column is the label, not available in online test
                    annotations.append(extended_row)
            else:
                extended_row.append(0)
                annotations.append(extended_row)
        gtFile.close()

        return annotations

    def count_annotation_classes(self, annotations):
        classes = []
        count = []
        for row in annotations:
            classes.append(row[8])
        classes = numpy.array(classes)
        for c in range(0, 43):
            # print "  class %d appears:"%c, (classes[:]==c).sum(),
            count.append((classes[:] == c).sum())
        return numpy.array(count)

    def enforce_max_samples_per_class(self, annotations, max_samples=150, repetition=1):
        annot_list = [None] * 43
        for i in range(len(annot_list)):
            annot_list[i] = []

        if isinstance(repetition, int):
            repetition = [repetition] * 43

        for row in annotations:
            c = int(row[8])
            for i in range(repetition[c]):
                annot_list[c] += [copy.deepcopy(row)]

        print("fixing max number of samples per class")
        for c in range(43):
            if len(annot_list[c]) > max_samples:
                #            print "len(class[%d])=%d"%(c, len(annot_list[c])),
                numpy.random.shuffle(annot_list[c])
                annot_list[c] = annot_list[c][0:max_samples]
        new_annotations = []
        for c in range(43):
            #        print " len(class[%d])="%c, len(annot_list[c]),
            new_annotations += annot_list[c]
        return new_annotations

    def sample_annotations(self, annotations, first_sample=0, num_samples=None):
        if len(annotations[0]) < 8 or (first_sample is None and num_samples is None):
            return annotations

        c_annotations = []
        for c in range(43):
            c_annotations.append([])
        for row in annotations:
            c_annotations[row[8]].append(row)

        out_annotations = []
        for c in range(43):
            if num_samples is None:
                num_samples = len(c_annotations[c]) - first_sample
            # print "Add", first_sample, first_sample+num_samples, len(c_annotations[c]),
            out_annotations += c_annotations[c][first_sample:first_sample + num_samples]
            #    print "out_annotations=", out_annotations
        return out_annotations

        # Switch either HOG or SFA here

    def load_HOG_data(self, base_GTSRB_dir="/home/eban/GTSRB", filenames=None, switch_SFA_over_HOG="HOG02",
                      feature_noise=0.0, padding=False):
        all_data = []
        print("HOG DATA LOADING %d images..." % len(filenames))
        # online_base_dir = GTSRB_HOG_dir_testing #GTSRB_Images_dir_UnlabeledOnline #Unlabeled data dir

        # warning, assumes all filenames belong to same directory!!!
        # make this a good function and apply it to every file
        if filenames[0][0:len(
                self.GTSRB_Images_dir_UnlabeledOnline)] == self.GTSRB_Images_dir_UnlabeledOnline:  # or final data base dir
            if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:
                base_hog_dir = self.GTSRB_HOG_dir_UnlabeledOnline
            elif switch_SFA_over_HOG in ["SFA"]:
                base_hog_dir = self.GTSRB_SFA_dir_UnlabeledOnline
            else:
                er = "Incorrect 1 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                raise Exception(er)
            unlabeled_data = True
        elif filenames[0][0:len(self.GTSRB_Images_dir_training)] == self.GTSRB_Images_dir_training:
            if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:
                base_hog_dir = self.GTSRB_HOG_dir_training
            elif switch_SFA_over_HOG in ["SFA02"]:
                base_hog_dir = self.GTSRB_SFA_dir_training
            else:
                er = "Incorrect 2 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                raise Exception(er)
            unlabeled_data = False
        elif filenames[0][0:len(self.GTSRB_Images_dir_test)] == self.GTSRB_Images_dir_test:
            if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:
                base_hog_dir = self.GTSRB_HOG_dir_test
            elif switch_SFA_over_HOG in ["SFA02"]:
                base_hog_dir = self.GTSRB_SFA_dir_training  # undefined
            else:
                er = "Incorrect 2 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                raise Exception(er)
            unlabeled_data = True
        elif filenames[0][0:len(self.GTSRB_Images_dir_Online)] == self.GTSRB_Images_dir_Online:
            if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:
                base_hog_dir = self.GTSRB_HOG_dir_Online
            elif switch_SFA_over_HOG in ["SFA02"]:
                base_hog_dir = self.GTSRB_SFA_dir_Online
            else:
                er = "Incorrect 2.3 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                raise Exception(er)
            unlabeled_data = False
        else:
            er = "Filename does not belong to known data sets: ", filenames[0]
            raise Exception(er)

        feat_set = switch_SFA_over_HOG[-2:]
        sample_hog_filename = "00000/00001_00000.txt"  # "00000/00001_00000.txt" (Older)
        print("HOG loading..." + str(feat_set) + filenames[0] + filenames[-1])
        for ii, image_filename in enumerate(filenames):
            if ii == 0:
                print (str("image_filename=" + image_filename))
            if unlabeled_data:
                if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:
                    hog_filename = base_hog_dir + "/HOG_" + feat_set + "/" + image_filename[-9:-3] + "txt"
                elif switch_SFA_over_HOG in ["SFA02"]:
                    hog_filename = base_hog_dir + "/SFA_" + feat_set + "/" + image_filename[-9:-3] + "txt"
                else:
                    er = "Incorrect 3 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                    raise Exception(er)
            else:
                if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03"]:
                    hog_filename = base_hog_dir + "/HOG_" + feat_set + "/" + image_filename[
                                                                             -len(sample_hog_filename):-3] + "txt"
                elif switch_SFA_over_HOG in ["SFA02"]:
                    hog_filename = base_hog_dir + "/SFA_" + feat_set + "/" + image_filename[
                                                                             -len(sample_hog_filename):-3] + "txt"
                else:
                    er = "Incorrect 4 switch_SFA_over_HOG value:", switch_SFA_over_HOG
                    raise Exception(er)
            if ii == 0:
                print("first hogh_filename:" + hog_filename)
            data_file = open(hog_filename, "rb")
            data = [float(line) for line in data_file.readlines()]
            data_file.close()
            data = numpy.array(data)
            # print data
            all_data.append(data)
        all_data = numpy.array(all_data)

        # 0.035 => New Id: 0.938 CR_CCC, CR_Gauss 0.943,
        # 0.025 => New Id: 0.938 CR_CCC, CR_Gauss 0.940,
        # Adding repetitions for balancing: 0.025 => New Id: 0.939 CR_CCC, CR_Gauss 0.949
        # Adding repetitions for balancing: 0.04 => New Id: 0.936 CR_CCC, CR_Gauss 0.956
        if feature_noise > 0.0:
            noise = numpy.random.normal(loc=0.0, scale=feature_noise, size=all_data.shape)
            print ("feature (SFA/HOG) Noise added in amount:", feature_noise)
            all_data += noise
        else:
            print ("NO feature (SFA/HOG) Noise added")

        if padding:
            if switch_SFA_over_HOG in ["HOG01", "HOG02"]:
                num_samples = all_data.shape[0]
                true_feature_data_shape = (num_samples, 14, 14, 8)
                desired_feature_data_shape = (num_samples, 16, 16, 8)
                all_data = numpy.reshape(all_data, true_feature_data_shape)
                noise_data = numpy.random.normal(loc=1.0, scale=0.05, size=desired_feature_data_shape)
                noise_data[:, 0:14, 0:14, 0:8] = all_data[:, :, :, :]
                all_data = numpy.reshape(noise_data, (num_samples, 16 * 16 * 8))
            else:
                er = "Padding not supported for data type: ", switch_SFA_over_HOG
                raise Exception(er)
        return all_data

    # Real last_track is num_tracks - skip_end_tracks -1
    def iSeqCreateRGTSRB_Labels(self, annotations, labels_available=True, repetition_factors=1, seed=-1):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)

        print ("***** Setting Training Information Parameters for German Traffic Sign Recognition Benchmark ******")
        iSeq = system_parameters.ParamsInput()
        iSeq.name = "German Traffic Sign Recognition Benchmark"

        iSeq.data_base_dir = ""
        #
        all_classes = []
        for row in annotations:
            all_classes.append(row[8])
            # print "adding ", row[8],
        print ("all_classes=" +  str( all_classes))
        all_classes = numpy.unique(all_classes)
        print ("all_classes=" +  str( all_classes))

        c_filenames = {}
        c_info = {}
        c_labels = {}
        for c in all_classes:
            c_filenames[c] = []
            c_info[c] = []
            c_labels[c] = []

        counter = 0
        for row in annotations:
            c = row[8]  # extract class number
            c_filenames[c].append(row[0])
            c_info[c].append(row[2:8])
            c_labels[c].append(c)
            counter += 1
        print ("Sample counter before repetition= %d" % counter)

        if repetition_factors is None:
            repetition_factors = 1
        if isinstance(repetition_factors, int):
            num_classes = len(all_classes)
            repetition_factors = [repetition_factors] * num_classes
        if isinstance(repetition_factors, list):
            dic = {}
            for i, c in enumerate(all_classes):
                dic[c] = repetition_factors[i]
        print (repetition_factors)
        for c in all_classes:
            # print "c=", c, len(c_info[c]), len(c_info), len(c_info[0]), len(c_info[1])
            c_filenames[c] = (c_filenames[c]) * (repetition_factors[c])
            c_info[c] = (c_info[c]) * (repetition_factors[c])
            c_labels[c] = (c_labels[c]) * (repetition_factors[c])

            # #    counter = 0
            # #    for row in annotations:
            # ##        print "T=%d"%row[1],
            # #        if row[1] >= first_track and row[1] < last_track:
            # #            c = row[8] #extract class number
            # ##            print c, len(c_filenames)
            # #            c_filenames[c].append(row[0])
            # #            c_info[c].append(row[2:8])
            # #            c_labels[c].append(c)
            # #            counter += 1
            # #    print "Sample counter after repetition=", counter
            #    print c_filenames[0][0], c_filenames[1][0]
            #    print c_filenames[0][1], c_filenames[1][1]

        conc_c_info = []
        for c in all_classes:
            conc_c_info += c_info[c]

            # print "conc_c_info is", conc_c_info
            # Avoids void entries
        iSeq.c_info = numpy.array(conc_c_info)

        print ("len(iSeq.c_info) = %d" % len(iSeq.c_info))
        # zeros((counter, 6))

        iSeq.ids = []
        iSeq.input_files = []
        iSeq.block_size = []

        for c in all_classes:
            if len(c_filenames[c]) > 0:
                iSeq.block_size.append(len(c_filenames[c]))
                print ("iSeq.block_size[%d]=%d" % (c, iSeq.block_size[-1]))
                for filename in c_filenames[c]:
                    iSeq.ids.append(c)
                    iSeq.input_files.append(filename)

        iSeq.block_size = numpy.array(iSeq.block_size)
        #    print iSeq.block_size
        #    quit()
        #    ii = 130
        #    print ii, iSeq.input_files[ii], iSeq.c_info[ii]
        #    print len(iSeq.input_files), len(iSeq.c_info)
        #    quit()

        iSeq.ids = numpy.array(iSeq.ids)
        # print "iSeq.ids", iSeq.ids
        #    quit()
        iSeq.ages = [None]
        iSeq.genders = [None]
        iSeq.racetweens = [None]
        iSeq.expressions = [None]
        iSeq.morphs = [None]
        iSeq.poses = [None]
        iSeq.lightings = [None]
        iSeq.slow_signal = 0  # real slow signal is the class number (type of sign)
        iSeq.step = 1
        iSeq.offset = 0

        iSeq.num_images = len(iSeq.input_files)
        # iSeq.params = [ids, expressions, morphs, poses, lightings]
        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                       iSeq.morphs, iSeq.poses, iSeq.lightings]
        if labels_available:
            iSeq.train_mode = "clustered"  # "compact_classes+5" #"clustered" #"compact_classes+31"
        else:
            iSeq.train_mode = "unlabeled"
        print ("BLOCK SIZE ="  +  str(iSeq.block_size))
        iSeq.correct_classes = copy.deepcopy(iSeq.ids)
        iSeq.correct_labels = copy.deepcopy(iSeq.ids)
        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateRGTSRB(self, iSeq, delta_translation=2.0, delta_scaling=0.1, delta_rotation=4.0,
                         contrast_enhance=True, activate_HOG=False, switch_SFA_over_HOG="HOG02", feature_noise=0.0,
                         seed=-1):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)
        # else seed <0 then, do not change seed

        print ("****** Setting Training Data Parameters for German Traffic Sign Recognition Benchmark  **************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        print ("iSec.c_info is" +  str(iSeq.c_info))
        sSeq.image_width = iSeq.c_info[:, 0]
        sSeq.image_height = iSeq.c_info[:, 1]
        sSeq.subimage_width = 32
        sSeq.subimage_height = 32

        #    sign_size = ((iSeq.c_info[:,4] - iSeq.c_info[:,2]) + (iSeq.c_info[:,5] - iSeq.c_info[:,3]))/2.0
        # Keep aspect ratio as in the original
        #    sSeq.pixelsampling_x = sign_size * 1.0 /  sSeq.subimage_width
        #    sSeq.pixelsampling_y = sign_size * 1.0 /  sSeq.subimage_height
        #    sSeq.subimage_first_row =  (iSeq.c_info[:,5] + iSeq.c_info[:,3])/2.0 - sign_size / 2.0
        #    sSeq.subimage_first_column = (iSeq.c_info[:,4] + iSeq.c_info[:,2])/2.0 - sign_size / 2.0

        sSeq.scales = 1 + numpy.random.uniform(-delta_scaling, delta_scaling, sSeq.num_images)

        sign_widths = (iSeq.c_info[:, 4] - iSeq.c_info[:, 2] + 1) * sSeq.scales
        sign_heights = iSeq.c_info[:, 5] - iSeq.c_info[:, 3] + 1 * sSeq.scales
        sign_centers_x = (iSeq.c_info[:, 4] + iSeq.c_info[:, 2]) * 0.5
        sign_centers_y = (iSeq.c_info[:, 5] + iSeq.c_info[:, 3]) * 0.5

        sSeq.pixelsampling_x = sign_widths / sSeq.subimage_width
        sSeq.pixelsampling_y = sign_heights / sSeq.subimage_height
        sSeq.subimage_first_row = sign_centers_y - sign_heights / 2
        sSeq.subimage_first_column = sign_centers_x - sign_widths / 2

        # sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0
        sSeq.add_noise_L0 = True
        if not activate_HOG:
            sSeq.convert_format = "RGB"
        else:
            sSeq.convert_format = switch_SFA_over_HOG

        sSeq.background_type = None

        sSeq.translations_x = numpy.random.uniform(-delta_translation, delta_translation, sSeq.num_images)
        sSeq.translations_y = numpy.random.uniform(-delta_translation, delta_translation, sSeq.num_images)

        sSeq.rotations = numpy.random.uniform(-delta_rotation, delta_rotation, sSeq.num_images)

        sSeq.contrast_enhance = contrast_enhance
        sSeq.trans_sampled = True  # Translations are given in subimage coordinates
        sSeq.name = "GTSRB"
        iSeq.name = sSeq.name

        if not activate_HOG:
            sSeq.load_data = load_data_from_sSeq
        else:
            if switch_SFA_over_HOG in ["HOG01", "HOG02", "HOG03", "SFA02"]:
                sSeq.load_data = lambda sSeq: self.load_HOG_data(base_GTSRB_dir="/local/scalafl/Alberto/GTSRB",
                                                                 filenames=sSeq.input_files,
                                                                 switch_SFA_over_HOG=switch_SFA_over_HOG,
                                                                 feature_noise=feature_noise, padding=True)
            elif switch_SFA_over_HOG == "SFA02_HOG02":
                sSeq.load_data = lambda sSeq: numpy.concatenate((
                    self.load_HOG_data(base_GTSRB_dir="/local/scalafl/Alberto/GTSRB", filenames=sSeq.input_files,
                                       switch_SFA_over_HOG="SFA02", feature_noise=feature_noise)[:, 0:49 * 2],
                    self.load_HOG_data(base_GTSRB_dir="/local/scalafl/Alberto/GTSRB", filenames=sSeq.input_files,
                                       switch_SFA_over_HOG="HOG02", feature_noise=feature_noise)), axis=1)
            else:
                er = "Unknown switch_SFA_over_HOG:", switch_SFA_over_HOG
                raise Exception(er)

            if switch_SFA_over_HOG in ["HOG01", "HOG02"]:
                sSeq.subimage_width = 16  # 14 #49
                sSeq.subimage_height = 16  # 14 #32 #channel dim = 8
                sSeq.convert_format = switch_SFA_over_HOG
            elif switch_SFA_over_HOG in ["HOG03"]:
                sSeq.subimage_width = 54
                sSeq.subimage_height = 54
                sSeq.convert_format = switch_SFA_over_HOG
            elif switch_SFA_over_HOG == "SFA02":
                sSeq.subimage_width = 7
                sSeq.subimage_height = 7
                sSeq.convert_format = switch_SFA_over_HOG
            elif switch_SFA_over_HOG == "SFA02_HOG02":
                sSeq.subimage_width = 49
                sSeq.subimage_height = 34
                sSeq.convert_format = switch_SFA_over_HOG
            else:
                quit()
        sSeq.train_mode = iSeq.train_mode
        system_parameters.test_object_contents(sSeq)
        return sSeq


ParamsRGTSRBFunc = ParamsRGTSRBExperiment(experiment_seed, experiment_basedir)


# This subsumes RTransX, RTransY, RTrainsScale
class ParamsRTransXYPAngScaleExperiment(system_parameters.ParamsSystem):
    """ Experiment for the localization (x,y,angle,scale) of faces in image patches (real photographs). """
    def __init__(self, experiment_seed, experiment_basedir, continuous=True, slow_var="All", iteration=0,
                 pipeline_fd_1Label=None, pipeline_fd_4Labels=None):
        super(ParamsRTransXYPAngScaleExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.experiment_basedir = experiment_basedir
        self.continuous = continuous
        self.slow_var = slow_var
        self.iteration = iteration
        if self.slow_var == "All":
            self.pipeline_fd = pipeline_fd_4Labels
        else:
            self.pipeline_fd = pipeline_fd_1Label

        self.analysis = None
        self.enable_reduced_image_sizes = True
        self.reduction_factor = 1.0  # WARNING 1.0, 2.0, 4.0, 8.0  / 1.0 2.0 ...
        self.hack_image_size = 128  # WARNING   128,  64,  32 , 16 / 160  80 ...
        self.enable_hack_image_size = True

    def create(self):
        numpy.random.seed(self.experiment_seed)

        print ("RTransXYPAngScale ***********************")
        # quit()

        alldbnormalized_available_images = numpy.arange(0, 64470)
        numpy.random.shuffle(alldbnormalized_available_images)

        # iSeq = iSeqCreateRTransY(4800, av_fim, first_image=0, repetition_factor=2, seed=-1)
        # sSeq = sSeqCreateRTransY(iSeq, seed=-1)
        # print ";) 3"
        # quit()
        normalized_base_dir_INIBilder = "/local/escalafl/Alberto/INIBilder/INIBilderNormalized"

        # # CASE [[F]]iSeqCreateRTransXYPAngScale(dx=45, dy=20, smin=0.55, smax=1.1, num_steps=20, slow_var = "X",
        # continuous=False, num_images_used=10000, images_base_dir= alldbnormalized_base_dir,
        # normalized_images=alldbnormalized_available_images, first_image_index=0, repetition_factor=1, seed=-1):

        pipeline_fd = self.pipeline_fd
        if self.slow_var == "X" or self.slow_var == "All":
            if self.iteration == 0:
                dx, dy, da, smin, smax = (pipeline_fd['dx0'], pipeline_fd['dy0'], pipeline_fd['da0'],
                                          pipeline_fd['smin0'], pipeline_fd['smax0'])
            elif self.iteration == 1:
                dx, dy, da, smin, smax = (pipeline_fd['dx1'], pipeline_fd['dy1'], pipeline_fd['da1'],
                                          pipeline_fd['smin1'], pipeline_fd['smax1'])
            elif self.iteration == 2 and slow_var == "All":
                dx, dy, da, smin, smax = (pipeline_fd['dx2'], pipeline_fd['dy2'], pipeline_fd['da2'],
                                          pipeline_fd['smin2'], pipeline_fd['smax2'])
            else:
                er = "incorrect iteration" + str(iteration)
                raise Exception(er)
        elif self.slow_var == "Y":
            if self.iteration == 0:
                dx, dy, da, smin, smax = (pipeline_fd['dx1'], pipeline_fd['dy0'], pipeline_fd['da0'],
                                          pipeline_fd['smin0'], pipeline_fd['smax0'])
            elif self.iteration == 1:
                dx, dy, da, smin, smax = (pipeline_fd['dx1'], pipeline_fd['dy1'], pipeline_fd['da1'],
                                          pipeline_fd['smin1'], pipeline_fd['smax1'])
            else:
                er = "incorrect iteration" + str(iteration)
                raise Exception(er)
        elif self.slow_var == "PAng":
            if self.iteration == 0:
                dx, dy, da, smin, smax = (pipeline_fd['dx1'], pipeline_fd['dy1'], pipeline_fd['da0'],
                                          pipeline_fd['smin0'], pipeline_fd['smax0'])
            elif self.iteration == 1:
                dx, dy, da, smin, smax = (pipeline_fd['dx1'], pipeline_fd['dy1'], pipeline_fd['da1'],
                                          pipeline_fd['smin1'], pipeline_fd['smax1'])
            else:
                er = "incorrect iteration" + str(iteration)
                raise Exception(er)
        elif self.slow_var == "Scale":
            if self.iteration == 0:
                dx, dy, da, smin, smax = (pipeline_fd['dx1'], pipeline_fd['dy1'], pipeline_fd['da1'],
                                          pipeline_fd['smin0'], pipeline_fd['smax0'])
            elif self.iteration == 1:
                dx, dy, da, smin, smax = (pipeline_fd['dx1'], pipeline_fd['dy1'], pipeline_fd['da1'],
                                          pipeline_fd['smin1'], pipeline_fd['smax1'])
            else:
                er = "incorrect iteration" + str(iteration)
                raise Exception(er)
        else:
            er = "incorrect slow_var " + str(self.slow_var)
            raise Exception(er)

        # dx=45.0
        # dy=20.0
        # da=10.0
        # smin = 0.55
        # smax = 1.1

        print ("dx=%g" % dx + "dy=%g" % dy + "da=%g" % da + "smin=%g" % smin + "smax=%g" % smax)
        iSeq_set = iTrainRTransXYPAngScale = \
            [[self.iSeqCreateRTransXYPAngScale(dx=dx, dy=dy, da=da, smin=smin, smax=smax, num_steps=50,
                                               slow_var=self.slow_var, continuous=self.continuous,
                                               num_images_used=55000,  # 30000
                                               images_base_dir=alldbnormalized_base_dir,
                                               normalized_images=alldbnormalized_available_images,
                                               first_image_index=0, pre_mirroring="none",
                                               repetition_factor=2,
                                               seed=-1)]]  # ###repetition factor is 2
        # Experiment below is just for display purposes!!! comment it out!
        # iSeq_set = iTrainRTransXYPAngScale = [[iSeqCreateRTransXYPAngScale(dx=45, dy=2, da=da, smin=0.85, smax=0.95,
        #  num_steps=50, slow_var = "X", continuous=continuous, num_images_used=15000, #30000
        #                                                      images_base_dir=alldbnormalized_base_dir,
        #  normalized_images = alldbnormalized_available_images,
        #                                                      first_image_index=0, repetition_factor=1, seed=-1)]]
        # For generating INI images
        # da=10.0
        # iSeq_set = iTrainRTransXYPAngScale = [[iSeqCreateRTransXYPAngScale(dx=45, dy=20, da=da, smin=0.55,
        #  smax=1.1, num_steps=79, slow_var = "X", continuous=continuous, num_images_used=79, #30000
        #                                                      images_base_dir=normalized_base_dir_INIBilder,
        #  normalized_images = numpy.arange(0, 79),
        #                                                      first_image_index=0, repetition_factor=10, seed=143)]]
        sTrainRTransXYPAngScale = [[self.sSeqCreateRTransXYPAngScale(iSeq_set[0][0], seed=-1)]]

        # iSeq_set = iTrainRTransY2 = [[copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=4500,
        #  repetition_factor=1, seed=-1)],
        #                             [copy.deepcopy(iSeq0), iSeqCreateRTransY(900, av_fim, first_image=8100,
        #  repetition_factor=1, seed=-1)],
        #                             ]
        # sSeq_set = sTrainRTransY2 = [[copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[0][1], seed=-1)],
        #                             [copy.deepcopy(sSeq0), sSeqCreateRTransY(iSeq_set[4][1], seed=-1)],
        #                             ]

        iSeq_set = iSeenidRTransXYPAngScale = self.iSeqCreateRTransXYPAngScale(dx=dx, dy=dy, da=da, smin=smin,
                                                                               smax=smax, num_steps=50,
                                                                               slow_var=self.slow_var, continuous=True,
                                                                               num_images_used=55000,  # 20000
                                                                               images_base_dir=alldbnormalized_base_dir,
                                                                               normalized_images=alldbnormalized_available_images,
                                                                               first_image_index=0,
                                                                               pre_mirroring="none",
                                                                               repetition_factor=2,
                                                                               seed=-1)  # repetition factor is 2
        sSeenidRTransXYPAngScale = self.sSeqCreateRTransXYPAngScale(iSeq_set, seed=-1)

        # WARNING, here continuous=continuous was wrong!!! we should always use the same test data!!!
        iSeq_set = iNewidRTransXYPAngScale = [[self.iSeqCreateRTransXYPAngScale(dx=dx, dy=dy, da=da, smin=smin,
                                                                                smax=smax, num_steps=50,
                                                                                slow_var=self.slow_var, continuous=True,
                                                                                num_images_used=9000,  # 9000
                                                                                images_base_dir=alldbnormalized_base_dir,
                                                                                normalized_images=alldbnormalized_available_images,
                                                                                first_image_index=55000,
                                                                                pre_mirroring="none",
                                                                                repetition_factor=4,
                                                                                seed=-1)]]  # repetition_factor=4
        # WARNING, code below is for display purposes only, it should be commented out!
        # iSeq_set = iNewidRTransXYPAngScale = [[iSeqCreateRTransXYPAngScale(dx=45, dy=2, da=da, smin=0.85,
        # smax=0.95, num_steps=50, slow_var = "X", continuous=True, num_images_used=2000, #9000
        #                                                      images_base_dir=alldbnormalized_base_dir,
        # normalized_images = alldbnormalized_available_images,
        #                                                      first_image_index=55000, repetition_factor=1, seed=-1)]]
        # iSeq_set = iNewidRTransXYPAngScale = [[iSeqCreateRTransXYPAngScale(dx=45, dy=20, da=da, smin=0.55,
        # smax=1.1, num_steps=79, slow_var = "X", continuous=continuous, num_images_used=79, #30000
        #                                                      images_base_dir=normalized_base_dir_INIBilder,
        # normalized_images = numpy.arange(0, 79),
        #                                                      first_image_index=0, repetition_factor=20, seed=143)]]

        sNewidRTransXYPAngScale = [[self.sSeqCreateRTransXYPAngScale(iSeq_set[0][0], seed=-1)]]

        # This fixes classes of Newid data

        print ("Orig iSeenidRTransXYPAngScale.correct_labels=", iSeenidRTransXYPAngScale.correct_labels)
        print ("Orig len(iSeenidRTransXYPAngScale.correct_labels)=%d" % len(iSeenidRTransXYPAngScale.correct_labels))
        print ("Orig len(iSeenidRTransXYPAngScale.correct_classes)=%d" % len(iSeenidRTransXYPAngScale.correct_classes))

        print ("iNewidRTransXYPAngScale[0][0].correct_labels=", iNewidRTransXYPAngScale[0][0].correct_labels)
        print ("Before correction: iNewidRTransXYPAngScale[0][0].correct_classes=" +
               str(iNewidRTransXYPAngScale[0][0].correct_classes))

        if len(iSeenidRTransXYPAngScale.correct_classes.shape) == 1:
            all_classes = numpy.unique(iSeenidRTransXYPAngScale.correct_classes)
            print ("all_classes=" + str(all_classes))
            avg_labels = more_nodes.compute_average_labels_for_each_class(iSeenidRTransXYPAngScale.correct_classes,
                                                                          iSeenidRTransXYPAngScale.correct_labels)
            print ("avg_labels=" + str(avg_labels))
            iNewidRTransXYPAngScale[0][0].correct_classes = \
                more_nodes.map_labels_to_class_number(all_classes, avg_labels,
                                                      iNewidRTransXYPAngScale[0][0].correct_labels)
        else:
            for j in range(4):
                correct_classes = iSeenidRTransXYPAngScale.correct_classes[:, j].copy()
                correct_classes.sort()
                correct_labels = iSeenidRTransXYPAngScale.correct_labels[:, j].copy()
                correct_labels.sort()
                print ("correct_labels=" + str(correct_labels))
                print ("correct_classes=" + str(correct_classes))
                all_classes = numpy.unique(correct_classes)
                print ("all_classes[:,%d]=" % j + str(all_classes))
                avg_labels = more_nodes.compute_average_labels_for_each_class(correct_classes, correct_labels)
                print ("avg_labels=", avg_labels)
                iNewidRTransXYPAngScale[0][0].correct_classes[:, j] = \
                    more_nodes.map_labels_to_class_number(all_classes, avg_labels,
                                                          iNewidRTransXYPAngScale[0][0].correct_labels[:, j])

        print ("iNewidRTransXYPAngScale[0][0].correct_classes=" +  str(iNewidRTransXYPAngScale[0][0].correct_classes))

        if self.slow_var == "All":
            iSeq = iTrainRTransXYPAngScale[0][0]
            sSeq = sTrainRTransXYPAngScale[0][0]
            sSeq.train_mode = [sSeq.train_mode,
                               ("serial_regression50", iSeq.correct_labels[:, 1], 1.25),  # 1.25, 1.5, 1.75
                               ("serial_regression50", iSeq.correct_labels[:, 2], 1.75),
                               ("serial_regression50", iSeq.correct_labels[:, 3],
                                2.25)]  # Using experimental weights, otherwise they should be 1.0

        self.name = "Function Based Data Creation for RTransXYPAngScale"
        self.network = "linearNetwork4L"  # Default Network, but ignored
        self.iTrain = iTrainRTransXYPAngScale
        self.sTrain = sTrainRTransXYPAngScale

        self.iSeenid = iSeenidRTransXYPAngScale

        self.sSeenid = sSeenidRTransXYPAngScale

        self.iNewid = iNewidRTransXYPAngScale
        self.sNewid = sNewidRTransXYPAngScale

        self.block_size = iTrainRTransXYPAngScale[0][0].block_size
        # if not continuous:
        #    self.train_mode = 'serial' #clustered improves final performance! mixed
        # else:
        #    self.train_mode = "window32" #clustered improves final performance! mixed

    # pipeline_fd:
    # dx0 = 45, 45 Steps,
    # dy0 = 20, 20 Steps,
    # smin0 = 0.55, smax0 = 1.1, 40 Steps
    def iSeqCreateRTransXYPAngScale(self, dx=45, dy=20, da=22.5, smin=0.55, smax=1.1, num_steps=20, slow_var="X",
                                    continuous=False, pre_mirroring="none", num_images_used=10000,
                                    images_base_dir=alldbnormalized_base_dir, normalized_images=None,
                                    first_image_index=0, repetition_factor=1, seed=-1):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)

        if first_image_index + num_images_used > len(normalized_images):
            err = "Images to be used (%d + %d) exceeds the number of available images (%d)" % (
            first_image_index, num_images_used, len(normalized_images))
            raise Exception(err)

        print ("***** Setting Information Parameters for RTransXYPAngScale ******")
        iSeq = system_parameters.ParamsInput()
        iSeq.name = "RTransXYPAngScale " + slow_var + \
                    " (DX%d, DY%d, DA%d, SMin%d, SMax%d) numsteps %d" % (dx, dy, da, smin * 1000,
                                                                         smax * 1000, num_steps)
        iSeq.data_base_dir = images_base_dir
        iSeq.ids = normalized_images[first_image_index:first_image_index + num_images_used]
        iSeq.slow_var = slow_var
        iSeq.num_steps = num_steps
        iSeq.dx = dx
        iSeq.dy = dy
        iSeq.da = da
        iSeq.smin = smin
        iSeq.smax = smax
        iSeq.pre_mirroring = pre_mirroring

        if continuous == True:
            real_num_steps = num_images_used * repetition_factor
        else:
            real_num_steps = num_steps

        if iSeq.pre_mirroring == "duplicate":
            real_num_steps *= 2

            # Here only the unique discrete values for each class are coumputed, these might need to be
            # repeated multiple times
        if slow_var == "X" or slow_var == "All":
            # iSeq.trans = numpy.arange(-1 * pipeline_fd['dx0'], pipeline_fd['dx0'], 2) # (-50, 50, 2)
            iSeq.trans_pangs_scales = numpy.linspace(-1.0 * dx, dx, real_num_steps)  # (-50, 50, 2)
        elif slow_var == "Y":
            iSeq.trans_pangs_scales = numpy.linspace(-1.0 * dy, dy, real_num_steps)
        elif slow_var == "PAng":
            iSeq.trans_pangs_scales = numpy.linspace(-1.0 * da, da, real_num_steps)
        elif slow_var == "Scale":
            iSeq.trans_pangs_scales = numpy.linspace(smin, smax, real_num_steps)
        else:
            er = "Wrong slow_variable: " + str(slow_var)
            raise Exception(er)

        if len(iSeq.ids) % len(iSeq.trans_pangs_scales) != 0 and not self.continuous:
            ex = "Here the number of translations/scalings must be a divisor of the number of identities"
            raise Exception(ex)

        iSeq.ages = [None]
        iSeq.genders = [None]
        iSeq.racetweens = [None]
        iSeq.expressions = [None]
        iSeq.morphs = [None]
        iSeq.poses = [None]
        iSeq.lightings = [None]
        iSeq.slow_signal = 0  # real slow signal is the translation in the x axis (correlated to identity),
        # added during image loading
        iSeq.step = 1
        iSeq.offset = 0
        iSeq.input_files = image_loader.create_image_filenames3(iSeq.data_base_dir, "image", iSeq.slow_signal, iSeq.ids,
                                                                iSeq.ages,
                                                                iSeq.genders, iSeq.racetweens, iSeq.expressions,
                                                                iSeq.morphs,
                                                                iSeq.poses, iSeq.lightings, iSeq.step, iSeq.offset,
                                                                len_ids=5, image_postfix=".jpg")
        ##WARNING! (comment this!)
        #    dir_list = os.listdir(iSeq.data_base_dir)
        #    iSeq.input_files = []
        #    for filename in dir_list:
        #        iSeq.input_files.append( os.path.join(iSeq.data_base_dir,filename) )

        print ("number of input files=%d" % len(iSeq.input_files))
        print ("number of iSeq.ids=%d" % len(iSeq.ids))

        iSeq.input_files = iSeq.input_files * repetition_factor  # warning!!! 4, 8
        iSeq.num_images = len(iSeq.input_files)

        print ("iSeq.num_images = %d" % iSeq.num_images)

        # To avoid grouping similar images next to one other, even though available images already shuffled
        numpy.random.shuffle(iSeq.input_files)

        if iSeq.pre_mirroring == "none":
            iSeq.pre_mirror_flags = [False] * iSeq.num_images
        elif iSeq.pre_mirroring == "all":
            iSeq.pre_mirror_flags = [True] * iSeq.num_images
        elif iSeq.pre_mirroring == "random":
            iSeq.pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images)
        elif iSeq.pre_mirroring == "duplicate":
            input_files_duplicated = list(iSeq.input_files)
            iSeq.pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images)

            shuffling = numpy.arange(0, iSeq.num_images)
            numpy.random.shuffle(shuffling)

            input_files_duplicated = [input_files_duplicated[i] for i in shuffling]
            pre_mirror_flags_duplicated = iSeq.pre_mirror_flags[shuffling] ^ True

            iSeq.input_files.extend(input_files_duplicated)
            iSeq.pre_mirror_flags = numpy.concatenate((iSeq.pre_mirror_flags, pre_mirror_flags_duplicated))
            iSeq.num_images *= 2
        else:
            er = "Erroneous parameter iSeq.pre_mirroring="+str(iSeq.pre_mirroring)
            raise Exception(er)

        # iSeq.params = [ids, expressions, morphs, poses, lightings]
        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions,
                       iSeq.morphs, iSeq.poses, iSeq.lightings]

        iSeq.block_size = iSeq.num_images // num_steps

        if not self.continuous:
            print ("before len(iSeq.trans_pangs_scales)= %d" % len(iSeq.trans_pangs_scales))
            iSeq.trans_pangs_scales = sfa_libs.wider_1Darray(iSeq.trans_pangs_scales, iSeq.block_size)
            print ("after len(iSeq.trans_pangs_scales)= %d" % len(iSeq.trans_pangs_scales))

            #    if not continuous:
            #        iSeq.train_mode = "serial" # = "serial" "mixed", None
            #    else:
            #        iSeq.train_mode = "mirror_window64" # "mirror_window32" "smirror_window32" # None, "regular",
            # "window32", "fwindow16", "fwindow32", "fwindow64", "fwindow128",

            #        quit()
        iSeq.train_mode = "serial"  # "mirror_window256" "mirror_window32", "regular", "fwindow16", "serial",
        # "mixed", None
        #        iSeq.train_mode = None

        print ("BLOCK SIZE =" +  str(iSeq.block_size))
        iSeq.correct_classes = numpy.arange(num_steps * iSeq.block_size) // iSeq.block_size
        #    sfa_libs.wider_1Darray(numpy.arange(len(iSeq.trans)), iSeq.block_size)
        #    if not continuous:
        #        iSeq.correct_labels = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
        #    else:
        iSeq.correct_labels = iSeq.trans_pangs_scales + 0.0
        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateRTransXYPAngScale(self, iSeq, seed=-1):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)

        print ("******** Setting Training Data Parameters for RTransXYPAngScale  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.include_latest = iSeq.include_latest
        sSeq.pre_mirror_flags = iSeq.pre_mirror_flags

        sSeq.image_width = 256
        sSeq.image_height = 192
        sSeq.subimage_width = 128
        sSeq.subimage_height = 128

        sSeq.trans_x_max = iSeq.dx
        sSeq.trans_x_min = -1 * iSeq.dx

        sSeq.trans_y_max = iSeq.dy
        sSeq.trans_y_min = -1 * iSeq.dy

        sSeq.da_max = iSeq.da
        sSeq.da_min = -1 * iSeq.da

        # iSeq.scales = numpy.linspace(0.5, 1.30, 16) # (-50, 50, 2)
        sSeq.min_sampling = iSeq.smin
        sSeq.max_sampling = iSeq.smax

        # sSeq.subimage_pixelsampling = 2
        # sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+
        # 5*sSeq.pixelsampling_x
        sSeq.add_noise_L0 = True
        sSeq.convert_format = "L"
        sSeq.background_type = None
        # random translation for th w coordinate
        # sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
        # sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
        if iSeq.slow_var == "X" or iSeq.slow_var == "All":
            sSeq.translations_x = iSeq.trans_pangs_scales
            # sSeq.translations_x = sfa_libs.wider_1Darray(iSeq.trans, iSeq.block_size)
            # print "sSeq.translations_x=", sSeq.translations_x
            # print "len( sSeq.translations_x)=",  len(sSeq.translations_x)
        elif sSeq.trans_x_max - sSeq.trans_x_min > 50 / 2:
            sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)
        else:
            sSeq.translations_x = numpy.random.uniform(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images)

        if iSeq.slow_var == "Y":
            sSeq.translations_y = iSeq.trans_pangs_scales
        elif sSeq.trans_y_max - sSeq.trans_y_min > 50 / 2:
            sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)
        else:
            sSeq.translations_y = numpy.random.uniform(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images)

        if iSeq.slow_var == "PAng":
            sSeq.rotations = iSeq.trans_pangs_scales
        else:
            sSeq.rotations = numpy.random.uniform(sSeq.da_min, sSeq.da_max, sSeq.num_images)

        if iSeq.slow_var == "Scale":
            sSeq.pixelsampling_x = iSeq.trans_pangs_scales  # sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
            sSeq.pixelsampling_y = iSeq.trans_pangs_scales  # sfa_libs.wider_1Darray(iSeq.scales,  iSeq.block_size)
        else:
            sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling,
                                                        size=sSeq.num_images)
            sSeq.pixelsampling_y = sSeq.pixelsampling_x + 0.0

        if iSeq.slow_var == "All":
            all_classes = (numpy.arange(sSeq.num_images) * 50.0 / sSeq.num_images).astype(int)
            iSeq.correct_labels = numpy.stack(
                (sSeq.translations_x, sSeq.translations_y, sSeq.rotations, sSeq.pixelsampling_x)).T

            translations_y_ordering = numpy.argsort(sSeq.translations_y)
            translations_y_classes = numpy.zeros(sSeq.num_images, dtype=int)
            translations_y_classes[translations_y_ordering] = all_classes

            rotation_ordering = numpy.argsort(sSeq.rotations)
            rotation_classes = numpy.zeros(sSeq.num_images, dtype=int)
            rotation_classes[rotation_ordering] = all_classes

            pixel_sampling_ordering = numpy.argsort(sSeq.pixelsampling_x)
            pixel_sampling_classes = numpy.zeros(sSeq.num_images, dtype=int)
            pixel_sampling_classes[pixel_sampling_ordering] = all_classes

            iSeq.correct_classes = numpy.stack(
                (iSeq.correct_classes, translations_y_classes, rotation_classes, pixel_sampling_classes)).T
            print ("iSeq.correct_labels=" + str(iSeq.correct_labels))
            print ("iSeq.correct_classes=" + str(iSeq.correct_classes))

        # Warning, code below seems to have been deleted at some point!!!
        sSeq.subimage_first_row = sSeq.image_height / 2.0 - sSeq.subimage_height * sSeq.pixelsampling_y / 2.0
        sSeq.subimage_first_column = sSeq.image_width / 2.0 - sSeq.subimage_width * sSeq.pixelsampling_x / 2.0

        sSeq.contrast_enhance = "array_mean_std-137.5-45.0"  # None #"array_mean_std-127.5-45.0"
        sSeq.trans_sampled = True  # TODO: Why are translations specified according to the sampled images?
        sSeq.rotate_before_translation = True

        sSeq.name = "RTansXYPAngScale %s Dx in (%d, %d) Dy in (%d, %d), sampling in (%d, %d)" % (
            iSeq.slow_var, sSeq.trans_x_min,
            sSeq.trans_x_max, sSeq.trans_y_min, sSeq.trans_y_max, int(sSeq.min_sampling * 100),
            int(sSeq.max_sampling * 100))

        print ("Var in trans X is: %g" % sSeq.translations_x.var())
        sSeq.load_data = load_data_from_sSeq
        system_parameters.test_object_contents(sSeq)
        return sSeq


pipeline_fd_1Label = dict(dx0=40.0, dy0=20.0, da0=22.5, smin0=0.55, smax0=1.1,
                          dx1=20.0, dy1=10.0, da1=11.3, smin1=0.775, smax1=1.05)

pipeline_fd_4Labels = dict(dx0=40.0, dy0=20.0, da0=22.5, smin0=0.694, smax0=0.981,
                           dx1=14.0, dy1=13.0, da1=21.3, smin1=0.694, smax1=0.981,
                           # why am I not reducing the scale range???!!!!!
                           dx2=7.5, dy2=8.0, da2=16.5, smin2=0.694,
                           smax2=0.981)  # ERROR, THIS IS A BUG

#####For 4 lables, try next options for iter 1
#####13, 13.5, 18, 0.73, 1.075,
#####After iter 0 the following RMSE are provided: X:3.78864, Y:3.66665, PAng:6.38798, Scale:0.08301
#####3.5 times=                                      (13.26024, 12.833275, 22.35793, 0.290535)
#####After iter 1 the following RMSE are provided: X:2.06688, Y:2.18546, PAng:4.65952, Scale:0.04956
#####3.5 times=                                      (7.23408, 7.64911, 16.30832, 0.17346)
#####After iter 2 the following RMSE are provided: X:
#####3.5 times=

# continuous = True  # and False
# slow_var = "All"  # "All", "X", "PAng", "Y", "Scale"
# iteration = 0  # 0, 1, 2 (only 4Labels)
ParamsRTransXYPAngScaleFunc_128x128 = ParamsRTransXYPAngScaleExperiment(experiment_seed, experiment_basedir,
                                                                        continuous=True,
                                                                        slow_var="All", iteration=0,
                                                                        pipeline_fd_1Label=pipeline_fd_1Label,
                                                                        pipeline_fd_4Labels=pipeline_fd_4Labels)

ParamsRTransXYPAngScaleFunc_64x64 = copy.deepcopy(ParamsRTransXYPAngScaleFunc_128x128)
ParamsRTransXYPAngScaleFunc_64x64.reduction_factor = 2.0
ParamsRTransXYPAngScaleFunc_64x64.hack_image_size = 64

ParamsRTransXYPAngScaleFunc_32x32 = copy.deepcopy(ParamsRTransXYPAngScaleFunc_128x128)
ParamsRTransXYPAngScaleFunc_32x32.reduction_factor = 4.0
ParamsRTransXYPAngScaleFunc_32x32.hack_image_size = 32


# This has not been loaded yet, wait until create() is called
# print ParamsRTransXYPAngScaleFunc_128x128.sTrain[0][0].num_images
# print ParamsRTransXYPAngScaleFunc_128x128.sSeenid.num_images
# print ParamsRTransXYPAngScaleFunc_128x128.sNewid[0][0].num_images
# print len(ParamsRTransXYPAngScaleFunc_128x128.sTrain[0][0].input_files)
# print len(ParamsRTransXYPAngScaleFunc_128x128.sNewid[0][0].input_files)


######################################################################################################
# ################          AGE Extraction Experiments                      ##########################
######################################################################################################
class ParamsRAgeExperiment(system_parameters.ParamsSystem):
    """ Experiment for the estimation of age from normalized faces in image patches (real photographs).
    This experiment also supports gender and 'race' estimation.
    """
    def __init__(self, experiment_seed, experiment_basedir, subimage_size=96,
                 use_setup_Guo=True, rep_Guo_train=22, rep_Guo_supervised=3,
                 training_set=1, pre_mirroring_supervised_test=False):
        super(ParamsRAgeExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.experiment_basedir = experiment_basedir
        self.age_use_RGB_images = False  # LRec_use_RGB_images

        self.train_mode = "Weird Mode"  # Ignored for the moment
        self.analysis = None
        self.enable_reduced_image_sizes = True
        self.reduction_factor = 160.0 / subimage_size  # 160.0/72 ## 160.0/96 # (article 2.0)
        self.hack_image_size = subimage_size  # 72 ## 96 # (article 80)
        self.enable_hack_image_size = True
        self.patch_network_for_RGB = False  #
        self.use_setup_Guo = use_setup_Guo
        self.rep_Guo_train = rep_Guo_train
        self.rep_Guo_supervised = rep_Guo_supervised
        self.pre_mirroring_supervised_test = pre_mirroring_supervised_test
        if training_set == 1 or training_set == 2:
            self.training_set = training_set
        else:
            raise Exception("training_set should be either 1 or 2")

    def create(self):
        print ("Age estimation. experiment_seed=%d" % self.experiment_seed)
        numpy.random.seed(self.experiment_seed)
        print ("experiment_seed=%d" % self.experiment_seed)
        print ("use_setup_Guo=%d" % self.use_setup_Guo)
        print ("training_set=" +  str(self.training_set))
        # TODO: Repeat computation of chance levels for MORPH, FG_Net and MORPH+FGNet. (orig labels, no rep, all images)
        # min_cluster_size_MORPH = 60000 # 1400 # 60000
        # max_cluster_size_MORPH = None  # 1400 # None
        # age_trim_number_MORPH = 1400 # 1400
        leave_k_out_MORPH = 2000  # 2000 #1000 #0 #Number of subjects left out for testing
        if leave_k_out_MORPH == 0:
            select_k_images_newid = 8000
            # select_k_images_seenid = 1500 #16000 #4000 #3000 #3750

            # 1=CNN setup, 0=my setup
        if self.use_setup_Guo:
            leave_k_out_MORPH = 0  # to speed on unnecessary computation
            select_k_images_newid = 1000
            # select_k_images_seenid = 1000 #4000 #3000 #3750

        if self.age_use_RGB_images:
            age_eyes_normalized_base_dir_MORPH = "/local/escalafl/Alberto/MORPH_normalizedEyesZ4_horiz_RGB_ByAge"
            # RGB: "/local/escalafl/Alberto/MORPH_normalizedEyesZ3_horiz_RGB_ByAge"
        else:
            age_eyes_normalized_base_dir_MORPH = "/local/escalafl/Alberto/MORPH_normalizedEyesZ4_horiz_ByAge"
        age_files_list_MORPH = self.list_available_images(age_eyes_normalized_base_dir_MORPH, from_subdirs=None)
        # change from_subdirs to select a subset of all ages!

        if leave_k_out_MORPH:
            print ("LKO enabled, k=%d" % leave_k_out_MORPH)
            age_files_list_MORPH, age_files_list_MORPH_out = self.MORPH_leave_k_identities_out_list(
                age_files_list_MORPH, k=leave_k_out_MORPH)
        # #     if leave_k_out_MORPH == 1000:
        # #         age_trim_number_MORPH = 1270 #age_trim_number_MORPH = 1270. This might be important to
        # preserve the number of clusters
        # #     else:
        # #         age_trim_number_MORPH = 1160
        # #     #print "age_files_dict_MORPH_out=", age_files_dict_MORPH_out
        else:
            age_files_list_MORPH, age_files_list_MORPH_out = self.MORPH_select_k_images_list(age_files_list_MORPH,
                                                                                             k=select_k_images_newid)

        select_k_images_seenid = len(age_files_list_MORPH)
        age_files_list_MORPH, age_files_list_MORPH_seenid = self.MORPH_select_k_images_list(age_files_list_MORPH,
                                                                                            k=select_k_images_seenid,
                                                                                            replacement=True)

        # counter = 0
        # for subdir in age_files_dict_MORPH.keys():
        #    counter += age_files_dict_MORPH[subdir][0]
        # print "age_files_dict_MORPH contains %d images"%counter
        # counter = 0
        # for subdir in age_files_dict_MORPH_seenid.keys():
        #    counter += age_files_dict_MORPH_seenid[subdir][0]
        # print "age_files_dict_MORPH_seenid contains %d images"%counter
        # counter = 0
        # for subdir in age_files_dict_MORPH_out.keys():
        #    counter += age_files_dict_MORPH_out[subdir][0]
        # print "age_files_dict_MORPH_out (LKO) contains %d images"%counter

        age_all_labels_map_MORPH = load_GT_labels( #/MORPH_setsS1S2S3_seed12345/
            "/local/escalafl/Alberto/GT_MORPH_AgeRAgeGenderRace.txt", age_included=True,
            rAge_included=True, gender_included=True, race_included=True, avgColor_included=False)
        age_all_labels_map_INI = load_GT_labels("/home/escalafl/Databases/faces/INIBilder/GT_INI_AgeRAgeGenderRace.txt",
                                                age_included=True, rAge_included=True, gender_included=True,
                                                race_included=True, avgColor_included=False)
        # "/local/escalafl/Alberto/GT_ETIT_AgeRAgeGenderRace.txt"
        # "/home/escalafl/workspace4/FaceDetectSFA/FGNet_FD/GT_FGNet_AgeRAgeGenderRace.txt"
        # "/local/escalafl/Alberto/GT_INIProf_AgeRAgeGenderRace.txt"
        age_all_labels_map_FGNet = load_GT_labels(
            "/home/escalafl/workspace4/FaceDetectSFA/FGNet_FD/GT_FGNet_AgeRAgeGenderRace.txt", age_included=True,
            rAge_included=True, gender_included=True, race_included=True, avgColor_included=False)
        # print age_all_labels_map_FGNet
        # quit()

        age_labeled_files_list_MORPH = self.append_GT_labels_to_files(age_files_list_MORPH, age_all_labels_map_MORPH,
                                                                      select_races=[-2, -1, 0, 1, 2], verbose=True)
        age_labeled_files_list_MORPH_seenid = self.append_GT_labels_to_files(age_files_list_MORPH_seenid,
                                                                             age_all_labels_map_MORPH,
                                                                             select_races=[-2, -1, 0, 1, 2],
                                                                             verbose=True)
        # age_labeled_files_list_MORPH_seenid = age_labeled_files_list_MORPH #WARNING!!!!!
        age_labeled_files_list_MORPH_out = self.append_GT_labels_to_files(age_files_list_MORPH_out,
                                                                          age_all_labels_map_MORPH,
                                                                          select_races=[-2, -1, 0, 1, 2], verbose=True)

        print ("age_labeled_files_list_MORPH contains %d images" % len(age_labeled_files_list_MORPH))
        print ("age_labeled_files_list_MORPH_seenid contains %d images" % len(age_labeled_files_list_MORPH_seenid))
        print ("age_labeled_files_list_MORPH_out contains %d images" % len(age_labeled_files_list_MORPH_out))

        print ("age_labeled_files_list_MORPH_seenid[0,1,2]=",  age_labeled_files_list_MORPH_seenid[0],
               age_labeled_files_list_MORPH_seenid[1], age_labeled_files_list_MORPH_seenid[2])

        # # if self.use_setup_Guo: #just to speed this up
        # #     pre_repetitions=1
        # #     age_trim_number_MORPH = (pre_repetitions*1075)
        # # else:
        # #     pre_repetitions = 4 #4
        # #     age_trim_number_MORPH = (pre_repetitions*1075)
        # #     age_trim_number_MORPH = (pre_repetitions*1250) #When replacement=True in seenid selection
        # from training and L1K0
        # #     #age_trim_number_MORPH = (pre_repetitions*1180) #When replacement=True in seenid selection
        # from training and L2K0
        # #
        # pre_repetitions = 1
        # age_trim_number_MORPH = (pre_repetitions*1250) #When replacement=True in seenid selection from training

        num_clusters_MORPH_serial = 32
        # age_trim_number_MORPH = None
        # print "age_labeled_files_list_INIBilder", age_labeled_files_list_INIBilder
        # age_trim_number_MORPH = 200
        # TODO: repetition=3 or 5, changed it only for a fast exercise

        age_clusters_MORPH = self.age_cluster_labeled_files(age_labeled_files_list_MORPH, repetition=1,
                                                            num_clusters=num_clusters_MORPH_serial, trim_number=None,
                                                            shuffle_each_cluster=False)  # r=5, r=6, trim_number=None
        # age_clusters_MORPH = age_cluster_list(age_files_dict_MORPH, repetition=pre_repetitions,
        # smallest_number_images=age_trim_number_MORPH, largest_number_images=age_trim_number_MORPH)
        # #Cluster so that all clusters have size at least 1400 or 1270 for L1KPO
        print ("len(age_clusters_MORPH)= %d" % len(age_clusters_MORPH))
        num_images_per_cluster_used_MORPH = age_clusters_MORPH[0][0]
        # len(age_labeled_files_list_MORPH) / num_clusters_MORPH_serial

        print ("num_images_per_cluster_used_MORPH=%d" % num_images_per_cluster_used_MORPH)
        # age_clusters_MORPH = cluster_to_labeled_samples(age_clusters_MORPH, trim_number=age_trim_number_MORPH)

        age_clusters_MORPH_seenid = self.age_cluster_labeled_files(age_labeled_files_list_MORPH_seenid, repetition=1,
                                                                   num_clusters=1, trim_number=None,
                                                                   shuffle_each_cluster=False)  # trim_number=None
        # age_clusters_MORPH_seenid = age_cluster_list(age_files_dict_MORPH_seenid, repetition=1,
        # smallest_number_images=800000, largest_number_images=None) #A single cluster for all images,
        # age_clusters_MORPH_seenid = cluster_to_labeled_samples(age_clusters_MORPH_seenid,
        # trim_number=None, shuffle_each_cluster=False)

        age_clusters_MORPH_out = self.age_cluster_labeled_files(age_labeled_files_list_MORPH_out, repetition=1,
                                                                num_clusters=1, trim_number=None,
                                                                shuffle_each_cluster=False)  # trim_number=None
        # age_clusters_MORPH_out = age_cluster_list(age_files_dict_MORPH_out, smallest_number_images=800000)
        # #A single cluster for all images
        # age_clusters_MORPH_out = cluster_to_labeled_samples(age_clusters_MORPH_out, trim_number=None,
        # shuffle_each_cluster=False)

        if len(age_clusters_MORPH_seenid) > 0:
            num_images_per_cluster_used_MORPH_seenid = age_clusters_MORPH_seenid[0][0]
        else:
            print ("???")
            quit()
            num_images_per_cluster_used_MORPH_seenid = 0
        print ("num_images_per_cluster_used_MORPH_seenid=%d" % num_images_per_cluster_used_MORPH_seenid)

        if len(age_clusters_MORPH_out) > 0:
            num_images_per_cluster_used_MORPH_out = age_clusters_MORPH_out[0][0]
        else:
            num_images_per_cluster_used_MORPH_out = 0
        print ("num_images_per_cluster_used_MORPH_out=%d" % num_images_per_cluster_used_MORPH_out)

        numpy.random.seed(self.experiment_seed + 123123)
        # min_cluster_size_FGNet = 5000 # 32 #5000
        # max_cluster_size_FGNet = None # 32 #None
        # TODO: use externally provided directory
        if self.age_use_RGB_images:
            age_eyes_normalized_base_dir_FGNet = \
                "/local/escalafl/Alberto/FGNet/FGNet_normalizedEyesZ4_horiz_RGB_2015_08_25"
        else:
            age_eyes_normalized_base_dir_FGNet = \
                "/local/escalafl/Alberto/FGNet/FGNet_normalizedEyesZ4_horiz_2015_08_25"
            # age_eyes_normalized_base_dir_FGNet = "/local/escalafl/Alberto/ETIT_normalizedEyesZ4_h"
            # #WARNING, TEMPORAL EXPERIMENT ONLY
            # age_eyes_normalized_base_dir_FGNet = "/local/escalafl/Alberto/INIProf_normalizedEyesZ4_h"
        # subdirs_FGNet=None
        subdirs_FGNet = ["%d" % i for i in range(16, 77)]  # 70 77 #OBSOLETE: all images are in a single directory

        age_files_list_FGNet = self.list_available_images(age_eyes_normalized_base_dir_FGNet, from_subdirs=None,
                                                          verbose=False)
        # age_files_dict_FGNet = find_available_images(age_eyes_normalized_base_dir_FGNet, from_subdirs=subdirs_FGNet)
        # #change from_subdirs to select a subset of all ages!

        age_labeled_files_list_FGNet = self.append_GT_labels_to_files(age_files_list_FGNet, age_all_labels_map_FGNet,
                                                                      min_age=15.99 * DAYS_IN_A_YEAR,
                                                                      max_age=77.01 * DAYS_IN_A_YEAR, verbose=True)
        # age_clusters_FGNet = age_cluster_list(age_files_dict_FGNet, smallest_number_images=min_cluster_size_FGNet)
        # #Cluster so that all clusters have size at least 1400
        # print "******************"
        # print len(age_clusters_FGNet), age_clusters_FGNet[0], ":)"
        # print "******************"

        age_clusters_FGNet = self.age_cluster_labeled_files(age_labeled_files_list_FGNet, repetition=1, num_clusters=1,
                                                            trim_number=None, shuffle_each_cluster=False)
        # age_clusters_FGNet = cluster_to_labeled_samples(age_clusters_FGNet, trim_number=max_cluster_size_FGNet)

        if len(age_clusters_FGNet) > 0:
            num_images_per_cluster_used_FGNet = age_clusters_FGNet[0][0]
        else:
            num_images_per_cluster_used_FGNet = 0
        # print "******************"
        # print len(age_clusters_FGNet), age_clusters_FGNet[0], ":)"
        # print "******************"
        print ("num_images_per_cluster_used_FGNet=%d" % num_images_per_cluster_used_FGNet)
        # quit()



        numpy.random.seed(self.experiment_seed + 432432)
        # age_trim_number_INIBilder = None
        if self.age_use_RGB_images:
            age_eyes_normalized_base_dir_INIBilder = \
                "/local/escalafl/Alberto/INIBilder/INIBilder_normalizedEyesZ4_horiz_RGB"
        else:
            age_eyes_normalized_base_dir_INIBilder = \
                "/local/escalafl/Alberto/INIBilder/INIBilder_normalizedEyesZ4_horiz_byAge"
        # age_files_dict_INIBilder = find_available_images(age_eyes_normalized_base_dir_INIBilder, from_subdirs=None)
        # #change from_subdirs to select a subset of all ages!
        # age_clusters_INIBilder = age_cluster_list(age_files_dict_INIBilder,
        # smallest_number_images=age_trim_number_INIBilder) #Cluster so that all clusters have size at least 1400
        # age_clusters_INIBilder = cluster_to_labeled_samples(age_clusters_INIBilder,
        # trim_number=age_trim_number_INIBilder, shuffle_each_cluster=False)
        age_files_list_INIBilder = self.list_available_images(age_eyes_normalized_base_dir_INIBilder, from_subdirs=None,
                                                              verbose=False)
        # print "age_files_list_INIBilder=",age_files_list_INIBilder
        age_labeled_files_list_INIBilder = self.append_GT_labels_to_files(age_files_list_INIBilder,
                                                                          age_all_labels_map_INI)
        # print "age_labeled_files_list_INIBilder", age_labeled_files_list_INIBilder
        age_clusters_INIBilder = self.age_cluster_labeled_files(age_labeled_files_list_INIBilder, repetition=1,
                                                                num_clusters=1, trim_number=None,
                                                                shuffle_each_cluster=False)

        if len(age_clusters_INIBilder) > 0:
            num_images_per_cluster_used_INIBilder = age_clusters_INIBilder[0][0]
        else:
            num_images_per_cluster_used_INIBilder = 0
        print ("num_images_per_cluster_used_INIBilder= %d" % num_images_per_cluster_used_INIBilder)
        # quit()

        numpy.random.seed(self.experiment_seed + 654654)
        age_trim_number_MORPH_FGNet = 1400
        if self.age_use_RGB_images:
            age_eyes_normalized_base_dir_MORPH_FGNet = \
                "/local/escalafl/Alberto/MORPH_FGNet_normalizedEyesZ3_horiz_ByAge"
        else:
            age_eyes_normalized_base_dir_MORPH_FGNet = "/local/escalafl/Alberto/MORPH_FGNet_normalizedEyesZ3_horiz"
        age_files_dict_MORPH_FGNet = self.find_available_images(age_eyes_normalized_base_dir_MORPH_FGNet,
                                                                from_subdirs=None)  # change from_subdirs to select
        # a subset of all ages!
        age_clusters_MORPH_FGNet = self.age_cluster_list(age_files_dict_MORPH_FGNet,
                                                         smallest_number_images=age_trim_number_MORPH_FGNet)
        # Cluster so that all clusters have size at least 1400
        age_clusters_MORPH_FGNet = self.cluster_to_labeled_samples(age_clusters_MORPH_FGNet,
                                                                   trim_number=age_trim_number_MORPH_FGNet)
        if len(age_clusters_MORPH_FGNet) > 0:
            num_images_per_cluster_used_MORPH_FGNet = age_clusters_MORPH_FGNet[0][0]
        else:
            num_images_per_cluster_used_MORPH_FGNet = 0
        print ("num_images_per_cluster_used_MORPH_FGNet= %d" % num_images_per_cluster_used_MORPH_FGNet)

        # ############################################### CRUCIAL FOR MCNN TESTING ################
        # age_all_labels_map = load_GT_labels("/local/escalafl/Alberto/MORPH_setsS1S2S3_seed12345/
        # GT_MORPH_AgeRAgeGenderRace.txt", age_included=True, rAge_included=True, gender_included=True,
        # race_included=True, avgColor_included=False)

        numpy.random.seed(self.experiment_seed + 432432)
        # #pre_repetitions = 2 #16 # 24 #18 # ** 16 #2, 4, 4*4
        # #age_trim_number_set1 = (pre_repetitions*250)

        MORPH_base_dir = "/local/escalafl/Alberto/MORPH_splitted_GUO_2015_09_02/"
        # MORPH_base_dir = "/local/escalafl/Alberto/MORPH_setsS1S2S3_seed12345/"   #WARNING!!!
        if self.age_use_RGB_images:
            age_eyes_normalized_base_dir_set1 = "/local/escalafl/Alberto/MORPH_splitted_GUO_2015_09_02/nonexistent"
        else:
            age_eyes_normalized_base_dir_set1 = MORPH_base_dir + "Fs" + str(
                self.training_set)  # F s 1 ot F s 2 #WARNING!
        # age_files_dict_set1 = find_available_images(age_eyes_normalized_base_dir_set1, from_subdirs=None)
        # #change from_subdirs to select a subset of all ages!
        age_files_list_set1 = self.list_available_images(age_eyes_normalized_base_dir_set1, from_subdirs=None,
                                                         verbose=False)
        age_labeled_files_list_set1 = self.append_GT_labels_to_files(age_files_list_set1, age_all_labels_map_MORPH)
        age_clusters_set1 = self.age_cluster_labeled_files(age_labeled_files_list_set1, repetition=self.rep_Guo_train,
                                                           num_clusters=32, trim_number=None,
                                                           shuffle_each_cluster=False)  # r=22
        # WARNING, should be: repetition=22
        # age_clusters_set1 = age_cluster_labeled_files(age_labeled_files_list_set1, repetition=16,
        # num_clusters=33, trim_number=None, shuffle_each_cluster=False)

        # age_clusters_set1 = age_cluster_list(age_files_dict_set1, repetition=pre_repetitions,
        # smallest_number_images=age_trim_number_set1, largest_number_images=age_trim_number_set1)
        # #Cluster so that all clusters have size at least 1400
        # age_clusters_set1 = cluster_to_labeled_samples(age_clusters_set1, trim_number=age_trim_number_set1,
        # shuffle_each_cluster=True)
        if len(age_clusters_set1) > 0:
            num_images_per_cluster_used_set1 = age_clusters_set1[0][0]
        else:
            num_images_per_cluster_used_set1 = 0
        print ("num_images_per_cluster_used_set1= %d" % num_images_per_cluster_used_set1)
        print ("num clusters: %d" % len(age_clusters_set1))
        # print "len(age_files_dict_set1)=", len(age_files_dict_set1)

        if self.age_use_RGB_images:
            age_eyes_normalized_base_dir_set1b = \
                "/local/escalafl/Alberto/MORPH_setsS1S2S3_seed12345/nonexistent/s1_byAge_mcnn"
        else:
            age_eyes_normalized_base_dir_set1b = MORPH_base_dir + "Fs" + str(self.training_set)
        # age_files_dict_set1b = find_available_images(age_eyes_normalized_base_dir_set1b, from_subdirs=None)
        # #change from_subdirs to select a subset of all ages!
        age_files_list_set1b = self.list_available_images(age_eyes_normalized_base_dir_set1b, from_subdirs=None,
                                                          verbose=False)
        age_labeled_files_list_set1b = self.append_GT_labels_to_files(age_files_list_set1b, age_all_labels_map_MORPH,
                                                                      select_races=[-2, -1, 0, 1, 2], verbose=True)  # [-2, 2] for only black and white (redundant)
        age_clusters_set1b = self.age_cluster_labeled_files(age_labeled_files_list_set1b,
                                                            repetition=self.rep_Guo_supervised, num_clusters=1,
                                                            trim_number=None, shuffle_each_cluster=False)  # r=3
        # WARNING, should be: repetition=3
        # age_clusters_set1b = age_cluster_labeled_files(age_labeled_files_list_set1b, repetition=2, num_clusters=1,
        # trim_number=None, shuffle_each_cluster=False)


        # print "len(age_files_dict_set1b)=", len(age_files_dict_set1b)
        # pre_repetitions = 2 # 2
        # age_clusters_set1b = age_cluster_list(age_files_dict_set1b, repetition=pre_repetitions,
        # smallest_number_images=800000, largest_number_images=None)
        # #Cluster so that all clusters have size at least 1400
        # age_clusters_set1b = cluster_to_labeled_samples(age_clusters_set1b, trim_number=None,
        # shuffle_each_cluster=True)
        if len(age_clusters_set1b) > 0:
            num_images_per_cluster_used_set1b = age_clusters_set1b[0][0]
        else:
            num_images_per_cluster_used_set1b = 0
        print ("num_images_per_cluster_used_set1b=%d" % num_images_per_cluster_used_set1b)

        # pre_repetitions = 1
        if self.age_use_RGB_images:
            age_eyes_normalized_base_dir_set1test = \
                "/local/escalafl/Alberto/MORPH_setsS1S2S3_seed12345/s1-test_byAge_mcnn/nonexistent"
        else:
            age_eyes_normalized_base_dir_set1test = MORPH_base_dir + "Fs" + str(
                self.training_set) + "-test"  # s2-test_byAge_mcnn2
            # age_files_dict_set1test = find_available_images(age_eyes_normalized_base_dir_set1test,
            # from_subdirs=None) #change from_subdirs to select a subset of all ages!
        age_files_list_set1test = self.list_available_images(age_eyes_normalized_base_dir_set1test, from_subdirs=None,
                                                             verbose=False)
        age_labeled_files_list_set1test = self.append_GT_labels_to_files(age_files_list_set1test,
                                                                         age_all_labels_map_MORPH,
                                                                         select_races=[-2, -1, 0, 1, 2], verbose=True) # [-2, 2] for only black and white
        age_clusters_set1test = self.age_cluster_labeled_files(age_labeled_files_list_set1test, repetition=1,
                                                               num_clusters=1, trim_number=None,
                                                               shuffle_each_cluster=False)

        # age_clusters_set1test = age_cluster_list(age_files_dict_set1test, repetition=1, smallest_number_images=800000,
        # largest_number_images=None) #Cluster so that all clusters have size at least 1400
        # age_clusters_set1test = cluster_to_labeled_samples(age_clusters_set1test, trim_number=None,
        # shuffle_each_cluster=False)
        if len(age_clusters_set1test) > 0:
            num_images_per_cluster_used_set1test = age_clusters_set1test[0][0]
        else:
            num_images_per_cluster_used_set1test = 0
        print ("num_images_per_cluster_used_set1test=%d" % num_images_per_cluster_used_set1test)

        numpy.random.seed(self.experiment_seed + 987987)
        if not self.use_setup_Guo:  # MY MORPH SETUP
            age_clusters = age_clusters_MORPH  # _MORPH #_FGNet
            age_clusters_seenid = age_clusters_MORPH_seenid
            age_clusters_newid = age_clusters_MORPH_out

            age_eyes_normalized_base_dir_train = age_eyes_normalized_base_dir_MORPH
            age_eyes_normalized_base_dir_seenid = age_eyes_normalized_base_dir_MORPH
            age_eyes_normalized_base_dir_newid = age_eyes_normalized_base_dir_MORPH
        else:  # Guo  SETUP
            age_clusters = age_clusters_set1  # _MORPH #_FGNet
            age_clusters_seenid = age_clusters_set1b
            age_clusters_newid = age_clusters_set1test

            #    age_trim_number_MORPH = num_images_per_cluster_used_set1 #age_trim_number_set1
            num_images_per_cluster_used_MORPH = num_images_per_cluster_used_set1
            num_images_per_cluster_used_MORPH_seenid = num_images_per_cluster_used_set1b
            num_images_per_cluster_used_MORPH_out = num_images_per_cluster_used_set1test
            # age_clusters_newid = age_clusters_set1b #WARNING!!!
            # num_images_per_cluster_used_MORPH_out = 15000 #num_images_per_cluster_used_set1b #WARNING!!!

            age_eyes_normalized_base_dir_train = MORPH_base_dir + "Fs" + str(self.training_set)
            age_eyes_normalized_base_dir_seenid = MORPH_base_dir + "Fs" + str(self.training_set)
            age_eyes_normalized_base_dir_newid = MORPH_base_dir + "Fs" + str(
                self.training_set) + "-test"  # s2-test_byAge_mcnn

            if num_images_per_cluster_used_set1 == 0:
                ex = "error: num_images_per_cluster_used_set1 = 0"
                raise Exception(ex)
            if num_images_per_cluster_used_set1b == 0:
                ex = "error: num_images_per_cluster_used_set1b = 0"
                raise Exception(ex)
            if num_images_per_cluster_used_set1test == 0:
                ex = "error: num_images_per_cluster_used_set1test = 0"
                raise Exception(ex)
                # quit()
        age_all_labels_map_newid = age_all_labels_map_MORPH

        print ("age_eyes_normalized_base_dir_newid=" + age_eyes_normalized_base_dir_newid)

        verbose = False or True
        if verbose:
            print ("num clusters = %d" % len(age_clusters))
            for age_cluster in age_clusters:
                print ("avg_label=%f, num_images=%d" % (age_cluster[1], age_cluster[0]))
                print ("filenames[0]=" + age_cluster[2][0])
                print ("orig_labels[0]=", age_cluster[3][0])
                print ("filenames[-1]=" + age_cluster[2][-1])
                print ("orig_labels[0]=", age_cluster[3][-1])
        # quit()
        if leave_k_out_MORPH > 0 and len(age_clusters) != 30 and len(
                age_clusters) != 0 and False:  # WARNING! 33 clusters now!
            er = "leave_k_out_MORPH is changing the number of clusters (%d clusters instead of 30)" % len(age_clusters)
            raise Exception(er)

        use_seenid_classes_to_generate_knownid_and_newid_classes = True  # and False #WARNING

        extra_distortions_for_out_of_database_test = True and False  # Warning! default is False
        if extra_distortions_for_out_of_database_test:
            print ("using even more extensive distortions")
            base_scale = 1.14  # 1.155 # 1.125 # 1.14 * (37.5/37.0) #1.14  #* 1.1 #*0.955 #*1.05 # 1.14 WARNING!!!
            factor_scale_training = 1.04  # 1.04! #*** 1.03573 #1.04 #1.03573 (article) # 1.032157 #1.0393 # 1.03573
            factor_scale_seenid = 1.01989  # 1.021879 #1.018943  #1.020885         # 1.01989
            delta_rotation = 2.5  # ** 2.0 #2.0
            factor_rotation_seenid = 1.65 / 3.0
            delta_pos = 2.8  # Net60,61,62: 3.3 #2.5! # ** 1.0 #1.2 #0.75 #eedit
            factor_pos_seenid = 0.5  # 0.5
            obj_avg_std = 0.18  # 0.17! # ** 0.0
            obj_std_base = 0.16  # ** 0.16
            obj_std_dif = 0.05  # Net60: 0.081 # 0.096! # ** 0.00 #WARNING!!! WHY min can be smaller than zero???!!!

            obj_std_min = obj_std_base - obj_std_dif  # ** 0.16
            obj_std_max = obj_std_base + obj_std_dif  # ** 0.16
        elif not self.use_setup_Guo:  # WARNING!!!!!  ERROR!!! #MY MORPH SETUP, eventually use only one setup? why,
            # nooo! # warning!!!
            print ("using distortions for my experimental setup")
            base_scale = 1.14  # 1.155 # 1.125 # 1.14 * (37.5/37.0) #1.14  #* 1.1 #*0.955 #*1.05 # 1.14 WARNING!!!
            factor_scale_training = 1.03573  # 1.04 #1.03573 (article) # 1.032157 #1.0393 # 1.03573
            factor_scale_seenid = 1.01989  # 1.021879 #1.018943  #1.020885         # 1.01989
            delta_rotation = 2.0  # ** 2.0 #2.0
            factor_rotation_seenid = 1.65 / 3.0
            delta_pos = 1.0  # ** 1.0 #1.2 #0.75 #eedit
            factor_pos_seenid = 0.5
            obj_avg_std = 0.0  # ** 0.0
            obj_std_base = 0.16  # ** 0.16
            obj_std_dif = 0.01  # WARNING! # ** 0.00

            obj_std_min = obj_std_base - obj_std_dif  # ** 0.16
            obj_std_max = obj_std_base + obj_std_dif  # ** 0.16
        else:  # GUO MORPH SETUP
            print ("using distortions for GUO setup")
            # 0.825 #0.55 # 1.1
            # smin=0.575, smax=0.625 (orig images)
            # (zoomed images)
            # smin=1.25, smax=1.40, 1.325
            base_scale = 1.14  # 1.155 # 1.125 # 1.14 * (37.5/37.0) #1.14  #* 1.1 #*0.955 #*1.05 # 1.14 WARNING!!!
            factor_scale_training = 1.04  # 1.04! #*** 1.03573 #1.04 #1.03573 (article) # 1.032157 #1.0393 # 1.03573
            factor_scale_seenid = 1.01989  # 1.021879 #1.018943  #1.020885         # 1.01989

            delta_rotation = 2.0  # ** 2.0 #2.0
            factor_rotation_seenid = 1.65 / 3.0
            delta_pos = 2.3  # 2.5! # ** 1.0 #1.2 #0.75 #eedit
            factor_pos_seenid = 0.5
            obj_avg_std = 0.165  # 0.17! # ** 0.0
            obj_std_base = 0.16  # ** 0.16
            obj_std_dif = 0.081  # 0.096! # ** 0.00 #WARNING!!! WHY min can be smaller than zero???!!!

            obj_std_min = obj_std_base - obj_std_dif  # ** 0.16
            obj_std_max = obj_std_base + obj_std_dif  # ** 0.16

        obj_avg_std_seenid = 0.0  # Image mean is thus constant and equal to 0.5
        obj_std_dif_seenid = 0.0  # std of image pixels is constant and equal to obj_std_base
        obj_std_min_seenid = obj_std_base - obj_std_dif_seenid  # ** 0.16
        obj_std_max_seenid = obj_std_base + obj_std_dif_seenid  # ** 0.16

        # 0.825 #0.55 # 1.1
        # smin=0.575, smax=0.625 (orig images)
        # (zoomed images)
        # smin=1.25, smax=1.40, 1.325
        # ##base_scale = 1.14 #1.155 # 1.125 # 1.14 * (37.5/37.0) #1.14  #* 1.1 #*0.955 #*1.05 # 1.14 WARNING!!!
        # ##factor_scale_training = 1.03573 #1.04 #1.03573 (article) # 1.032157 #1.0393 # 1.03573
        # ##factor_scale_seenid = 1.01989 #1.021879 #1.018943  #1.020885         # 1.01989
        # ##delta_rotation = 2.0 #2.0
        # ##factor_rotation_seenid = 1.65 / 3.0
        # ##delta_pos = 1.0 #1.0 #1.2 #0.75 #eedit
        # ##factor_pos_seenid = 0.5

        print ("Randomization parameters base_scale=" + \
               "%f, factor_scale_training=%f, factor_scale_seenid=%f" % ( base_scale, factor_scale_training,
                                                                          factor_scale_seenid))
        print ("Randomization parameters delta_rotation=" + \
               "%f, factor_rotation_seenid=%f, delta_pos=%f, factor_pos_seenid=%f" % (delta_rotation,
                                                                                      factor_rotation_seenid,
                                                                                      delta_pos, factor_pos_seenid))

        # if leave_k_out_MORPH == 1000:
        #     extra_images_LKO_train = (age_trim_number_MORPH - 1200)*2/3 # (1270-1200)=70, 70/3 = 23
        #     extra_images_LKO_seenid = (age_trim_number_MORPH - 1200)/3
        #     images_after_LKO_train = 1075 + extra_images_LKO_train # (1270-1200)=70, 70/3 = 23
        #     images_after_LKO_seenid = 125 + extra_images_LKO_seenid
        #     print "Effective number of images for Train appears to be:", images_after_LKO_train
        #     print "Effective number of images for SeenId appears to be:", images_after_LKO_seenid
        # elif leave_k_out_MORPH == 2000:
        #     extra_images_LKO_train = (age_trim_number_MORPH - 1100)*2/3 # (1270-1200)=70, 70/3 = 23
        #     extra_images_LKO_seenid = (age_trim_number_MORPH - 1100)/3
        #     images_after_LKO_train = 975 + extra_images_LKO_train # (1270-1200)=70, 70/3 = 23
        #     images_after_LKO_seenid = 125 + extra_images_LKO_seenid
        #     print "Effective number of images for Train appears to be:", images_after_LKO_train
        #     print "Effective number of images for SeenId appears to be:", images_after_LKO_seenid
        #
        # else:
        #     images_after_LKO_train = 1075 # (1270-1200)=70, 70/3 = 23
        #     images_after_LKO_seenid = 125

        # DEFINITIVE:
        # 128x128: iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=0.0, dy=0.0, smin=1.275, smax=1.375, delta_rotation=3.0,
        # pre_mirroring="none" "duplicate", contrast_enhance=True,
        # 160x160:
        iSeq_set = iTrainRAge = [[self.iSeqCreateRAge(dx=delta_pos, dy=delta_pos,
                                                      smin=base_scale / factor_scale_training,
                                                      smax=base_scale * factor_scale_training,
                                                      delta_rotation=delta_rotation, pre_mirroring="none",
                                                      contrast_enhance=True,  # WARNING!!! pre_mirroring="none"
                                                      # -0.05
                                                      # 192x192: iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=0.0,
                                                      # dy=0.0, smin=0.85, smax=0.91666, delta_rotation=3.0,
                                                      # pre_mirroring="none", contrast_enhance=True,
                                                      obj_avg_std=obj_avg_std, obj_std_min=obj_std_min,
                                                      obj_std_max=obj_std_max, new_clusters=age_clusters,
                                                      num_images_per_cluster_used=num_images_per_cluster_used_MORPH,
                                                      # =>=>=>1075 #1000 #1000, 900=>27000
                                                      images_base_dir=age_eyes_normalized_base_dir_train,
                                                      first_image_index=0, repetition_factor=1, seed=-1,
                                                      use_orig_label_as_class=False,
                                                      use_orig_label=True)]]  # repetition_factor=8 (article 8), 6 or
        # at least 4 #T: dx=2, dy=2, smin=1.25, smax=1.40, repetition_factor=5
        # Experimental:
        # iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=1.0, smax=1.0, delta_rotation=0.0,
        # pre_mirroring="none", contrast_enhance=True,
        #                                         obj_avg_std=0.00, obj_std_min=0.20, obj_std_max=0.20,
        # clusters=age_clusters, num_images_per_cluster_used=1000,  #1000 #1000, 900=>27000
        #                                         images_base_dir=age_eyes_normalized_base_dir,
        # first_image_index=0, repetition_factor=1, seed=-1, use_orig_label=True)]] #repetition_factor=6 or at
        # least 4 #T: dx=2, dy=2, smin=1.25, smax=1.40, repetition_factor=5
        sTrainRAge = [[self.sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=self.age_use_RGB_images)]]

        if self.pre_mirroring_supervised_test:
            pre_mirroring_supervised_test = "all"
        else:
            pre_mirroring_supervised_test = "none"
        print ("pre_mirroring_supervised_test = ", pre_mirroring_supervised_test)

        # MORPH+FGNet
        # iSeq_set = iTrainRAge = [[iSeqCreateRAge(dx=2, dy=2, smin=1.25, smax=1.40, clusters=age_clusters,
        # num_images_per_cluster_used=num_images_per_cluster_used_MORPH_FGNet,  #1000 =>30000, 900=>27000
        #                                         images_base_dir=age_eyes_normalized_base_dir, first_image_index=0,
        # repetition_factor=5, seed=-1, use_orig_label=False)]] #rep=5
        # sSeq_set = sTrainRAge = [[sSeqCreateRAge(iSeq_set[0][0], seed=-1)]]
        # smin=0.595, smax=0.605 (orig images)
        # 128x128: iSeq_set = iSeenidRAge = iSeqCreateRAge(dx=0.0, dy=0.0, smin=1.3, smax=1.35, delta_rotation=1.5,
        # pre_mirroring="none", contrast_enhance=True, 160x160: WARNING!!! WHY OBJ_STD_MIN/MAX is fixed???
        # WARNING! USED age_clusters instead of age_clusters_seenid
        iSeq_set = iSeenidRAge = \
            self.iSeqCreateRAge(dx=delta_pos * factor_pos_seenid, dy=delta_pos * factor_pos_seenid,
                                smin=base_scale / factor_scale_seenid,
                                smax=base_scale * factor_scale_seenid,
                                delta_rotation=delta_rotation * factor_rotation_seenid,
                                pre_mirroring=pre_mirroring_supervised_test, contrast_enhance=True,  # Activate for mirroring experiments
                                # 192x192:iSeq_set = iSeenidRAge = iSeqCreateRAge(dx=0.0, dy=0.0,
                                # smin=0.86667, smax=0.9, delta_rotation=1.5, pre_mirroring="none",
                                # contrast_enhance=True,
                                obj_avg_std=obj_avg_std_seenid, obj_std_min=obj_std_min_seenid,
                                obj_std_max=obj_std_max_seenid, new_clusters=age_clusters_seenid,
                                num_images_per_cluster_used=num_images_per_cluster_used_MORPH_seenid,
                                # 125+extra_images_LKO_third  #200 #300=>9000
                                images_base_dir=age_eyes_normalized_base_dir_seenid,
                                first_image_index=0, repetition_factor=1, seed=-1,
                                # repetition_factor= 6 (article 6), 12,8,4
                                use_orig_label_as_class=\
                                    use_seenid_classes_to_generate_knownid_and_newid_classes and False,
                                use_orig_label=True)  # repetition_factor=8, 4
        # #T=repetition_factor=3
        sSeenidRAge = self.sSeqCreateRAge(iSeq_set, seed=-1, use_RGB_images=self.age_use_RGB_images)
        ###Testing Original MORPH:
        # 128x128: iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=1.325, smax=1.326, delta_rotation=0.0,
        # pre_mirroring="none", contrast_enhance=True,
        # 192x192:
        # iSeq_set = iNewidRAge = [[iSeqCreateRAge(dx=0, dy=0, smin=0.8833, smax=0.8833, delta_rotation=0.0,
        # pre_mirroring="none", contrast_enhance=True,
        # 160x160: use_orig_label_as_class
        # TODO:get rid of this conditional. Add Leave-k-out for MORPH+FGNet
        testing_INIBilder = (num_images_per_cluster_used_INIBilder > 0) and False
        testing_FGNet = True and False  # Warning, should be False normally
        if (not testing_INIBilder) and (not testing_FGNet):
            if leave_k_out_MORPH == 0 and not self.use_setup_Guo:
                print ("Selecting NewId without using leave_k_out_strategy")
                # ##WARNING!!!! HERE 0.16 instead of 0.2 should be used???
                iSeq_set = iNewidRAge = [[self.iSeqCreateRAge(dx=0, dy=0, smin=base_scale, smax=base_scale,
                                                              delta_rotation=0.0, pre_mirroring="none",
                                                              contrast_enhance=True,
                                                              obj_avg_std=0.0, obj_std_min=obj_std_base,
                                                              obj_std_max=obj_std_base, new_clusters=age_clusters,
                                                              num_images_per_cluster_used=200,
                                                              # 200=>6000 pre_mirroring="none"
                                                              images_base_dir=age_eyes_normalized_base_dir_newid,
                                                              first_image_index=1200, repetition_factor=1, seed=-1,
                                                              use_orig_label_as_class=False, use_orig_label=True,
                                                              increasing_orig_label=True)]]
            else:
                print ("Selecting NewId data using leave_k_out_strategy, with k=%d" % leave_k_out_MORPH)
                iSeq_set = iNewidRAge = [[self.iSeqCreateRAge(dx=0, dy=0, smin=base_scale, smax=base_scale,
                                                              delta_rotation=0.0, pre_mirroring=pre_mirroring_supervised_test,   # Activate for mirroring experiments
                                                              contrast_enhance=True,
                                                              obj_avg_std=0.0, obj_std_min=obj_std_base,
                                                              obj_std_max=obj_std_base, new_clusters=age_clusters_newid,
                                                              num_images_per_cluster_used=\
                                                                  num_images_per_cluster_used_MORPH_out,
                                                              # 200=>6000
                                                              images_base_dir=age_eyes_normalized_base_dir_newid,
                                                              first_image_index=0, repetition_factor=1, seed=-1,
                                                              use_orig_label_as_class=False, use_orig_label=True,
                                                              increasing_orig_label=True)]]
            sNewidRAge = [[self.sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=self.age_use_RGB_images)]]

        # iSeenidRAge = iTrainRAge[0][0]
        # sSeenidRAge = sTrainRAge[0][0]
        # iNewidRAge = iTrainRAge
        # sNewidRAge = sTrainRAge
        # Testing with INI Bilder:
        if testing_INIBilder:
            print ("Using INI Bilder for testing")
            iSeq_set = iNewidRAge = [[self.iSeqCreateRAge(dx=0, dy=0, smin=base_scale, smax=base_scale,
                                                          delta_rotation=0.0, pre_mirroring="none",
                                                          contrast_enhance=True,  # pre_mirroring="none",
                                                          obj_avg_std=0.0, obj_std_min=obj_std_base,
                                                          obj_std_max=obj_std_base, new_clusters=age_clusters_INIBilder,
                                                          num_images_per_cluster_used=\
                                                              num_images_per_cluster_used_INIBilder,
                                                          images_base_dir=age_eyes_normalized_base_dir_INIBilder,
                                                          first_image_index=0, repetition_factor=1, seed=-1,
                                                          use_orig_label_as_class=False, use_orig_label=True)]]
            sNewidRAge = [[self.sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=self.age_use_RGB_images)]]
            age_all_labels_map_newid = age_all_labels_map_INI
        # Testing with FGNet:
        if testing_FGNet:
            print ("Using FGNet for testing")
            iSeq_set = iNewidRAge = [[self.iSeqCreateRAge(dx=0, dy=0, smin=base_scale, smax=base_scale,
                                                          delta_rotation=0.0, pre_mirroring="none",
                                                          contrast_enhance=True,
                                                          obj_avg_std=0.0, obj_std_min=obj_std_base,
                                                          obj_std_max=obj_std_base, new_clusters=age_clusters_FGNet,
                                                          num_images_per_cluster_used=num_images_per_cluster_used_FGNet,
                                                          images_base_dir=age_eyes_normalized_base_dir_FGNet,
                                                          first_image_index=0, repetition_factor=1, seed=-1,
                                                          use_orig_label_as_class=False, use_orig_label=True,
                                                          increasing_orig_label=True)]]
            sNewidRAge = [[self.sSeqCreateRAge(iSeq_set[0][0], seed=-1, use_RGB_images=self.age_use_RGB_images)]]
            age_all_labels_map_newid = age_all_labels_map_FGNet

        self.name = "Function Based Data Creation for RAge"
        self.network = "linearNetwork4L"  # Default Network, but ignored
        self.iTrain = iTrainRAge
        self.sTrain = sTrainRAge

        self.iSeenid = iSeenidRAge
        self.sSeenid = sSeenidRAge

        self.iNewid = iNewidRAge
        self.sNewid = sNewidRAge

        if iTrainRAge is not None and iTrainRAge[0][0] is not None:
            self.block_size = iTrainRAge[0][0].block_size

        # print sTrainRAge[0][0].translations_x
        # print sSeenidRAge.translations_x
        # print sNewidRAge[0][0].translations_x
        # quit()

        # TODO:This code seems broken, fix it!
        if iSeenidRAge is not None:
            all_classes = numpy.unique(iSeenidRAge.correct_classes)
            min_num_classes = 7  # 40
            max_num_classes = 42  # 42

            num_classes = len(all_classes)
            print ("Updating num_classes in Seenid. Originally %d classes. Afterwards at least %d and at most %d." % (
                num_classes, min_num_classes, max_num_classes))

            if min_num_classes > num_classes:
                ex = "insufficient number of classes in seenid data: %d >= %d " % (min_num_classes, num_classes)
                raise Exception(ex)

            smallest_number_of_samples_per_class = 30
            class_list = list(all_classes)  # This is the actual mapping from orig. classes to final classes
            # print "Original class list: ", class_list
            num_classes = len(class_list)
            factor_increment_samples = 1.1

            if num_classes > 1000:
                print ("using optimized parameters for faster class partitioning for Seenid")
                min_num_classes = max_num_classes = 39
                smallest_number_of_samples_per_class = len(iSeenidRAge.correct_classes) / (max_num_classes + 0.3)  # 780
                factor_increment_samples = 1.02
            while num_classes > max_num_classes:
                current_count = 0
                current_class = 0
                class_list = []
                for classnr in all_classes:  # [::-1]:
                    class_list.append(current_class)
                    current_count += (iSeenidRAge.correct_classes == classnr).sum()
                    if current_count >= smallest_number_of_samples_per_class:
                        current_count = 0
                        current_class += 1
                if current_count >= smallest_number_of_samples_per_class:
                    print ("last class cluster is fine")
                elif current_count > 0:
                    print ("fixing last class cluster by adding it to its predecesor")
                    class_list = numpy.array(class_list)
                    class_list[class_list == current_class] = current_class - 1
                num_classes = len(numpy.unique(class_list))
                print ("In iteration with smallest_number_of_samples_per_class=%d, we have %d clases" % (
                    smallest_number_of_samples_per_class, num_classes))
                smallest_number_of_samples_per_class = \
                    factor_increment_samples * smallest_number_of_samples_per_class + 1
                # class_list.reverse()
                # class_list = numpy.array(class_list)
                # class_list = class_list.max()-class_list
            if min_num_classes > num_classes:
                ex = "Resulting number of classes %d is smaller than min_num_classes %d" % (
                num_classes, min_num_classes)
                raise Exception(ex)

            print ("class_list=", class_list)
            for i, classnr in enumerate(all_classes):
                iSeenidRAge.correct_classes[iSeenidRAge.correct_classes == classnr] = class_list[i]
                # quit()

            if use_seenid_classes_to_generate_knownid_and_newid_classes:
                print ("Orig iSeenidRAge.correct_labels=" +  str(iSeenidRAge.correct_labels))
                print ("Orig len(iSeenidRAge.correct_labels)=%d" % len(iSeenidRAge.correct_labels))
                print ("Orig len(iSeenidRAge.correct_classes)=%d" % len(iSeenidRAge.correct_classes))
                all_classes = numpy.unique(iSeenidRAge.correct_classes)
                print ("all_classes=" +  str(all_classes))
                avg_labels = more_nodes.compute_average_labels_for_each_class(iSeenidRAge.correct_classes,
                                                                              iSeenidRAge.correct_labels)
                print ("avg_labels=" +  str(avg_labels))
                iTrainRAge[0][0].correct_classes = more_nodes.map_labels_to_class_number(all_classes, avg_labels,
                                                                                         iTrainRAge[0][
                                                                                             0].correct_labels)
                iNewidRAge[0][0].correct_classes = more_nodes.map_labels_to_class_number(all_classes, avg_labels,
                                                                                         iNewidRAge[0][
                                                                                             0].correct_labels)

                print ("Unique classes for Training: %d" % len(numpy.unique(iTrainRAge[0][0].correct_classes)))
                print ("Unique classes for Newid: %d" % len(numpy.unique(iNewidRAge[0][0].correct_classes)))

                for classnr in all_classes:
                    print ("|class %d| = %d," % (classnr, (iSeenidRAge.correct_classes == classnr).sum()))
            # quit()

            print ("Verifying that train and test image filenames are disjoint (not verifying leave-k-out here)")
            input_files_network = set(iTrainRAge[0][0].input_files)
            input_files_supervised = set(iSeenidRAge.input_files)
            input_files_testing = set(iNewidRAge[0][0].input_files)
            intersection_network = input_files_testing.intersection(input_files_network)
            intersection_supervised = input_files_testing.intersection(input_files_supervised)
            if len(intersection_network) > 0:
                ex = "Test data and training data have %d non disjoint input images!: " % len(
                    intersection_network) + str(intersection_network)
                raise Exception(ex)
            elif len(intersection_supervised) > 0:  # WARNING!!!!
                ex = "Test data and supervised data have %d non disjoint input images!: " % len(
                    intersection_supervised) + str(intersection_supervised)
                raise Exception(ex)
            else:
                print ("Test data is disjoint to training data and supervised data")

        age_add_other_label_classes = True  # and False
        age_multiple_labels = age_add_other_label_classes  # and False
        # age_first_age_label = True and False
        age_label_ordering = "AgeRaceGenderIAge"
        age_ordered_by_increasing_label = 0  # 0=Age,1=Race, 2=Gender
        append_gender_classification = True  # and False #Add training graph for gender
        append_race_classification = True  # and False #Add training graph for race

        # False, *, *, * => learn only age
        # True, True, True, False, True  => learn both age and skin color, age is first label and determines ordering,
        # color determines subordering inside the clusters
        # True, True, True, False, False => learn both age and skin color, age is first label and determines ordering
        # True, True, True, True, False => learn both age and skin color, color is first label and determines ordering
        # True, False, *, *  => learn only color

        # WARNING!!!! THERE SEEMS TO BE A BUG WITH THE FUNCTION load_skin_color_labels or with python get_lines()!!!!
        # Fixed by repeating first line twice!!!
        if age_add_other_label_classes:
            # age_all_labels_map = load_GT_labels("/local/escalafl/Alberto/MORPH_setsS1S2S3_seed12345/
            # GT_MORPH_AgeRAgeGenderRace.txt", age_included=True, rAge_included=True, gender_included=True,
            # race_included=True, avgColor_included=False)

            #     if "EyeNZ4292740_05M30.JPG" not in age_skin_color_labels_map.keys():
            #         print "key EyeNZ4292740_05M30.JPG still missing 1"
            #         quit()
            #     else:
            #         print "OK 1"
            #         print 'age_skin_color_labels_map["EyeNZ4292740_05M30.JPG"] =',
            # age_skin_color_labels_map["EyeNZ4292740_05M30.JPG"]

            # print "age_skin_color_labels_map= ", age_skin_color_labels_map
            print ("has %d entries" % (len(age_all_labels_map_MORPH.keys())))
            #    for keys in age_skin_color_labels_map.keys():
            #        if keys.startswith("EyeNZ4292740"):
            #            print "Key <%s><%f>"%(keys, age_skin_color_labels_map[keys])
            iAge_labels_TrainRAge = []
            rAge_labels_TrainRAge = []
            gender_labels_TrainRAge = []
            race_labels_TrainRAge = []
            for input_file in iTrainRAge[0][0].input_files:
                input_file_short = string.split(input_file, "/")[-1]
                all_labels = age_all_labels_map_MORPH[input_file_short]
                iAge_labels_TrainRAge.append(all_labels[0])
                rAge_labels_TrainRAge.append(all_labels[1])
                gender_labels_TrainRAge.append(all_labels[2])
                race_labels_TrainRAge.append(all_labels[3])
            iAge_labels_TrainRAge = numpy.array(iAge_labels_TrainRAge).reshape((-1, 1))
            rAge_labels_TrainRAge = numpy.array(rAge_labels_TrainRAge).reshape((-1, 1))
            race_labels_TrainRAge = numpy.array(race_labels_TrainRAge).reshape((-1, 1))
            gender_labels_TrainRAge = numpy.array(gender_labels_TrainRAge).reshape((-1, 1))

            #    if "EyeNZ4292740_05M30.JPG" not in age_skin_color_labels_map.keys():
            #        print "key EyeNZ4292740_05M30.JPG still missing 2"
            #        #quit()
            #    else:
            #        print "OK 2"

            iAge_labels_SeenidRAge = []
            rAge_labels_SeenidRAge = []
            gender_labels_SeenidRAge = []
            race_labels_SeenidRAge = []
            for input_file in iSeenidRAge.input_files:
                input_file_short = string.split(input_file, "/")[-1]
                all_labels = age_all_labels_map_MORPH[input_file_short]
                iAge_labels_SeenidRAge.append(all_labels[0])
                rAge_labels_SeenidRAge.append(all_labels[1])
                gender_labels_SeenidRAge.append(all_labels[2])
                race_labels_SeenidRAge.append(all_labels[3])
            iAge_labels_SeenidRAge = numpy.array(iAge_labels_SeenidRAge).reshape((-1, 1))
            rAge_labels_SeenidRAge = numpy.array(rAge_labels_SeenidRAge).reshape((-1, 1))
            gender_labels_SeenidRAge = numpy.array(gender_labels_SeenidRAge).reshape((-1, 1))
            race_labels_SeenidRAge = numpy.array(race_labels_SeenidRAge).reshape((-1, 1))
            #    if "EyeNZ4292740_05M30.JPG" not in age_skin_color_labels_map.keys():
            #        print "key EyeNZ4292740_05M30.JPG still missing 3"
            #        #quit()
            #    else:
            #        print "OK 3"
            iAge_labels_NewidRAge = []
            rAge_labels_NewidRAge = []
            gender_labels_NewidRAge = []
            race_labels_NewidRAge = []
            for input_file in iNewidRAge[0][0].input_files:
                input_file_short = string.split(input_file, "/")[-1]
                #       if "EyeNZ4292740_05M30.JPG" == input_file_short:
                #           print "FOUND but not found?"
                #        if input_file_short in age_skin_color_labels_map.keys():
                #        xx = age_skin_color_labels_map[input_file_short]
                all_labels = age_all_labels_map_newid[input_file_short]
                iAge_labels_NewidRAge.append(all_labels[0])
                rAge_labels_NewidRAge.append(all_labels[1])
                gender_labels_NewidRAge.append(all_labels[2])
                race_labels_NewidRAge.append(all_labels[3])
                #        else:
                #            print "entry-", input_file_short, "not found!!!", ":("
            iAge_labels_NewidRAge = numpy.array(iAge_labels_NewidRAge).reshape((-1, 1))
            rAge_labels_NewidRAge = numpy.array(rAge_labels_NewidRAge).reshape((-1, 1))
            gender_labels_NewidRAge = numpy.array(gender_labels_NewidRAge).reshape((-1, 1))
            race_labels_NewidRAge = numpy.array(race_labels_NewidRAge).reshape((-1, 1))

            print ("comparing integer age vs. int(rAge/365.242+0.0006)")
            riAge_labels_NewidRAge = (rAge_labels_NewidRAge / DAYS_IN_A_YEAR + 0.0006).astype(int)
            print ("riAge_labels_NewidRAge=" + str(riAge_labels_NewidRAge.flatten()))
            print ("iAge_labels_NewidRAge=" + str(iAge_labels_NewidRAge.flatten()))
            print ("rAge_labels_NewidRAge=" + str(rAge_labels_NewidRAge.flatten()))
            print ("Same: %d" % ((riAge_labels_NewidRAge == iAge_labels_NewidRAge).sum()))
            print ("Diff: %d" % ((riAge_labels_NewidRAge != iAge_labels_NewidRAge).sum()))
            indices = numpy.arange(len(iAge_labels_NewidRAge))[
                riAge_labels_NewidRAge.flatten() != iAge_labels_NewidRAge.flatten()]
            print ("Different indices:" + str(indices))
            print ("iAge[indices]=" + str(iAge_labels_NewidRAge.flatten()[indices]))
            print ("riAge[indices]=" + str(riAge_labels_NewidRAge.flatten()[indices]))
            print ("rAge[indices]=" + str(rAge_labels_NewidRAge.flatten()[indices]))

            #    race_labels = race_labels_SeenidRAge.flatten()
            #    sorting = numpy.argsort(race_labels)
            #    num_classes = 10
            #    skin_color_classes_SeenidRAge = numpy.zeros(iSeenidRAge.num_images)
            #    skin_color_classes_SeenidRAge[sorting] = numpy.arange(iSeenidRAge.num_images)*
            # num_classes/iSeenidRAge.num_images
            #    avg_labels = more_nodes.compute_average_labels_for_each_class(skin_color_classes_SeenidRAge,
            # skin_color_labels)
            #    all_classes = numpy.unique(skin_color_classes_SeenidRAge)
            #    print "skin color avg_labels=", avg_labels
            #    skin_color_classes_TrainRAge = more_nodes.map_labels_to_class_number(all_classes, avg_labels,
            # skin_color_labels_TrainRAge.flatten())
            #    skin_color_classes_NewidRAge = more_nodes.map_labels_to_class_number(all_classes, avg_labels,
            # skin_color_labels_NewidRAge.flatten())

            #    skin_color_classes_TrainRAge = skin_color_classes_TrainRAge.reshape((-1,1))
            #    skin_color_classes_SeenidRAge = skin_color_classes_SeenidRAge.reshape((-1,1))
            #    skin_color_classes_NewidRAge = skin_color_classes_NewidRAge.reshape((-1,1))

            age_classes_TrainRAge = iTrainRAge[0][0].correct_classes.reshape((-1, 1))
            age_classes_SeenidRAge = iSeenidRAge.correct_classes.reshape((-1, 1))
            age_classes_NewidRAge = iNewidRAge[0][0].correct_classes.reshape((-1, 1))

            # TODO: FIX class assignment for rAge
            min_rAge = int(rAge_labels_SeenidRAge.min() / DAYS_IN_A_YEAR)
            max_rAge = int(rAge_labels_SeenidRAge.max() / DAYS_IN_A_YEAR)

            rAge_classes_TrainRAge = (rAge_labels_TrainRAge / DAYS_IN_A_YEAR + 0.0006).astype(int)
            rAge_classes_SeenidRAge = (rAge_labels_SeenidRAge / DAYS_IN_A_YEAR + 0.0006).astype(int)
            rAge_classes_NewidRAge = (rAge_labels_NewidRAge / DAYS_IN_A_YEAR + 0.0006).astype(int)

            # race_classes_tmp = numpy.ones_like(race_labels_TrainRAge) #B=-2, O=-1, A=0, H=1, W=2
            # race_classes_tmp[race_labels_TrainRAge == -2] = 0 #B
            # race_classes_tmp[race_labels_TrainRAge == 1] = 0  #H
            # race_classes_TrainRAge = race_classes_tmp

            # race_classes_tmp = numpy.ones_like(race_labels_SeenidRAge) #B=-2, O=-1, A=0, H=1, W=2
            # race_classes_tmp[race_labels_SeenidRAge == -2] = 0 #B
            # race_classes_tmp[race_labels_SeenidRAge == 1] = 0  #H
            # race_classes_SeenidRAge = race_classes_tmp

            # race_classes_tmp = numpy.ones_like(race_labels_NewidRAge) #B=-2, O=-1, A=0, H=1, W=2
            # race_classes_tmp[race_labels_NewidRAge == -2] = 0 #B
            # race_classes_tmp[race_labels_NewidRAge == 1] = 0  #H
            # race_classes_NewidRAge = race_classes_tmp

            race_classes_TrainRAge = (race_labels_TrainRAge + 2 + 3).astype(int) // 4  # B=0 O=1 A=1 H=1 W=1
            race_classes_SeenidRAge = (race_labels_SeenidRAge + 2 + 3).astype(int) // 4
            race_classes_NewidRAge = (race_labels_NewidRAge + 2 + 3).astype(int) // 4

            gender_classes_TrainRAge = gender_labels_TrainRAge.reshape((-1, 1))
            gender_classes_SeenidRAge = gender_labels_SeenidRAge.reshape((-1, 1))
            gender_classes_NewidRAge = gender_labels_NewidRAge.reshape((-1, 1))

            print ("average gender_labels for training data: %g" % gender_labels_TrainRAge.mean())
            num_males_TrainRAge = (gender_classes_TrainRAge == 0).sum()
            num_females_TrainRAge = iTrainRAge[0][0].num_images - num_males_TrainRAge

            print ("average race_labels for training data:" + str(race_labels_TrainRAge.mean()))
            num_BO_TrainRAge = (race_classes_TrainRAge == 0).sum()
            num_AHW_TrainRAge = (race_classes_TrainRAge == 1).sum()

            age_labels_TrainRAge = rAge_labels_TrainRAge
            age_labels_SeenidRAge = rAge_labels_SeenidRAge
            # age_labels_NewidRAge_test = iAge_labels_NewidRAge

        # print "num unique age_classes_NewidRAge_test =", len(numpy.unique(age_classes_NewidRAge_test))

        pure_integer_years = True  # and False #DEFAULT: TRUE
        if pure_integer_years:
            #    age_labels_TrainRAge = iAge_labels_TrainRAge
            #    age_labels_SeenidRAge = iAge_labels_SeenidRAge
            print ("newid/test age labels are integers (integer years coded in days)")
            age_labels_NewidRAge_test = iAge_labels_NewidRAge * DAYS_IN_A_YEAR
        else:
            print ("newid/test age labels are real valued (in exact number of days)")
            age_labels_NewidRAge_test = rAge_labels_NewidRAge

        # print "age_labels_SeenidRAge =", age_labels_SeenidRAge
        print ("age_labels_NewidRAge_test =" + str(age_labels_NewidRAge_test))

        if age_multiple_labels:
            list_all_labels = [
                [age_labels_TrainRAge, race_labels_TrainRAge, gender_labels_TrainRAge, iAge_labels_TrainRAge],
                [age_labels_SeenidRAge, race_labels_SeenidRAge, gender_labels_SeenidRAge, iAge_labels_SeenidRAge],
                [age_labels_NewidRAge_test, race_labels_NewidRAge, gender_labels_NewidRAge, iAge_labels_NewidRAge]]

            list_all_classes = [
                [age_classes_TrainRAge, race_classes_TrainRAge, gender_classes_TrainRAge, iAge_labels_TrainRAge],
                [age_classes_SeenidRAge, race_classes_SeenidRAge, gender_classes_SeenidRAge, iAge_labels_SeenidRAge],
                [age_classes_NewidRAge, race_classes_NewidRAge, gender_classes_NewidRAge, iAge_labels_NewidRAge]]

            if age_label_ordering == "AgeRaceGenderIAge":
                label_ordering = [0, 1, 2, 3]
            elif age_label_ordering == "RaceAgeGenderIAge":
                label_ordering = [1, 0, 2, 3]
            elif age_label_ordering == "GenderAgeRaceIAge":
                label_ordering = [2, 0, 1, 3]
            elif age_label_ordering == "AgeGenderRaceIAge":
                label_ordering = [0, 2, 1, 3]
            elif age_label_ordering == "RaceGenderAgerIAge":
                label_ordering = [1, 2, 0, 3]
            elif age_label_ordering == "GenderRaceAgeIAge":
                label_ordering = [2, 1, 0, 3]
            else:
                print ("age_label_ordering unknown: " + str(age_label_ordering))
                quit()

            print ("iAge_labels_TrainRAge=" + str(iAge_labels_TrainRAge))
            print ("iAge_labels_TrainRAge.shape=" + str(iAge_labels_TrainRAge.shape))
            iTrainRAge[0][0].correct_labels = numpy.concatenate((list_all_labels[0][label_ordering[0]],
                                                                 list_all_labels[0][label_ordering[1]],
                                                                 list_all_labels[0][label_ordering[2]],
                                                                 list_all_labels[0][label_ordering[3]]), axis=1)
            iSeenidRAge.correct_labels = numpy.concatenate((list_all_labels[1][label_ordering[0]],
                                                            list_all_labels[1][label_ordering[1]],
                                                            list_all_labels[1][label_ordering[2]],
                                                            list_all_labels[1][label_ordering[3]]), axis=1)
            iNewidRAge[0][0].correct_labels = numpy.concatenate((list_all_labels[2][label_ordering[0]],
                                                                 list_all_labels[2][label_ordering[1]],
                                                                 list_all_labels[2][label_ordering[2]],
                                                                 list_all_labels[2][label_ordering[3]]), axis=1)

            iTrainRAge[0][0].correct_classes = numpy.concatenate((list_all_classes[0][label_ordering[0]],
                                                                  list_all_classes[0][label_ordering[1]],
                                                                  list_all_classes[0][label_ordering[2]],
                                                                  list_all_classes[0][label_ordering[3]]), axis=1)
            iSeenidRAge.correct_classes = numpy.concatenate((list_all_classes[1][label_ordering[0]],
                                                             list_all_classes[1][label_ordering[1]],
                                                             list_all_classes[1][label_ordering[2]],
                                                             list_all_classes[1][label_ordering[3]]), axis=1)
            iNewidRAge[0][0].correct_classes = numpy.concatenate((list_all_classes[2][label_ordering[0]],
                                                                  list_all_classes[2][label_ordering[1]],
                                                                  list_all_classes[2][label_ordering[2]],
                                                                  list_all_classes[2][label_ordering[3]]), axis=1)

        else:
            print ("Learning only a single label")
            single_label = "Age"  # "Gender"
            if single_label == "Age":
                print ("Learning age only")
            elif single_label == "Race":
                print ("Learning race instead of age")
                iTrainRAge[0][0].correct_labels = race_labels_TrainRAge.flatten()
                iSeenidRAge.correct_labels = race_labels_SeenidRAge.flatten()
                iNewidRAge[0][0].correct_labels = race_labels_NewidRAge.flatten()
                iTrainRAge[0][0].correct_classes = race_classes_TrainRAge.flatten()
                iSeenidRAge.correct_classes = race_classes_SeenidRAge.flatten()
                iNewidRAge[0][0].correct_classes = race_classes_NewidRAge.flatten()
            elif single_label == "Gender":
                print ("Learning gender instead of age")
                iTrainRAge[0][0].correct_labels = gender_labels_TrainRAge.flatten()
                iSeenidRAge.correct_labels = gender_labels_SeenidRAge.flatten()
                iNewidRAge[0][0].correct_labels = gender_labels_NewidRAge.flatten()
                iTrainRAge[0][0].correct_classes = gender_classes_TrainRAge.flatten()
                iSeenidRAge.correct_classes = gender_classes_SeenidRAge.flatten()
                iNewidRAge[0][0].correct_classes = gender_classes_NewidRAge.flatten()
            else:
                print ("single_label has unknown value:" + str(single_label))
                quit()

        if age_ordered_by_increasing_label > 0 and age_add_other_label_classes:
            if age_ordered_by_increasing_label == 1:
                print ("reordering by increasing race label")
                iSeqs_labels = [[iTrainRAge[0][0], race_labels_TrainRAge],
                                [iSeenidRAge, race_labels_SeenidRAge],
                                [iNewidRAge[0][0], race_labels_NewidRAge]]
            elif age_ordered_by_increasing_label == 2:
                print ("reordering by increasing gender (male then female)")
                iSeqs_labels = [[iTrainRAge[0][0], gender_labels_TrainRAge],
                                [iSeenidRAge, gender_labels_SeenidRAge],
                                [iNewidRAge[0][0], gender_labels_NewidRAge]]
            else:
                print ("age_ordered_by_increasing_label has unknown value: " + str(age_ordered_by_increasing_label))
                quit()
            for iSeq, labels in iSeqs_labels:
                all_labels = labels.flatten()
                reordering = numpy.argsort(all_labels)
                if age_multiple_labels:
                    iSeq.correct_labels = iSeq.correct_labels[reordering, :]
                    iSeq.correct_classes = iSeq.correct_classes[reordering, :]
                else:
                    iSeq.correct_labels = iSeq.correct_labels[reordering]
                    iSeq.correct_classes = iSeq.correct_classes[reordering]
                reordered_files = []
                for i in range(len(iSeq.input_files)):
                    reordered_files.append(iSeq.input_files[reordering[i]])
                iSeq.input_files = reordered_files
                # print "reordered_files=", reordered_files
                print ("iSeq.input_files[0] =" + iSeq.input_files[0])
                print ("iSeq.input_files[-1] =" + iSeq.input_files[-1])
                # quit()

        if age_add_other_label_classes:
            print ("Unique classes for Training2: %d" % len(numpy.unique(iTrainRAge[0][0].correct_classes[:, 0])))
            print ("Unique classes for Newid2: %d" % len(numpy.unique(iNewidRAge[0][0].correct_classes[:, 0])))
        else:
            print ("Unique classes for Training2: %d" % len(numpy.unique(iTrainRAge[0][0].correct_classes)))
            print ("Unique classes for Newid2:  %d" % len(numpy.unique(iNewidRAge[0][0].correct_classes)))
        # quit()

        print ("iTrainRAge[0][0].input_files[0]=" + iTrainRAge[0][0].input_files[0])
        print ("iTrainRAge[0][0].input_files[1]=" + iTrainRAge[0][0].input_files[1])
        print ("iTrainRAge[0][0].input_files[-2]=" + iTrainRAge[0][0].input_files[-2])
        print ("iTrainRAge[0][0].input_files[-1]=" + iTrainRAge[0][0].input_files[-1])

        # quit()

        # Update input_files in sSeqs
        for (iSeq, sSeq) in ((iTrainRAge[0][0], sTrainRAge[0][0]),
                             (iSeenidRAge, sSeenidRAge),
                             (iNewidRAge[0][0], sNewidRAge[0][0])):
            sSeq.input_files = [os.path.join(iSeq.data_base_dir, file_name) for file_name in iSeq.input_files]

        if age_ordered_by_increasing_label == 2:  # Gender
            sTrainRAge[0][0].train_mode = "clustered"
            iTrainRAge[0][0].block_size = [num_males_TrainRAge, num_females_TrainRAge]
            sTrainRAge[0][0].block_size = [num_males_TrainRAge, num_females_TrainRAge]

        if age_ordered_by_increasing_label == 1:  # Race
            sTrainRAge[0][0].train_mode = "clustered"
            iTrainRAge[0][0].block_size = [num_BO_TrainRAge, num_AHW_TrainRAge]
            sTrainRAge[0][0].block_size = [num_BO_TrainRAge, num_AHW_TrainRAge]

        # WARNING, EXPERIMENTING WITH WEIGHT=2.0, it should be 1.0
        if append_gender_classification and append_race_classification and \
                        age_label_ordering == "AgeRaceGenderIAge" and age_add_other_label_classes:
            sTrainRAge[0][0].train_mode = [sTrainRAge[0][0].train_mode,
                                           ("classification", iTrainRAge[0][0].correct_classes[:, 1], 1.0),
                                           ("classification", iTrainRAge[0][0].correct_classes[:, 2], 1.0)]
        elif append_gender_classification and age_label_ordering == "AgeRaceGenderIAge" and age_add_other_label_classes:
            sTrainRAge[0][0].train_mode = [sTrainRAge[0][0].train_mode,
                                           ("classification", iTrainRAge[0][0].correct_classes[:, 2], 1.0)]
        elif append_race_classification and age_label_ordering == "AgeRaceGenderIAge" and age_add_other_label_classes:
            sTrainRAge[0][0].train_mode = [sTrainRAge[0][0].train_mode,
                                           ("classification", iTrainRAge[0][0].correct_classes[:, 1], 1.0)]
        else:
            print ("Not adding any graph to the current train_mode")

    def find_available_images(self, base_dir, from_subdirs=None, verbose=False):
        """Counts how many files are in each subdirectory.
        Returns a dictionary d, with entries: d[subdir] = (num_files_in_subfolder, label, [file names])
        where the file names have a relative path w.r.t. the base directory """

        files_dict = {}
        if os.path.lexists(base_dir):
            dirList = os.listdir(base_dir)
        else:
            dirList = []
        for subdir in dirList:
            if from_subdirs is None or subdir in from_subdirs:
                subdir_full = os.path.join(base_dir, subdir)
                subdirList = os.listdir(subdir_full)
                subdirList.sort()
                if subdir in ["Male", "Female", "Unknown"]:
                    if subdir == "Male":
                        label = 1
                    elif subdir == "Female":
                        label = -1
                    else:
                        label = 0
                else:
                    if verbose:
                        print("Subdir: " + str(subdir))
                    label = float(subdir)
                files_dict[subdir] = (len(subdirList), label, subdirList)
        return files_dict

    def list_available_images(self, base_dir, from_subdirs=None, verbose=False):
        """Finds the files in all subdirectory.
        Returns a list l, with entries: l[k] = (file_name, subfolder)
        where the file names have a relative path w.r.t. the base directory """

        if os.path.lexists(base_dir):
            dirList = os.listdir(base_dir)
            #        dirList.append("")
        else:
            dirList = []
            print("WARNING! base_dir " + base_dir + " does not exist!")

        files_list = []
        for filename in dirList:
            file_full = os.path.join(base_dir, filename)
            if os.path.isdir(file_full):
                subdir = filename
                if from_subdirs is None or subdir in from_subdirs:
                    subdir_full = os.path.join(base_dir, subdir)
                    subdirList = os.listdir(subdir_full)
                    subdirList.sort()

                    for filename in subdirList:
                        files_list.append((os.path.join(subdir, filename), subdir))
            else:
                if verbose:
                    print("filename " + filename)
                files_list.append((filename, "100"))

                #    for subdir in dirList:
                # %        if from_subdirs is None or subdir in from_subdirs:
                #            subdir_full = os.path.join(base_dir, subdir)
                #            if os.path.isdir(subdir_full):
                #                subdirList = os.listdir(subdir_full)
                #                subdirList.sort()
                #
                #                for filename in subdirList:
                #                    files_list.append((os.path.join(subdir, filename), subdir))

        return files_list

    def append_GT_labels_to_files(self, files_list, age_all_labels_map, min_age=0.0, max_age=365000.0,
                                  select_races=[-2, -1, 0, 1, 2], verbose=False):
        # min_age and max_age in days
        print("age_all_labels_map has %d entries" % (len(age_all_labels_map.keys())))
        #    rAge_labels_TrainRAge = []
        #    gender_labels_TrainRAge = []
        #    race_labels_TrainRAge = []
        labeled_files_list = []
        for (input_file, subdir) in files_list:
            input_file_short = string.split(input_file, "/")[-1]
            #        rAge_labels_TrainRAge.append(age_all_labels_map[input_file_short][1])
            #        gender_labels_TrainRAge.append(age_all_labels_map[input_file_short][2])
            #        race_labels_TrainRAge.append(age_all_labels_map[input_file_short][3])
            #               (input_file, age, rAge, race, gender)
            iage = int(subdir)
            if iage == 100:
                iage = age_all_labels_map[input_file_short][0]

            entry = (input_file, int(subdir), age_all_labels_map[input_file_short][1],
                     age_all_labels_map[input_file_short][3], age_all_labels_map[input_file_short][2])
            if entry[2] < min_age or entry[2] > max_age:
                if verbose:
                    print("entry " + str(entry) + " discarded due to age out of range: " + str(entry[2]))
            elif entry[3] not in select_races:
                if verbose:
                    print("entry " + str(entry) + " discarded due to race not in select_races: " + str(entry[3]))
            else:
                labeled_files_list.append(entry)
        print("after appending label and filtering samples, the number of original filenames is: %d" % len(
            labeled_files_list))
        #    rAge_labels_TrainRAge = numpy.array(rAge_labels_TrainRAge).reshape((-1,1))
        #    race_labels_TrainRAge = numpy.array(race_labels_TrainRAge).reshape((-1,1))
        #    gender_labels_TrainRAge = numpy.array(gender_labels_TrainRAge).reshape((-1,1))

        return labeled_files_list

    def age_cluster_labeled_files(self, labeled_files_list, repetition=1, num_clusters=1, trim_number=None,
                                  shuffle_each_cluster=False):
        """ clusters/joins the entries of files_dict, so that
            each cluster has the same size
            The set of clusters is represented as a list of clusters [cluster1, cluster2, ...],
            and each cluster is a list of one tuples [(num_files, label, files_with_subdir_dict, ages,
            rAges, races, genders)]
            """
        labeled_files_list = sorted(labeled_files_list, key=lambda labeled_file: labeled_file[2])
        num_files = len(labeled_files_list)
        print("num_files =" + str(num_files))

        cluster_size = num_files // num_clusters
        if cluster_size * num_clusters != num_files:
            print("Number of files %d is not a multiple of num_clusters %d. Fixing this discarding first few samples" % (
            num_files, num_clusters))
            labeled_files_list = labeled_files_list[-cluster_size * num_clusters:]
            num_files = len(labeled_files_list)
            print("new num_files=" + str(num_files))

        if num_files == 0:
            num_clusters = 0

        clusters = []
        for c in range(num_clusters):
            cluster_labeled_files_list = labeled_files_list[c * cluster_size:(c + 1) * cluster_size]

            cluster_files_list = [labeled_file[0] for _ in range(repetition) for labeled_file in
                                  cluster_labeled_files_list]

            cluster_ages_rAges_races_genders_list = [
                (labeled_file[1], labeled_file[2], labeled_file[3], labeled_file[4]) for _ in range(repetition) for
                labeled_file in cluster_labeled_files_list]
            cluster_ages_rAges_races_genders_list = numpy.array(cluster_ages_rAges_races_genders_list)

            #        cluster_ages_list = [labeled_file[1] for labeled_file in cluster_labeled_files_list]
            #        cluster_rAges_list = [labeled_file[2] for labeled_file in cluster_labeled_files_list]
            #        cluster_races_list = [labeled_file[3] for labeled_file in cluster_labeled_files_list]
            #        cluster_genders_list = [labeled_file[4] for labeled_file in cluster_labeled_files_list]

            #        cluster_files_list = sfa_libs.repeat_list_elements(cluster_files_list, rep=repetition)
            #        cluster_ages_list = sfa_libs.repeat_list_elements(cluster_ages_list, rep=repetition)
            #        cluster_rAges_list = sfa_libs.repeat_list_elements(cluster_rAges_list, rep=repetition)
            #        cluster_races_list = sfa_libs.repeat_list_elements(cluster_races_list, rep=repetition)
            #        cluster_genders_list = sfa_libs.repeat_list_elements(cluster_genders_list, rep=repetition)

            if trim_number > 0:
                cluster_num_files = trim_number
            else:
                cluster_num_files = cluster_size * repetition

            if shuffle_each_cluster:
                ordering = numpy.arange(cluster_size * repetition)
                numpy.random.shuffle(ordering)
                cluster_files_list = cluster_files_list[ordering][0:cluster_num_files]
                cluster_ages_rAges_races_genders_list = cluster_ages_rAges_races_genders_list[ordering, :]
                #            cluster_rAges_list = cluster_rAges_list[ordering][0:cluster_num_files]
                #            cluster_races_list = cluster_races_list[ordering][0:cluster_num_files]
                #            cluster_genders_list = cluster_genders_list[ordering][0:cluster_num_files]

            # print "cluster_ages_rAges_races_genders_list=",cluster_ages_rAges_races_genders_list
            cluster_label = cluster_ages_rAges_races_genders_list[:, 0].mean()

            cluster = [cluster_num_files, cluster_label, cluster_files_list, cluster_ages_rAges_races_genders_list]
            clusters.append(cluster)
        return clusters

    def age_cluster_list(self, files_dict, repetition=1, smallest_number_images=None, largest_number_images=None):
        """ clusters/joins the entries of files_dict, so that
            each cluster has at least smallest_number_images images.
            The set of clusters is represented as a list of clusters [cluster1, cluster2, ...],
            and each cluster is a list of tuples [(num_files, label, files_with_subdir_dict), ...]
            """
        subdirs = files_dict.keys()
        subdirs = sorted(subdirs, key=lambda subdir: float(subdir))
        subdirs.reverse()

        clusters = []
        cluster = []
        cluster_size = 0
        for subdir in subdirs:
            num_files_subdir, label, files_subdir = files_dict[subdir]
            if num_files_subdir > 0:
                if largest_number_images is not None:
                    max_repetition = int(
                        numpy.min((numpy.ceil(largest_number_images * 1.0 / num_files_subdir), repetition)))
                else:
                    max_repetition = repetition
                print("max_repetition=%d from %d to %d" % (max_repetition, num_files_subdir,
                                                           num_files_subdir * max_repetition))
                num_files_subdir *= max_repetition
                files_subdir *= max_repetition

                files_subdir_with_subdir = []
                for file_name in files_subdir:
                    files_subdir_with_subdir.append(os.path.join(subdir, file_name))
                cluster.append((num_files_subdir, label, files_subdir_with_subdir))
                cluster_size += num_files_subdir
                if smallest_number_images is None or cluster_size >= smallest_number_images:
                    # Save cluster
                    clusters.append(cluster)
                    # Start a new cluster
                    cluster = []
                    cluster_size = 0
        if cluster_size != 0:  # Something has not reached proper size, add it to lastly created cluster
            if len(clusters) > 0:
                clusters[-1].extend(cluster)
            else:
                clusters.append(cluster)
        return clusters

    def cluster_to_labeled_samples(self, clusters, trim_number=None, shuffle_each_cluster=True, verbose=False):
        """ Reads a cluster structure in a nested list, generates representative labels,
        joins filenames and trims them.
        If trim_number is None, no trimming is done.
        Otherwise clusters are of size at most trim_number
        new_clusters[0]= a cluster
        cluster = (total_files_cluster, avg_label, files_subdirs, orig_labels))
        """
        new_clusters = []
        # print "clusters[0] is", clusters[0]
        # print "len(clusters[0]) is", len(clusters[0])

        for cluster in clusters:
            total_files_cluster = 0
            files_subdirs = []
            orig_labels = []
            sum_labels = 0
            # print "Processing cluster:", cluster
            if verbose:
                print("ClustersAdded:")
            for (num_files_subdir, label, files_subdir) in cluster:
                if verbose:
                    print(" With label:" +  str(label) + "(%d imgs)" % num_files_subdir)
                total_files_cluster += num_files_subdir
                files_subdirs.extend(files_subdir)
                orig_labels += [label] * num_files_subdir
                # TODO: handle non-float labels
                sum_labels += label * num_files_subdir
            avg_label = sum_labels / total_files_cluster
            if verbose:
                print("")

            if shuffle_each_cluster:
                selected = list(range(total_files_cluster))
                numpy.random.shuffle(selected)
                files_subdirs = [files_subdirs[i] for i in selected]
                orig_labels = [orig_labels[i] for i in selected]

            if len(files_subdirs) != len(orig_labels):
                print("Wrong cluster files and orig labels lengths")
                print(len(files_subdirs))
                print(len(orig_labels))

            if trim_number is not None:
                files_subdirs = files_subdirs[0:trim_number]
                orig_labels = orig_labels[0:trim_number]
                total_files_cluster = min(total_files_cluster, trim_number)
            new_clusters.append((total_files_cluster, avg_label, files_subdirs, orig_labels))

        new_clusters.reverse()
        return new_clusters

    def MORPH_leave_k_identities_out_list(self, available_images_list, k=0):
        if k == 0:
            return available_images_list, []

        # subdirs = available_images_dict.keys()
        # all_filenames = []
        # for subdir in subdirs:
        #    all_filenames.extend(available_images_dict[subdir][2])
        all_identities_list = []
        for (filename, subdir) in available_images_list:
            identity = string.split(filename, "_")[0]
            identity = string.split(identity, "/")[-1]
            all_identities_list.append(identity)

        all_identities = set(all_identities_list)
        all_identities_unique = list(all_identities)
        print("%d unique identities detected" % len(all_identities_unique))

        num_identities = numpy.arange(len(all_identities_unique))
        numpy.random.shuffle(num_identities)
        separated_identities = all_identities_unique[0:k]

        # print "separated_identities=", separated_identities

        # now create two dictionaries
        available_images_list_orig = []
        available_images_list_separated = []
        num_separated = 0
        num_orig = 0
        for current_index, (filename, subdir) in enumerate(available_images_list):
            if all_identities_list[current_index] in separated_identities:
                available_images_list_separated.append((filename, subdir))
                num_separated += 1
            else:
                available_images_list_orig.append((filename, subdir))
                num_orig += 1

        print("Separating %d/%d (LKO)" % (num_orig, num_separated))
        print("available_images_list_orig[0]= %d" % available_images_list_orig[0])
        print("available_images_list_separated[0]= %d" % available_images_list_separated[0])
        return available_images_list_orig, available_images_list_separated

    def MORPH_select_k_images_list(self, available_images_list, k=0, replacement=False):
        # available_images_dict[subdir] = (num_files_in_subfolder, label, [file names])
        if k == 0:
            return available_images_list, []

        num_filenames = len(available_images_list)
        if k > num_filenames:
            er = "selecting too many images k=%d, but available=%d" % (k, num_filenames)
            raise Exception(er)

        selection = numpy.arange(num_filenames)
        numpy.random.shuffle(selection)
        selection = selection[0:k]
        separated = numpy.zeros(num_filenames)
        separated[selection] = 1
        available_images_list_orig = []
        available_images_list_separated = []
        num_orig = 0
        num_separated = 0
        for current_index, filename_subdir in enumerate(available_images_list):
            if separated[current_index]:
                available_images_list_separated.append(filename_subdir)
                num_separated += 1
            else:
                available_images_list_orig.append(filename_subdir)
                num_orig += 1
        if num_separated + num_orig != num_filenames:
            er = "Unexpected error in the sizes of the filename lists"
            raise Exception(er)
        print("Separating %d/%d (selecting k=%d) ..." % (num_orig, num_separated, k))
        # print "available_images_list_separated=", available_images_list_separated
        if replacement:
            available_images_list_orig = available_images_list

        print("available_images_list_orig[0]=", available_images_list_orig[0])
        print("available_images_list_separated[0]=", available_images_list_separated[0])

        return available_images_list_orig, available_images_list_separated

    def iSeqCreateRAge(self, dx=2, dy=2, smin=0.95, smax=1.05, delta_rotation=0.0, pre_mirroring="none",
                       contrast_enhance=False, obj_avg_std=0.0, obj_std_min=0.20, obj_std_max=0.20, new_clusters=None,
                       num_images_per_cluster_used=10,
                       images_base_dir="wrong dir", first_image_index=0, repetition_factor=1, seed=-1,
                       use_orig_label_as_class=False, use_orig_label=True, increasing_orig_label=True):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)

        if len(new_clusters) > 0:
            min_cluster_size = new_clusters[0][0]
        else:
            return None

        for cluster in new_clusters:
            min_cluster_size = min(min_cluster_size, cluster[0])

        if first_image_index + num_images_per_cluster_used > min_cluster_size and num_images_per_cluster_used > 0:
            err = "Images to be used %d + %d exceeds the number of available images %d of at least one cluster" % (
                first_image_index, num_images_per_cluster_used, min_cluster_size)
            raise Exception(err)

        print("***** Setting Information Parameters for Real Translation RAge ******")
        iSeq = system_parameters.ParamsInput()
        # iSeq.name = "Real Translation " + slow_var + ": (%d, %d, %d, %d) numsteps %d"%(dx, dy, smin, smax, num_steps)
        iSeq.data_base_dir = images_base_dir
        # TODO: create clusters here.
        iSeq.ids = []
        iSeq.input_files = []
        iSeq.orig_labels = []
        for cluster in new_clusters:
            cluster_id = [cluster[1]] * repetition_factor * num_images_per_cluster_used  # avg_label. The average
            # label of the current cluster being considered
            iSeq.ids.extend(cluster_id)

            increasing_indices = numpy.arange(num_images_per_cluster_used * repetition_factor) // repetition_factor
            cluster_labels = (cluster[3][first_image_index:first_image_index + num_images_per_cluster_used, :])[
                             increasing_indices, :]  # orig_label
            if len(cluster_id) != len(cluster_labels):
                print("ERROR: Wrong number of cluster labels and original labels")
                print("len(cluster_id)= %d" % len(cluster_id))
                print("len(cluster_labels)= %d" % len(cluster_labels))
                quit()
            iSeq.orig_labels.extend(cluster_labels)
            selected_image_filenames = repeat_list_elements(
                cluster[2][first_image_index:first_image_index + num_images_per_cluster_used],
                repetition_factor)  # filenames
            iSeq.input_files.extend(selected_image_filenames)
        if len(iSeq.orig_labels) > 1:
            iSeq.orig_labels = numpy.vstack(iSeq.orig_labels)
            print("iSeq.orig_labels.shape=" +  str(iSeq.orig_labels.shape))

        if use_orig_label and increasing_orig_label:
            if iSeq.orig_labels.shape[1] > 1:
                original_label_index = 1  # rAge
            else:
                original_label_index = 0  # age
            print("Reordering image filenames of data set according to the original labels")
            ordering = numpy.argsort(iSeq.orig_labels[:, original_label_index])
            ordered_input_files = [iSeq.input_files[i] for i in ordering]
            ordered_orig_labels = iSeq.orig_labels[ordering, :]
            ordered_ids = [iSeq.ids[i] for i in ordering]
            iSeq.input_files = ordered_input_files
            iSeq.orig_labels = ordered_orig_labels
            iSeq.ids = ordered_ids
        else:
            print("Image filenames of data set not reordered according to the original labels")

        iSeq.num_images = len(iSeq.input_files)
        iSeq.pre_mirroring = pre_mirroring

        if iSeq.pre_mirroring == "none":
            iSeq.pre_mirror_flags = [False] * iSeq.num_images
        elif iSeq.pre_mirroring == "all":
            iSeq.pre_mirror_flags = [True] * iSeq.num_images
        elif iSeq.pre_mirroring == "random":
            iSeq.pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images)
        elif iSeq.pre_mirroring == "duplicate":
            iSeq.input_files = sfa_libs.repeat_list_elements(iSeq.input_files, rep=2)
            iSeq.ids = sfa_libs.repeat_list_elements(iSeq.ids)
            increasing_indices = numpy.arange(iSeq.num_images * repetition_factor) // repetition_factor
            iSeq.orig_labels = iSeq.orig_labels[increasing_indices, :]
            tmp_pre_mirror_flags = more_nodes.random_boolean_array(iSeq.num_images)  # e.g., [T, T, F, F, T, F]
            iSeq.pre_mirror_flags = numpy.array([item ^ val for item in tmp_pre_mirror_flags for val in
                                                 (False, True)])  # e.g., [T,F, T,F, F,T, F,T, T,F, F,T])
            iSeq.num_images *= 2
            repetition_factor *= 2
        else:
            er = "Erroneous parameter iSeq.pre_mirroring=", iSeq.pre_mirroring
            raise Exception(er)

        # iSeq.slow_var = slow_var
        iSeq.dx = dx
        iSeq.dy = dy
        iSeq.smin = smin
        iSeq.smax = smax
        iSeq.delta_rotation = delta_rotation
        if contrast_enhance == True:
            contrast_enhance = "AgeContrastEnhancement_Avg_Std"  # "PostEqualizeHistogram" # "AgeContrastEnhancement"
        if contrast_enhance in ["AgeContrastEnhancement_Avg_Std", "AgeContrastEnhancement25",
                                "AgeContrastEnhancement20", "AgeContrastEnhancement15", "AgeContrastEnhancement",
                                "PostEqualizeHistogram", "SmartEqualizeHistogram", False, ]:
            iSeq.contrast_enhance = contrast_enhance
        else:
            ex = "Contrast method unknown"
            raise Exception(ex)

        iSeq.obj_avg_std = obj_avg_std
        iSeq.obj_std_min = obj_std_min
        iSeq.obj_std_max = obj_std_max

        #    if len(iSeq.ids) % len(iSeq.trans) != 0 and not continuous:
        #        ex="Here the number of translations/scalings must be a divisor of the number of identities"
        #        raise Exception(ex)
        iSeq.ages = [None]
        iSeq.genders = [None]
        iSeq.racetweens = [None]
        iSeq.expressions = [None]
        iSeq.morphs = [None]
        iSeq.poses = [None]
        iSeq.lightings = [None]
        iSeq.slow_signal = 0  # real slow signal is the translation in the x axis (correlated to identity),
        # added during image loading
        iSeq.step = 1
        iSeq.offset = 0

        # iSeq.params = [ids, expressions, morphs, poses, lightings]
        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                       iSeq.morphs, iSeq.poses, iSeq.lightings]

        iSeq.block_size = num_images_per_cluster_used * repetition_factor

        iSeq.train_mode = "serial"  # "regular" #"serial" # = "serial" "mixed", None
        # None, "regular", "fwindow16", "fwindow32", "fwindow64", "fwindow128"
        #        quit()
        #        iSeq.train_mode = None

        print("BLOCK SIZE =" +  str(iSeq.block_size))
        #     if not use_orig_label_as_class: #Use cluster (average) labels as classes, or the true original labels
        #         unique_ids, iSeq.correct_classes = numpy.unique(iSeq.ids, return_inverse=True)
        # #        iSeq.correct_classes_from_zero = iSeq.correct_classes
        # #        iSeq.correct_classes = sfa_libs.wider_1Darray(numpy.arange(len(clusters)),  iSeq.block_size)
        #     else:
        #         unique_labels, iSeq.correct_classes = numpy.unique(iSeq.orig_labels, return_inverse=True)
        # #        iSeq.correct_classes_from_zero = iSeq.correct_classes


        if not use_orig_label:  # Use cluster (average) labels as labels, or the true original labels
            iSeq.correct_labels = numpy.array(iSeq.ids)
        else:
            iSeq.correct_labels = numpy.array(iSeq.orig_labels[:, 1])
        unique_labels, iSeq.correct_classes = numpy.unique(iSeq.correct_labels, return_inverse=True)
        # iSeq.correct_classes_from_zero = iSeq.correct_labels[:,1]


        if len(iSeq.ids) != len(iSeq.orig_labels) or len(iSeq.orig_labels) != len(iSeq.input_files):
            er = "Computation of orig_labels failed:" + str(iSeq.ids) + str(iSeq.orig_labels)
            er += "len(iSeq.ids)=%d" % len(iSeq.ids) + "len(iSeq.orig_labels)=%d" % len(
                iSeq.orig_labels) + "len(iSeq.input_files)=%d" % len(iSeq.input_files)
            raise Exception(er)

        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateRAge(self, iSeq, seed=-1, use_RGB_images=False):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)
        # else seed <0 then, do not change seed

        if iSeq is None:
            print("iSeq was None, this might be an indication that the data is not available")
            sSeq = system_parameters.ParamsDataLoading()
            return sSeq

        print("******** Setting Training Data Parameters for Real Age  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        # print "iSeq.input_files=", iSeq.input_files
        # print "iSeq.input_files[0]=", iSeq.input_files[0]
        # print "iSeq.data_base_dir", iSeq.data_base_dir
        sSeq.input_files = [os.path.join(iSeq.data_base_dir, file_name) for file_name in iSeq.input_files]
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.include_latest = iSeq.include_latest
        sSeq.image_width = 256
        sSeq.image_height = 260  # 192
        sSeq.subimage_width = 160  # (article 160) #192 # 160 #128
        sSeq.subimage_height = 160  # (article 160) #192 # 160 #128
        sSeq.pre_mirror_flags = iSeq.pre_mirror_flags

        sSeq.trans_x_max = iSeq.dx
        sSeq.trans_x_min = -1 * iSeq.dx
        sSeq.trans_y_max = iSeq.dy
        sSeq.trans_y_min = -1 * iSeq.dy
        sSeq.min_sampling = iSeq.smin
        sSeq.max_sampling = iSeq.smax
        sSeq.delta_rotation = iSeq.delta_rotation
        sSeq.contrast_enhance = iSeq.contrast_enhance
        sSeq.obj_avg_std = iSeq.obj_avg_std
        sSeq.obj_std_min = iSeq.obj_std_min
        sSeq.obj_std_max = iSeq.obj_std_max

        # sSeq.subimage_pixelsampling = 2
        # sSeq.subimage_first_column = sSeq.image_width/2-sSeq.subimage_width*sSeq.pixelsampling_x/2+ 5*sSeq.pixelsampling_x
        sSeq.add_noise_L0 = True
        if use_RGB_images:
            sSeq.convert_format = "RGB"  # "RGB", "L"
        else:
            sSeq.convert_format = "L"
        sSeq.background_type = None
        # random translation for th w coordinate
        # sSeq.translation = 20 # 25, Should be 20!!! Warning: 25
        # sSeq.translations_x = numpy.random.random_integers(-sSeq.translation, sSeq.translation, sSeq.num_images)
        #    print "=>", sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images, sSeq.translations_x
        #    quit()

        #    Do integer displacements make more sense? depends if respect to original image or not. Translation logic needs urgent review!!!
        #    sSeq.translations_x = numpy.random.uniform(low=sSeq.trans_x_min, high=sSeq.trans_x_max, size=sSeq.num_images)
        #    sSeq.translations_y = numpy.random.uniform(low=sSeq.trans_y_min, high=sSeq.trans_y_max, size=sSeq.num_images)
        # BUG0: why is this an integer offset? also frationary offsets should be supported and give acceptable results.

        # Continuous translations:
        #    sSeq.translations_x = numpy.random.uniform(low=sSeq.trans_x_min, high=sSeq.trans_x_max, size=sSeq.num_images)
        #    sSeq.translations_y = numpy.random.uniform(low=sSeq.trans_y_min, high=sSeq.trans_y_max, size=sSeq.num_images)
        # Or alternatively, discrete ofsets:
        sSeq.offset_translation_x = 0.0  # WARNING, experimental recentering!!!
        sSeq.offset_translation_y = -6.0  # -6.0
        #    sSeq.offset_translation_x = -15.0
        #    sSeq.offset_translation_y = -10.0
        integer_translations = True and False
        if integer_translations:
            sSeq.translations_x = numpy.random.randint(sSeq.trans_x_min, sSeq.trans_x_max + 1,
                                                       sSeq.num_images) + sSeq.offset_translation_x
            sSeq.translations_y = numpy.random.randint(sSeq.trans_y_min, sSeq.trans_y_max + 1,
                                                       sSeq.num_images) + sSeq.offset_translation_y
        else:
            sSeq.translations_x = numpy.random.uniform(sSeq.trans_x_min, sSeq.trans_x_max,
                                                       sSeq.num_images) + sSeq.offset_translation_x
            sSeq.translations_y = numpy.random.uniform(sSeq.trans_y_min, sSeq.trans_y_max,
                                                       sSeq.num_images) + sSeq.offset_translation_y

        print("GSP: sSeq.translations_x=" +  str(sSeq.translations_x))

        sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
        sSeq.pixelsampling_y = sSeq.pixelsampling_x + 0.0

        sSeq.rotations = numpy.random.uniform(-sSeq.delta_rotation, sSeq.delta_rotation, sSeq.num_images)
        if iSeq.obj_avg_std is None:
            sSeq.obj_avgs = [None, ] * sSeq.num_images
            print("Good 1/3")
        elif iSeq.obj_avg_std > 0:
            sSeq.obj_avgs = numpy.random.normal(0.0, iSeq.obj_avg_std,
                                                size=sSeq.num_images)  # mean intensity, centered at zero here
        else:
            sSeq.obj_avgs = numpy.zeros(sSeq.num_images)
        sSeq.obj_stds = numpy.random.uniform(sSeq.obj_std_min, sSeq.obj_std_max, sSeq.num_images)  # contrast

        # BUG1 fix
        sSeq.subimage_first_row = (sSeq.image_height - sSeq.subimage_height * sSeq.pixelsampling_y) / 2.0
        sSeq.subimage_first_column = (sSeq.image_width - sSeq.subimage_width * sSeq.pixelsampling_x) / 2.0

        # #BUG1: image center is not computed that way!!! also half(width-1) computation is wrong!!!
        # sSeq.subimage_first_row =  sSeq.image_height/2.0-sSeq.subimage_height*sSeq.pixelsampling_y/2.0
        # sSeq.subimage_first_column = sSeq.image_width/2.0-sSeq.subimage_width*sSeq.pixelsampling_x/2.0

        sSeq.trans_sampled = True  # TODO:check semantics, when is sampling/translation done? why does this value matter?
        sSeq.name = "RAge Dx in (%d, %d) Dy in (%d, %d), sampling in (%d perc, %d perc)" % (sSeq.trans_x_min,
                                                                                            sSeq.trans_x_max,
                                                                                            sSeq.trans_y_min,
                                                                                            sSeq.trans_y_max, int(
            sSeq.min_sampling * 100), int(sSeq.max_sampling * 100))

        print("Mean in correct_labels is: %g" % iSeq.correct_labels.mean())
        print("Var in correct_labels is: %g" % iSeq.correct_labels.var())
        sSeq.load_data = load_data_from_sSeq
        system_parameters.test_object_contents(sSeq)
        return sSeq


ParamsRAgeFunc_set1_96x96 = ParamsRAgeFunc = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 96,
                                                                  use_setup_Guo=True, training_set=1,
                                                                  pre_mirroring_supervised_test=False)
ParamsRAgeFunc_set1_96x96_small = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 96,
                                                        use_setup_Guo=True, rep_Guo_train=5,
                                                        rep_Guo_supervised=1, training_set=1,
                                                        pre_mirroring_supervised_test=False)
ParamsRAgeFunc_set1_96x96_mini = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 96,
                                                                       use_setup_Guo=True, rep_Guo_train=1,
                                                                       rep_Guo_supervised=1, training_set=1,
                                                                       pre_mirroring_supervised_test=False)
ParamsRAgeFunc_set1_48x48 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 48, use_setup_Guo=True,
                                                 training_set=1, pre_mirroring_supervised_test=False)
ParamsRAgeFunc_set1_48x48_mini = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 48,
                                                                       use_setup_Guo=True, rep_Guo_train=1,
                                                                       rep_Guo_supervised=1, training_set=1,
                                                                       pre_mirroring_supervised_test=False)

ParamsRAgeFunc_set1_24x24 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 24, use_setup_Guo=True, training_set=1, pre_mirroring_supervised_test=False)
ParamsRAgeFunc_mirror_set1_96x96 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 96,
                                                             use_setup_Guo=True, training_set=1, pre_mirroring_supervised_test=True)
ParamsRAgeFunc_mirror_set1_48x48 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 48, use_setup_Guo=True, training_set=1, pre_mirroring_supervised_test=True)
ParamsRAgeFunc_mirror_set1_24x24 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 24, use_setup_Guo=True, training_set=1, pre_mirroring_supervised_test=True)

ParamsRAgeFunc_set2_96x96 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 96,
                                                             use_setup_Guo=True, training_set=2, pre_mirroring_supervised_test=False)
ParamsRAgeFunc_set2_48x48 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 48, use_setup_Guo=True, training_set=2, pre_mirroring_supervised_test=False)
ParamsRAgeFunc_set2_24x24 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 24, use_setup_Guo=True, training_set=2, pre_mirroring_supervised_test=False)
ParamsRAgeFunc_mirror_set2_96x96 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 96,
                                                             use_setup_Guo=True, training_set=2, pre_mirroring_supervised_test=True)
ParamsRAgeFunc_mirror_set2_48x48 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 48, use_setup_Guo=True, training_set=2, pre_mirroring_supervised_test=True)
ParamsRAgeFunc_mirror_set2_24x24 = ParamsRAgeExperiment(experiment_seed, experiment_basedir, 24, use_setup_Guo=True, training_set=2, pre_mirroring_supervised_test=True)


#####################################################################################################################
#################                   MNIST Number Database Experiments                     ###########################
#####################################################################################################################
class ParamsMNISTExperiment(system_parameters.ParamsSystem):
    """ Experiment for the recognition of hand-written digits (MNIST).
    Notice the original resolution is 28x28, but the images extracted here have resolution 24x24.
    """
    def __init__(self, experiment_seed, experiment_basedir, hold_out_test=True):
        super(ParamsMNISTExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.experiment_basedir = experiment_basedir
        self.hold_out_test = hold_out_test

        # TODO: determine if the following variables are still necessary or can be deleted
        self.train_mode = "Weird Mode"  # Ignored for the moment
        self.analysis = None
        self.enable_reduced_image_sizes = False  # True
        self.reduction_factor = 1.0  # T=1.0
        self.hack_image_size = 24  # 28, T=24, 28=2*2*7, 24=2*2*2*3
        self.enable_hack_image_size = False  # True
        self.patch_network_for_RGB = False  #

    def create(self):
        # import mnist

        print("MNIST: starting with experiment_seed= %d" % self.experiment_seed)
        numpy.random.seed(self.experiment_seed + 123451)
        shuffle_digits = True  # and False
        num_digits_used = 10  # 8 #8 # 10
        digits_used = numpy.arange(10)  # (8) numpy.arange(10) #[4,9] #(4,9) or (9,7) or numpy.arange(10)
        if shuffle_digits:
            numpy.random.shuffle(digits_used)
        digits_used = digits_used[0:num_digits_used]
        print("digits used for MNIST:" +  str(digits_used))

        clusters_MNIST, images_array_MNIST = \
            self.load_MNIST_clusters(digits_used=digits_used, image_set='training',
                                     images_base_dir=self.experiment_basedir + '/MNIST')
        if self.hold_out_test:
            clusters_MNIST_test, images_array_MNIST_test = clusters_MNIST, images_array_MNIST
            first_image_index_test=0
            num_images_per_cluster_used_test=1000
            first_image_index_seenid=1000
            num_images_per_cluster_used_seenid=2000
            first_image_index_train=1500
            num_images_per_cluster_used_train=3500+421
            rep_train = 4
            rep_seenid = 3
        else:
            clusters_MNIST_test, images_array_MNIST_test = \
                self.load_MNIST_clusters(digits_used=digits_used, image_set='testing',
                                         images_base_dir=self.experiment_basedir + '/MNIST')
            first_image_index_test=0
            num_images_per_cluster_used_test=-1
            first_image_index_seenid=0
            num_images_per_cluster_used_seenid=-1  # 1421
            first_image_index_train=0  # 421
            num_images_per_cluster_used_train=-1  # 5000
            rep_train = 3
            rep_seenid = 1

        numpy.random.seed(self.experiment_seed + 987987)

        # enable_distortions = False
        dx = 1.0  # * 0
        dy = 1.0  # * 0
        delta_rotation = None

        # base_scale = 1.0
        # factor_training = 1.0
        # factor_seenid = 1.0
        # scale_offset = 0.00

        # 5421 images for digit 5 WARNING, here use only 5000!!!!
        iSeq_set1 = self.iSeqCreateMNIST(dx=dx, dy=dy, smin=1.0, smax=1.0, delta_rotation=delta_rotation,
                                         pre_mirroring="none", contrast_enhance=None, obj_avg_std=0.0, obj_std_min=0.20,
                                         obj_std_max=0.20,
                                         clusters=clusters_MNIST,
                                         first_image_index=first_image_index_train,
                                         num_images_per_cluster_used=num_images_per_cluster_used_train,
                                         repetition_factor=rep_train, seed=-1,
                                         use_orig_label=True,
                                         increasing_orig_label=True)  # 5000, num_images_per_cluster_used=-1
        # WARNING! changed repetition factor to 2
        sSeq_set1 = self.sSeqCreateMNIST(iSeq_set1, images_array_MNIST, seed=-1, use_RGB_images=False)

        semi_supervised_learning = True and False
        if not semi_supervised_learning:
            iSeq_set = iTrainMNIST = [[iSeq_set1]]
            sSeq_set = sTrainMNIST = [[sSeq_set1]]
        else:
            iSeq_set2 = self.iSeqCreateMNIST(dx=0.0, dy=0.0, smin=1.0, smax=1.0, delta_rotation=None,
                                             pre_mirroring="none", contrast_enhance=None, obj_avg_std=0.0,
                                             obj_std_min=0.20, obj_std_max=0.20,
                                             clusters=clusters_MNIST, first_image_index=921,
                                             num_images_per_cluster_used=4500, repetition_factor=1, seed=-1,
                                             use_orig_label=True,
                                             increasing_orig_label=True)  # 5000, num_images_per_cluster_used=-1
            iSeq_set2.train_mode = "smart_unlabeled2"  # smart_unlabeled2" #"ignore_data" #"smart_unlabeled3"
            sSeq_set2 = self.sSeqCreateMNIST(iSeq_set2, images_array_MNIST, seed=-1, use_RGB_images=False)
            iSeq_set = iTrainMNIST = [[iSeq_set1, iSeq_set2]]
            sSeq_set = sTrainMNIST = [[sSeq_set1, sSeq_set2]]

            # print "iSeq_set.input_files=",iSeq_set[0][0].input_files
        print("len(iSeq_set.input_files)= %d" % len(iSeq_set[0][0].input_files))
        # print "sSeq_set.input_files=",sSeq_set[0][0].input_files
        print("len(sSeq_set.input_files)= %d" % len(sSeq_set[0][0].input_files))

        # WARNING HERE USE index 5000 and 421 images per cluster!!!
        iSeq_set = iSeenidMNIST = self.iSeqCreateMNIST(dx=dx, dy=dy, smin=1.0, smax=1.0, delta_rotation=delta_rotation,
                                                       pre_mirroring="none", contrast_enhance=None, obj_avg_std=0.0,
                                                       obj_std_min=0.20, obj_std_max=0.20,
                                                       clusters=clusters_MNIST,
                                                       first_image_index=first_image_index_seenid,
                                                       num_images_per_cluster_used=num_images_per_cluster_used_seenid,
                                                       repetition_factor=rep_seenid, seed=-1,
                                                       use_orig_label=True, increasing_orig_label=True)  # 421
        # WARNING, should be 421, not 1421. The latter causes overlap between training and seenid. WARNING!!! changed it to 421
        # WARNING, changed dx and dy to 1.0!!!
        # warning, changed repetition factor to 3

        sSeq_set = sSeenidMNIST = self.sSeqCreateMNIST(iSeq_set, images_array_MNIST, seed=-1, use_RGB_images=False)

        iSeq_set = iNewidMNIST = [[self.iSeqCreateMNIST(dx=0.0, dy=0.0, smin=1.0, smax=1.0, delta_rotation=None,
                                                        pre_mirroring="none", contrast_enhance=None, obj_avg_std=0.0,
                                                        obj_std_min=0.20, obj_std_max=0.20,
                                                        clusters=clusters_MNIST_test,
                                                        first_image_index=first_image_index_test,
                                                        num_images_per_cluster_used=num_images_per_cluster_used_test,
                                                        repetition_factor=1, seed=-1,
                                                        use_orig_label=True, increasing_orig_label=True)]]

        sSeq_set = sNewidMNIST = [
            [self.sSeqCreateMNIST(iSeq_set[0][0], images_array_MNIST_test, seed=-1, use_RGB_images=False)]]

        self.name = "Function Based Data Creation for MNIST"
        self.network = "linearNetwork4L"  # Default Network, but ignored
        self.iTrain = iTrainMNIST
        self.sTrain = sTrainMNIST

        self.iSeenid = iSeenidMNIST
        self.sSeenid = sSeenidMNIST
        # images_array_MNIST_test
        self.iNewid = iNewidMNIST
        self.sNewid = sNewidMNIST

        if iTrainMNIST is not None and iTrainMNIST[0][0] is not None:
            self.block_size = iTrainMNIST[0][0].block_size

    def load_MNIST_clusters(self, digits_used=[2, 8], image_set='training',
                            images_base_dir='/home/escalafl/Databases/MNIST'):
        """Computes a dictionary containing the indices used to index the images of each digit (cluster) and 
        also returns an array containing all images.
        
        image_set: either 'training' or 'testing'
        
        Notice that the images of each cluster are shuffled
        """
        from . import mnist

        images, labels = mnist.read(digits_used, image_set, images_base_dir)
        labels = labels.reshape((-1,))

        num_images = len(labels)
        all_indices = numpy.arange(num_images)

        clusters = {}
        for i, digit in enumerate(digits_used):
            # print "cluster ", i, " for digit ", digit
            cluster_indices = all_indices[(labels == digit)]
            clusters[digit] = cluster_indices
            numpy.random.shuffle(clusters[digit])
            print("cluster %d for digit %d has %d images" % (i, digit, len(cluster_indices)))
        return clusters, images

    def iSeqCreateMNIST(self, dx=0, dy=0, smin=1.0, smax=1.00, delta_rotation=0.0, pre_mirroring="none",
                        contrast_enhance=False, obj_avg_std=0.0, obj_std_min=0.20, obj_std_max=0.20,
                        clusters={}, first_image_index=0, num_images_per_cluster_used=0, repetition_factor=1, seed=-1,
                        use_orig_label=False, increasing_orig_label=False):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)
            print("MNIST. Using seed %d" % seed)

        if num_images_per_cluster_used > 0:
            for digit in clusters.keys():
                if first_image_index + num_images_per_cluster_used > len(clusters[digit]):
                    err = "first_image_index + num_images_per_digit_used > len(cluster_indices)." + "%d + %d > %d" % (
                    first_image_index, num_images_per_cluster_used, len(clusters[digit]))
                    raise Exception(err)

        print("***** Setting Information Parameters for MNIST ******")
        iSeq = system_parameters.ParamsInput()
        # iSeq.name = "MNISTReal Translation " + slow_var + ": (%d, %d, %d, %d) numsteps %d"%(dx, dy, smin, smax,
        # num_steps)
        iSeq.data_base_dir = "wrong dir"
        # TODO: create clusters here.
        iSeq.ids = []
        iSeq.orig_labels = []
        iSeq.block_sizes = []
        for digit in clusters.keys():
            cluster = clusters[digit]
            if num_images_per_cluster_used >= 0:
                cluster_id = cluster[first_image_index:first_image_index + num_images_per_cluster_used].repeat(
                    repetition_factor)  # image numbers
            else:
                cluster_id = cluster[first_image_index:].repeat(repetition_factor)
            iSeq.ids.extend(cluster_id)

            if num_images_per_cluster_used >= 0:
                cluster_labels = [digit] * repetition_factor * num_images_per_cluster_used
            else:
                cluster_labels = [digit] * repetition_factor * (len(cluster) - first_image_index)
            if len(cluster_id) != len(cluster_labels):
                print("ERROR: Wrong number of cluster labels and original labels")
                print("len(cluster_id)=", len(cluster_id))
                print("len(cluster_labels)=", len(cluster_labels))
                quit()
            iSeq.orig_labels.extend(cluster_labels)

        iSeq.input_files = iSeq.ids

        if use_orig_label and increasing_orig_label:
            print("Reordering image filenames of data set according to the original labels")
            ordering = numpy.argsort(iSeq.orig_labels)
            ordered_orig_labels = [iSeq.orig_labels[i] for i in ordering]
            ordered_ids = [iSeq.ids[i] for i in ordering]
            iSeq.orig_labels = ordered_orig_labels
            iSeq.ids = ordered_ids
        else:
            print("Image filenames of data set not reordered according to the original labels")

        iSeq.num_images = len(iSeq.ids)

        iSeq.dx = dx
        iSeq.dy = dy
        iSeq.smin = smin
        iSeq.smax = smax
        iSeq.delta_rotation = delta_rotation

        iSeq.contrast_enhance = contrast_enhance

        iSeq.pre_mirror_flags = False

        #    if contrast_enhance == True:
        #        contrast_enhance = "AgeContrastEnhancement_Avg_Std" # "PostEqualizeHistogram"
        #  "AgeContrastEnhancement"
        #    if contrast_enhance in ["AgeContrastEnhancement_Avg_Std", "AgeContrastEnhancement25",
        # "AgeContrastEnhancement20", "AgeContrastEnhancement15", "AgeContrastEnhancement", "PostEqualizeHistogram",
        # "SmartEqualizeHistogram", False]:
        #        iSeq.contrast_enhance = contrast_enhance
        #    else:
        #        ex = "Contrast method unknown"
        #        raise Exception(ex)

        iSeq.obj_avg_std = obj_avg_std
        iSeq.obj_std_min = obj_std_min
        iSeq.obj_std_max = obj_std_max

        #    if len(iSeq.ids) % len(iSeq.trans) != 0 and not continuous:
        #        ex="Here the number of translations/scalings must be a divisor of the number of identities"
        #        raise Exception(ex)
        iSeq.ages = [None]
        iSeq.genders = [None]
        iSeq.racetweens = [None]
        iSeq.expressions = [None]
        iSeq.morphs = [None]
        iSeq.poses = [None]
        iSeq.lightings = [None]
        iSeq.slow_signal = 0  # real slow signal is the translation in the x axis (correlated to identity),
        # added during image loading
        iSeq.step = 1
        iSeq.offset = 0

        # iSeq.params = [ids, expressions, morphs, poses, lightings]
        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions, \
                       iSeq.morphs, iSeq.poses, iSeq.lightings]

        diff_labels = numpy.unique(iSeq.orig_labels)
        iSeq.block_size = [(iSeq.orig_labels == label).sum() for label in diff_labels]
        print("iSeq.block_size =" +  str(iSeq.block_size))

        #    if num_images_per_cluster_used >= 0:
        #        iSeq.block_size = num_images_per_cluster_used * repetition_factor
        #    else:
        #        iSeq.block_size = 1
        iSeq.train_mode = "clustered"  # compact_classes+7" #"compact_classes+7" #"compact_classes3" #"serial"
        # #"clustered" #"serial" #"mixed", None, "regular", "fwindow16", "fwindow32", "fwindow64", "fwindow128"

        unique_labels, iSeq.correct_classes = numpy.unique(iSeq.orig_labels, return_inverse=True)
        print("unique labels = " +  str(unique_labels))
        # print "(unique) correct_classes = ", iSeq.correct_classes
        # quit()
        iSeq.correct_labels = numpy.array(iSeq.orig_labels)

        use_compact_labels = 0  # 2, 3
        if use_compact_labels > 0:
            classes_mod_2 = iSeq.correct_classes % 2
            classes_mod_3 = iSeq.correct_classes % 3
            classes_mod_5 = iSeq.correct_classes % 5
            if use_compact_labels == 2:
                iSeq.train_mode = [("classification", classes_mod_2, 1.0), ("classification", classes_mod_5, 1.0)]
            elif use_compact_labels == 3:
                iSeq.train_mode = [("classification", classes_mod_2, 1.0), ("classification", classes_mod_3, 1.0),
                                   ("classification", classes_mod_5, 1.0)]
            else:
                er = "incorrect value for use_compact_labels (%d). Use either 2 or 3" % use_compact_labels
                raise Exception(er)

        if len(iSeq.ids) != len(iSeq.orig_labels):
            er = "Computation of orig_labels failed:" + str(iSeq.ids) + str(iSeq.orig_labels)
            er += "len(iSeq.ids)=%d" % len(iSeq.ids) + "len(iSeq.orig_labels)=%d" % len(
                iSeq.orig_labels) + "len(iSeq.input_files)=%d" % len(iSeq.input_files)
            raise Exception(er)

        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateMNIST(self, iSeq, images_array, seed=-1, use_RGB_images=False):
        if seed >= 0 or seed is None:  # also works for
            numpy.random.seed(seed)
        # else seed <0 then, do not change seed

        if iSeq is None:
            print("iSeq was None, this might be an indication that the data is not available")
            sSeq = system_parameters.ParamsDataLoading()
            return sSeq

        print("******** Setting Training Data Parameters for MNIST  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.images_array = images_array
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.include_latest = iSeq.include_latest
        sSeq.image_width = 28
        sSeq.image_height = 28  # 192
        sSeq.subimage_width = 24  # 28 #192 # 160 #128
        sSeq.subimage_height = 24  # 28 #192 # 160 #128
        sSeq.pre_mirror_flags = iSeq.pre_mirror_flags

        sSeq.trans_x_max = iSeq.dx
        sSeq.trans_x_min = -1 * iSeq.dx
        sSeq.trans_y_max = iSeq.dy
        sSeq.trans_y_min = -1 * iSeq.dy
        sSeq.min_sampling = iSeq.smin
        sSeq.max_sampling = iSeq.smax
        sSeq.delta_rotation = iSeq.delta_rotation
        sSeq.contrast_enhance = iSeq.contrast_enhance
        sSeq.obj_avg_std = iSeq.obj_avg_std
        sSeq.obj_std_min = iSeq.obj_std_min
        sSeq.obj_std_max = iSeq.obj_std_max

        sSeq.add_noise_L0 = True
        if use_RGB_images:
            sSeq.convert_format = "RGB"  # "RGB", "L"
        else:
            sSeq.convert_format = "L"
        sSeq.background_type = None

        sSeq.offset_translation_x = 0.0
        sSeq.offset_translation_y = 0.0

        # sSeq.translations_x = numpy.random.uniform(sSeq.trans_x_min, sSeq.trans_x_max, sSeq.num_images) +
        # sSeq.offset_translation_x
        # sSeq.translations_y = numpy.random.uniform(sSeq.trans_y_min, sSeq.trans_y_max, sSeq.num_images) +
        # sSeq.offset_translation_y

        sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max,
                                                           sSeq.num_images) + sSeq.offset_translation_x
        sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max,
                                                           sSeq.num_images) + sSeq.offset_translation_y

        # print "sSeq.translations_x=", sSeq.translations_x

        sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
        sSeq.pixelsampling_y = sSeq.pixelsampling_x + 0.0
        if sSeq.delta_rotation is not None:
            sSeq.rotations = numpy.random.uniform(-sSeq.delta_rotation, sSeq.delta_rotation, sSeq.num_images)
        else:
            sSeq.rotations = None
        if iSeq.obj_avg_std > 0:
            sSeq.obj_avgs = numpy.random.normal(0.0, iSeq.obj_avg_std, size=sSeq.num_images)
        else:
            sSeq.obj_avgs = numpy.zeros(sSeq.num_images)
        sSeq.obj_stds = numpy.random.uniform(sSeq.obj_std_min, sSeq.obj_std_max, sSeq.num_images)

        # BUG1: image center is not computed that way!!! also half(width-1) computation is wrong!!!
        sSeq.subimage_first_row = sSeq.image_height / 2.0 - sSeq.subimage_height * sSeq.pixelsampling_y / 2.0
        sSeq.subimage_first_column = sSeq.image_width / 2.0 - sSeq.subimage_width * sSeq.pixelsampling_x / 2.0

        # TODO:check semantics, when is sampling/translation done? why does this value matter?
        sSeq.trans_sampled = True
        sSeq.name = "MNIST Dx in (%d, %d) Dy in (%d, %d)" % (sSeq.trans_x_min, sSeq.trans_x_max, sSeq.trans_y_min,
                                                             sSeq.trans_y_max) + \
                    ", sampling in (%d perc, %d perc)" % (int(sSeq.min_sampling * 100), int(sSeq.max_sampling * 100))

        print("Mean in correct_labels is: %g" % iSeq.correct_labels.mean())
        print("Var in correct_labels is: %g" % iSeq.correct_labels.var())
        sSeq.load_data = load_data_from_sSeq
        system_parameters.test_object_contents(sSeq)
        return sSeq


ParamsMNISTFunc = ParamsMNISTExperiment(experiment_seed, experiment_basedir, hold_out_test=True)
ParamsMNISTFunc_test = ParamsMNISTExperiment(experiment_seed, experiment_basedir, hold_out_test=False)


# TODO: IS a mapping from classes to zero-starting classes needed????
# all_classes = numpy.unique(iSeenidRAge.correct_classes)
#
# min_num_classes = 7 #40
# max_num_classes = 42 #42
#
# num_classes = len(all_classes)
# if min_num_classes > num_classes:
#     ex = "insufficient number of classes in seenid data: %d >= %d "%(min_num_classes, num_classes)
#     raise Exception(ex)
#
# smallest_number_of_samples_per_class = 30
# class_list = list(all_classes) #This is the actual mapping from orig. classes to final classes
# num_classes = len(class_list)    IEVMLRecNet_out_dims = [ 28, 52, 75, 85, 90, 90, 85, 100, 75,75,75 ]
# #Experiment with control2: Dt=1.9999

# while num_classes > max_num_classes:
#     current_count = 0
#     current_class = 0
#     class_list = []
#     for classnr in all_classes: #[::-1]:
#         class_list.append(current_class)
#         current_count += (iSeenidRAge.correct_classes==classnr).sum()
#         if current_count >= smallest_number_of_samples_per_class:
#             current_count = 0
#             current_class += 1
#     if current_count >= smallest_number_of_samples_per_class:
#         print "last class cluster is fine"
#     elif current_count > 0:
#         print "fixing last class cluster by adding it to its predecesor"
#         class_list = numpy.array(class_list)
#         class_list[class_list == current_class] = current_class-1
#     num_classes = len(numpy.unique(class_list))
#     print "In iteration with smallest_number_of_samples_per_class=%diSeenidGender = iSeqCreateGender(first_id = 0,
# num_ids = 25, user_base_dir=user_base_dir, data_dir="Alberto/RenderingsGender60x200", gender_continuous=True,
# seed=None)
####, we have %d clases"%(smallest_number_of_samples_per_class,num_classes)
#     smallest_number_of_samples_per_class = 1.1 * smallest_number_of_samples_per_class + 1
# #class_list.reverse()
# #class_list = numpy.array(class_list)
# #class_list = class_list.max()-class_list
# if min_num_classes > num_classes:pirate
#     ex = "Resulting number of classes %d is smaller than min_num_classes %d"%(num_classes, min_num_classes)
#     raise Exception(ex)
#
# print "class_list=", class_list
# for i, classnr in enumerate(all_classes):
#     iSeenidRAge.correct_classes[iSeenidRAge.correct_classes==classnr] = class_list[i]

############################################################################################
# #######################          RATLAB experiment!!!!     ###############################
############################################################################################
class ParamsRatlabExperiment(system_parameters.ParamsSystem):
    """ Experiment for learning unsupervised features from the view of a simulated rat (artificial images). """
    def __init__(self, experiment_seed, experiment_basedir, num_images_training=80000, num_images_test=40000):
        super(ParamsRatlabExperiment, self).__init__()
        self.experiment_seed = experiment_seed
        self.experiment_basedir = experiment_basedir
        self.num_images_training = num_images_training
        self.num_images_test = num_images_test
        self.num_available_images = 200000
        self.train_mode = "Weird Mode"  # This value should not be used anymore
        self.analysis = None
        self.enable_reduced_image_sizes = False  # True
        self.reduction_factor = 1.0  # T=1.0 WARNING 1.0, 2.0, 3, 4
        self.hack_image_size = 24  # 28, T=24 WARNING      24, 12, 8, 6 #IS IT 24 or 28!!!!! 28=2*2*7, 24=2*2*2*3
        self.enable_hack_image_size = False  # True
        self.patch_network_for_RGB = False  #

    def compute_indices_first_training_test_images(self):
        success = False
        x0_training = numpy.random.randint(0, self.num_available_images - self.num_images_training)
        while not success:
            x0_test = numpy.random.randint(0, self.num_available_images - self.num_images_test)
            if x0_test >= x0_training + self.num_images_training:
                success = True
            elif x0_training >= x0_test + self.num_images_test:
                success = True
        return x0_training, x0_test

    def create(self):
        print("Ratlab: starting with experiment_seed= %d" % self.experiment_seed)
        numpy.random.seed(self.experiment_seed + 123451)

        num_available_images = self.num_available_images
        num_images_training = self.num_images_training  # 100000  # 6800
        num_images_test = self.num_images_test
        ratlab_data_base_dir = self.experiment_basedir + "/ratlab_sequence_200000"
        ratlab_images = ["frame_%05d.png" % d for d in
                         range(num_available_images)]  # frame_00000.png ... frame_09599.png
        first_image_training, first_image_test = self.compute_indices_first_training_test_images()

        self.iTrain = [[self.iSeqCreateRatlab(ratlab_images, first_image_index=first_image_training,
                                              num_images_used=num_images_training, data_base_dir=ratlab_data_base_dir)]]
        self.sTrain = [[self.sSeqCreateRatlab(self.iTrain[0][0], seed=-1, use_RGB_images=True)]]

        self.iSeenid = self.iSeqCreateRatlab(ratlab_images, first_image_index=first_image_training,
                                             num_images_used=num_images_test // 4, data_base_dir=ratlab_data_base_dir)
        self.sSeenid = self.sSeqCreateRatlab(self.iSeenid, seed=-1, use_RGB_images=True)

        self.iNewid = [[self.iSeqCreateRatlab(ratlab_images, first_image_index=first_image_test,
                                              num_images_used=num_images_test, data_base_dir=ratlab_data_base_dir)]]
        self.sNewid = [[self.sSeqCreateRatlab(self.iNewid[0][0], seed=-1, use_RGB_images=True)]]

        print("len(self.iTrain[0][0].input_files)= %d" % len(self.iTrain[0][0].input_files))
        print("len(self.sTrain[0][0].input_files)= %d" % len(self.sTrain[0][0].input_files))

        self.name = "Function Based Data Creation for ratlab"
        self.network = "linearNetwork4L"  # Default Network, but ignored

    def iSeqCreateRatlab(self, ratlab_images, first_image_index=0, num_images_used=0, data_base_dir=""):
        if first_image_index + num_images_used > len(ratlab_images):
            err = "first_image_index + num_images_per_digit_used > len(ratlab_images." + \
                  "%d + %d > %d" % (first_image_index, num_images_used, len(ratlab_images))
            raise Exception(err)

        print("***** Setting Information Parameters for Ratlab ******")
        iSeq = system_parameters.ParamsInput()
        iSeq.data_base_dir = data_base_dir

        iSeq.ids = [data_base_dir + "/" + image_filename for image_filename in
                    ratlab_images[first_image_index: first_image_index + num_images_used]]
        iSeq.orig_labels = []
        iSeq.block_sizes = []

        iSeq.input_files = iSeq.ids
        iSeq.orig_labels = numpy.arange(first_image_index, first_image_index + num_images_used)
        iSeq.num_images = num_images_used

        iSeq.dx = 0.0
        iSeq.dy = 0.0
        iSeq.smin = 1.0
        iSeq.smax = 1.0
        iSeq.delta_rotation = None
        iSeq.contrast_enhance = None
        iSeq.pre_mirror_flags = False
        iSeq.obj_avg_std = 0.0
        iSeq.obj_std_min = 0.0
        iSeq.obj_std_max = 0.0

        iSeq.ages = [None]
        iSeq.genders = [None]
        iSeq.racetweens = [None]
        iSeq.expressions = [None]
        iSeq.morphs = [None]
        iSeq.poses = [None]
        iSeq.lightings = [None]
        iSeq.slow_signal = 0  # real slow signal is the translation in the x axis (correlated to identity),
        # added during image loading
        iSeq.step = 1
        iSeq.offset = 0

        iSeq.params = [iSeq.ids, iSeq.ages, iSeq.genders, iSeq.racetweens, iSeq.expressions,
                       iSeq.morphs, iSeq.poses, iSeq.lightings]

        iSeq.block_size = 1
        print("iSeq.block_size =" +  str(iSeq.block_size))
        iSeq.train_mode = "regular"
        iSeq.correct_labels = 1.0 * numpy.random.randint(0, 2, num_images_used)
        iSeq.correct_classes = iSeq.correct_labels.astype(int)
        system_parameters.test_object_contents(iSeq)
        return iSeq

    def sSeqCreateRatlab(self, iSeq, seed=-1, use_RGB_images=True):
        if seed >= 0 or seed is None:
            numpy.random.seed(seed)

        if iSeq is None:
            print("Ratlab experiment: iSeq was None, this might be an indication that the data is not available")
            sSeq = system_parameters.ParamsDataLoading()
            return sSeq

        print("******** Setting Training Data Parameters for Ratlab data  ****************")
        sSeq = system_parameters.ParamsDataLoading()
        sSeq.input_files = iSeq.input_files
        sSeq.num_images = iSeq.num_images
        sSeq.block_size = iSeq.block_size
        sSeq.train_mode = iSeq.train_mode
        sSeq.include_latest = iSeq.include_latest
        sSeq.image_width = 320
        sSeq.image_height = 40
        sSeq.subimage_width = 320
        sSeq.subimage_height = 40
        sSeq.pre_mirror_flags = iSeq.pre_mirror_flags

        sSeq.trans_x_max = iSeq.dx
        sSeq.trans_x_min = -1 * iSeq.dx
        sSeq.trans_y_max = iSeq.dy
        sSeq.trans_y_min = -1 * iSeq.dy
        sSeq.min_sampling = iSeq.smin
        sSeq.max_sampling = iSeq.smax
        sSeq.delta_rotation = iSeq.delta_rotation
        sSeq.contrast_enhance = iSeq.contrast_enhance
        sSeq.obj_avg_std = iSeq.obj_avg_std
        sSeq.obj_std_min = iSeq.obj_std_min
        sSeq.obj_std_max = iSeq.obj_std_max

        sSeq.add_noise_L0 = False
        if use_RGB_images:
            sSeq.convert_format = "RGB"  # "RGB", "L"
        else:
            sSeq.convert_format = "L"
        sSeq.background_type = None

        sSeq.offset_translation_x = 0.0
        sSeq.offset_translation_y = 0.0

        sSeq.translations_x = numpy.random.random_integers(sSeq.trans_x_min, sSeq.trans_x_max,
                                                           sSeq.num_images) + sSeq.offset_translation_x
        sSeq.translations_y = numpy.random.random_integers(sSeq.trans_y_min, sSeq.trans_y_max,
                                                           sSeq.num_images) + sSeq.offset_translation_y

        sSeq.pixelsampling_x = numpy.random.uniform(low=sSeq.min_sampling, high=sSeq.max_sampling, size=sSeq.num_images)
        sSeq.pixelsampling_y = sSeq.pixelsampling_x + 0.0
        if sSeq.delta_rotation is not None:
            sSeq.rotations = numpy.random.uniform(-sSeq.delta_rotation, sSeq.delta_rotation, sSeq.num_images)
        else:
            sSeq.rotations = None
        if iSeq.obj_avg_std > 0:
            sSeq.obj_avgs = numpy.random.normal(0.0, iSeq.obj_avg_std, size=sSeq.num_images)
        else:
            sSeq.obj_avgs = numpy.zeros(sSeq.num_images)
        sSeq.obj_stds = numpy.random.uniform(sSeq.obj_std_min, sSeq.obj_std_max, sSeq.num_images)

        # BUG1: image center is not computed that way!!! also half(width-1) computation is wrong!!!
        sSeq.subimage_first_row = sSeq.image_height / 2.0 - sSeq.subimage_height * sSeq.pixelsampling_y / 2.0
        sSeq.subimage_first_column = sSeq.image_width / 2.0 - sSeq.subimage_width * sSeq.pixelsampling_x / 2.0

        # TODO:check semantics, when is sampling/translation done? why does this value matter?
        sSeq.trans_sampled = True
        sSeq.name = "Ratlab_data"

        sSeq.load_data = load_data_from_sSeq
        system_parameters.test_object_contents(sSeq)
        return sSeq


ParamsRatlabFunc_320x40_training_80k = ParamsRatlabExperiment(experiment_seed, experiment_basedir, 80000, 40000)
ParamsRatlabFunc_320x40_training_40k = ParamsRatlabExperiment(experiment_seed, experiment_basedir, 40000, 40000)
ParamsRatlabFunc_320x40_training_20k = ParamsRatlabExperiment(experiment_seed, experiment_basedir, 20000, 40000)
ParamsRatlabFunc_320x40_training_10k = ParamsRatlabExperiment(experiment_seed, experiment_basedir, 10000, 40000)
ParamsRatlabFunc_320x40_training_5k = ParamsRatlabExperiment(experiment_seed, experiment_basedir, 5000, 40000)
ParamsRatlabFunc_320x40_training_2500 = ParamsRatlabExperiment(experiment_seed, experiment_basedir, 2500, 40000)
ParamsRatlabFunc_320x40_training_1250 = ParamsRatlabExperiment(experiment_seed, experiment_basedir, 1250, 40000)


