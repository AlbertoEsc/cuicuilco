import numpy

def cumulative_score(ground_truth, estimation, largest_error, integer_rounding=True):
    if len(ground_truth) != len(estimation):
        er = "ground_truth and estimation have different number of elements"
        raise Exception(er)

    if integer_rounding:
        _estimation = numpy.rint(estimation)
    else:
        _estimation = estimation

    N_e_le_j = (numpy.absolute(_estimation-ground_truth) <= largest_error).sum()
    return N_e_le_j * 1.0 / len(ground_truth)

ground_truth = numpy.array([0, 1, 2, 5, 10, 14, 19, 25, 35, 28])
estimation = numpy.array([0.2, 1.1, 1.9, 5.6, 9.2, 15.5, 16.5, 28.9, 45, 18])
for largest_error in [0, 1, 2, 3, 4, 5]:
    cs = cumulative_score(ground_truth, estimation, largest_error, integer_rounding=True)
    print "cs(%d)="%largest_error,cs