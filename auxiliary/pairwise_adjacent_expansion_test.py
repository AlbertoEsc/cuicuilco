from  eban_SFA_libs import *


multiply2 = lambda x: pairwise_adjacent_expansion(x, adj=2, func=multiply, reflexive=True)


print "*************************************************************************"
print "***     TESTING BASIC FUNCTIONALITY OF PAIRWISE_ADJACENT_EXPANSION    ***"
print "*************************************************************************"

x = numpy.array([[1., 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]])
print x

adjs=range(1,4)
reflexives = [True, False]
funcs = [multiply]

for adj in adjs:
    for reflexive in reflexives:
        for func in funcs:
            x_exp = pairwise_adjacent_expansion(x, adj, func, reflexive)
            print "adj=%d, reflexive=%s, func=%s"%(adj, str(reflexive), str(func))
            print "x_exp=", x_exp


x_exp2 = pair_prod_adj3_ex(x)
print "x_exp2=", x_exp2

x_exp3 = pair_sqrt_abs_dif_adj3_ex(x)
print "x_exp3=", x_exp3

print "*************************************************************************"
print "***            BENCHMARKING    PAIRWISE_ADJACENT_EXPANSION... WORK PENDING!!!           ***"
print "*************************************************************************"

#x = numpy.array([[1., 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]])
#print x
#
#adjs=range(1,4)
#reflexives = [True, False]
#funcs = [multiply]
#
#for adj in adjs:
#    for reflexive in reflexives:
#        for func in funcs:
#            x_exp = pairwise_adjacent_expansion(x, adj, func, reflexive)
#            print "adj=%d, reflexive=%s, func=%s"%(adj, str(reflexive), str(func))
#            print "x_exp=", x_exp
#
#
#    
#    
#    
#    
#    