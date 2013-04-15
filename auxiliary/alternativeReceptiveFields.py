import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image

factor_local = 0.4
factor_sparse = 0.2
num_layers = 9
all_l = []

l0 = numpy.zeros((3,3))
l0[0][0] = 1.0
l0[1][1] = 1.0
l0[2][2] = 1.0
print "l0=", l0
all_l.append(l0)

med = len(l0[0])
size = 2 * med
l1 = numpy.zeros((3,size))
n1= size / 2
for sig in range(3):
    for i, x in enumerate(l0[sig]):
        l1[sig][i*2] += x * factor_local
        l1[sig][i*2+1] += x * factor_local
        l1[sig][(n1+i*2)%size] += x * factor_sparse
    l1[sig] /= l1[sig].sum()
print "l1=", l1
all_l.append(l1)

prev_l = l1
prev_n = n1
for rep in range(num_layers-2):    
    med = len(prev_l[0])
    size = 2 * med
    new_l = numpy.zeros((3,size))
    new_n= prev_n*2+2
    for sig in range(3):
        for i, x in enumerate(prev_l[sig]):
            new_l[sig][i*2] += x * factor_local
            new_l[sig][i*2+1] += x * factor_local
            new_l[sig][(new_n+i*2)%size] += x * factor_sparse
        new_l[sig] /= new_l[sig].sum()
    print "l%d="%(rep+2), new_l
    all_l.append(new_l)
    prev_l = new_l
    prev_n = new_n
    

print "************ Displaying First Receptive Field **************"
#Create Figure
f0 = plt.figure()
plt.suptitle("Receptive Fields of First Unit at several layers. L0(3x3)")
  
layers = []

for i in range(num_layers):
    layers.append(all_l[i][0].reshape((3 * 2**i,1)) * all_l[i][0].reshape((1,3 * 2**i)))

for i, layer in enumerate(layers):
    ax_tmp = plt.subplot(3,3,1+i)
    plt.title("Layer %d (Reversed Order)"%i)
#    ax_tmp.imshow(layer, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
    ax_tmp.imshow(layer, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)


print "Displaying square receptive fields"
num_layers = 9
all_l = []

l0 = numpy.zeros((2,2))
l0[0][0] = 1.0
l0[1][1] = 1.0
print "l0=", l0
all_l.append(l0)

med = len(l0[0])
size = 2 * med
l1 = numpy.zeros((2,size))
n1= size / 2
for sig in range(2):
    for i, x in enumerate(l0[sig]):
        l1[sig][i*2] += x * factor_local
        l1[sig][i*2+1] += x * factor_local
        l1[sig][(n1+i*2)%size] += x * factor_sparse
    l1[sig] /= l1[sig].sum()
print "l1=", l1
all_l.append(l1)

prev_l = l1
prev_n = n1
for rep in range(num_layers-2):    
    med = len(prev_l[0])
    size = 2 * med
    new_l = numpy.zeros((2,size))
    new_n= prev_n*2+2
    for sig in range(2):
        for i, x in enumerate(prev_l[sig]):
            new_l[sig][i*2] += x * factor_local
            new_l[sig][i*2+1] += x * factor_local 
            new_l[sig][(new_n+i*2)%size] += x * factor_sparse
        new_l[sig] /= new_l[sig].sum()
    print "l%d="%(rep+2), new_l
    all_l.append(new_l)
    prev_l = new_l
    prev_n = new_n
    

print "************ Displaying SQUARE Receptive Field **************"
#Create Figure
f0 = plt.figure()
plt.suptitle("Receptive Fields of First Unit (Square) at several layers L0(2x2)")
  
layers = []

for i in range(num_layers):
    layers.append(all_l[i][0].reshape((2 * 2**i,1)) * all_l[i][0].reshape((1,2 * 2**i)))

for i, layer in enumerate(layers):
    ax_tmp = plt.subplot(3,4,1+i)
    plt.title("Layer %d (Reversed Order)"%i)
#    ax_tmp.imshow(layer, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
    ax_tmp.imshow(layer, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)









print "GUI Created, showing!!!!"
plt.show()
print "GUI Finished!"