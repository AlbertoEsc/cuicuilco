import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

f0 = plt.figure()
plt.suptitle("ROC Curves (k-largest-faces heuristic)")
  
#p11 = plt.plot()
#plt.title("input dim = 64")
RR = {}
FAR = {}

#BioID Database
ks= [numpy.inf, 6, 5, 4, 3, 2, 1]
RR = [0.961598, 0.961598, 0.960061, 0.955, 0.9210, 0.851, 0.5975]
FAR = [3.782, 2.954, 2.614, 2.193, 1.659, 1.0246, 0.39]

RR2 = [0.961598, 0.961598, 0.960061, 0.955, 0.9210, 0.851, 0.5975]
FAR2 = [5.2, 3.782, 2.954, 2.614, 2.193, 1.659, 1.0246]

pltfunc = plt.plot #plt.plot, plt.semilogy, plt.loglog, plt.semilogx

legends = []
for i, k in enumerate(ks):
    pltfunc(FAR[i], RR[i], "o")
    legends.append("k=%s"%str(k))

plt.grid(True)
plt.legend(legends, loc=4)

plt.ylim(0, 1.0)
plt.xlim(0, 5.0)
plt.xlabel("False positives per each complete image")
plt.ylabel("True positives per face")

plt.figure()
plt.plot(FAR, RR, "*-b")
plt.plot(FAR2, RR2, "^-g")
plt.legend(["alg 1", "alg 2"], loc=2)

plt.show()