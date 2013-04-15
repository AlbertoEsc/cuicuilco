import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

f0 = plt.figure()
plt.suptitle("Evolution of Delta values vs Iterations")
  
p11 = plt.plot()
#plt.title("input dim = 64")

t=[100, 5000, 10000, 15000, 20000]
delta_sfa = [0.01504806]*len(t)
delta_rbm64 = [0.02877733, 0.02219387, 0.02199444, 0.02209395, 0.0221391]
delta_nsfa = [0.01419556]*len(t)
delta_nsfa_rbm64 = [0.0230359, 0.02129932, 0.02184318, 0.02193603, 0.0223157]
delta_rbm128 = [0.02510091, 0.02281067, 0.02193792, 0.02152184, 0.02149291]

plt.plot(t, delta_sfa, "o-r")
plt.plot(t, delta_nsfa, "o-m")
plt.plot(t, delta_rbm64, "o-g")
plt.plot(t, delta_nsfa_rbm64, "o-y")
plt.plot(t, delta_rbm128, "o-b")
plt.legend(["SFA", "NL-SFA", "RBM64 + SFA", "RBM64 + NL-SFA", "RBM128 + SFA"], loc=4)
plt.ylim(0, 0.03)
plt.xlabel("Number of iterations in RBM")
plt.ylabel("Delta values of the output")

plt.show()