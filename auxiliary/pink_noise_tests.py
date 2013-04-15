from  imageLoader import *

##EXAMPLE of Pink Noise Geration:
size = (128,64)
alphas = [8.0, 6.0, 5.0, 4.0, 
          3.0, 2.0, 1.5, 1.25, 
          1.0, 0.75, 0.5, 0.25]
amp = 25
mean = 127.5
plt.figure()
plt.suptitle("Pink Noise. Amplitude = k1 * 1 / f^(alpha/2), Energy = k2 * 1 / f^alpha")

for i, alpha in enumerate(alphas):
    plt.subplot(3,4,i+1)
    filter = filter_colored_noise2D_imp(size, alpha) # 1/f^(3/2)
    yy =  random_filtered_noise2D(size, filter)
    yy4 = change_mean(yy, mean, amp)
#    yy2 = (yy - yy.mean()) 
#    yy3 = yy2 / yy2.std()
#    yy4 = yy3 * amp + mean
    plt.imshow(yy4, vmin=0, vmax=255, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)  
    plt.xlabel("Alpha=%0.1f, std=%0.1f, mean=%0.1f"%(alpha, amp, mean))

plt.show()