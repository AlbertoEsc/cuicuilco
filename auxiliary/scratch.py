import numpy
import Image

def mark_center_pixels(image, colorstr):
    im_width, im_height = image.size
    if image.mode == "RGB":
        if colorstr == "green":
            color = (0,255,0)
        elif colorstr == "blue":
            color = (0,0,255)
        elif colorstr == "white":
            color = (255,255,255)
    else:
        if colorstr == "black":
            color = 0
        elif colorstr == "gray":
            color = 127
        elif colorstr == "white":
            color = 255
        
    cx = (im_width-1)/2
    cy = (im_height-1)/2
    image.putpixel((cx,cy),color)
    if im_width & 1 == 0:
        image.putpixel((cx+1,cy),color)
    if im_height & 1 == 0:
        image.putpixel((cx,cy+1),color)
    if im_width & 1 == 0 and im_height & 1 == 0:
        image.putpixel((cx+1,cy+1),color)

im_width = 250
im_height = 250
scale = 5

test_im_ar = numpy.zeros((im_height,im_width),dtype="float32") #+0.01
for px in range(im_width):
    for py in range(im_height):
        pc = ((px+py+1)%3)/2.0
        test_im_ar[py,px] = pc * 255

#test_im = Image.fromarray(test_im_ar) #, mode="L")
test_im = Image.fromarray(numpy.uint8(test_im_ar))
mark_center_pixels(test_im, "white")

for r in range(10000):
    test_im = test_im.transpose(Image.FLIP_LEFT_RIGHT)
test_im_flip = test_im.transpose(Image.FLIP_LEFT_RIGHT)

im_out_flip = test_im_flip.transform((im_width*scale, im_height*scale), Image.EXTENT, (0,0,im_width,im_height), Image.NEAREST) 
im_out_flip.save("flipped_image.png")

    
quit()


from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorLocator   = MultipleLocator(20)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(5)


t = arange(0.0, 100.0, 0.1)
s = sin(0.1*pi*t)*exp(-t*0.01)

ax = subplot(111)
plot(t,s)

ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)

#for the minor ticks, use no labels; default NullFormatter
ax.xaxis.set_minor_locator(minorLocator)
ax.grid(True)
show()
