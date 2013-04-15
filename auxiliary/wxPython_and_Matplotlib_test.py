import wx
import pylab
from matplotlib.numerix import arange, sin, cos, pi
import matplotlib

matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
matplotlib.interactive(False)

class App(wx.App):
    def OnInit(self):
        self.frame = MainFrame("BioXtas - Autoplotter", (50,60), (700,700))
        self.frame.Show()
        return True

class MainFrame(wx.Frame):
    def __init__(self, title, pos, size):
        wx.Frame.__init__(self, None, -1, title, pos, size)
        
        pPanel = PlotPanel(self, -1) # Plot panel
        
        bPanel = ButtonPanel(self, 100,500, (200,100)) # button        panel
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        sizer.Add(pPanel,0)
        sizer.Add(bPanel,0)
        
        self.SetSizer(sizer)

class ButtonPanel(wx.Panel):

    def __init__(self, Parent, xPos, yPos, insize):
    
        pos = (xPos, yPos)
        wx.Panel.__init__(self, Parent, -1, pos, style =
        wx.RAISED_BORDER, size = insize)
        
        button = wx.Button(self, -1, 'HELLO!!', (10,10), (150,50))

class NoRepaintCanvas(FigureCanvasWxAgg):
    """We subclass FigureCanvasWxAgg, overriding the _onPaint method,
    so that
    the draw method is only called for the first two paint events.
    After that,
    the canvas will only be redrawn when it is resized.
    """
    def __init__(self, *args, **kwargs):
        FigureCanvasWxAgg.__init__(self, *args, **kwargs)
        self._drawn = 0
    
    def _onPaint(self, evt):
        """
        Called when wxPaintEvt is generated
        """
#        if not self._isRealized:
#            self.realize()
    
        if self._drawn < 2:
            #self.draw(repaint = False)
            self._drawn += 1
        
            self.gui_repaint(drawDC=wx.PaintDC(self))


class PlotPanel(wx.Panel):

    def __init__(self, parent, id = -1, color = None,\
        dpi = None, style = wx.NO_FULL_REPAINT_ON_RESIZE,
        **kwargs):
        
        wx.Panel.__init__(self, parent, id = id, style = style,
        **kwargs)
        
        self.figure = Figure(None, dpi)
        self.canvas = NoRepaintCanvas(self, -1, self.figure)
        self._resizeflag = True
        
        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)
        
        self._SetSize()

    def draw(self):
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        theta = arange(0, 45*2*pi, 0.02)
        rad = (0.8*theta/(2*pi)+1)
        r = rad*(8 + sin(theta*7+rad/1.8))
        x = r*cos(theta)
        y = r*sin(theta)
        #Now draw it
        self.subplot.plot(x,y, '-r')
        
    def _onSize(self, event):
        self._resizeflag = True
    
    def _onIdle(self, evt):
        if self._resizeflag:
            self._resizeflag = False
            self._SetSize()
            self.draw()
    
    def _SetSize(self, pixels = None):
        """
        This method can be called to force the Plot to be a desired
        size, which defaults to
        the ClientSize of the panel
        """
        if not pixels:
            pixels = self.GetClientSize()
            self.canvas.SetSize(pixels)
            self.figure.set_size_inches(pixels[0]/
            self.figure.get_dpi(),
            pixels[1]/self.figure.get_dpi())


if __name__ == "__main__":

    app = App(0)
    app.MainLoop()