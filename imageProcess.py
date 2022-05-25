from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int, Bool
from openmdao.main.api import Component, Assembly
import numpy as np
import time
import cv2

class RGBSplit(Component):

    def __init__(self):
        super(RGBSplit, self).__init__()
        self.add("frame_in", Array(iotype="in"))

        self.add("R", Array(iotype="out"))
        self.add("G", Array(iotype="out"))
        self.add("B", Array(iotype="out"))

    def execute(self):
        self.R = self.frame_in[:, :, 0]
        self.G = self.frame_in[:, :, 1]
        self.B = self.frame_in[:, :, 2]


class RGBmuxer(Component):

    def __init__(self):
        super(RGBmuxer, self).__init__()
        self.add("R", Array(iotype="in"))
        self.add("G", Array(iotype="in"))
        self.add("B", Array(iotype="in"))

        self.add("frame_out", Array(iotype="out"))

    def execute(self):
        m, n = self.R.shape
        self.frame_out = cv2.merge([self.R, self.G, self.B])


class CVwrapped(Component):

    def __init__(self, func, *args, **kwargs):
        super(CVwrapped, self).__init__()
        self.add("frame_in", Array(iotype="in"))
        self.add("frame_out", Array(iotype="out"))
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def execute(self):
        self.frame_out = self._func(self.frame_in, *self._args, **self._kwargs)


class Grayscale(CVwrapped):

    def __init__(self):
        super(Grayscale, self).__init__(cv2.cvtColor, cv2.COLOR_BGR2GRAY)


class equalizeContrast(CVwrapped):

    def __init__(self):
        super(equalizeContrast, self).__init__(cv2.equalizeHist)


class showBPMtext(Component):
    ready = Bool(False, iotype="in")
    bpm = Float(iotype="in")
    x = Int(iotype="in")
    y = Int(iotype="in")
    fps = Float(iotype="in")
    size = Float(iotype="in")
    n = Int(iotype="in")

    def __init__(self):
        super(showBPMtext, self).__init__()
        self.add("frame_in", Array(iotype="in"))
        self.add("frame_out", Array(iotype="out"))
        self.bpms = []

    def execute(self):
        self.bpms.append([time.time(), self.bpm])
        if self.ready:
            col = (0, 255, 0)
            text = "%0.1f bpm" % self.bpm
            tsize = 2
        else:
            col = (100, 255, 100)
            gap = (self.n - self.size) / self.fps
            text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            tsize = 1
        cv2.putText(self.frame_in, text,
                    (self.x, self.y), cv2.FONT_HERSHEY_PLAIN, tsize, col)
        self.frame_out = self.frame_in
