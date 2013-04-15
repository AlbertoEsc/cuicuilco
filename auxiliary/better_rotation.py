import Image as Im

if True:
    def rotate(self, angle, resample=Im.NEAREST, expand=0):
        "Rotate image.  Angle given as degrees counter-clockwise."

        if expand:
            import math
            angle = -angle * math.pi / 180
            matrix = [
                 math.cos(angle), math.sin(angle), 0.0,
                -math.sin(angle), math.cos(angle), 0.0
                 ]
            def transform(x, y, (a, b, c, d, e, f)=matrix):
                return a*x + b*y + c, d*x + e*y + f

            # calculate output size
            w0, h0 = w, h = self.size
            xx = []
            yy = []
            for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
                x, y = transform(x, y)
                xx.append(x)
                yy.append(y)
            w = int(math.ceil(max(xx)) - math.floor(min(xx)))
            if w & 1 == 0:
                w += 1
            h = int(math.ceil(max(yy)) - math.floor(min(yy)))
            if h & 1 == 0:
                h += 1

            # adjust center
            x, y = transform(w / 2.0, h / 2.0)
            matrix[2] = w0 / 2.0 - x
            matrix[5] = h0 / 2.0 - y

            return self.transform((w, h), Im.AFFINE, matrix)

        if resample not in (Im.NEAREST, Im.BILINEAR, Im.BICUBIC):
            raise ValueError("unknown resampling filter")

        self.load()

        if self.mode in ("1", "P"):
            resample = Im.NEAREST

        return self._new(self.im.rotate(angle, resample))