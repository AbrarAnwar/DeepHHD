import os
import io
import numpy as np
TAG_FLOAT = 202021.25

def read(path):
    if not isinstance(path, io.BufferedReader):
        if not isinstance(path, str):
            raise AssertionError(
                "Input [{p}] is not a string".format(p=path))
        if not os.path.isfile(path):
            raise AssertionError(
                "Path [{p}] does not exist".format(p=path))
        if not path.split('.')[-1] == 'flo':
            raise AssertionError(
                "File extension [flo] required, [{f}] given".format(f=path.split('.')[-1]))

        flo = open(path, 'rb')
    else:
        flo = path

    tag = np.frombuffer(flo.read(4), np.float32, count=1)[0]
    if not TAG_FLOAT == tag:
        raise AssertionError("Wrong Tag [{t}]".format(t=tag))

    width = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal width [{w}]".format(w=width))

    height = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal height [{h}]".format(h=height))

    nbands = 2
    tmp = np.frombuffer(flo.read(nbands * width * height * 4),
                        np.float32, count=nbands * width * height)
    flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    flo.close()

    return flow
