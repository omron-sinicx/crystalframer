import numpy

def to_uint8(scalar:int):
    return numpy.array([scalar], dtype=numpy.uint8)
def to_uint16(scalar:int):
    return numpy.array([scalar], dtype=numpy.uint16)
def to_uint32(scalar:int):
    return numpy.array([scalar], dtype=numpy.uint32)
def to_uint64(scalar:int):
    return numpy.array([scalar], dtype=numpy.uint64)

def to_int8(scalar:int):
    return numpy.array([scalar], dtype=numpy.int8)
def to_int16(scalar:int):
    return numpy.array([scalar], dtype=numpy.int16)
def to_int32(scalar:int):
    return numpy.array([scalar], dtype=numpy.int32)

def to_float32(scalar:float):
    return numpy.array([scalar], dtype=numpy.float32)
