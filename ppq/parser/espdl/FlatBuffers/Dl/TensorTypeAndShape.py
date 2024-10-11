# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Dl

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TensorTypeAndShape(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TensorTypeAndShape()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensorTypeAndShape(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # TensorTypeAndShape
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TensorTypeAndShape
    def ElemType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TensorTypeAndShape
    def Shape(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from FlatBuffers.Dl.TensorShape import TensorShape
            obj = TensorShape()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TensorTypeAndShapeStart(builder):
    builder.StartObject(2)

def Start(builder):
    TensorTypeAndShapeStart(builder)

def TensorTypeAndShapeAddElemType(builder, elemType):
    builder.PrependInt32Slot(0, elemType, 0)

def AddElemType(builder, elemType):
    TensorTypeAndShapeAddElemType(builder, elemType)

def TensorTypeAndShapeAddShape(builder, shape):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(shape), 0)

def AddShape(builder, shape):
    TensorTypeAndShapeAddShape(builder, shape)

def TensorTypeAndShapeEnd(builder):
    return builder.EndObject()

def End(builder):
    return TensorTypeAndShapeEnd(builder)

import FlatBuffers.Dl.TensorShape
try:
    from typing import Optional
except:
    pass

class TensorTypeAndShapeT(object):

    # TensorTypeAndShapeT
    def __init__(self):
        self.elemType = 0  # type: int
        self.shape = None  # type: Optional[FlatBuffers.Dl.TensorShape.TensorShapeT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        tensorTypeAndShape = TensorTypeAndShape()
        tensorTypeAndShape.Init(buf, pos)
        return cls.InitFromObj(tensorTypeAndShape)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, tensorTypeAndShape):
        x = TensorTypeAndShapeT()
        x._UnPack(tensorTypeAndShape)
        return x

    # TensorTypeAndShapeT
    def _UnPack(self, tensorTypeAndShape):
        if tensorTypeAndShape is None:
            return
        self.elemType = tensorTypeAndShape.ElemType()
        if tensorTypeAndShape.Shape() is not None:
            self.shape = FlatBuffers.Dl.TensorShape.TensorShapeT.InitFromObj(tensorTypeAndShape.Shape())

    # TensorTypeAndShapeT
    def Pack(self, builder):
        if self.shape is not None:
            shape = self.shape.Pack(builder)
        TensorTypeAndShapeStart(builder)
        TensorTypeAndShapeAddElemType(builder, self.elemType)
        if self.shape is not None:
            TensorTypeAndShapeAddShape(builder, shape)
        tensorTypeAndShape = TensorTypeAndShapeEnd(builder)
        return tensorTypeAndShape