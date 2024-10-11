# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Dl

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Graph(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Graph()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsGraph(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Graph
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Graph
    def Node(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.Node import Node
            obj = Node()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def NodeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def NodeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Graph
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Graph
    def Initializer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def InitializerLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def InitializerIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Graph
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Graph
    def Input(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.ValueInfo import ValueInfo
            obj = ValueInfo()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def InputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def InputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # Graph
    def Output(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.ValueInfo import ValueInfo
            obj = ValueInfo()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def OutputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def OutputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Graph
    def ValueInfo(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.ValueInfo import ValueInfo
            obj = ValueInfo()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def ValueInfoLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def ValueInfoIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # Graph
    def QuantizationAnnotation(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.TensorAnnotation import TensorAnnotation
            obj = TensorAnnotation()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def QuantizationAnnotationLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def QuantizationAnnotationIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # Graph
    def TestInputsValue(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def TestInputsValueLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def TestInputsValueIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

    # Graph
    def TestOutputsValue(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def TestOutputsValueLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def TestOutputsValueIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

def GraphStart(builder):
    builder.StartObject(10)

def Start(builder):
    GraphStart(builder)

def GraphAddNode(builder, node):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(node), 0)

def AddNode(builder, node):
    GraphAddNode(builder, node)

def GraphStartNodeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodeVector(builder, numElems):
    return GraphStartNodeVector(builder, numElems)

def GraphAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    GraphAddName(builder, name)

def GraphAddInitializer(builder, initializer):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(initializer), 0)

def AddInitializer(builder, initializer):
    GraphAddInitializer(builder, initializer)

def GraphStartInitializerVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInitializerVector(builder, numElems):
    return GraphStartInitializerVector(builder, numElems)

def GraphAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    GraphAddDocString(builder, docString)

def GraphAddInput(builder, input):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(input), 0)

def AddInput(builder, input):
    GraphAddInput(builder, input)

def GraphStartInputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputVector(builder, numElems):
    return GraphStartInputVector(builder, numElems)

def GraphAddOutput(builder, output):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(output), 0)

def AddOutput(builder, output):
    GraphAddOutput(builder, output)

def GraphStartOutputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOutputVector(builder, numElems):
    return GraphStartOutputVector(builder, numElems)

def GraphAddValueInfo(builder, valueInfo):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(valueInfo), 0)

def AddValueInfo(builder, valueInfo):
    GraphAddValueInfo(builder, valueInfo)

def GraphStartValueInfoVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartValueInfoVector(builder, numElems):
    return GraphStartValueInfoVector(builder, numElems)

def GraphAddQuantizationAnnotation(builder, quantizationAnnotation):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(quantizationAnnotation), 0)

def AddQuantizationAnnotation(builder, quantizationAnnotation):
    GraphAddQuantizationAnnotation(builder, quantizationAnnotation)

def GraphStartQuantizationAnnotationVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartQuantizationAnnotationVector(builder, numElems):
    return GraphStartQuantizationAnnotationVector(builder, numElems)

def GraphAddTestInputsValue(builder, testInputsValue):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(testInputsValue), 0)

def AddTestInputsValue(builder, testInputsValue):
    GraphAddTestInputsValue(builder, testInputsValue)

def GraphStartTestInputsValueVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartTestInputsValueVector(builder, numElems):
    return GraphStartTestInputsValueVector(builder, numElems)

def GraphAddTestOutputsValue(builder, testOutputsValue):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(testOutputsValue), 0)

def AddTestOutputsValue(builder, testOutputsValue):
    GraphAddTestOutputsValue(builder, testOutputsValue)

def GraphStartTestOutputsValueVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartTestOutputsValueVector(builder, numElems):
    return GraphStartTestOutputsValueVector(builder, numElems)

def GraphEnd(builder):
    return builder.EndObject()

def End(builder):
    return GraphEnd(builder)

import FlatBuffers.Dl.Node
import FlatBuffers.Dl.Tensor
import FlatBuffers.Dl.TensorAnnotation
import FlatBuffers.Dl.ValueInfo
try:
    from typing import List
except:
    pass

class GraphT(object):

    # GraphT
    def __init__(self):
        self.node = None  # type: List[FlatBuffers.Dl.Node.NodeT]
        self.name = None  # type: str
        self.initializer = None  # type: List[FlatBuffers.Dl.Tensor.TensorT]
        self.docString = None  # type: str
        self.input = None  # type: List[FlatBuffers.Dl.ValueInfo.ValueInfoT]
        self.output = None  # type: List[FlatBuffers.Dl.ValueInfo.ValueInfoT]
        self.valueInfo = None  # type: List[FlatBuffers.Dl.ValueInfo.ValueInfoT]
        self.quantizationAnnotation = None  # type: List[FlatBuffers.Dl.TensorAnnotation.TensorAnnotationT]
        self.testInputsValue = None  # type: List[FlatBuffers.Dl.Tensor.TensorT]
        self.testOutputsValue = None  # type: List[FlatBuffers.Dl.Tensor.TensorT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        graph = Graph()
        graph.Init(buf, pos)
        return cls.InitFromObj(graph)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, graph):
        x = GraphT()
        x._UnPack(graph)
        return x

    # GraphT
    def _UnPack(self, graph):
        if graph is None:
            return
        if not graph.NodeIsNone():
            self.node = []
            for i in range(graph.NodeLength()):
                if graph.Node(i) is None:
                    self.node.append(None)
                else:
                    node_ = FlatBuffers.Dl.Node.NodeT.InitFromObj(graph.Node(i))
                    self.node.append(node_)
        self.name = graph.Name()
        if not graph.InitializerIsNone():
            self.initializer = []
            for i in range(graph.InitializerLength()):
                if graph.Initializer(i) is None:
                    self.initializer.append(None)
                else:
                    tensor_ = FlatBuffers.Dl.Tensor.TensorT.InitFromObj(graph.Initializer(i))
                    self.initializer.append(tensor_)
        self.docString = graph.DocString()
        if not graph.InputIsNone():
            self.input = []
            for i in range(graph.InputLength()):
                if graph.Input(i) is None:
                    self.input.append(None)
                else:
                    valueInfo_ = FlatBuffers.Dl.ValueInfo.ValueInfoT.InitFromObj(graph.Input(i))
                    self.input.append(valueInfo_)
        if not graph.OutputIsNone():
            self.output = []
            for i in range(graph.OutputLength()):
                if graph.Output(i) is None:
                    self.output.append(None)
                else:
                    valueInfo_ = FlatBuffers.Dl.ValueInfo.ValueInfoT.InitFromObj(graph.Output(i))
                    self.output.append(valueInfo_)
        if not graph.ValueInfoIsNone():
            self.valueInfo = []
            for i in range(graph.ValueInfoLength()):
                if graph.ValueInfo(i) is None:
                    self.valueInfo.append(None)
                else:
                    valueInfo_ = FlatBuffers.Dl.ValueInfo.ValueInfoT.InitFromObj(graph.ValueInfo(i))
                    self.valueInfo.append(valueInfo_)
        if not graph.QuantizationAnnotationIsNone():
            self.quantizationAnnotation = []
            for i in range(graph.QuantizationAnnotationLength()):
                if graph.QuantizationAnnotation(i) is None:
                    self.quantizationAnnotation.append(None)
                else:
                    tensorAnnotation_ = FlatBuffers.Dl.TensorAnnotation.TensorAnnotationT.InitFromObj(graph.QuantizationAnnotation(i))
                    self.quantizationAnnotation.append(tensorAnnotation_)
        if not graph.TestInputsValueIsNone():
            self.testInputsValue = []
            for i in range(graph.TestInputsValueLength()):
                if graph.TestInputsValue(i) is None:
                    self.testInputsValue.append(None)
                else:
                    tensor_ = FlatBuffers.Dl.Tensor.TensorT.InitFromObj(graph.TestInputsValue(i))
                    self.testInputsValue.append(tensor_)
        if not graph.TestOutputsValueIsNone():
            self.testOutputsValue = []
            for i in range(graph.TestOutputsValueLength()):
                if graph.TestOutputsValue(i) is None:
                    self.testOutputsValue.append(None)
                else:
                    tensor_ = FlatBuffers.Dl.Tensor.TensorT.InitFromObj(graph.TestOutputsValue(i))
                    self.testOutputsValue.append(tensor_)

    # GraphT
    def Pack(self, builder):
        if self.node is not None:
            nodelist = []
            for i in range(len(self.node)):
                nodelist.append(self.node[i].Pack(builder))
            GraphStartNodeVector(builder, len(self.node))
            for i in reversed(range(len(self.node))):
                builder.PrependUOffsetTRelative(nodelist[i])
            node = builder.EndVector()
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.initializer is not None:
            initializerlist = []
            for i in range(len(self.initializer)):
                initializerlist.append(self.initializer[i].Pack(builder))
            GraphStartInitializerVector(builder, len(self.initializer))
            for i in reversed(range(len(self.initializer))):
                builder.PrependUOffsetTRelative(initializerlist[i])
            initializer = builder.EndVector()
        if self.docString is not None:
            docString = builder.CreateString(self.docString)
        if self.input is not None:
            inputlist = []
            for i in range(len(self.input)):
                inputlist.append(self.input[i].Pack(builder))
            GraphStartInputVector(builder, len(self.input))
            for i in reversed(range(len(self.input))):
                builder.PrependUOffsetTRelative(inputlist[i])
            input = builder.EndVector()
        if self.output is not None:
            outputlist = []
            for i in range(len(self.output)):
                outputlist.append(self.output[i].Pack(builder))
            GraphStartOutputVector(builder, len(self.output))
            for i in reversed(range(len(self.output))):
                builder.PrependUOffsetTRelative(outputlist[i])
            output = builder.EndVector()
        if self.valueInfo is not None:
            valueInfolist = []
            for i in range(len(self.valueInfo)):
                valueInfolist.append(self.valueInfo[i].Pack(builder))
            GraphStartValueInfoVector(builder, len(self.valueInfo))
            for i in reversed(range(len(self.valueInfo))):
                builder.PrependUOffsetTRelative(valueInfolist[i])
            valueInfo = builder.EndVector()
        if self.quantizationAnnotation is not None:
            quantizationAnnotationlist = []
            for i in range(len(self.quantizationAnnotation)):
                quantizationAnnotationlist.append(self.quantizationAnnotation[i].Pack(builder))
            GraphStartQuantizationAnnotationVector(builder, len(self.quantizationAnnotation))
            for i in reversed(range(len(self.quantizationAnnotation))):
                builder.PrependUOffsetTRelative(quantizationAnnotationlist[i])
            quantizationAnnotation = builder.EndVector()
        if self.testInputsValue is not None:
            testInputsValuelist = []
            for i in range(len(self.testInputsValue)):
                testInputsValuelist.append(self.testInputsValue[i].Pack(builder))
            GraphStartTestInputsValueVector(builder, len(self.testInputsValue))
            for i in reversed(range(len(self.testInputsValue))):
                builder.PrependUOffsetTRelative(testInputsValuelist[i])
            testInputsValue = builder.EndVector()
        if self.testOutputsValue is not None:
            testOutputsValuelist = []
            for i in range(len(self.testOutputsValue)):
                testOutputsValuelist.append(self.testOutputsValue[i].Pack(builder))
            GraphStartTestOutputsValueVector(builder, len(self.testOutputsValue))
            for i in reversed(range(len(self.testOutputsValue))):
                builder.PrependUOffsetTRelative(testOutputsValuelist[i])
            testOutputsValue = builder.EndVector()
        GraphStart(builder)
        if self.node is not None:
            GraphAddNode(builder, node)
        if self.name is not None:
            GraphAddName(builder, name)
        if self.initializer is not None:
            GraphAddInitializer(builder, initializer)
        if self.docString is not None:
            GraphAddDocString(builder, docString)
        if self.input is not None:
            GraphAddInput(builder, input)
        if self.output is not None:
            GraphAddOutput(builder, output)
        if self.valueInfo is not None:
            GraphAddValueInfo(builder, valueInfo)
        if self.quantizationAnnotation is not None:
            GraphAddQuantizationAnnotation(builder, quantizationAnnotation)
        if self.testInputsValue is not None:
            GraphAddTestInputsValue(builder, testInputsValue)
        if self.testOutputsValue is not None:
            GraphAddTestOutputsValue(builder, testOutputsValue)
        graph = GraphEnd(builder)
        return graph