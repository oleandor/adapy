# automatically generated by the FlatBuffers compiler, do not modify

# namespace: wsock

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class ProcedureStart(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ProcedureStart()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsProcedureStart(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # ProcedureStart
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ProcedureStart
    def ProcedureName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ProcedureStart
    def ProcedureIdString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ProcedureStart
    def Parameters(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ada.comms.wsock.Parameter import Parameter

            obj = Parameter()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ProcedureStart
    def ParametersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ProcedureStart
    def ParametersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0


def ProcedureStartStart(builder):
    builder.StartObject(3)


def Start(builder):
    ProcedureStartStart(builder)


def ProcedureStartAddProcedureName(builder, procedureName):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(procedureName), 0)


def AddProcedureName(builder, procedureName):
    ProcedureStartAddProcedureName(builder, procedureName)


def ProcedureStartAddProcedureIdString(builder, procedureIdString):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(procedureIdString), 0)


def AddProcedureIdString(builder, procedureIdString):
    ProcedureStartAddProcedureIdString(builder, procedureIdString)


def ProcedureStartAddParameters(builder, parameters):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(parameters), 0)


def AddParameters(builder, parameters):
    ProcedureStartAddParameters(builder, parameters)


def ProcedureStartStartParametersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def StartParametersVector(builder, numElems):
    return ProcedureStartStartParametersVector(builder, numElems)


def ProcedureStartEnd(builder):
    return builder.EndObject()


def End(builder):
    return ProcedureStartEnd(builder)
