# automatically generated by the FlatBuffers compiler, do not modify

# namespace: wsock

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class Procedure(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Procedure()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsProcedure(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # Procedure
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Procedure
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Procedure
    def Description(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Procedure
    def ScriptFileLocation(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Procedure
    def Parameters(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ada.comms.wsock.Parameter import Parameter

            obj = Parameter()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Procedure
    def ParametersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Procedure
    def ParametersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # Procedure
    def InputFileVar(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Procedure
    def InputFileType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Procedure
    def ExportFileType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Procedure
    def ExportFileVar(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Procedure
    def State(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Procedure
    def IsComponent(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False


def ProcedureStart(builder):
    builder.StartObject(10)


def Start(builder):
    ProcedureStart(builder)


def ProcedureAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)


def AddName(builder, name):
    ProcedureAddName(builder, name)


def ProcedureAddDescription(builder, description):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(description), 0)


def AddDescription(builder, description):
    ProcedureAddDescription(builder, description)


def ProcedureAddScriptFileLocation(builder, scriptFileLocation):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(scriptFileLocation), 0)


def AddScriptFileLocation(builder, scriptFileLocation):
    ProcedureAddScriptFileLocation(builder, scriptFileLocation)


def ProcedureAddParameters(builder, parameters):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(parameters), 0)


def AddParameters(builder, parameters):
    ProcedureAddParameters(builder, parameters)


def ProcedureStartParametersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def StartParametersVector(builder, numElems):
    return ProcedureStartParametersVector(builder, numElems)


def ProcedureAddInputFileVar(builder, inputFileVar):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(inputFileVar), 0)


def AddInputFileVar(builder, inputFileVar):
    ProcedureAddInputFileVar(builder, inputFileVar)


def ProcedureAddInputFileType(builder, inputFileType):
    builder.PrependInt8Slot(5, inputFileType, 0)


def AddInputFileType(builder, inputFileType):
    ProcedureAddInputFileType(builder, inputFileType)


def ProcedureAddExportFileType(builder, exportFileType):
    builder.PrependInt8Slot(6, exportFileType, 0)


def AddExportFileType(builder, exportFileType):
    ProcedureAddExportFileType(builder, exportFileType)


def ProcedureAddExportFileVar(builder, exportFileVar):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(exportFileVar), 0)


def AddExportFileVar(builder, exportFileVar):
    ProcedureAddExportFileVar(builder, exportFileVar)


def ProcedureAddState(builder, state):
    builder.PrependInt8Slot(8, state, 0)


def AddState(builder, state):
    ProcedureAddState(builder, state)


def ProcedureAddIsComponent(builder, isComponent):
    builder.PrependBoolSlot(9, isComponent, 0)


def AddIsComponent(builder, isComponent):
    ProcedureAddIsComponent(builder, isComponent)


def ProcedureEnd(builder):
    return builder.EndObject()


def End(builder):
    return ProcedureEnd(builder)
