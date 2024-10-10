# automatically generated by the FlatBuffers compiler, do not modify

# namespace: wsock

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class ServerReply(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ServerReply()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsServerReply(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # ServerReply
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ServerReply
    def Message(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ServerReply
    def FileObject(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ada.comms.wsock.FileObject import FileObject

            obj = FileObject()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ServerReply
    def ReplyTo(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # ServerReply
    def Error(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ada.comms.wsock.Error import Error

            obj = Error()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None


def ServerReplyStart(builder):
    builder.StartObject(4)


def Start(builder):
    ServerReplyStart(builder)


def ServerReplyAddMessage(builder, message):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(message), 0)


def AddMessage(builder, message):
    ServerReplyAddMessage(builder, message)


def ServerReplyAddFileObject(builder, fileObject):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(fileObject), 0)


def AddFileObject(builder, fileObject):
    ServerReplyAddFileObject(builder, fileObject)


def ServerReplyAddReplyTo(builder, replyTo):
    builder.PrependInt8Slot(2, replyTo, 0)


def AddReplyTo(builder, replyTo):
    ServerReplyAddReplyTo(builder, replyTo)


def ServerReplyAddError(builder, error):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(error), 0)


def AddError(builder, error):
    ServerReplyAddError(builder, error)


def ServerReplyEnd(builder):
    return builder.EndObject()


def End(builder):
    return ServerReplyEnd(builder)
