from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import ada
from ada.cadit.sat.utils import make_ints_if_possible


# ------------------------
# Base ACIS Entity
# ------------------------
@dataclass
class SATEntity:
    id: int

    def to_string(self) -> str:
        raise NotImplementedError


# ------------------------
#   Body / Lump / Shell
# ------------------------
@dataclass
class Body(SATEntity):
    lump: Lump
    bbox: list[float]

    def to_string(self):
        bbox_str = " ".join(str(x) for x in self.bbox)
        return f"-{self.id} body $-1 -1 -1 $-1 ${self.lump.id} $-1 $-1 T {bbox_str} #"


@dataclass
class Lump(SATEntity):
    shell: Shell
    body: Body
    bbox: list[float]
    next_lump: Lump = None

    def to_string(self):
        bbox_str = " ".join(str(x) for x in self.bbox)
        nl = -1 if self.next_lump is None else self.next_lump.id
        return f"-{self.id} lump $-1 -1 -1 $-1 ${nl} ${self.shell.id} ${self.body.id} T {bbox_str} #"


@dataclass
class Shell(SATEntity):
    wire: Wire
    lump: Lump
    bbox: list[float]

    def to_string(self) -> str:
        bbox_str = " ".join(str(coord) for coord in self.bbox)
        # Match GeniE beam SAT exactly:
        # -2 shell $-1 -1 -1 $-1 $-1 $-1 $-1 $3 $1 T ...
        return (
            f"-{self.id} shell $-1 -1 -1 $-1 $-1 $-1 "
            f"$-1 ${self.wire.id} ${self.lump.id} T {bbox_str} #"
        )



# ------------------------
#   Wire (instead of Face/Loop)
# ------------------------
@dataclass
class Wire(SATEntity):
    coedge: CoEdge
    shell: Shell
    bbox: list[float]

    def to_string(self):
        bbox_str = " ".join(str(x) for x in self.bbox)
        return f"-{self.id} wire $-1 -1 -1 $-1 $-1 ${self.coedge.id} ${self.shell.id} $-1 out T {bbox_str} #"


# ------------------------
#   CoEdge
# ------------------------
@dataclass
class CoEdge(SATEntity):
    next_coedge: CoEdge
    prev_coedge: CoEdge
    edge: Edge
    wire: Wire
    orientation: Literal["forward", "reverse"]

    def to_string(self):
        nxt = self.next_coedge.id if self.next_coedge else self.id
        prv = self.prev_coedge.id if self.prev_coedge else self.id
        return (
            f"-{self.id} coedge $-1 -1 -1 $-1 "
            f"${nxt} ${prv} $-1 ${self.edge.id} {self.orientation} "
            f"${self.wire.id} $-1 #"
        )


# ------------------------
#   Vertex / Point
# ------------------------
@dataclass
class SatPoint(SATEntity):
    point: ada.Point

    def to_string(self):
        p = " ".join(str(x) for x in make_ints_if_possible(self.point))
        return f"-{self.id} point $-1 -1 -1 $-1 {p} #"


@dataclass
class Vertex(SATEntity):
    edge: Edge
    point: SatPoint

    def to_string(self):
        return f"-{self.id} vertex $-1 -1 -1 $-1 ${self.edge.id} ${self.point.id} #"


# ------------------------
#   Straight Curve
# ------------------------
@dataclass
class StraightCurve(SATEntity):
    start_pt: ada.Point
    direction: ada.Direction

    def to_string(self):
        p = " ".join(str(x) for x in make_ints_if_possible(self.start_pt))
        d = " ".join(str(x) for x in make_ints_if_possible(self.direction.get_normalized()))
        return f"-{self.id} straight-curve $-1 -1 -1 $-1 {p} {d} I I #"


# ------------------------
#   Edge
# ------------------------
@dataclass
class Edge(SATEntity):
    vertex_start: Vertex
    vertex_end: Vertex
    coedge: CoEdge
    straight_curve: StraightCurve

    start_pt: ada.Point
    end_pt: ada.Point
    attrib_name: StringAttribName = None

    def to_string(self):
        attr = self.attrib_name.id if self.attrib_name else -1
        p1 = " ".join(str(x) for x in make_ints_if_possible(self.start_pt))
        p2 = " ".join(str(x) for x in make_ints_if_possible(self.end_pt))
        return (
            f"-{self.id} edge ${attr} -1 -1 $-1 "
            f"${self.vertex_start.id} 0 ${self.vertex_end.id} 1 "
            f"${self.coedge.id} ${self.straight_curve.id} forward @7 unknown T {p1} {p2} #"
        )


# ------------------------
#   String Attribute
# ------------------------
@dataclass
class StringAttribName(SATEntity):
    name: str
    entity: SATEntity

    def to_string(self):
        return (
            f"-{self.id} string_attrib-name_attrib-gen-attrib "
            f"$-1 -1 $-1 $-1 ${self.entity.id} "
            f"2 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 "
            f"@6 dnvscp @12 {self.name} #"
        )
