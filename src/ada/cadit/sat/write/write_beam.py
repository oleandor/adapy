from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ada
from ada.base.types import GeomRepr  # Not strictly needed here, but keeps style consistent
from ada.cadit.sat.write import sat_entities as se

if TYPE_CHECKING:
    from ada.cadit.sat.write.writer import SatWriter


def beam_to_sat_entities(bm: ada.Beam, sw: SatWriter, wire: se.Wire):
    id_gen = sw.id_generator
    entities = []

    # ---------------------------
    # Coordinates
    # ---------------------------
    p1 = bm.n1.p.copy()
    p2 = bm.n2.p.copy()

    if bm.e1 is not None:
        p1 = p1 + bm.e1
    if bm.e2 is not None:
        p2 = p2 + bm.e2

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    direction = ada.Direction(p2 - p1).get_normalized()

    # ---------------------------
    # Points
    # ---------------------------
    spt1 = se.SatPoint(id_gen.next_id(), p1)
    spt2 = se.SatPoint(id_gen.next_id(), p2)
    entities += [spt1, spt2]

    # ---------------------------
    # Vertices
    # ---------------------------
    v1 = se.Vertex(id_gen.next_id(), None, spt1)
    v2 = se.Vertex(id_gen.next_id(), None, spt2)
    entities += [v1, v2]

    # ---------------------------
    # Straight curve
    # ---------------------------
    curve = se.StraightCurve(id_gen.next_id(), p1, direction)
    entities.append(curve)

    # ---------------------------
    # Edge
    # ---------------------------
    edge = se.Edge(
        id_gen.next_id(),
        v1, v2,
        None,       # coedge placeholder
        curve,
        p1, p2
    )
    v1.edge = edge
    v2.edge = edge
    entities.append(edge)

    # ---------------------------
    # CoEdge
    # ---------------------------
    co = se.CoEdge(
        id_gen.next_id(),
        None, None,     # next/prev will be self-references
        edge,
        wire,
        "forward"
    )

    co.next_coedge = co
    co.prev_coedge = co
    edge.coedge = co
    wire.coedge = wire.coedge or co  # set only for first beam

    entities.append(co)

    # ---------------------------
    # Attribute name
    # ---------------------------
    name_id = id_gen.next_id()
    name_str = f"EDGE{sw.edge_name_id:08d}"
    sw.edge_name_id += 1

    name_att = se.StringAttribName(name_id, name_str, edge)
    edge.attrib_name = name_att
    entities.append(name_att)

    return entities, co
