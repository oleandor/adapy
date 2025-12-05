from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from ada.base.types import GeomRepr
from ada.cadit.sat.write import sat_entities as se
from ada.cadit.sat.write.sat_entities import SATEntity
from ada.cadit.sat.write.utils import IDGenerator
from ada.cadit.sat.write.write_beam import beam_to_sat_entities
from ada.cadit.sat.write.write_plate import plate_to_sat_entities

if TYPE_CHECKING:
    from ada import Assembly, Part

HEADER_STR = """2000 0 1 0
18 SESAM - gmGeometry 14 ACIS 33.0.1 NT 24 Tue Jan 17 20:39:08 2023
1000 9.9999999999999995e-07 1e-10
"""


from ada import Beam

def part_to_sat_writer(part: Part | Assembly) -> SatWriter:
    from ada import Beam
    import numpy as np

    sw = SatWriter(part)
    id_gen = sw.id_generator
    bbox = sw.bbox

    # ---------------------------------------------------------
    # 1) Create core topological ACIS structure
    # ---------------------------------------------------------

    # Body
    body_id = id_gen.next_id()
    body = se.Body(body_id, None, bbox)

    # Lump
    lump_id = id_gen.next_id()
    lump = se.Lump(lump_id, None, body, bbox)

    # Link body <-> lump
    body.lump = lump

    # Shell
    shell_id = id_gen.next_id()
    shell = se.Shell(shell_id, None, lump, bbox)

    # Link lump <-> shell
    lump.shell = shell

    # ----------------------------
    # Temporary placeholder wire
    # (actual Wire must be created after coedge exists)
    # ----------------------------
    wire_id = id_gen.next_id()
    wire = se.Wire(wire_id, None, shell, bbox)

    # Link shell <-> wire
    shell.wire = wire

    # Register core entities
    for ent in (body, lump, shell, wire):
        sw.add_entity(ent)

    # ---------------------------------------------------------
    # 2) Create beam edges inside the wire
    # ---------------------------------------------------------
    for bm in part.get_all_physical_objects(by_type=Beam):
        # This returns ONLY edge/coedge/vertex/point/curve
        new_entities, first_coedge = beam_to_sat_entities(bm, sw, wire)
        for ent in new_entities:
            sw.add_entity(ent)

    # After beams are created, update wire.coedge to first coedge
    wire.coedge = first_coedge

    # ---------------------------------------------------------
    # 3) Final reordering and renumbering
    # ---------------------------------------------------------
    all_entities = sorted(sw.entities.values(), key=lambda x: x.id)
    sw.entities = {e.id: e for e in all_entities}

    return sw


def part_to_sat_writer_old(part: Part | Assembly) -> SatWriter:
    from ada import Plate

    sw = SatWriter(part)

    # Beams
    for edge_id, bm in enumerate(part.get_all_physical_objects(by_type=Beam), start=1):
        edge_name = f"EDGE{edge_id:08d}"
        # Optionally store mapping if useful
        # sw.edge_map[bm.guid] = edge_name
        new_entities = beam_to_sat_entities(bm, edge_name, sw)
        for entity in new_entities:
            sw.add_entity(entity)

    # Plates
    '''
    for face_id, pl in enumerate(part.get_all_physical_objects(by_type=Plate), start=1):
        face_name = f"FACE{face_id:08d}"
        sw.face_map[pl.guid] = face_name
        new_entities = plate_to_sat_entities(pl, face_name, GeomRepr.SHELL, sw)
        for entity in new_entities:
            sw.add_entity(entity)
    '''

    # re-arrange entities and make sure all body, lump, shell and face elements are before any further elements
    bodies = sw.get_entities_by_type(se.Body)
    lumps = sw.get_entities_by_type(se.Lump)
    shells = sw.get_entities_by_type(se.Shell)
    faces = sw.get_entities_by_type(se.Face)
    new_id = 0

    first_entities = list(chain(bodies, lumps, shells, faces))
    for i, entity in enumerate(first_entities):
        if isinstance(entity, se.Lump) and len(lumps) > 1:
            next_entity = first_entities[i + 1]
            if isinstance(next_entity, se.Lump):
                entity.next_lump = next_entity

        entity.id = new_id
        new_id += 1
    for rest in sw.entities.values():
        if type(rest) not in [se.Body, se.Lump, se.Shell, se.Face]:
            rest.id = new_id
            new_id += 1
    # update map
    sw.entities = {entity.id: entity for entity in sorted(sw.entities.values(), key=lambda x: x.id)}
    return sw


@dataclass
class SatWriter:
    part: Part | Assembly
    entities: dict = field(default_factory=dict)

    header: str = HEADER_STR
    bbox: list[float] = field(default_factory=list)
    id_generator: IDGenerator = field(default_factory=IDGenerator)
    face_map: dict[str, str] = field(default_factory=dict)  # face_name -> plate guid

    edge_name_id: int = 1

    def __post_init__(self):
        bboxes = []
        for part in self.part.get_all_parts_in_assembly(include_self=True):
            if len(part.nodes.nodes) == 0:
                continue
            bboxes.append(list(chain.from_iterable(zip(*part.nodes.bbox()))))
        # Find the minimum values for the first 3 and the maximum values for the last 3
        if len(bboxes) == 0:
            raise ValueError("No nodes found in the part")

        bbox_min = [min([bbox[i] for bbox in bboxes]) for i in range(3)]
        bbox_max = [max([bbox[i + 3] for bbox in bboxes]) for i in range(3)]
        self.bbox = bbox_min + bbox_max

    def add_entity(self, entity: SATEntity) -> None:
        self.entities[entity.id] = entity

    def write(self, file_path: str | pathlib.Path) -> None:
        with open(file_path, "w") as f:
            f.write(self.to_str())

    def to_str(self):
        sorted_values = sorted(self.entities.values(), key=lambda x: x.id)
        return self.header + "\n".join(entity.to_string() for entity in sorted_values) + "\nEnd-of-ACIS-data"

    def get_entities_by_type(self, by_type) -> list[SATEntity]:
        return list(filter(lambda x: type(x) is by_type, self.entities.values()))
