from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from ada.api.spatial.equipment import Equipment, EquipRepr
from ada.cadit.sat.write.writer import SatWriter

from .write_utils import add_local_system

if TYPE_CHECKING:
    from ada import Beam, Part


import base64
import zipfile
import io
import xml.etree.ElementTree as ET


def create_sat_embedded_xml_element(sat_text: str) -> ET.Element:
    """
    Compresses the SAT text into a ZIP archive and returns a <sat_embedded> XML element
    exactly matching GeniE's format.
    """

    # ------------------------------------------------
    # 1. Create in-memory ZIP with a single file
    # ------------------------------------------------
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("b64temp.sat", sat_text)

    zipped_bytes = zip_buffer.getvalue()

    # ------------------------------------------------
    # 2. Base64 encode
    # ------------------------------------------------
    b64_encoded = base64.b64encode(zipped_bytes).decode("ascii")

    # ------------------------------------------------
    # 3. Build <sat_embedded> element
    # ------------------------------------------------
    sat_elem = ET.Element(
        "sat_embedded",
        {
            "encoding": "base64",
            "compression": "zip",
            "tag_name": "dnvscp"
        }
    )

    # Insert CDATA block
    sat_elem.text = "<![CDATA[" + b64_encoded + "]]>"

    return sat_elem

def insert_sat_into_xml(root: ET.Element, sat_text: str):
    """
    Finds or creates the <geometry> node and inserts <sat_embedded>.
    """
    structure_domain = root.find(".//structure_domain")

    if structure_domain is None:
        raise RuntimeError("XML root does not contain <structure_domain>")

    geometry = structure_domain.find("geometry")
    if geometry is None:
        geometry = ET.SubElement(structure_domain, "geometry")

    # Create embedded SAT
    sat_elem = create_sat_embedded_xml_element(sat_text)

    # Add as child
    geometry.append(sat_elem)

def export_genie_xml_with_sat(xml_root: ET.Element, sat_writer: SatWriter, filename: str):
    """
    Creates a GeniE .xml file with ACIS SAT embedded as zipped+base64.
    """

    # Get SAT text
    sat_text = sat_writer.dump_sat()

    # Insert into XML
    insert_sat_into_xml(xml_root, sat_text)

    # Serialize XML (we need manual CDATA injection afterward)
    raw_xml = ET.tostring(xml_root, encoding="utf-8").decode("utf-8")

    # Fix CDATA: ElementTree escaped our <![CDATA[]]>
    raw_xml = raw_xml.replace("&lt;![CDATA[", "<![CDATA[").replace("]]&gt;", "]]>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(raw_xml)


class SatWriter:
    def __init__(self):
        self.next_edge_id = 1
        self.next_vertex_id = 1
        self.next_point_id = 1
        self.next_curve_id = 1
        self.next_coedge_id = 1
        self.next_wire_id = 1

        # Maps (x,y,z) → vertex_id for reuse
        self.vertex_map = {}

        # SAT entity lines
        self.entities = []

    def _new_edge_name(self):
        name = f"EDGE{self.next_edge_id:08d}"
        self.next_edge_id += 1
        return name

    def _get_or_create_vertex(self, x, y, z):
        key = (round(x,6), round(y,6), round(z,6))

        if key in self.vertex_map:
            return self.vertex_map[key]

        vid = self.next_vertex_id
        pid = self.next_point_id

        self.vertex_map[key] = vid

        # Create point entity
        self.entities.append(
            f"-{pid} point $-1 -1 -1 $-1 {x} {y} {z} #"
        )
        # Create vertex entity
        self.entities.append(
            f"-{vid} vertex $-1 -1 -1 $-1 ${pid} $-1 #"
        )

        self.next_point_id += 1
        self.next_vertex_id += 1

        return vid

    def add_straight_edge(self, p1, p2):
        (x1, y1, z1) = p1
        (x2, y2, z2) = p2

        v1 = self._get_or_create_vertex(x1, y1, z1)
        v2 = self._get_or_create_vertex(x2, y2, z2)

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        length = (dx*dx + dy*dy + dz*dz) ** 0.5
        if length == 0:
            raise ValueError("Beam length is zero")

        ux = dx / length
        uy = dy / length
        uz = dz / length

        curve_id = self.next_curve_id
        self.next_curve_id += 1

        coedge_id = self.next_coedge_id
        self.next_coedge_id += 1

        wire_id = self.next_wire_id
        self.next_wire_id += 1

        edge_name = self._new_edge_name()

        # Straight curve
        self.entities.append(
            f"-{curve_id} straight-curve $-1 -1 -1 $-1 {x1} {y1} {z1} {ux} {uy} {uz} I I #"
        )

        # Edge
        edge_id = curve_id + 100  # ensure unique negative IDs
        self.entities.append(
            f"-{edge_id} edge $-1 -1 -1 $-1 ${v1} 0 ${v2} {length} ${coedge_id} ${curve_id} forward @7 unknown T "
            f"{min(x1,x2)} {min(y1,y2)} {min(z1,z2)} {max(x1,x2)} {max(y1,y2)} {max(z1,z2)} #"
        )

        # Name attribute for edge
        attrib_id = edge_id + 1
        self.entities.append(
            f"-{attrib_id} string_attrib-name_attrib-gen-attrib $-1 -1 $-1 $-1 ${edge_id} 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 "
            f"@6 dnvscp @12 {edge_name} #"
        )

        # Coedge & wire
        self.entities.append(
            f"-{coedge_id} coedge $-1 -1 -1 $-1 ${coedge_id} ${coedge_id} $-1 ${edge_id} forward ${wire_id} $-1 #"
        )
        self.entities.append(
            f"-{wire_id} wire $-1 -1 -1 $-1 $-1 ${coedge_id} $-1 $-1 out F #"
        )

        return edge_name

    def dump_sat(self):
        header = [
            "2000 0 1 0",
            "18 SESAM - gmGeometry 14 ACIS 33.0.1 NT 24 Fri Jan 01 00:00:00 2026",
            "1000 1e-6 1e-10"
        ]
        return "\n".join(header + self.entities + ["End-of-ACIS-data\n"])


def add_beams(root: ET.Element, part: Part, sw: SatWriter):
    from ada import Beam, BeamTapered

    iter_beams = part.get_all_physical_objects(by_type=Beam)
    iter_taper = part.get_all_physical_objects(by_type=BeamTapered)

    for beam in itertools.chain(iter_beams, iter_taper):
        parent = beam.parent
        if isinstance(parent, Equipment) and parent.eq_repr != EquipRepr.AS_IS:
            continue
        add_straight_beam(beam, root, sw)


def add_straight_beam(beam: Beam, xml_root: ET.Element, sw: SatWriter):
    import numpy as np
    from ada import Placement

    # ------------------------------------------------------
    # XML: Create structure + straight_beam container
    # ------------------------------------------------------
    structure_elem = ET.SubElement(xml_root, "structure")
    straight_beam = ET.SubElement(structure_elem, "straight_beam", {"name": beam.name})

    # ------------------------------------------------------
    # ORIENTATION HANDLING (unchanged from your implementation)
    # ------------------------------------------------------
    xvec = beam.xvec
    yvec = beam.yvec
    up   = beam.up

    if not beam.placement.is_identity():
        ident_place = Placement()
        place_abs = beam.placement.get_absolute_placement(include_rotations=True)

        if not np.allclose(place_abs.rot_matrix, ident_place.rot_matrix):
            ori_vectors = place_abs.transform_array_from_other_place(
                np.asarray([xvec, yvec, up]),
                ident_place,
                ignore_translation=True
            )
            xvec = ori_vectors[0]
            yvec = ori_vectors[1]
            up   = ori_vectors[2]

    straight_beam.append(add_local_system(xvec, yvec, up))

    # ------------------------------------------------------
    # XML: SEGMENTS (<guide> positions but no SAT yet)
    # ------------------------------------------------------
    segments_node = add_segments(beam)
    straight_beam.append(segments_node)

    # ------------------------------------------------------
    # GEOMETRY EXTRACTION FOR SAT
    # Correct handling of eccentricities e1, e2 if present
    # ------------------------------------------------------
    p1 = beam.n1.p.copy()
    p2 = beam.n2.p.copy()

    if beam.e1 is not None:
        p1 = p1 + beam.e1
    if beam.e2 is not None:
        p2 = p2 + beam.e2

    p1_tuple = (float(p1[0]), float(p1[1]), float(p1[2]))
    p2_tuple = (float(p2[0]), float(p2[1]), float(p2[2]))

    # ------------------------------------------------------
    # SAT GENERATION — create ACIS straight edge + get the EDGE name
    # ------------------------------------------------------
    #edge_name = sw.add_straight_edge(p1_tuple, p2_tuple)

    # ------------------------------------------------------
    # INSERT <sat_reference><edge edge_ref="EDGE####" /></sat_reference>
    # ------------------------------------------------------
    '''
    wire = segments_node.find(".//wire")
    sat_ref = wire.find("sat_reference")

    if sat_ref is None:
        sat_ref = ET.SubElement(wire, "sat_reference")

    ET.SubElement(sat_ref, "edge", {"edge_ref": edge_name})
    '''
    # ------------------------------------------------------
    # OPTIONAL HINGES (unchanged)
    # ------------------------------------------------------
    if beam.hinge1 is not None:
        ET.SubElement(straight_beam, "end1", {"hinge_ref": beam.hinge1.name})

    if beam.hinge2 is not None:
        ET.SubElement(straight_beam, "end2", {"hinge_ref": beam.hinge2.name})

    # ------------------------------------------------------
    # CURVE OFFSET (unchanged)
    # ------------------------------------------------------
    curve_offset = ET.SubElement(straight_beam, "curve_offset")
    ET.SubElement(curve_offset, "reparameterized_beam_curve_offset")


def add_straight_beam_old(beam: Beam, xml_root: ET.Element):
    import numpy as np

    from ada import Placement

    structure_elem = ET.SubElement(xml_root, "structure")
    straight_beam = ET.SubElement(structure_elem, "straight_beam", {"name": beam.name})

    xvec = beam.xvec
    yvec = beam.yvec
    up = beam.up

    if beam.placement.is_identity() is False:
        ident_place = Placement()
        place_abs = beam.placement.get_absolute_placement(include_rotations=True)
        place_abs_rot_mat = place_abs.rot_matrix
        ident_rot_mat = ident_place.rot_matrix
        # check if the 3x3 rotational np arrays are identical
        if not np.allclose(place_abs_rot_mat, ident_rot_mat):
            ori_vectors = place_abs.transform_array_from_other_place(
                np.asarray([xvec, yvec, up]), ident_place, ignore_translation=True
            )
            xvec = ori_vectors[0]
            yvec = ori_vectors[1]

    straight_beam.append(add_local_system(xvec, yvec, up))
    straight_beam.append(add_segments(beam))
    if beam.hinge1 is not None:
        ET.SubElement(straight_beam, "end1", {"hinge_ref": beam.hinge1.name})
    if beam.hinge2 is not None:
        ET.SubElement(straight_beam, "end2", {"hinge_ref": beam.hinge2.name})

    curve_offset = ET.SubElement(straight_beam, "curve_offset")
    ET.SubElement(curve_offset, "reparameterized_beam_curve_offset")


def add_curve_orientation(beam: Beam, straight_beam: ET.Element):
    curve_orientation = ET.SubElement(straight_beam, "curve_orientation")
    cco = ET.SubElement(curve_orientation, "customizable_curve_orientation", {"use_default_rule": "true"})
    orientation = ET.SubElement(cco, "orientation")
    local_system = ET.SubElement(orientation, "local_system")
    ET.SubElement(local_system, "x_vector", {"x": str(beam.xvec[0]), "y": str(beam.xvec[1]), "z": str(beam.xvec[2])})
    ET.SubElement(local_system, "y_vector", {"x": str(beam.yvec[0]), "y": str(beam.yvec[1]), "z": str(beam.yvec[2])})
    ET.SubElement(local_system, "up_vector", {"x": str(beam.up[0]), "y": str(beam.up[1]), "z": str(beam.up[2])})


def add_segments(beam: Beam):
    import numpy as np

    from ada import BeamTapered, Placement

    segments = ET.Element("segments")
    props = dict(index="1", section_ref=beam.section.name, material_ref=beam.material.name)
    if isinstance(beam, BeamTapered):
        props.update(dict(section_ref=f"{beam.section.name}_{beam.taper.name}"))

    straight_segment = ET.SubElement(segments, "straight_segment", props)

    d = ["x", "y", "z"]
    p1 = beam.n1.p
    p2 = beam.n2.p
    if beam.placement.is_identity() is False:
        ident_place = Placement()
        place_abs = beam.placement.get_absolute_placement(include_rotations=True)
        place_abs_rot_mat = place_abs.rot_matrix
        ident_rot_mat = ident_place.rot_matrix
        # check if the 3x3 rotational np arrays are identical
        if not np.allclose(place_abs_rot_mat, ident_rot_mat):
            tra_vectors = place_abs.transform_array_from_other_place(np.asarray([p1, p2]), ident_place)
            p1 = tra_vectors[0]
            p2 = tra_vectors[1]
        else:
            p1 = place_abs.origin + p1
            p2 = place_abs.origin + p2

    geom = ET.SubElement(straight_segment, "geometry")
    wire = ET.SubElement(geom, "wire")
    guide = ET.SubElement(wire, "guide")
    for i, pos in enumerate([p1, p2], start=1):
        props = {d[i]: str(k) for i, k in enumerate(pos)}
        props.update(dict(end=str(i)))
        ET.SubElement(guide, "position", props)

    ET.SubElement(wire, "sat_reference")

    # TODO: add SAT embedded geometry and include the reference to the EDGE geometry here
    # ET.SubElement(sat_ref, "edge_ref", dict(edge_ref=""))

    return segments
