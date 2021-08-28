import os
from dataclasses import dataclass
from itertools import chain
from typing import Iterable, List, Union

import gmsh
import numpy as np

from ada.concepts.containers import Nodes
from ada.concepts.points import Node
from ada.concepts.primitives import Shape
from ada.concepts.structural import Beam, Plate
from ada.config import Settings
from ada.core.utils import make_name_fem_ready
from ada.fem import FEM, Elem, FemSection, FemSet
from ada.fem.containers import FemElements
from ada.ifc.utils import create_guid

from .gmshapi import eval_thick_normal_from_cog_of_beam_plate, gmsh_map

GmshOptions = {
    "Mesh.Algorithm": 1,
    "Mesh.MeshSizeFromCurvature": True,
    "Mesh.MinimumElementsPerTwoPi": 12,
    "Mesh.MeshSizeMax": 0.1,
    "Mesh.ElementOrder": 1,
    "Mesh.SecondOrderIncomplete": 1,
    "Mesh.Smoothing": 3,
    "Geometry.Tolerance": 1e-3,
}


@dataclass
class GmshData:
    entities: Iterable
    geom_repr: str
    order: int


class GmshSession:
    def __init__(self, silent=False, persist=True, geom_repr="shall", settings: dict = None):
        print("init method called")
        self._gmsh = None
        self.settings = settings if settings is not None else GmshOptions
        if silent is True:
            self.settings["General.Terminal"] = 0
        self.geom_repr = geom_repr
        self.persist = persist
        self.model_map = dict()

    def add_obj(self, obj: Union[Shape, Beam, Plate], geom_repr, el_order=1):
        temp_dir = Settings.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        name = f"{obj.name}_{create_guid()}"

        if geom_repr == "line" and type(obj) is Beam:
            obj.to_stp(temp_dir / name, geom_repr=geom_repr, silent=True)
            entities = self.model.occ.importShapes(str(temp_dir / f"{name}.stp"))
            self.model.occ.synchronize()
            self.model_map[obj] = GmshData(entities, geom_repr, el_order)
        else:
            obj.to_stp(temp_dir / name, geom_repr=geom_repr, silent=True)
            entities = self.model.occ.importShapes(str(temp_dir / f"{name}.stp"))
            self.model.occ.synchronize()
            self.model_map[obj] = GmshData(entities, geom_repr, el_order)

    def mesh(self, size: float = None):
        if size is not None:
            self.gmsh.option.setNumber("Mesh.MeshSizeMax", size)

        model = self.model
        model.geo.synchronize()
        model.mesh.setRecombine(3, 1)
        model.mesh.generate(3)
        model.mesh.removeDuplicateNodes()

    def get_fem(self) -> FEM:
        fem = FEM("AdaFEM")
        fem._nodes = Nodes(get_nodes_from_gmsh(self.model, fem), parent=fem)
        elements = []

        # Get Elements
        for model_obj, gmsh_data in self.model_map.items():
            elements += get_bm_elements(self.model, fem, gmsh_data)
        fem._elements = FemElements(elements, fem_obj=fem)
        # Add FEM sections
        for model_obj, gmsh_data in self.model_map.items():
            if type(model_obj) is Beam:
                add_fem_sections(self.model, fem, model_obj, gmsh_data)
        return fem

    def _add_settings(self):
        if self.settings is not None:
            for setting, value in self.settings.items():
                self.gmsh.option.setNumber(setting, value)

    def open_gui(self):
        self.gmsh.fltk.run()

    def __enter__(self):
        print("Starting GMSH session")

        self._gmsh = gmsh
        self.gmsh.initialize()
        self._add_settings()
        self.model.add("ada")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print("Closing GMSH")
        self.gmsh.finalize()

    @property
    def gmsh(self) -> gmsh:
        return self._gmsh

    @property
    def model(self) -> "gmsh.model":
        return self.gmsh.model


def get_bm_elements(model: gmsh.model, fem: FEM, gmsh_data: GmshData):
    if gmsh_data.geom_repr == "shell":
        elements = get_elements_from_entities(model, gmsh_data.entities, fem)
    elif gmsh_data.geom_repr == "solid":
        elements = get_elements_from_entities(model, gmsh_data.entities, fem)
    else:  # beam
        elements = get_elements_from_entities(model, gmsh_data.entities, fem)
    return elements


def add_fem_sections(model: gmsh.model, fem: FEM, model_obj: Union[Beam, Plate], gmsh_data: GmshData):
    for _, ent in gmsh_data.entities:
        if gmsh_data.geom_repr == "shell":
            get_sh_sections(model, model_obj, ent, fem)
        elif gmsh_data.geom_repr == "solid":
            get_so_sections(model, model_obj, ent, fem)
        else:  # beam
            get_bm_sections(model, model_obj, ent, fem)


def get_sh_sections(model: gmsh.model, model_obj: Union[Beam, Plate], ent, fem: FEM):
    _, tags, _ = model.mesh.getElements(2, ent)
    r = model.occ.getCenterOfMass(2, ent)
    if type(model_obj) is Beam:
        t, n, _ = eval_thick_normal_from_cog_of_beam_plate(model_obj, r)
    else:
        t, n = model_obj.t, model_obj.n

    set_name = make_name_fem_ready(f"el{model_obj.name}e{ent}")
    fem_sec_name = make_name_fem_ready(f"d{model_obj.name}e{ent}")

    fem_set = fem.add_set(FemSet(set_name, [fem.elements.from_id(x) for x in chain.from_iterable(tags)], "elset"))
    props = dict(local_z=n, thickness=t, int_points=5)
    fem_sec = FemSection(fem_sec_name, "shell", fem_set, model_obj.material, **props)
    add_sec_to_fem(fem, fem_sec, fem_set)


def get_bm_sections(model: gmsh.model, beam: Beam, ent, fem: FEM):

    elem_types, elem_tags, elem_node_tags = model.mesh.getElements(1, ent)
    elements = [fem.elements.from_id(tag) for tag in elem_tags[0]]

    set_name = make_name_fem_ready(f"el{beam.name}_set")
    fem_sec_name = make_name_fem_ready(f"d{beam.name}_sec")
    fem_set = FemSet(set_name, elements, "elset", parent=fem)
    fem_sec = FemSection(fem_sec_name, "beam", fem_set, beam.material, beam.section, beam.ori[2])

    add_sec_to_fem(fem, fem_sec, fem_set)


def get_so_sections(model: gmsh.model, beam: Beam, ent, fem: FEM):
    _, tags, _ = model.mesh.getElements(3, ent)

    set_name = make_name_fem_ready(f"el{beam.name}_set")
    fem_sec_name = make_name_fem_ready(f"d{beam.name}_sec")

    elements = [fem.elements.from_id(tag) for tag in tags[0]]
    fem_set = fem.sets.add(FemSet(set_name, elements, "elset", parent=fem))
    fem_sec = FemSection(fem_sec_name, "solid", fem_set, beam.material)

    add_sec_to_fem(fem, fem_sec, fem_set)


def add_sec_to_fem(fem: FEM, fem_section: FemSection, fem_set: FemSet):
    if fem_set.name in fem.elsets.keys():
        fem_set = fem.elsets[fem_set.name]
        for el in fem_set.members:
            el.fem_sec = fem_set.members[0].fem_sec
        fem_set.add_members(fem_set.members)
    else:
        fem.sets.add(fem_set)
        fem.add_section(fem_section)


def get_point(gmsh_session: gmsh, p, tol):
    tol_vec = np.array([tol, tol, tol])
    lower = np.array(p) - tol_vec
    upper = np.array(p) + tol_vec
    return gmsh_session.model.getEntitiesInBoundingBox(*lower.tolist(), *upper.tolist(), 0)


def get_nodes_from_gmsh(model: gmsh.model, fem: FEM) -> List[Node]:
    nodes = list(model.mesh.getNodes(-1, -1))
    node_ids = nodes[0]
    node_coords = nodes[1].reshape(len(node_ids), 3)
    return [Node(coord, nid, parent=fem) for nid, coord in zip(node_ids, node_coords)]


def get_elements_from_entity(model: gmsh.model, ent, fem: FEM, dim) -> List[Elem]:
    elem_types, elem_tags, elem_node_tags = model.mesh.getElements(dim, ent)
    elements = []
    for k, element_list in enumerate(elem_tags):
        el_name, dim, _, numv, _, _ = model.mesh.getElementProperties(elem_types[k])
        if el_name == "Point":
            continue
        elem_type = gmsh_map[el_name]
        for j, eltag in enumerate(element_list):
            nodes = []
            for i in range(numv):
                idtag = numv * j + i
                p1 = elem_node_tags[k][idtag]
                nodes.append(fem.nodes.from_id(p1))

            el = Elem(eltag, nodes, elem_type, parent=fem)
            elements.append(el)
    return elements


def get_elements_from_entities(model: gmsh.model, entities, fem: FEM) -> List[Elem]:
    elements = []
    for dim, ent in entities:
        elements += get_elements_from_entity(model, ent, fem, dim)
    return elements
