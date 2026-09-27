"""The curved concept model: a quarter of an open-ended cylinder under internal pressure.

The third case in this package, and the first on a **non-planar** plate. Written once, here,
and both solver inputs derived from it -- the premise of :mod:`model` and :mod:`plate_model`
applied to a curved shell. What is new is that the discretised cylinder is **faceted**: both
meshers replace the arc by chords, so both answers carry a chord error that vanishes as the
mesh refines, and the comparison is again a convergence bracket between extrapolants
(:mod:`curved_compare`) rather than a single-mesh table.

The structure
=============

A quarter of a circular cylindrical shell: radius ``r = 2.0`` m, axial length ``L`` along
``z``, ``t = 10`` mm, S355, spanning ``theta = 0 .. 90`` degrees in the x-y plane, under a
uniform **internal** pressure of 1 bar. Supports:

* ``theta = 0`` -- the straight generator at ``y = 0, x = r``, lying in the symmetry plane
  ``y = 0``: ``u_y = 0, ur_x = 0, ur_z = 0``;
* ``theta = 90`` -- the straight generator at ``x = 0, y = r``, in the plane ``x = 0``:
  ``u_x = 0, ur_y = 0, ur_z = 0``;
* the two arc ends ``z = 0`` and ``z = L`` are **free** -- which is what makes the ends open
  and the axial stress zero;
* ``u_z = 0`` at the single corner node ``(r, 0, 0)``, which is the *only* rigid-body mode the
  two symmetry planes leave (see :data:`AXIAL_REFERENCE_SUPPORT`).

The closed forms are exact and there are two of them; :mod:`curved_hand_check` derives both.
The panel is a pure membrane hoop state: ``w = p r^2 / (E t)`` radially, uniform over the
whole panel, with zero bending and zero axial stress; and ``u_z = -nu p r z / (E t)``, the
axial Poisson contraction, measured from the plane the reference node fixes.

Why these dimensions, and why ``L`` is not a round number
=========================================================

``L`` is :data:`PANEL_LENGTH` ``= pi r / 2 = pi`` m -- **the quarter arc's own length**. That
is not decoration. Both meshers divide an edge into ``round(length / seed)`` elements
(``seedPart``'s documented behaviour, and what the analytic grid below does by construction),
so making the two edge lengths equal means one seed divides *both* directions into the same
count. At :data:`MESH_COUNTS` ``= (8, 16, 32)`` that puts a node at every
:data:`PROBE_POINTS` coordinate on both sides at all three densities, and it is checked
rather than assumed (:func:`assert_probes_are_seeded`, and
:func:`displacements.sample_fea_result` on each result).

``p = 1e5`` Pa gives ``w = 1.9048e-04`` m and a hoop stress of 20 MPa. The displacement is
four orders above a ``.SIN``'s single-precision resolution at that magnitude (about 1e-11 m,
i.e. 6e-08 relative) and the stress is well inside S355, so the model is linear and both
solvers are run geometrically linear.

Every probe is on a **boundary edge**
=====================================

:data:`PROBE_POINTS` holds ten points on the two straight generators and two at
``theta = 45`` on the free arc ends. Nothing is probed in the interior, and that is a
measurement rather than a preference: a node's position on a boundary edge is fixed by that
edge's own 1D division, which both meshers do by arc length, so ``theta = 45`` and
``z = L/4, L/2, 3L/4`` are exact on both sides. An **interior** node is placed by the
surface mesher, and the two disagree -- measured, gmsh places interior rows at uniform
*parameter* of the rational patch, which on a 90-degree span puts the first interior row at
``theta = 21.60`` degrees where the boundary arc has its node at ``22.50``. That is a 0.9
degree, 3.1e-02 m difference: it is not a defect in either mesher (both place nodes on the
exact surface, radius error 4e-16), it is what makes an interior probe unusable.

What neither route can be given, and what is used instead
=========================================================

Three gaps, all in ``src``, all reported here by name rather than worked around silently.
They are reproduced in :func:`reproduce_meshing_refusals` and
:func:`curved_abaqus_runner.writer_pressure_resultant`.

**1. adapy cannot mesh a ``PlateCurved`` at all.** Three refusals in a row, measured:

    ``Part.to_fem_obj``        -- ``isinstance(obj, Plate)`` is False for a ``PlateCurved``
                                  (it derives from ``BackendGeom``), so the object falls to
                                  the ``else`` branch, logs ``Unsupported object type ...
                                  Should be either plate or beam objects`` at ERROR, and the
                                  returned FEM has **0 nodes and 0 elements**;
    ``PlateCurved.shell_occ``  -- not implemented; ``BackendGeom.shell_occ`` raises
                                  ``NotImplementedError``, so ``GmshSession.add_obj(pc,
                                  geom_repr="shell")`` cannot even get the face;
    ``add_fem_sections``       -- ``NotImplementedError: Unsupported combination of
                                  geom_repr=GeomRepr.SHELL, and <class ...PlateCurved>``,
                                  reached if the first two are worked around.

So this module builds the shell mesh itself, and builds it **analytically**
(:func:`shell_grid`): the node at ``(i, j)`` is at ``(r cos theta_i, r sin theta_i, z_j)``
with ``theta_i = i pi / (2 n)`` and ``z_j = j L / n``. That is the nearest expressible
equivalent and it is strictly better here than a mesher would be, for a stated reason: every
facet is then an exact **rectangle** -- two equal chords at the same theta-span, offset by
``L/n`` along ``z``, perpendicular to it -- so ``integral(N_i) dA = A / 4`` holds exactly and
:func:`consistent_nodal_loads` is the pressure's own load vector rather than a lumping of it.
A gmsh mesh of the same patch gives **warped** quads (measured: 3.3e-03 m out of plane at
``n = 8``) on which that identity is only approximate.

**2. The CAE writer's equilibrium guard cannot check a pressure on a curved face.**
``ada.cadit.cae.analysis.AnalysisPlan.applied_pressure`` computes a pressure's resultant as
``-magnitude * area * normal`` -- the flat-plate formula, correct only where the normal is
constant. Measured on this very panel at ``n = 8``, the emitted script carries

    APPLIED_PRESSURE = (724863.7521754879, 669823.4477113917, 0.0)

against the exact ``integral(p n dA) = (p r L, p r L, 0) = (628318.53, 628318.53, 0)``: 15.4%
high in ``x``, 6.6% high in ``y``, and **asymmetric** where the physics is symmetric to the
last bit. The guard's own tolerance is 1e-04 relative, so the residual is about 1000x it and
the build fails before the displacement sidecar is written -- there is no result at all. The
Abaqus route therefore emits with ``submit=False`` and appends a driver carrying only the
solve, the equilibrium check against the *exact* resultant, and the sidecar; the geometry, the
section, the mesh, the element type, the three supports and the pressure itself are all still
the writer's. See :mod:`curved_abaqus_runner`.

**3. adapy's Sesam writer still emits no distributed load.** As :mod:`plate_sestra_runner`
records, ``write_loads.load_str`` reports ``[OMITTED] a "pressure" load is not written by the
Sesam writer``. ``BEUSLO`` exists on ``origin/feat/sesam-pressure-loads`` (#404) and was
**not** merged: that branch is cut from 0.91.1 and has diverged from this stack, and the
Sesam side does not need it -- on a rectangular facet the consistent nodal vector *is* the
pressure's load vector, and giving both routes the same vector would only have moved the
asymmetry. The Sestra side gets :func:`consistent_nodal_loads`; the Abaqus side gets the
writer's own ``*Dsload``, which is how the pressure sign is measured on a curved face at all
(:data:`curved_abaqus_runner.PRESSURE_SIGN_MEASUREMENT`).
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import numpy as np

import ada
from ada.fem import FEM, Bc, Elem, FemSection, FemSet, Load, StepImplicitStatic
from ada.fem.shapes import ElemType
from ada.fem.shapes.definitions import ShellShapes
from ada.materials.metals import CarbonSteel

from .model import ProbePoint

#: Cylinder radius, metres, to the shell's mid-surface.
PANEL_RADIUS = 2.0

#: The quarter arc's length, metres. ``pi r / 2 = 3.14159...``, and :data:`PANEL_LENGTH` is
#: set equal to it -- see the module docstring.
PANEL_ARC = 0.5 * math.pi * PANEL_RADIUS

#: Axial length along ``z``, metres. Equal to :data:`PANEL_ARC` so that one seed divides both
#: directions into the same count and every probe lands on a node.
PANEL_LENGTH = PANEL_ARC

#: Shell thickness, metres. ``r / t = 200``, so this is a thin shell and the membrane closed
#: form is exact to the order of ``t / r`` -- far below the chord error the study measures.
PANEL_THICKNESS = 0.010

#: S355: E = 210 GPa, nu = 0.3 (adapy's ``CarbonSteel`` defaults).
MATERIAL_NAME = "S355"

#: Internal pressure, pascals. 1 bar: ``w = 1.9048e-04`` m and a hoop stress of 20 MPa.
PRESSURE = 1.0e5

#: The pressure's sign as an ``ada.fem.Load`` magnitude on the Abaqus route.
#:
#: **Negative**, and that is measured rather than reasoned: the CAE writer builds a pressure's
#: region as ``assembly.Surface(side1Faces=...)`` and Abaqus takes a positive magnitude as
#: acting *into* side1, i.e. against the face's own outward normal. This panel's face normal
#: is the **outward radial** direction (adapy's own ``thickness_direction()`` reads
#: ``(0.7071, 0.7071, 0.0)`` at the mid-face, and the SAT import preserves it), so a positive
#: magnitude would be external pressure and would *contract* the shell. See
#: :data:`curved_abaqus_runner.PRESSURE_SIGN_MEASUREMENT` for the solved numbers on both
#: routes.
SIGNED_PRESSURE_MAGNITUDE = -PRESSURE

#: Elements per direction, coarse to fine. A factor of two each time, which is what lets an
#: order be read off three values (:func:`plate_hand_check.observed_order`) and a Richardson
#: extrapolation be taken without assuming one. Every count is divisible by four, which is
#: what puts a node at ``z = L/4, L/2, 3L/4``, and even, which puts one at ``theta = 45``.
MESH_COUNTS: tuple[int, ...] = (8, 16, 32)

#: The seed each of :data:`MESH_COUNTS` corresponds to, metres. Both edge lengths are
#: :data:`PANEL_ARC`, so one number serves both directions: ``round(PANEL_ARC / seed) == n``.
MESH_SIZES: tuple[float, ...] = tuple(PANEL_ARC / n for n in MESH_COUNTS)

#: Name of the single load case / step.
LOAD_CASE = "LC1"
STEP_NAME = "static"

#: How close a solver's node must be to a probe point, metres. 1e-05 -- **ten times** the
#: package-wide :data:`displacements.MATCH_TOL`, and the reason is a measurement rather than a
#: convenience.
#:
#: :mod:`displacements`' rule 3 is that the tolerance is a round-off tolerance and not a search
#: radius, because both meshes descend from the same adapy geometry and a probe coordinate is
#: exact in both to floating-point. That holds for every probe on a **straight** edge here: the
#: adapy grid puts a node at each of the ten generator probes to 1e-15 m, and so does CAE. It
#: does **not** hold for the two probes at ``theta = 45`` on the arc ends, and that is not
#: round-off: CAE divides a curved edge by its own arc-length algorithm at
#: ``deviationFactor=0.1``, and the node it puts at the arc's midpoint lands **1.77e-06 m** from
#: the exact ``(r cos 45, r sin 45, 0)`` at ``n = 8`` -- measured, with ``x`` and ``y`` differing
#: from each other by 2.5e-06 where the point is symmetric. So it is a mesher tolerance, 1.25e-06
#: relative on a 2 m radius, and it is a property of sampling a point on a curved edge at all.
#:
#: 1e-05 is 5.6x that measurement and it cannot reach a neighbouring node: the hoop node spacing
#: is 0.39 m at the coarsest grid and **0.098 m** at the finest, so the tolerance is 9800x below
#: the closest thing it could confuse this probe with. The ambiguity guard is therefore still a
#: guard rather than a formality, and :func:`assert_probes_are_seeded` keeps the strict 1e-06 on
#: the adapy grid, where the exactness is real.
PROBE_MATCH_TOL = 1.0e-05

#: Which writer the model is being built for, and therefore the one forced choice it gets --
#: the form of the load. See gap 3 in the module docstring.
ROUTES = ("sestra", "abaqus")

#: ``"nodal"`` is the exact consistent load vector of the pressure on the faceted mesh;
#: ``"pressure"`` is the ``ada.fem.Load`` of type ``pressure`` the CAE writer turns into a
#: ``Surface`` and a ``*Dsload``.
LOAD_STYLES = {"sestra": "nodal", "abaqus": "pressure"}

#: The two symmetry supports, as ``(set name, adapy dof list, what it means)``.
#:
#: Written out rather than inlined because **both** decks are generated from exactly this --
#: the Sesam writer's ``BNBCD`` FIX codes and the CAE writer's ``DisplacementBC`` keywords.
#: "Symmetry" has to mean the same thing in both, and the only way to be sure is for there to
#: be one statement of it.
#:
#: Symmetry about a plane of normal ``n`` is: the displacement along ``n`` fixed, and the
#: rotations about the **two in-plane axes** fixed. The ``theta = 0`` generator lies in the
#: plane ``y = 0``, whose in-plane axes are ``x`` and ``z``, hence ``u_y = ur_x = ur_z = 0``;
#: the ``theta = 90`` generator lies in ``x = 0``, hence ``u_x = ur_y = ur_z = 0``. Every one
#: of those six is satisfied *identically* by the exact solution -- it has no rotation
#: anywhere and no hoop displacement across either plane -- so none of them carries a moment,
#: and the two in-plane reactions are the whole of the pressure resultant
#: (:func:`curved_hand_check.edge_reaction`).
SYMMETRY_SUPPORTS = (
    ("SYM_T0", (2, 4, 6), "symmetry about y = 0 on the theta = 0 generator: u_y = ur_x = ur_z = 0"),
    ("SYM_T90", (1, 5, 6), "symmetry about x = 0 on the theta = 90 generator: u_x = ur_y = ur_z = 0"),
)

#: The axial rigid-body restraint: ``u_z = 0`` at the single corner node ``(r, 0, 0)``.
#:
#: **One node, and the corner rather than a mid-edge point.** Both halves of that are forced.
#: One node, because translation along ``z`` is the only rigid-body mode the two symmetry
#: planes leave: ``u_x = 0`` on one generator kills translation in ``x`` and rotation about
#: ``y``, ``u_y = 0`` on the other kills translation in ``y`` and rotation about ``x``, and
#: ``ur_z = 0`` kills rotation about ``z``. And the corner, because the CAE writer resolves a
#: support against its own geometry: ``(r, 0, 0)`` is a **vertex** of the authored body, so
#: this record comes out a vertex region (measured, ``_analysis_region(assembly, 'AXIAL_REF',
#: 'Panel-1', ((2.0, 0.0, 0.0),), 'AXIAL_REF')``). A node halfway along the generator is on no
#: vertex, is not a whole edge and is not a plate's whole mesh, so it would be refused by
#: name -- and ``u_z = 0`` along a *whole* generator would be worse than refused: it would
#: restrain the axial Poisson contraction and destroy the second closed form.
#:
#: It carries no load, and that is checked rather than asserted: the exact solution has
#: ``u_z(z = 0) = 0`` for every ``theta``, so the reaction at this node is zero to
#: discretisation -- see :func:`curved_compare.assert_reference_node_carries_nothing`.
AXIAL_REFERENCE_SUPPORT = ("AXIAL_REF", (3,), "the one rigid-body mode the symmetry planes leave: u_z = 0 at (r, 0, 0)")

#: Every support the model declares, in the order both writers see them.
EDGE_SUPPORTS = SYMMETRY_SUPPORTS + (AXIAL_REFERENCE_SUPPORT,)


def _probe_points() -> tuple[ProbePoint, ...]:
    """:data:`PROBE_POINTS`, built once so the three coordinate families are stated once.

    ``cos(45) = sin(45)`` exactly in binary at this radius (``2.0 * 0.7071067811865476``),
    which is why the two arc probes can be written as one expression and still land on a node
    to 1e-15 on both sides -- measured.
    """
    r, length = PANEL_RADIUS, PANEL_LENGTH
    mid = r * math.cos(math.radians(45.0))
    fractions = (("Z0", 0.0), ("Q", 0.25), ("MID", 0.5), ("3Q", 0.75), ("ZL", 1.0))
    points: list[ProbePoint] = []
    for label, fraction in fractions:
        points.append(
            ProbePoint(
                f"T0_{label}",
                (r, 0.0, fraction * length),
                f"theta = 0 generator at z = {fraction} L: u1 is the radial expansion, u3 the axial contraction",
            )
        )
    for label, fraction in fractions:
        points.append(
            ProbePoint(
                f"T90_{label}",
                (0.0, r, fraction * length),
                f"theta = 90 generator at z = {fraction} L: u2 is the radial expansion, u3 the axial contraction",
            )
        )
    points.append(
        ProbePoint("ARC0_MID", (mid, mid, 0.0), "theta = 45 on the free z = 0 arc: mid-hoop, at the reference plane")
    )
    points.append(
        ProbePoint("ARCL_MID", (mid, mid, length), "theta = 45 on the free z = L arc: mid-hoop, at the far end")
    )
    return tuple(points)


#: The twelve points compared between solvers. Ten on the two straight generators at
#: ``z = 0, L/4, L/2, 3L/4, L``, and two at ``theta = 45`` on the free arc ends -- all of them
#: on a boundary edge, for the reason the module docstring gives.
#:
#: The generator probes carry the radial expansion in a *single* component (``u1`` at
#: ``theta = 0``, ``u2`` at ``theta = 90``) and the axial contraction in ``u3``; the two arc
#: probes carry the radial expansion split evenly between ``u1`` and ``u2``, which is the
#: mid-hoop statement the generators cannot make on their own -- they are both symmetry planes
#: and would agree with each other even if the hoop state were wrong.
PROBE_POINTS: tuple[ProbePoint, ...] = _probe_points()

#: The probe whose radial displacement the headline convergence table follows: mid-length on
#: the ``theta = 0`` generator, where the radial expansion is the whole of ``u1``.
RADIAL_PROBE = "T0_MID"

#: ...and the component that carries it there.
RADIAL_COMPONENT = "u1"

#: The probe whose ``u3`` the axial-contraction closed form is read at: the far end of the
#: ``theta = 0`` generator, where ``|u_z|`` is largest.
AXIAL_PROBE = "T0_ZL"

#: ``{probe name: (component, the unit radial direction's component there)}`` -- how the
#: radial displacement is recovered from the table at each probe.
#:
#: Carried as data because the recovery differs by probe and a comparison that got it wrong
#: would report a plausible number: at ``theta = 45`` the radial displacement is
#: ``(u1 + u2) / sqrt(2)`` and reading ``u1`` alone would be 29% low.
RADIAL_BASIS: dict[str, tuple[tuple[str, float], ...]] = {
    **{f"T0_{label}": (("u1", 1.0),) for label, _ in (("Z0", 0), ("Q", 0), ("MID", 0), ("3Q", 0), ("ZL", 0))},
    **{f"T90_{label}": (("u2", 1.0),) for label, _ in (("Z0", 0), ("Q", 0), ("MID", 0), ("3Q", 0), ("ZL", 0))},
    "ARC0_MID": (("u1", math.cos(math.radians(45.0))), ("u2", math.sin(math.radians(45.0)))),
    "ARCL_MID": (("u1", math.cos(math.radians(45.0))), ("u2", math.sin(math.radians(45.0)))),
}

#: The three probes whose radial displacement must agree for the panel to be in the uniform
#: hoop state the closed form is for. One per symmetry plane and one at mid-hoop.
HOOP_PROBES = ("T0_MID", "ARC0_MID", "T90_MID")

#: adapy's own names. The CAE part instance is ``Panel-1``.
PART_NAME = "Panel"
PANEL_NAME = "panel"
ASSEMBLY_NAME = "PanelSite"

#: The element set every shell of the panel is in -- what the ``pressure`` load acts on.
SHELL_SET_NAME = "PANEL_SHELLS"


class CurvedModelInvalid(ValueError):
    """The model or mesh handed in is not the curved shell model this comparison is about.

    Raised rather than worked around, for the reason :class:`plate_model.PlateModelInvalid`
    gives: a curved-panel comparison run on a mesh with no shells in it, or on a grid whose
    nodes are not on the cylinder, would compare two numbers neither of which came from a
    cylinder, and every table would look perfectly ordinary.
    """


@dataclass(frozen=True)
class NodalLoadGroup:
    """One force vector and every node that carries exactly it.

    A group, not a node, because that is the shape adapy's Sesam writer wants: it emits one
    ``BNLOAD`` per member of a ``Load``'s node set with the same vector on each, so a uniform
    pressure on a faceted cylinder needs one ``Load`` per **distinct vector**. On this grid
    that is a small number -- the vector depends on the node's hoop position and on whether it
    is on an end arc, not on where along the cylinder it sits -- measured 18 groups for 81
    nodes at ``n = 8`` and 66 for 1089 at ``n = 32``.
    """

    #: The force, newtons, global axes. Points outward: this is an *internal* pressure.
    force: tuple[float, float, float]
    #: The node ids, sorted.
    node_ids: tuple[int, ...]

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(np.asarray(self.force, dtype=float)))


def material() -> ada.Material:
    """The one material object both closed forms and both decks come from."""
    return ada.Material(MATERIAL_NAME, CarbonSteel(MATERIAL_NAME))


def section_properties() -> dict[str, float]:
    """``E``, ``nu`` and ``rho``, read off the model rather than retyped.

    For the reason :func:`plate_model.section_properties` gives: a changed material must not
    leave a stale modulus in the hand check. Measured: ``E = 2.1e11``, ``nu = 0.3``.
    """
    model_props = material().model
    return {
        "E": float(model_props.E),
        "nu": float(model_props.v),
        "rho": float(model_props.rho),
    }


def panel_face():
    """The quarter cylinder as an exact rational NURBS patch, as an ``AdvancedFace``.

    **Exact, not fitted**, and that is the whole reason for the rational form: a quadratic
    Bezier with control points ``(r, 0), (r, r), (0, r)`` and weights ``1, cos 45, 1`` *is* a
    90-degree circular arc, to the last bit. Ruled linearly along ``z``, it is the quarter
    cylinder. Measured against ``pi r L / 2``: OCC builds the face and reports an area of
    6.283185307121142 against the exact 6.283185307179586, 9.3e-12 relative, and every node
    the grid below places sits on radius ``r`` to 4.4e-16.

    Every ``OrientedEdge`` carries **both** its parameter range and its ``pcurve``, and both
    are load-bearing rather than tidy. ``ada.cadit.sat.write.write_curved_plate`` refuses an
    edge with no parameters by name (``BSplineCurveWithKnots edge without authored
    parameters``), and ACIS refuses a coedge on a spline surface with no UV image
    (``coedge on spline surface has no PCURVE``) -- which is exactly why the synthetic arch
    patch recorded earlier in this project imported with ``getSize() == 0.0`` and could not be
    meshed. With them, Abaqus 2025 imports this face, measures it against adapy's own area and
    meshes it (see :mod:`curved_abaqus_runner`).

    The two arc edges are authored as the *same* rational quadratic as the surface's own hoop
    direction, so an edge's parameter and the surface's ``u`` are one parameterisation and the
    pcurve is the straight line ``v = const`` in UV. The two generators are ``Line``s, whose
    ACIS parameterisation is arc length, so their range is ``0 .. L`` and their pcurve runs
    ``0 .. L`` too -- ACIS rejects a pcurve whose range does not cover its coedge's.
    """
    from ada.geom import Geometry
    from ada.geom import curves as geo_cu
    from ada.geom import surfaces as geo_su

    r, length = PANEL_RADIUS, PANEL_LENGTH
    weight = math.cos(math.radians(45.0))

    surface = geo_su.RationalBSplineSurfaceWithKnots(
        u_degree=2,
        v_degree=1,
        control_points_list=[
            [ada.Point(r, 0.0, 0.0), ada.Point(r, 0.0, length)],
            [ada.Point(r, r, 0.0), ada.Point(r, r, length)],
            [ada.Point(0.0, r, 0.0), ada.Point(0.0, r, length)],
        ],
        surface_form=geo_su.BSplineSurfaceForm.UNSPECIFIED,
        u_closed=False,
        v_closed=False,
        self_intersect=False,
        u_multiplicities=[3, 3],
        v_multiplicities=[2, 2],
        u_knots=[0.0, 1.0],
        v_knots=[0.0, length],
        knot_spec=geo_cu.KnotType.UNSPECIFIED,
        weights_data=[[1.0, 1.0], [weight, weight], [1.0, 1.0]],
    )

    def arc(z: float) -> geo_cu.RationalBSplineCurveWithKnots:
        return geo_cu.RationalBSplineCurveWithKnots(
            degree=2,
            control_points_list=[ada.Point(r, 0.0, z), ada.Point(r, r, z), ada.Point(0.0, r, z)],
            curve_form=geo_cu.BSplineCurveFormEnum.UNSPECIFIED,
            closed_curve=False,
            self_intersect=False,
            knot_multiplicities=[3, 3],
            knots=[0.0, 1.0],
            knot_spec=geo_cu.KnotType.UNSPECIFIED,
            weights_data=[1.0, weight, 1.0],
        )

    def pcurve(start, end, knots, *, same_sense: bool = True) -> geo_cu.Pcurve2dBSpline:
        return geo_cu.Pcurve2dBSpline(
            degree=1,
            control_points_2d=[list(start), list(end)],
            knots=list(knots),
            knot_multiplicities=[2, 2],
            weights=None,
            closed=False,
            # 0.0 is a claim that the pcurve is exact, and here it is one: a straight line in
            # UV is the exact image of an isoparametric boundary curve.
            fit_tolerance=0.0,
            same_sense=same_sense,
        )

    corners = {
        "a": (ada.Point(r, 0.0, 0.0), ada.Point(0.0, r, 0.0)),
        "b": (ada.Point(0.0, r, 0.0), ada.Point(0.0, r, length)),
        "c": (ada.Point(0.0, r, length), ada.Point(r, 0.0, length)),
        "d": (ada.Point(r, 0.0, length), ada.Point(r, 0.0, 0.0)),
    }
    # The loop runs the UV boundary counter-clockwise -- (0,0) -> (1,0) -> (1,L) -> (0,L) --
    # which makes the face normal dS/du x dS/dv, i.e. the hoop tangent crossed with +z, i.e.
    # the OUTWARD radial direction. That is what fixes the pressure's sign
    # (:data:`SIGNED_PRESSURE_MAGNITUDE`), so it is not an arbitrary traversal.
    #
    # Two of the four coedges therefore run their edge BACKWARDS -- ``c`` and ``d``, whose
    # ``t_start > t_end`` -- and those two need **both** halves of the pcurve contract, which is
    # the measurement this face cost the most to find:
    #
    #   * the pcurve's control points run in the **edge's own ascending** direction, not the
    #     loop's, because a pcurve is the edge curve's image in UV and shares its
    #     parameterisation: at parameter 0 it must give the UV of the edge's *low* end;
    #   * and ``same_sense`` is **False**, because that flag says whether the pcurve runs with
    #     the **coedge** -- and on a reversed coedge an edge-ascending pcurve runs against it.
    #
    # Get either one wrong and Abaqus 2025 still imports the face, still reports
    # ``faces 1 edges 4 vertices 4``, and still measures ``getSize()`` as 9.869604675509118
    # against adapy's own 9.869604401089358 (2.8e-08) -- and then ``part.geometryValidity`` reads
    # **0** and ``seedPart`` refuses with ``AbaqusException: At least one of the selected
    # instances contains invalid geometry and cannot be meshed or assigned mesh attributes``.
    # Measured across nine variants: the curve types (rational B-spline, ellipse, straight,
    # degree-1 B-spline), the surface's ``v`` degree (1 or 3) and its ``u`` knot span (0..1 or
    # 0..pi/2) changed **nothing** -- every one of them read ``geometryValidity = 0``; flipping
    # ``same_sense`` on the two reversed coedges alone took it to **1** and meshed to
    # ``{'S4R': 64}`` on 81 nodes. So a pcurve's presence is not the condition (the no-pcurve
    # control measures ``getSize() == 0.0``, which is the §8 failure) -- its *sense* is.
    edge_specs = (
        ("a", arc(0.0), pcurve((0.0, 0.0), (1.0, 0.0), [0.0, 1.0]), 0.0, 1.0),
        (
            "b",
            geo_cu.Line(ada.Point(0.0, r, 0.0), ada.Direction(0.0, 0.0, 1.0)),
            pcurve((1.0, 0.0), (1.0, length), [0.0, length]),
            0.0,
            length,
        ),  # noqa: E501
        ("c", arc(length), pcurve((0.0, length), (1.0, length), [0.0, 1.0], same_sense=False), 1.0, 0.0),
        (
            "d",
            geo_cu.Line(ada.Point(r, 0.0, 0.0), ada.Direction(0.0, 0.0, 1.0)),
            pcurve((0.0, 0.0), (0.0, length), [0.0, length], same_sense=False),
            length,
            0.0,
        ),  # noqa: E501
    )
    oriented = []
    for key, curve, pcurve_2d, t_start, t_end in edge_specs:
        start, end = corners[key]
        oriented.append(
            geo_cu.OrientedEdge(
                start,
                end,
                geo_cu.EdgeCurve(start, end, curve, True),
                True,
                pcurve=pcurve_2d,
                t_start=t_start,
                t_end=t_end,
            )
        )
    face = geo_su.AdvancedFace([geo_su.FaceBound(geo_cu.EdgeLoop(oriented), True)], surface, True)
    return Geometry(f"{PANEL_NAME}_geom", face, None)


def panel_plate() -> ada.PlateCurved:
    """The panel as an ``ada.PlateCurved``. The one geometry object both routes derive from.

    A plain ``PlateCurved`` and never a subclass, and that is not stylistic: adapy's
    ``Part.get_all_physical_objects(by_type=...)`` filters by **exact type**
    (``type(x) in by_type``), so a subclass is invisible to the SAT writer and therefore to the
    whole CAE route -- measured, ``part_to_sat_writer`` returned ``is_empty=True`` and the CAE
    writer refused with ``adapy's SAT writer authored no geometry for any of them``.
    """
    return ada.PlateCurved(PANEL_NAME, panel_face(), PANEL_THICKNESS, mat=material())


def face_area() -> float:
    """``pi r L / 2`` -- the quarter cylinder's exact surface area, square metres.

    ``6.283185307179586``. Stated in closed form rather than measured off the kernel, because
    it is what the kernel's own number is *checked against*: OCC reports 6.283185307121142
    (9.3e-12) and CAE's ``getSize()`` is what the emitted script's ``_guard_plate_faces``
    compares against adapy's. Note that this is the **bare face**: ``PlateCurved.solid_occ()``
    is a thickness-``t`` ClosedShell when ``Config().geom_thicken_curved_shells`` is on, which
    is the default, and its surface area is about twice this.
    """
    return 0.5 * math.pi * PANEL_RADIUS * PANEL_LENGTH


def shell_grid(count: int) -> tuple[dict[tuple[int, int], ada.Node], list[ada.Node]]:
    """``({(i, j): node}, nodes)`` -- the analytic structured grid, ``count`` elements a side.

    Node ``(i, j)`` is at ``(r cos theta_i, r sin theta_i, z_j)`` with
    ``theta_i = i pi / (2 count)`` and ``z_j = j L / count``. Built rather than meshed for
    gap 1 of the module docstring, and the exactness is what it buys: each facet's four
    corners are two equal chords of the same hoop span offset along ``z``, so the facet is a
    **rectangle**, ``integral(N_i) dA = A / 4`` exactly, and every
    :data:`PROBE_POINTS` coordinate is a node to 1e-15.
    """
    if count < 2 or count % 4 != 0:
        raise CurvedModelInvalid(
            f"the grid count must be a multiple of four and at least 4; got {count}. A count "
            f"divisible by four puts a node at z = L/4, L/2 and 3L/4 and at theta = 45, which is "
            f"every one of the {len(PROBE_POINTS)} probe points. MESH_COUNTS={MESH_COUNTS} all are."
        )
    grid: dict[tuple[int, int], ada.Node] = {}
    nodes: list[ada.Node] = []
    node_id = 1
    for i in range(count + 1):
        theta = 0.5 * math.pi * i / count
        x, y = PANEL_RADIUS * math.cos(theta), PANEL_RADIUS * math.sin(theta)
        for j in range(count + 1):
            node = ada.Node((x, y, PANEL_LENGTH * j / count), node_id)
            grid[(i, j)] = node
            nodes.append(node)
            node_id += 1
    return grid, nodes


def build_panel(count: int, *, route: str = "sestra") -> ada.Assembly:
    """The one definition of the panel. Both solver inputs are derived from this.

    Returns an :class:`ada.Assembly` carrying:

    * one ``PlateCurved`` on the exact rational NURBS quarter cylinder (:func:`panel_face`);
    * a ``count x count`` structured shell FEM on it (:func:`shell_grid`), quadrilateral --
      ``FQUS`` in Sesam, ``S4R`` in Abaqus, and the closed form is for a shell, so a
      triangulated mesh would be comparing two discretisations of two element families;
    * a node set and a ``Bc`` for each entry of :data:`EDGE_SUPPORTS`, on **both** routes:
      the Sesam writer turns them into ``BNBCD`` FIX codes and the CAE writer into three
      ``DisplacementBC``, two on edge regions and one on a vertex. Checked before the model
      leaves here: :func:`assert_supports_declared`;
    * one ``StepImplicitStatic`` carrying the load form the route can be given.

    ``route`` changes exactly one thing -- which of two equivalent forms the uniform pressure
    takes (:data:`LOAD_STYLES`); see gap 3 of the module docstring.
    """
    if route not in ROUTES:
        raise CurvedModelInvalid(f"route must be one of {ROUTES}, got {route!r}")

    plate = panel_plate()
    part = ada.Part(PART_NAME) / [plate]
    assembly = ada.Assembly(ASSEMBLY_NAME) / part

    fem = _build_fem(plate, count)
    part.fem = fem
    fem.parent = part

    assert_shell_grid(fem, count=count)
    assert_probes_are_seeded(fem, count=count)

    _add_supports(fem)
    assert_supports_declared(fem)

    step = assembly.fem.add_step(StepImplicitStatic(STEP_NAME, nl_geom=False, total_time=1, init_incr=1, max_incr=1))
    if LOAD_STYLES[route] == "nodal":
        _add_nodal_loads(fem, step)
    else:
        _add_pressure(fem, step)
    return assembly


def _build_fem(plate, count: int) -> FEM:
    """The shell FEM: the analytic grid, one quad per cell, one shell section over all of them.

    ``el.refs`` is appended with the plate on every element, which is adapy's own linkage from
    a shell element back to what it was meshed from -- and it is what
    ``ada.cadit.cae.analysis._pressure_plate`` resolves a ``pressure`` load through. An element
    with an empty ``refs`` would leave the pressure unable to name a plate and be refused.
    """
    from ada.api.containers import Nodes
    from ada.fem.containers import FemElements

    fem = FEM(f"{PART_NAME}-FEM")
    grid, nodes = shell_grid(count)
    fem.nodes = Nodes(nodes, parent=fem)

    elements: list[Elem] = []
    element_id = 1
    for i in range(count):
        for j in range(count):
            # Hoop-then-axial ordering, so the facet normal is cross(hoop, +z) = OUTWARD
            # radial. FQUS takes its own normal from this ordering, so it is the ordering that
            # makes the two solvers agree about which side of the shell is which; checked in
            # assert_shell_grid rather than left to the comment.
            corners = [grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]]
            element = Elem(element_id, corners, ShellShapes.QUAD, parent=fem)
            element.refs.append(plate)
            elements.append(element)
            element_id += 1
    fem.elements = FemElements(elements, fem_obj=fem)
    plate.elem_refs = elements

    shell_set = fem.add_set(FemSet(SHELL_SET_NAME, elements, FemSet.TYPES.ELSET, parent=fem))
    fem.add_section(
        FemSection(
            f"d_{PANEL_NAME}_sh",
            ElemType.SHELL,
            shell_set,
            plate.material,
            local_z=plate.thickness_direction(),
            thickness=PANEL_THICKNESS,
            int_points=5,
        )
    )
    return fem


def shell_elements(fem) -> list:
    """Every quadrilateral shell element of ``fem``, sorted by id."""
    return sorted((el for el in fem.elements if str(el.type).endswith("QUAD")), key=lambda el: el.id)


def element_normal(element) -> np.ndarray:
    """The unit normal of one facet, from its own node ordering.

    ``cross(n1 - n0, n3 - n0)`` normalised. Exact for this grid because every facet is a
    rectangle; the function does not assume that, and :func:`assert_shell_grid` is what
    checks it.
    """
    points = np.asarray([node.p for node in element.nodes], dtype=float)
    normal = np.cross(points[1] - points[0], points[3] - points[0])
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        raise CurvedModelInvalid(f"element {element.id} is degenerate: its two edge vectors are parallel")
    return normal / length


def element_area(element) -> float:
    """One facet's area, square metres, from the cross product of its two edge vectors.

    Exact for a rectangle, which every facet of this grid is.
    """
    points = np.asarray([node.p for node in element.nodes], dtype=float)
    return float(np.linalg.norm(np.cross(points[1] - points[0], points[3] - points[0])))


def assert_shell_grid(fem, *, count: int | None = None) -> None:
    """Raise unless ``fem`` is the faceted quarter cylinder this comparison is about.

    Five clauses, and each one is a defect that would otherwise produce a full table of
    plausible numbers:

    * quadrilateral shells exist at all -- a comparison whose mesh is all beams still prints
      six components at twelve probes, and the consistent nodal loads would silently be zero;
    * every node is on radius ``r`` -- a grid built on the chords rather than on the arc, or
      one whose radius was scaled, changes ``w = p r^2 / (E t)`` by the square of the error
      while leaving the field's *shape* right;
    * every facet is **planar** -- a warped quad breaks ``integral(N_i) dA = A / 4``, so the
      consistent nodal load would stop being the pressure's own vector. Measured on this grid:
      8.9e-16 m out of plane at ``n = 32``, against 3.3e-03 for a gmsh mesh of the same patch;
    * every facet normal points **outward** -- it is the node ordering that says which side of
      the shell is which, in both solvers, and an inward facet would reverse the pressure on
      it;
    * the facet areas sum to ``pi r L / 2`` -- the chord sum is *below* the arc area and
      approaches it as ``O(h^2)``, so this is checked against the chord area the grid actually
      has rather than against the exact one, which would fail on a correct coarse mesh.
    """
    shells = shell_elements(fem)
    if not shells:
        kinds = sorted({str(el.type) for el in fem.elements})
        raise CurvedModelInvalid(
            f"this FEM carries no quadrilateral shell elements at all -- element types present: "
            f"{kinds or '(none)'}, {len(fem.elements)} element(s), count={count}. Every number this "
            f"comparison produces comes from a membrane hoop state, and a uniform pressure resolved "
            f"onto no shells is a zero load vector."
        )
    if count is not None and len(shells) != count * count:
        raise CurvedModelInvalid(
            f"the grid holds {len(shells)} shell element(s) and a {count} x {count} grid has "
            f"{count * count}. A panel meshed to fewer shells than the model asked for is a panel "
            f"with a different chord error, and its place in the refinement sequence is not the one "
            f"the extrapolation assumes."
        )

    radii = np.hypot(
        np.asarray([node.x for node in fem.nodes], dtype=float), np.asarray([node.y for node in fem.nodes], dtype=float)
    )
    radial_error = float(np.abs(radii - PANEL_RADIUS).max())
    if radial_error > 1.0e-09:
        raise CurvedModelInvalid(
            f"the grid's nodes are up to {radial_error:.3e} m off radius {PANEL_RADIUS}, so this is not "
            f"the cylinder the closed form w = p r^2 / (E t) is for -- and that form goes as r^2, so a "
            f"radius wrong by 1% is an answer wrong by 2%."
        )

    worst_warp = 0.0
    inward = []
    chord_area = 0.0
    for element in shells:
        points = np.asarray([node.p for node in element.nodes], dtype=float)
        normal = element_normal(element)
        worst_warp = max(worst_warp, abs(float(np.dot(points[3] - points[0] + points[1] - points[2], normal))))
        centre = points.mean(axis=0)
        if float(np.dot(normal, np.array([centre[0], centre[1], 0.0]))) <= 0.0:
            inward.append(element.id)
        chord_area += element_area(element)
    if worst_warp > 1.0e-09:
        raise CurvedModelInvalid(
            f"a facet of this grid is {worst_warp:.3e} m out of plane, so it is not a rectangle -- and "
            f"integral(N_i) dA = A / 4, which is what makes consistent_nodal_loads the pressure's own "
            f"load vector rather than a lumping of it, holds only on one. A gmsh mesh of the same patch "
            f"warps by 3.3e-03 m at n = 8, which is why this grid is built analytically."
        )
    if inward:
        raise CurvedModelInvalid(
            f"{len(inward)} facet normal(s) point at the cylinder axis rather than away from it (element "
            f"id(s) {inward[:8]}{'...' if len(inward) > 8 else ''}). The node ordering is what tells both "
            f"solvers which side of the shell is which, so an inward facet carries the internal pressure "
            f"inwards and contracts where the rest expands."
        )
    if chord_area > face_area() * (1.0 + 1.0e-12):
        raise CurvedModelInvalid(
            f"the facets sum to {chord_area!r} m2 and the exact quarter cylinder is {face_area()!r} m2. A "
            f"chord is shorter than its arc, so the faceted area is always the smaller of the two; a grid "
            f"whose area exceeds the arc's is not inscribed in the cylinder."
        )


def assert_probes_are_seeded(fem, *, count: int | None = None) -> None:
    """Raise unless every :data:`PROBE_POINTS` coordinate is a unique node of ``fem``.

    Checked on the adapy grid before a solve is spent on it, exactly as
    :func:`plate_model.assert_probes_are_seeded` does. The Abaqus mesh is checked by
    :func:`displacements.sample_fea_result`, which raises rather than settling for the nearest
    node -- and that is the check that matters most here, because the Abaqus mesh is CAE's own
    and not this grid.
    """
    missing = []
    for probe in PROBE_POINTS:
        hits = fem.nodes.get_by_volume(p=probe.xyz, tol=1.0e-06)
        if len(hits) != 1:
            missing.append(f"{probe.name} at {probe.xyz} -> {len(hits)} node(s)")
    if missing:
        raise CurvedModelInvalid(
            f"the grid has no unique node at these probe points, so the comparison would be sampling "
            f"the wrong place: {missing}. count={count} must be a multiple of four for z = L/4, L/2 and "
            f"3L/4 and even for theta = 45; MESH_COUNTS={MESH_COUNTS} all are."
        )


def assert_supports_declared(fem) -> None:
    """Raise unless ``fem`` carries one ``Bc`` per :data:`EDGE_SUPPORTS` entry, with its dofs.

    The supports are the model's own records on **both** routes, which makes this the single
    point where their absence can be caught. And a dropped support here is quiet rather than
    loud: losing ``ur_z`` from either symmetry set leaves a model that still solves, still
    reacts the whole pressure and is simply not the symmetric quarter panel the closed form is
    for; losing ``AXIAL_REF`` leaves the one singular direction, which Abaqus refuses but
    Sestra may solve with a warning.
    """
    declared = {bc.name: tuple(sorted(int(d) for d in bc.dofs)) for bc in fem.bcs}
    problems = []
    for set_name, dofs, why in EDGE_SUPPORTS:
        expected = tuple(sorted(dofs))
        if set_name not in declared:
            problems.append(f"{set_name!r} ({why}) has no Bc at all")
        elif declared[set_name] != expected:
            problems.append(
                f"{set_name!r} fixes dofs {list(declared[set_name])}, and EDGE_SUPPORTS says {list(expected)}"
            )
    if problems:
        raise CurvedModelInvalid(
            f"the panel's supports are not the {len(EDGE_SUPPORTS)} EDGE_SUPPORTS records both writers "
            f"translate: {'; '.join(problems)}. The model holds {sorted(declared)}. 'Symmetry on both "
            f"generators, free arc ends, one axial reference' is written once, in EDGE_SUPPORTS, and it is "
            f"what both decks are generated from -- so a record missing here is a support missing from "
            f"both solvers."
        )


def _support_nodes(fem, set_name: str) -> list:
    """The nodes one :data:`EDGE_SUPPORTS` entry acts on, by position.

    By position and not by a mesher's own grouping, so the same three sentences describe the
    support at every density and in both decks. It is also what the CAE writer classifies:
    the two symmetry sets are every node on a whole straight boundary edge (hence edge
    regions) and the reference set is the one node at a vertex (hence a vertex region).
    """
    tol = 1.0e-09
    if set_name == "SYM_T0":
        nodes = [n for n in fem.nodes if abs(n.y) < tol]
    elif set_name == "SYM_T90":
        nodes = [n for n in fem.nodes if abs(n.x) < tol]
    elif set_name == "AXIAL_REF":
        nodes = [n for n in fem.nodes if abs(n.y) < tol and abs(n.z) < tol]
    else:  # pragma: no cover - EDGE_SUPPORTS is closed
        raise CurvedModelInvalid(f"no edge is defined for support set {set_name!r}")
    if not nodes:
        raise CurvedModelInvalid(f"support set {set_name!r} matched no node of the grid")
    if set_name == "AXIAL_REF" and len(nodes) != 1:
        raise CurvedModelInvalid(
            f"the axial reference must be exactly one node and {len(nodes)} matched (r, 0, 0). More than "
            f"one would restrain a rotation as well as the translation, and the reaction check that says "
            f"this node carries nothing would then be checking two things at once."
        )
    return sorted(nodes, key=lambda n: n.id)


def _add_supports(fem) -> None:
    for set_name, dofs, _why in EDGE_SUPPORTS:
        fem_set = fem.add_set(FemSet(set_name, _support_nodes(fem, set_name), FemSet.TYPES.NSET, parent=fem))
        fem.add_bc(Bc(set_name, fem_set, list(dofs)))


def consistent_nodal_loads(fem, *, pressure: float = PRESSURE) -> tuple[NodalLoadGroup, ...]:
    """The nodal load vector of ``pressure`` on ``fem``'s facets, grouped by force vector.

    **Not an approximation.** For a 4-node bilinear quadrilateral ``integral(N_i) dA = A / 4``
    for each shape function *when the element is a parallelogram* -- and every facet of
    :func:`shell_grid` is a rectangle, which :func:`assert_shell_grid` checks to 1e-09 m. So
    the work-equivalent nodal load of a uniform pressure on a facet is ``q A / 4`` along that
    facet's own normal at each of its four nodes, which is the vector Abaqus' ``*Dsload``
    assembles element by element -- not a lumping of it. The normal is **per facet**, not the
    panel's mid-face normal: that is the whole difference from the flat case, and getting it
    wrong is what :meth:`AnalysisPlan.applied_pressure` does (gap 2 in the module docstring).

    Measured: the vectors sum to ``(628318.53071796, 628318.53071796, 0.0)`` N at every
    density, against ``(p r L, p r L, 0)`` = ``(628318.5307179586, ...)`` -- exact to 1e-15
    relative, which is the identity :func:`curved_hand_check.edge_reaction` derives.
    """
    accumulated: dict[int, np.ndarray] = collections.defaultdict(lambda: np.zeros(3))
    for element in shell_elements(fem):
        share = pressure * element_area(element) / 4.0 * element_normal(element)
        for node in element.nodes:
            accumulated[node.id] = accumulated[node.id] + share

    groups: dict[tuple[float, float, float], list[int]] = collections.defaultdict(list)
    for node_id in sorted(accumulated):
        # Rounded to pick the distinct vectors of a structured grid out of float noise; the
        # *emitted* force uses the rounded value so the two sides cannot disagree by a bit.
        key = tuple(round(float(v), 10) for v in accumulated[node_id])
        groups[key].append(node_id)  # type: ignore[index]
    return tuple(NodalLoadGroup(force=key, node_ids=tuple(groups[key])) for key in sorted(groups))


def applied_load_total(groups) -> tuple[float, float, float]:
    """The resultant of ``groups``, newtons. Compared against :func:`expected_load_total`."""
    total = np.zeros(3)
    for group in groups:
        total = total + np.asarray(group.force, dtype=float) * len(group.node_ids)
    return (float(total[0]), float(total[1]), float(total[2]))


def expected_load_total() -> tuple[float, float, float]:
    """``(p r L, p r L, 0)`` -- what the internal pressure integrates to on the quarter panel.

    Derived in :func:`curved_hand_check.edge_reaction`: the outward normal is
    ``(cos theta, sin theta, 0)`` and ``dA = r L dtheta``, so the ``y`` component is
    ``integral p sin(theta) r L dtheta`` over ``0 .. pi/2`` = ``p r L``, and the ``x`` one is
    the same by symmetry. The ``z`` component is zero because the normal has none -- which is
    the same statement as "the ends are open".
    """
    resultant = PRESSURE * PANEL_RADIUS * PANEL_LENGTH
    return (resultant, resultant, 0.0)


def _add_nodal_loads(fem, step) -> None:
    """One ``Load`` per distinct force vector, over a node set of every node carrying it.

    One per *vector* rather than one per node because ``write_loads.load_force`` emits a
    ``BNLOAD`` for every member of the set with the same vector on each. Measured on this
    model: 18 ``Load`` objects and 81 ``BNLOAD`` records at ``n = 8``, summing to
    ``(628318.53, 628318.53, 0.0)`` N.
    """
    groups = consistent_nodal_loads(fem)
    total = applied_load_total(groups)
    expected = expected_load_total()
    worst = max(abs(total[axis] - expected[axis]) for axis in range(3))
    scale = max(abs(v) for v in expected)
    if worst > 1.0e-09 * scale:
        raise CurvedModelInvalid(
            f"the consistent nodal loads sum to {total!r} N and the pressure integrates to {expected!r} "
            f"N, off by {worst!r}. A load vector that does not carry the whole pressure makes every "
            f"displacement below wrong by the same factor, which no comparison between two solvers given "
            f"the same wrong load could see."
        )
    by_id = {node.id: node for node in fem.nodes}
    for index, group in enumerate(groups, start=1):
        members = [by_id[node_id] for node_id in group.node_ids]
        fem_set = fem.add_set(FemSet(f"Q_{index}", members, FemSet.TYPES.NSET, parent=fem))
        magnitude = group.magnitude
        if magnitude == 0.0:  # pragma: no cover - every node of a pressurised panel carries some
            raise CurvedModelInvalid(f"nodal load group {index} carries a zero force on {len(members)} node(s)")
        direction = [component / magnitude for component in group.force]
        step.add_load(
            Load(f"QP_{index}", Load.TYPES.FORCE, magnitude, fem_set=fem_set, dof=direction + [0.0, 0.0, 0.0])
        )


def _add_pressure(fem, step) -> None:
    """One ``Load`` of type ``pressure`` over the panel's whole shell element set.

    The whole panel, because ``ada.cadit.cae.analysis._pressure_plate`` refuses a set covering
    only part of one: a CAE ``Surface`` is made of whole faces, so a partial set would silently
    become a pressure over all of it.

    The magnitude is :data:`SIGNED_PRESSURE_MAGNITUDE`, i.e. **negative**, because the writer
    builds the surface as ``side1Faces`` and Abaqus takes a positive magnitude as acting into
    it -- and this face's normal is outward. ``Load.TYPES`` does not list ``pressure`` (the
    constructor sets ``_type`` directly rather than through the validating setter), which is
    why the string is written out here.
    """
    fem_set = fem.sets.get_elset_from_name(SHELL_SET_NAME)
    step.add_load(Load("p", "pressure", SIGNED_PRESSURE_MAGNITUDE, fem_set=fem_set))


def reproduce_meshing_refusals() -> dict[str, str]:
    """Reproduce gap 1 of the module docstring, and return what each of the three said.

    Kept as code rather than as prose because a gap that is only written down stops being
    checked the day it is closed. Each value is the exception (or the empty FEM) the call
    produced; :func:`curved_compare.assert_meshing_gap_is_still_open` reads them.
    """
    from ada.base.types import GeomRepr
    from ada.fem.meshing.utils import add_fem_sections

    plate = panel_plate()
    findings: dict[str, str] = {}

    part = ada.Part(f"{PART_NAME}Probe") / [plate]
    ada.Assembly(f"{ASSEMBLY_NAME}Probe") / part
    fem = part.to_fem_obj(MESH_SIZES[0], "line", use_quads=True, interactive=False)
    findings["Part.to_fem_obj"] = f"{len(fem.nodes)} node(s) and {len(fem.elements)} element(s)"

    try:
        plate.shell_occ()
        findings["PlateCurved.shell_occ"] = "returned a shape"
    except NotImplementedError as exc:
        findings["PlateCurved.shell_occ"] = f"NotImplementedError({str(exc) or 'no message'})"

    class _Data:
        geom_repr = GeomRepr.SHELL
        entities = ()

    try:
        add_fem_sections(None, FEM("probe"), plate, _Data())
        findings["add_fem_sections"] = "accepted a PlateCurved"
    except NotImplementedError as exc:
        findings["add_fem_sections"] = f"NotImplementedError({exc})"
    return findings
