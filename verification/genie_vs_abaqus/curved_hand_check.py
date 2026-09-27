"""Closed forms for the curved panel, and the convergence arithmetic that uses them.

Two exact closed forms, a third that follows from them, and one equilibrium identity -- all
for the model :mod:`curved_model` builds. The convergence arithmetic itself
(:func:`plate_hand_check.observed_order`, :func:`plate_hand_check.richardson`) is reused
unchanged: it is the same three-mesh estimate, and writing it twice would let the two copies
drift.

The derivation, in two lines each
=================================

**Radial expansion.** An open-ended cylinder under internal pressure ``p`` is in a pure
membrane state: cutting it on a diametral plane, the pressure on the projected area ``2 r L``
is carried by two wall sections of area ``2 t L``, so the hoop stress is
``sigma_theta = p r / t``; the ends are free so ``sigma_z = 0``. Hooke then gives
``eps_theta = (sigma_theta - nu sigma_z) / E = p r / (E t)``, and the radial displacement is
``w = r eps_theta =`` **``p r^2 / (E t)``** -- with **no** ``nu`` in it, which is the whole
point of leaving the ends open, and uniform over the entire panel, which is what makes the
comparison a single number rather than a shape.

**Axial contraction.** The same two stresses in the other Hooke equation:
``eps_z = (sigma_z - nu sigma_theta) / E = -nu p r / (E t)``, constant, so
``u_z(z) =`` **``-nu p r z / (E t)``** measured from the plane where ``u_z`` is fixed -- which
is ``z = 0``, the plane :data:`curved_model.AXIAL_REFERENCE_SUPPORT` pins. Exact, linear in
``z``, and free: it comes out of the same solves the radial expansion does, which is what makes
it an independent statement about the model rather than a restatement of the first.

**The edge reaction.** The outward normal of the quarter panel is
``n = (cos theta, sin theta, 0)`` and ``dA = r L dtheta``, so
``integral p n dA`` over ``theta = 0 .. pi/2`` is ``(p r L, p r L, 0)``. The ``theta = 0``
generator is the only place ``u_y`` is restrained and the ``theta = 90`` generator the only
place ``u_x`` is, so each edge's in-plane reaction is exactly ``p r L`` -- the same number by
symmetry, and the same number the hoop membrane force ``N_theta L = sigma_theta t L = p r L``
gives when read off the wall instead of off the pressure. Two routes to one number is why this
is the check that the *whole* load arrived.

**And the ``z`` reaction is zero**, which is the third closed form and costs nothing: the
normal has no ``z`` component, so an open-ended cylinder reacts nothing axially, and the one
node that fixes ``u_z`` therefore carries no load. That is checked
(:func:`curved_compare.assert_reference_node_carries_nothing`) rather than assumed, because a
non-zero reaction there would mean the axial Poisson contraction was being restrained and the
second closed form would be the wrong one.

What the discretisation does to all of that, measured
=====================================================

Both meshers replace the arc by ``n`` chords, so both solve a **faceted** cylinder: the
pressure acts normal to each facet, each facet carries membrane force *and* bending, and the
enclosed area is smaller than the circle's. Every one of those errors is ``O(h^2)``, and both
solvers measure it. Sestra ``FQUS``, radial displacement at :data:`curved_model.RADIAL_PROBE`
against ``p r^2 / (E t) = 1.904761904762e-04``:

===========  ======  ==================  ==========
n per side   nodes   Sestra ``FQUS``     rel
===========  ======  ==================  ==========
8            81      1.895564346e-04     4.829e-03
16           289     1.902506338e-04     1.184e-03
32           1089    1.904212550e-04     2.884e-04
===========  ======  ==================  ==========

The single error ratio three values give is **4.0687**, i.e. ``p = 2.0246`` -- second order,
measured rather than assumed, and slightly above two because the chord error and the facet
bending are two ``O(h^2)`` terms with different coefficients. The extrapolant is
``1.904768562e-04``, **3.495e-06** above the closed form, which is the size of its own
next-order term. The Abaqus ``S4R`` column and the cross-solver residual are in
:mod:`curved_compare`.

Note the sign of the residual: both sequences approach the closed form **from below**. That is
physics, not a coincidence -- a polygon inscribed in the cylinder encloses less area and its
chords are shorter than the arcs they replace, so the faceted panel is stiffer in hoop than the
cylinder is. A sequence that approached from above would mean the nodes were *outside* radius
``r``, which :func:`curved_model.assert_shell_grid` refuses.

Why the tolerances are what they are
====================================

:data:`RADIAL_REL_TOL` is 2e-04, :data:`AXIAL_REL_TOL` 2e-03 and :data:`HOOP_REL_TOL` 1e-03, and
each is read off its own measurement rather than felt -- see each one's note for the table it came
from. The gap between them is the point rather than an accident: the radial expansion is the
quantity the extrapolation removes the leading error from and both solvers land on it to 1e-04 or
better, while the axial check is three statements at once and its loosest clause -- linearity in
``z`` -- carries the ``S4R`` orientation asymmetry :mod:`curved_compare` measures. Three
questions, three tolerances, in the manner :mod:`hand_check` already argues for.
"""

from __future__ import annotations

import math

from .plate_hand_check import (  # noqa: F401 - re-exported so a reader finds the arithmetic here
    ORDER_BAND,
    REFINEMENT,
    Convergence,
    NotConverging,
    observed_order,
    richardson,
)

#: Tolerance on the radial closed form, relative. 2e-04.
#:
#: Budgeted from the measurements in the module docstring, not tuned. The extrapolated radial
#: displacements miss ``p r^2 / (E t)`` by **3.495e-06** (Sestra) and **8.420e-05** (Abaqus), so
#: 2e-04 is 2.4x the worse of the two -- the next-order term the extrapolation leaves behind, plus
#: the ``S4R`` orientation asymmetry described in :mod:`curved_compare`, is what it has to
#: accommodate. Every defect it exists for is orders larger: a pressure with the wrong sign
#: changes the answer's sign, a radius wrong by 1% moves it 2%, a thickness read as 12 mm instead
#: of 10 moves it 17%, and the coarsest grid is 4.8e-03 away.
RADIAL_REL_TOL = 2.0e-04

#: Tolerance on the axial-contraction closed form, relative. 2e-03.
#:
#: Ten times looser than :data:`RADIAL_REL_TOL`, for a measured reason rather than for comfort.
#: The axial check is three statements at once (:func:`curved_compare.assert_axial_contraction`) --
#: the slope, the zero at ``z = 0``, and the **linearity** in ``z`` -- and it is the linearity that
#: sets this number. Measured on the extrapolated tables, ``u3 / z`` over the ten probes at
#: non-zero ``z``:
#:
#: ==========  ====================  ====================  ==================
#: solver      slope, rel vs form    worst nonlinearity    ``|u3|`` at z = 0
#: ==========  ====================  ====================  ==================
#: Sestra      3.247e-07             1.131e-07             3.08e-13
#: Abaqus      2.305e-04             **7.553e-04**         2.54e-08
#: ==========  ====================  ====================  ==================
#:
#: 2e-03 is 2.6x that 7.553e-04, which is the ``S4R`` asymmetry between the two symmetry
#: generators showing up in the axial field (Abaqus' own *finest grid* fits the slope to
#: 2.161e-05 -- better than its extrapolant, because the asymmetry is not a single-rate term and
#: Richardson amplifies it). It is still 100x below what the check exists to catch: an axial
#: restraint that was not released gives ``u_z = 0`` at every probe, i.e. 100% away.
AXIAL_REL_TOL = 2.0e-03

#: How closely the three :data:`curved_model.HOOP_PROBES` must agree, relative. 1e-03.
#:
#: Applied to the **finest grid and the extrapolant**, and reported at every grid -- because the
#: non-uniformity is discretisation and shrinks with the mesh, so a tolerance a coarse grid had to
#: pass would have to be the coarse grid's number. Measured spread over the three hoop probes:
#:
#: ==========  ==========  ==========  ==========  ==============
#: solver      n = 8       n = 16      n = 32      extrapolant
#: ==========  ==========  ==========  ==========  ==============
#: Sestra      4.519e-05   3.054e-05   1.298e-05   9.176e-06
#: Abaqus      1.325e-03   3.291e-04   1.126e-04   **3.653e-04**
#: ==========  ==========  ==========  ==========  ==============
#:
#: 1e-03 is 2.7x the worst of the two applied values (Abaqus' extrapolant) and 8.9x its finest
#: grid; Abaqus' coarsest grid at 1.325e-03 is deliberately outside it. Sestra's *two generator*
#: probes are mirror images of one another and agree to the last digit a ``.SIN`` stores at every
#: grid -- its whole spread is the mid-hoop probe -- while Abaqus' two generators differ from each
#: other, which is the ``S4R`` orientation asymmetry :mod:`curved_compare` measures. Losing the
#: hoop state altogether -- an arc end restrained, or a symmetry dof dropped -- fans these three
#: out by percent.
HOOP_REL_TOL = 1.0e-03

#: How closely a solver's reaction total must match ``(p r L, p r L, 0)``, relative. 1e-06.
#:
#: Measured: Sestra returns ``(-628318.529, -628318.529, 6.4e-09)`` against ``628318.5307``
#: (2.7e-09 relative) and Abaqus' own equilibrium check closes to the same order. So this is
#: hundreds of times the worst residual either solver leaves, and it is an equilibrium
#: identity rather than a modelling approximation -- there is nothing here for a physical
#: effect to consume.
REACTION_REL_TOL = 1.0e-06


def hoop_stress() -> float:
    """``p r / t`` -- the membrane hoop stress, pascals. ``2.0e+07``.

    Not read off either solver, and deliberately not compared against one: the displacement
    sidecar the Abaqus route writes carries no stress, so a cross-solver stress comparison
    would be one-sided. It is here because ``w / r`` *is* ``sigma_theta / E``, so the radial
    check is the hoop-stress check with a modulus in it -- which is stated rather than left
    for a reader to notice.
    """
    from . import curved_model as cm

    return cm.PRESSURE * cm.PANEL_RADIUS / cm.PANEL_THICKNESS


def radial_displacement() -> float:
    """``p r^2 / (E t)`` -- the radial expansion, metres, uniform over the whole panel.

    ``1.9047619047619048e-04``. No ``nu``: the ends are open, so ``sigma_z = 0`` and the hoop
    strain is ``sigma_theta / E`` with nothing subtracted. That independence is itself a check
    -- a run in which the arc ends were restrained would pick up a ``1 - nu^2`` and come out
    9% stiffer.
    """
    from . import curved_model as cm

    props = cm.section_properties()
    return cm.PRESSURE * cm.PANEL_RADIUS**2 / (props["E"] * cm.PANEL_THICKNESS)


def axial_strain() -> float:
    """``-nu p r / (E t)`` -- the axial Poisson strain, dimensionless. ``-2.857142857e-05``.

    Negative: an internally pressurised open cylinder gets **shorter** as it swells. The slope
    of ``u_z`` against ``z``, and what :func:`axial_displacement` integrates.
    """
    from . import curved_model as cm

    props = cm.section_properties()
    return -props["nu"] * cm.PRESSURE * cm.PANEL_RADIUS / (props["E"] * cm.PANEL_THICKNESS)


def axial_displacement(z: float) -> float:
    """``-nu p r z / (E t)`` -- the axial displacement at height ``z``, metres.

    Measured from ``z = 0``, the plane :data:`curved_model.AXIAL_REFERENCE_SUPPORT` pins.
    ``-8.975979010e-05`` at ``z = L``. Exactly linear, which is a stronger statement than the
    value at one point: measured on the Sestra side at ``n = 32``, ``u_z`` at ``L/4``, ``L/2``
    and ``3L/4`` is ``-2.243319e-05``, ``-4.486638e-05`` and ``-6.729957e-05`` -- ratios
    1:2:3 to every digit stored.
    """
    return axial_strain() * float(z)


def edge_reaction() -> float:
    """``p r L`` -- each symmetry edge's in-plane reaction, newtons. ``628318.5307179586``.

    Derived twice in the module docstring, from the pressure and from the wall, which is what
    makes it the check that the whole load arrived rather than a restatement of the input. The
    ``theta = 0`` edge reacts it in ``-y`` and the ``theta = 90`` edge in ``-x``, and no other
    support restrains either direction -- so the model's whole reaction total is
    ``(-p r L, -p r L, 0)`` and the per-edge statement and the global one are the same check.
    """
    from . import curved_model as cm

    return cm.PRESSURE * cm.PANEL_RADIUS * cm.PANEL_LENGTH


def faceted_area(count: int) -> float:
    """The inscribed polygon's surface area at ``count`` facets a side, square metres.

    ``2 count r L sin(pi / (4 count))``: each chord is ``2 r sin(dtheta / 2)`` with
    ``dtheta = pi / (2 count)``, and there are ``count`` of them times the length ``L``. Below
    the exact ``pi r L / 2`` by 8.0e-04 relative at ``count = 8`` and 5.0e-05 at 32 -- an
    ``O(h^2)`` deficit, and the cheapest independent statement of the chord error the whole
    convergence study is about.
    """
    from . import curved_model as cm

    step = 0.5 * math.pi / count
    return 2.0 * count * cm.PANEL_RADIUS * cm.PANEL_LENGTH * math.sin(0.5 * step)


def expected_components(probe_name: str) -> dict[str, float]:
    """The exact displacement of one probe, component by component, in the table's own names.

    The membrane solution is known in closed form *everywhere*, so every probe has six exact
    numbers rather than one: the radial expansion split over ``u1``/``u2`` by the probe's hoop
    angle, the axial contraction in ``u3``, and **zero** in all three rotations -- a pure
    membrane state has no bending anywhere. That last part is the half a single-quantity check
    would miss: measured, both solvers put every rotation below 6e-10 rad at every probe and
    every density, which is what says the faceted panel is carrying the load as membrane and
    not as a shallow arch.
    """
    from . import curved_model as cm

    basis = cm.RADIAL_BASIS.get(probe_name)
    if basis is None:
        raise KeyError(
            f"{probe_name!r} is not one of the panel's probes, so its exact displacement is not "
            f"stated here; the model's probes are {[p.name for p in cm.PROBE_POINTS]}"
        )
    probe = next(p for p in cm.PROBE_POINTS if p.name == probe_name)
    radial = radial_displacement()
    expected = {"u1": 0.0, "u2": 0.0, "u3": axial_displacement(probe.xyz[2]), "r1": 0.0, "r2": 0.0, "r3": 0.0}
    for component, direction in basis:
        expected[component] = radial * direction
    return expected


def predictions() -> dict[str, float]:
    """Every closed form for this model, by name. What the report prints beside the solvers."""
    from . import curved_model as cm

    props = cm.section_properties()
    return {
        "E": props["E"],
        "nu": props["nu"],
        "hoop_stress": hoop_stress(),
        "radial_displacement": radial_displacement(),
        "axial_strain": axial_strain(),
        "axial_displacement_at_L": axial_displacement(cm.PANEL_LENGTH),
        "edge_reaction": edge_reaction(),
        "face_area": cm.face_area(),
    }
