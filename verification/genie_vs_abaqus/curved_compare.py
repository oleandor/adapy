"""The curved case's comparison: convergence, extrapolation, and the guards a cylinder needs.

:mod:`compare` is reused unchanged for the cross-solver verdict -- the per-point, per-component
table, the probe-set rule and the all-zero rule all apply as written -- and
:func:`plate_hand_check.observed_order` / :func:`plate_hand_check.richardson` unchanged for the
arithmetic. What this module adds is the part a **curved** shell needs and a flat one did not.

Why the tolerances are what they are
====================================

:data:`CURVED_REL_TOL` is **2e-03** and the two hand checks are 2e-04 and 2e-03; all three are
read off the convergence study. Three steps, each one a measurement, and the third is the finding
of this case.

**1. Both solvers converge at second order on the headline quantity.** Radial displacement at
:data:`curved_model.RADIAL_PROBE`, against ``p r^2 / (E t) = 1.904761904762e-04``:

===========  ======  ===================  ==========  ===================  ==========
n per side   nodes   Sestra ``FQUS``      rel         Abaqus ``S4R``       rel
===========  ======  ===================  ==========  ===================  ==========
8            81      1.895564346341e-04   4.829e-03   1.896738685900e-04   4.212e-03
16           289     1.902506337501e-04   1.184e-03   1.902670483105e-04   1.098e-03
32           1089    1.904212549562e-04   2.884e-04   1.904127275338e-04   3.332e-04
===========  ======  ===================  ==========  ===================  ==========

Both approach the closed form **from below** -- a polygon inscribed in the cylinder is stiffer in
hoop than the cylinder -- and the single error ratio three values give is **4.0687** for Sestra
and **4.0709** for Abaqus, i.e. ``p = 2.0246`` and ``p = 2.0257``. Second order to three figures
on both sides, measured.

**2. The extrapolants land on the closed form, one much closer than the other.**

    Sestra   1.904768562247e-04   (closed form 1.904761904762e-04, rel **+3.495e-06**)
    Abaqus   1.904601519194e-04   (closed form 1.904761904762e-04, rel **-8.420e-05**)
    between them                                                    rel **8.769e-05**

and the axial contraction, which is the second closed form and cost nothing:

    Sestra   -8.975982203491e-05  against ``-nu p r L / (E t) = -8.975979010257e-05``, +3.558e-07
    Abaqus   -8.976010266464e-05  against the same,                                  +3.482e-06

**3. What the extrapolation buys here is the closed form, not the cross-solver number.** That is
the finding of this case and it is the opposite of the flat strip's, so it is stated rather than
left in a table. Between the two extrapolated tables the worst significant component is
**9.847e-04** (``T90_Q.u3``); at the finest grid it is **1.133e-03**. Those are the same number:
extrapolating did **not** tighten the cross-solver agreement at all, where on the flat strip it
tightened it 44x. What it did do is bring each solver onto an **exact** closed form -- 3.495e-06
and 8.420e-05 radially, 3.247e-07 and 2.305e-04 axially -- which is a statement the flat case
could only make against one closed form and which this case makes against two.

The reason is measured, and it is not the translation. adapy's grid is built analytically and is
**exactly** mirror-symmetric about the ``theta = 45`` plane; CAE's own mesh of the imported face
is too, to the float32 the ODB stores node coordinates in (measured: radius error 5.9e-08 m,
departure from a uniform hoop division 1.9e-05 degrees, departure from mirror symmetry 3.9e-05
degrees at ``n = 8``). So both solvers are given the same symmetric structure. Sestra answers
symmetrically: its ``theta = 0`` and ``theta = 90`` probes carry the **same** radial displacement
to every digit a ``.SIN`` stores, at every grid. Abaqus does not -- its two mirror-image probes
differ by 1.325e-03, 3.291e-04 and 1.126e-04 at the three grids, **oscillating in sign** between
the last two. That is an ``S4R`` property: a reduced-integration element's stiffness is not
invariant to which of its corners is node 1, and every facet of this panel is generated with the
same hoop-then-axial ordering, so the stiffness field is not mirror-symmetric even though the mesh
is. It was checked against the one alternative explanation available: restraining ``u_z`` at
**both** ``z = 0`` corners instead of one -- a symmetric constraint set, both vertices, both
exactly satisfied by the closed form -- removed the 2.54e-08 m axial offset at ``theta = 90``
entirely (to 8.8e-39) and left the radial asymmetry at **1.3244e-03**, unchanged in the fourth
digit. So the asymmetry is the element and not the support, and the production model keeps the
single corner because one node is the minimum that removes the one free mode.

Because the asymmetry oscillates in sign it is not a single-rate term, and Richardson amplifies
rather than removes it: Abaqus' extrapolated hoop spread (3.653e-04) is **larger** than its
finest grid's (1.126e-04), and its extrapolated axial slope (2.305e-04 from the closed form) is
worse than its finest grid's (2.161e-05). That is also why its per-component orders scatter.
Measured over the 23 significant components of each solver's twelve probes:

==========  =============================  ================================================
solver      observed order, min .. max     what sets the spread
==========  =============================  ================================================
Sestra      1.9664 .. 2.0246               nothing: two clean families, the radial at 2.0246
                                           and the axial at 1.9988
Abaqus      1.4570 .. 2.9551               the orientation-asymmetry term, which is not a
                                           smooth power of h, riding on an ``O(h^2)`` answer
==========  =============================  ================================================

So :data:`CURVED_ORDER_BAND` is **(1.3, 3.2)** rather than the plate case's ``(1.5, 3.0)``, set
from those numbers with about 11% of margin at each end, and the range is *reported* on every run
(:attr:`ConvergenceReport.order_range`) so a sequence that moved is visible rather than merely
admitted.

:data:`CURVED_REL_TOL` is then **2e-03**, 2.0x the measured 9.847e-04, and
:data:`CURVED_MESH_REL_TOL` is the same 2e-03 at the finest grid, 1.8x its measured 1.133e-03.
They are equal here where the flat case's differ by ten, and that equality is the finding above
rather than an oversight. What still holds is the discipline: the **coarsest** grid's worst
cross-solver component is 3.422e-03 and **fails** 2e-03, so the tolerance is not one a coarse mesh
would pass. And it is orders below what it exists to catch:

===================================================  =========================================
defect                                                what it does to the answer
===================================================  =========================================
the pressure's sign reversed on the curved face       the panel **contracts**: the answer
                                                      changes sign (that is
                                                      :func:`assert_pressure_expands`)
a symmetry dof dropped (``ur_z`` on either edge)      the panel is no longer a quarter of a
                                                      cylinder; percent
the arc ends restrained instead of free               a ``1 - nu^2`` appears in ``w``: 9%
the radius wrong by 1%                                2% -- ``w`` goes as ``r^2``
the thickness read as 12 mm instead of 10             17% -- ``w`` goes as ``1 / t``
half the pressure                                     2x
the panel meshed to fewer facets than asked for       its place in the refinement sequence is
                                                      not the one the extrapolation assumes
                                                      (:func:`curved_model.assert_shell_grid`)
===================================================  =========================================

The tight statements of this case are therefore the two **hand checks** and not the cross-solver
table: :data:`curved_hand_check.RADIAL_REL_TOL` at 2e-04 against an exact closed form, and
:data:`curved_hand_check.AXIAL_REL_TOL` at 2e-03 against a second one. Both extrapolants meet
both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import compare, curved_hand_check, curved_model, plate_hand_check
from .displacements import COMPONENTS, DisplacementTable
from .plate_compare import NotASequence, _assert_one_sequence

#: Cross-solver relative tolerance on the **extrapolated** tables. 2e-03, which is 2.0x the
#: measured worst significant component (9.847e-04, ``T90_Q.u3``). See step 3 of the module
#: docstring for why that number is an ``S4R`` orientation asymmetry rather than a translation
#: residual, and why extrapolating does not reduce it here.
CURVED_REL_TOL = 2.0e-03

#: Cross-solver relative tolerance on a **single grid**, applied to the finest. 2e-03 -- the same
#: as :data:`CURVED_REL_TOL`, because both are dominated by the same asymmetry and the
#: extrapolation removes none of it. Measured: 1.133e-03 at the finest grid (1.8x of headroom) and
#: 3.422e-03 at the coarsest, which therefore **fails** it -- a tolerance a coarse mesh would pass
#: is not a tolerance.
CURVED_MESH_REL_TOL = 2.0e-03

#: The admissible observed-convergence-order band for this case. ``(1.3, 3.2)``.
#:
#: Wider than :data:`plate_hand_check.ORDER_BAND` for a measured reason and not for room: see
#: step 3 of the module docstring. Sestra's 23 significant components all land in
#: 1.9664 .. 2.0246; Abaqus' scatter 1.4570 .. 2.9551 because CAE's own mesh on this patch is not
#: mirror-symmetric about ``theta = 45`` while adapy's analytic grid is, and that asymmetry term
#: is not a smooth power of ``h``. The band still excludes a first-order sequence -- which is what
#: a locking element or a mesh that was not actually refined would give -- and the measured range
#: is reported on every run so widening it further would be visible.
CURVED_ORDER_BAND = (1.3, 3.2)


class NotUniform(AssertionError):
    """The panel's radial expansion is not uniform, so this is not the membrane hoop state.

    Which means the closed form the whole case is checked against -- ``p r^2 / (E t)``, with no
    ``nu`` and no bending -- is the wrong one. Fatal, because the numbers would otherwise look
    plausible: a panel whose arc ends were restrained still expands, still reacts the whole
    pressure, and is simply 9% stiffer in the middle.
    """


class ContractsUnderInternalPressure(AssertionError):
    """The panel moved **inwards** under an internal pressure.

    The one failure a curved face makes possible that a flat plate does not, because "outward" is
    a direction that turns along the surface: the pressure's sign is fixed by the face's own
    normal (``side1Faces`` in CAE, the facet's node ordering in Sesam), so a normal that came
    through flipped, or a magnitude with the wrong sign, gives a perfectly converged, perfectly
    equilibrated answer of the wrong sign. See
    :data:`curved_abaqus_runner.PRESSURE_SIGN_MEASUREMENT`.
    """


class ReactionMismatch(AssertionError):
    """A solver did not react the load the pressure integrates to over the quarter panel."""


class AxialRestraintPresent(AssertionError):
    """The axial Poisson contraction is not free, so the second closed form is the wrong one."""


class MeshingGapClosed(AssertionError):
    """adapy can now mesh a ``PlateCurved``, and this package's own grid should give way to it.

    Not a failure of the model -- a failure of this module's *premise*. Raised loudly rather
    than left as a comment, because the day the gap closes is the day
    :func:`curved_model.shell_grid` stops being the nearest expressible equivalent and starts
    being a second opinion about the mesh.
    """


@dataclass(frozen=True)
class ComponentConvergence:
    """One component at one probe, over the refinement sequence."""

    probe: str
    component: str
    values: tuple[float, ...]
    #: ``None`` when the component is zero to within :data:`compare.ABS_FLOOR` at every grid, in
    #: which case there is no rate to read and the finest value is carried through as it stands.
    convergence: plate_hand_check.Convergence | None
    extrapolated: float

    @property
    def negligible(self) -> bool:
        return self.convergence is None


@dataclass
class ConvergenceReport:
    """One solver: every component's sequence, plus the two closed-form quantities.

    :attr:`radial` is the sequence of *radial* displacements at
    :data:`curved_model.RADIAL_PROBE` -- recovered through
    :func:`curved_abaqus_runner.radial_displacement` rather than read off one component, because
    at a mid-hoop probe the radial displacement is split between ``u1`` and ``u2``. It is what
    the printed table leads with, because it is the quantity the first closed form predicts.
    """

    solver: str
    solver_version: str
    counts: tuple[int, ...]
    node_counts: tuple[int, ...]
    radial_values: tuple[float, ...] = ()
    radial: plate_hand_check.Convergence | None = None
    axial_values: tuple[float, ...] = ()
    axial: plate_hand_check.Convergence | None = None
    components: list[ComponentConvergence] = field(default_factory=list)

    def at(self, probe: str, component: str) -> ComponentConvergence:
        for entry in self.components:
            if entry.probe == probe and entry.component == component:
                return entry
        raise KeyError(f"{probe}.{component} is not in this report")

    @property
    def order_range(self) -> tuple[float, float]:
        """``(min, max)`` observed order over every significant component.

        Reported on every run, and that is the point: this case's admissible band
        (:data:`CURVED_ORDER_BAND`) is wider than the plate strip's because one solver's scatter
        made it so, so the scatter itself has to be visible or the band is just slack.
        """
        orders = [entry.convergence.order for entry in self.components if entry.convergence is not None]
        if not orders:
            return (0.0, 0.0)
        return (min(orders), max(orders))

    @property
    def radial_residuals(self) -> tuple[float, ...]:
        """``|value| / w - 1`` at each grid, for the headline radial displacement."""
        closed = curved_hand_check.radial_displacement()
        return tuple(abs(abs(v) / closed - 1.0) for v in self.radial_values)

    def format_table(self) -> str:
        closed = curved_hand_check.radial_displacement()
        head = f"{'n/side':>7} {'nodes':>7} {'radial [m]':>21} {'|rel| vs p r^2/(E t)':>21}"
        lines = [
            f"{self.solver} {self.solver_version} -- quarter cylinder, radial expansion at "
            f"{curved_model.RADIAL_PROBE}",
            head,
            "-" * len(head),
        ]
        residuals = self.radial_residuals
        for index, count in enumerate(self.counts):
            lines.append(
                f"{count:>7} {self.node_counts[index]:>7} {self.radial_values[index]:>21.12e} "
                f"{residuals[index]:>21.3e}"
            )
        lines.append("-" * len(head))
        if self.radial is None:
            lines.append("the radial displacement is zero to within the floor -- nothing converged here")
            return "\n".join(lines)
        rel = abs(abs(self.radial.extrapolated) / closed - 1.0)
        lines.append(
            f"observed order {self.radial.order:.4f} (error ratio "
            f"{_ratio_text(self.radial_values)}), Richardson extrapolant {self.radial.extrapolated:.12e}"
        )
        lines.append(
            f"closed form {closed:.12e}: the finest grid is {residuals[-1]:.3e} from it, " f"the extrapolant {rel:.3e}"
        )
        if self.axial is not None:
            axial_closed = curved_hand_check.axial_displacement(curved_model.PANEL_LENGTH)
            axial_rel = abs(self.axial.extrapolated / axial_closed - 1.0)
            lines.append(
                f"axial contraction at {curved_model.AXIAL_PROBE}: order {self.axial.order:.4f}, "
                f"extrapolant {self.axial.extrapolated:.12e} against -nu p r L / (E t) = "
                f"{axial_closed:.12e}, rel {axial_rel:.3e}"
            )
        low, high = self.order_range
        lines.append(
            f"observed order over all {sum(1 for c in self.components if c.convergence is not None)} "
            f"significant component(s): {low:.4f} .. {high:.4f}, admissible {CURVED_ORDER_BAND}"
        )
        return "\n".join(lines)


def _ratio_text(values) -> str:
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    parts = []
    for index in range(len(diffs) - 1):
        parts.append("inf" if diffs[index + 1] == 0.0 else f"{abs(diffs[index] / diffs[index + 1]):.4f}")
    return ", ".join(parts) or "-"


def grid_count(solve) -> int:
    """The elements-per-side this solve was run at, recovered from its seed.

    ``PlateSolve`` carries a ``mesh_size`` because it is the plate case's carrier, and for this
    case the seed and the count are one-to-one (:data:`curved_model.MESH_SIZES`). Recovered
    rather than stored so there is one statement of the relation.
    """
    count = int(round(curved_model.PANEL_ARC / solve.mesh_size))
    if abs(curved_model.PANEL_ARC / count - solve.mesh_size) > 1.0e-09 * solve.mesh_size:
        raise NotASequence(
            f"a seed of {solve.mesh_size!r} is not PANEL_ARC / n for any whole n (the nearest is "
            f"{count}). This case's grids are named by their count and every seed is derived from "
            f"one, so a seed that is not is a solve from a different study."
        )
    return count


def radial_displacement(table: DisplacementTable, probe_name: str) -> float:
    """The radial displacement at one probe, positive outward.

    Delegates to :func:`curved_abaqus_runner.radial_displacement` so the recovery -- ``u1`` on
    one generator, ``u2`` on the other, ``(u1 + u2) / sqrt(2)`` at ``theta = 45`` -- is written
    once. Imported lazily because that module imports this one's siblings.
    """
    from .curved_abaqus_runner import radial_displacement as recover

    return recover(table, probe_name)


def extrapolate(solves, *, abs_floor: float = compare.ABS_FLOOR) -> DisplacementTable:
    """One :class:`DisplacementTable` from a refinement sequence, Richardson per component.

    ``solves`` is a list of :class:`plate_sestra_runner.PlateSolve`, coarse to fine. Each
    component of each probe is extrapolated independently, with the order read off its own three
    values -- not borrowed from the radial sequence's -- because a component converging at a
    different rate is exactly what should be visible rather than smoothed over. On this model it
    is visible: see :data:`CURVED_ORDER_BAND`.

    A component whose magnitude never exceeds ``abs_floor`` is carried through at its finest
    value instead of extrapolated. That is not a convenience: the three rotations are zero by
    construction here -- a pure membrane state has no bending anywhere -- and measured at 5.3e-10
    (Sestra) and 3.5e-08 (Abaqus), from which :func:`plate_hand_check.observed_order` would
    compute a meaningless rate and :func:`plate_hand_check.richardson` would then amplify it.
    They are still *compared*, by :func:`compare.compare`'s absolute rule.

    Raises :class:`plate_compare.NotASequence` when the list is not one solver coarse to fine,
    and :class:`plate_hand_check.NotConverging` when a **significant** component does not form a
    converging sequence -- naming the probe and the component.
    """
    _assert_one_sequence(solves)
    finest = solves[-1]
    tables = [solve.table for solve in solves]

    displacements: dict[str, tuple[float, float, float, float, float, float]] = {}
    for probe in sorted(finest.table.displacements):
        row = []
        for component in COMPONENTS:
            values = [table.component(probe, component) for table in tables]
            if max(abs(v) for v in values) <= abs_floor:
                row.append(values[-1])
                continue
            try:
                row.append(plate_hand_check.richardson(values, order_band=CURVED_ORDER_BAND).extrapolated)
            except plate_hand_check.NotConverging as exc:
                raise plate_hand_check.NotConverging(
                    f"{finest.table.solver}: {probe}.{component} does not converge over "
                    f"{tuple(grid_count(solve) for solve in solves)} elements a side: {exc}"
                ) from exc
        displacements[probe] = tuple(row)  # type: ignore[arg-type]

    return DisplacementTable(
        solver=finest.table.solver,
        solver_version=finest.table.solver_version,
        model=finest.table.model,
        load_case=finest.table.load_case,
        displacements=displacements,
        node_ids=dict(finest.table.node_ids),
        node_coords=dict(finest.table.node_coords),
        source="Richardson extrapolation of {0} at {1} elements a side".format(
            finest.source or finest.table.source, ", ".join(str(grid_count(solve)) for solve in solves)
        ),
    )


def convergence_report(solves) -> ConvergenceReport:
    """Every component's sequence for one solver, plus the two closed-form quantities."""
    _assert_one_sequence(solves)
    finest = solves[-1]
    tables = [solve.table for solve in solves]
    report = ConvergenceReport(
        solver=finest.table.solver,
        solver_version=finest.table.solver_version,
        counts=tuple(grid_count(solve) for solve in solves),
        node_counts=tuple(solve.node_count for solve in solves),
    )
    radial_values = tuple(radial_displacement(table, curved_model.RADIAL_PROBE) for table in tables)
    report.radial_values = radial_values
    if max(abs(v) for v in radial_values) > compare.ABS_FLOOR:
        report.radial = plate_hand_check.richardson(radial_values, order_band=CURVED_ORDER_BAND)
    axial_values = tuple(table.component(curved_model.AXIAL_PROBE, "u3") for table in tables)
    report.axial_values = axial_values
    if max(abs(v) for v in axial_values) > compare.ABS_FLOOR:
        report.axial = plate_hand_check.richardson(axial_values, order_band=CURVED_ORDER_BAND)

    for probe in sorted(finest.table.displacements):
        for component in COMPONENTS:
            values = tuple(table.component(probe, component) for table in tables)
            if max(abs(v) for v in values) <= compare.ABS_FLOOR:
                report.components.append(ComponentConvergence(probe, component, values, None, values[-1]))
                continue
            conv = plate_hand_check.richardson(values, order_band=CURVED_ORDER_BAND)
            report.components.append(ComponentConvergence(probe, component, values, conv, conv.extrapolated))
    return report


def assert_pressure_expands(table: DisplacementTable) -> float:
    """Raise :class:`ContractsUnderInternalPressure` unless every probe moved **outward**.

    Returns the smallest radial displacement over the twelve probes, so a caller can report how
    much margin the sign has -- and on a correct run that margin is the whole answer, 1.9e-04 m.

    The check this case exists for as much as any. A pressure's direction on a curved face is
    decided by the face's own normal, which turns through 90 degrees across this panel: Abaqus
    takes a positive ``*Dsload`` magnitude as acting *into* ``side1Faces``, so
    :data:`curved_model.SIGNED_PRESSURE_MAGNITUDE` is negative, and the Sesam side's consistent
    nodal vector is built along each facet's own outward normal. Get either wrong -- a flipped
    normal, a positive magnitude, a facet whose node ordering reversed -- and the panel collapses
    inwards while converging beautifully at second order, balancing its reactions exactly, and
    passing every other check in this module. Measured on the Abaqus route with the magnitude
    reversed: ``u1`` at ``T0_MID`` came back **negative** and the reaction total reversed with it
    (:data:`curved_abaqus_runner.PRESSURE_SIGN_MEASUREMENT`).
    """
    radial = {probe.name: radial_displacement(table, probe.name) for probe in curved_model.PROBE_POINTS}
    worst = min(radial.values())
    peak = max(abs(v) for v in radial.values())
    if peak <= compare.ABS_FLOOR:
        raise ContractsUnderInternalPressure(
            f"{table.solver}: every probe's radial displacement is at or below the absolute floor "
            f"({peak:.3e}), so there is no sign to read. An unloaded panel neither expands nor "
            f"contracts, which is not the same as an internal pressure that arrived."
        )
    if worst <= 0.0:
        inward = sorted(name for name, value in radial.items() if value <= 0.0)
        raise ContractsUnderInternalPressure(
            f"{table.solver}: {len(inward)} of {len(radial)} probe(s) moved INWARD under an internal "
            f"pressure of {curved_model.PRESSURE} Pa -- {inward}, the worst at {worst:.6e} m. An "
            f"internal pressure must expand the shell. On the Abaqus route the sign is "
            f"side1Faces': a positive Pressure magnitude acts into the face's own outward radial "
            f"normal, so curved_model.SIGNED_PRESSURE_MAGNITUDE is negative; on the Sesam route it "
            f"is each facet's node ordering, which curved_model.assert_shell_grid checks points "
            f"away from the cylinder axis."
        )
    return worst


def assert_hoop_uniform(table: DisplacementTable, *, rel_tol: float = curved_hand_check.HOOP_REL_TOL) -> float:
    """Raise :class:`NotUniform` unless the three hoop probes carry the same radial displacement.

    Returns the measured spread, relative, so a caller can report it on a pass -- which is worth
    doing, because the two solvers' spreads are the whole explanation of their cross-solver
    residual: **1.298e-05** (Sestra, on an analytically symmetric grid) against **1.126e-04**
    (Abaqus, on CAE's own mesh, which is not mirror-symmetric about ``theta = 45``).

    This is the check that the membrane hoop state survived translation. ``w = p r^2 / (E t)`` is
    uniform over the *whole* panel, so any variation across the hoop is discretisation -- and if
    the arc ends were restrained instead of free, or a symmetry dof were lost, these three would
    fan out by percent while every number in the report stayed plausible. Both solvers given the
    same wrong boundary would agree with each other beautifully; that is what this exists for.
    """
    values = []
    for probe in curved_model.HOOP_PROBES:
        if probe not in table.displacements:
            raise NotUniform(
                f"{table.solver} has no probe {probe!r}, so whether the panel is in a uniform hoop "
                f"state cannot be established; it carries {sorted(table.displacements)}."
            )
        values.append(radial_displacement(table, probe))
    peak = max(abs(v) for v in values)
    if peak <= compare.ABS_FLOOR:
        raise NotUniform(
            f"{table.solver}: all three hoop probes are at or below the absolute floor "
            f"({peak:.3e}), so there is no expansion whose uniformity could be checked. An "
            f"unloaded cylinder is uniform round its hoop too."
        )
    spread = (max(values) - min(values)) / peak
    if spread > rel_tol:
        pairs = ", ".join(f"{p}={radial_displacement(table, p):.9e}" for p in curved_model.HOOP_PROBES)
        raise NotUniform(
            f"{table.solver}: the radial expansion varies by {spread:.3e} round the panel's hoop "
            f"({pairs}), above {rel_tol:.1e}. The membrane solution w = p r^2 / (E t) is uniform "
            f"over the whole panel, so this much variation means it is not the right closed form: "
            f"check that both arc ends are free (restraining one puts a 1 - nu^2 into w, 9%) and "
            f"that both symmetry records reached the deck with all three of their dofs."
        )
    return spread


def assert_axial_contraction(
    table: DisplacementTable, *, rel_tol: float = curved_hand_check.AXIAL_REL_TOL
) -> dict[str, float]:
    """Raise unless ``u_z`` is ``-nu p r z / (E t)`` -- linear in ``z``, through zero, at that slope.

    Returns ``{"slope": ..., "relative": ..., "worst_nonlinearity": ...}``. Three statements, not
    one, because they fail separately:

    * the **slope** is the closed form and it is what says the ends are open: an axial restraint
      anywhere gives a smaller one, and restraining both gives zero;
    * the value at ``z = 0`` must be zero, which is what says
      :data:`curved_model.AXIAL_REFERENCE_SUPPORT` is the *only* axial restraint -- a second one
      elsewhere would shift the whole field;
    * and the field must be **linear** in ``z``, which no single probe could say. Measured on the
      Sestra side at the finest grid, ``u_z`` at ``L/4``, ``L/2``, ``3L/4`` and ``L`` is
      ``-2.243319e-05``, ``-4.486638e-05``, ``-6.729957e-05`` and ``-8.973276e-05`` -- ratios
      1 : 2 : 3 : 4 to every digit stored, i.e. a nonlinearity of 0.0.

    Raises :class:`AxialRestraintPresent`, because that is what a failure here means: the axial
    Poisson contraction is being held somewhere, and the second closed form is then the wrong one.
    """
    strain = curved_hand_check.axial_strain()
    fitted: list[float] = []
    worst_nonlinear = 0.0
    for probe in curved_model.PROBE_POINTS:
        z = probe.xyz[2]
        value = table.component(probe.name, "u3")
        if z == 0.0:
            # The reference plane: u_z is zero there by the closed form and by the one support.
            if abs(value) > abs(strain * curved_model.PANEL_LENGTH) * rel_tol:
                raise AxialRestraintPresent(
                    f"{table.solver}: u3 at {probe.name} (z = 0, the reference plane) is {value:.6e} m "
                    f"and the closed form says 0. Only one node fixes u_z, at (r, 0, 0), and the exact "
                    f"solution has u_z = 0 for every theta at z = 0 -- so a non-zero value here means "
                    f"something else is restraining the axial direction, or the reference node is not "
                    f"where the model says."
                )
            continue
        fitted.append(value / z)
    if not fitted:
        raise AxialRestraintPresent(
            f"{table.solver}: no probe at a non-zero z, so there is no axial slope to read; the "
            f"table holds {sorted(table.displacements)}."
        )
    slope = math.fsum(fitted) / len(fitted)
    relative = abs(slope / strain - 1.0)
    if abs(slope) <= abs(strain) * rel_tol:
        # u_z held everywhere. Checked before the nonlinearity, which would divide by this slope --
        # and reported as the restraint it is rather than as a division error.
        raise AxialRestraintPresent(
            f"{table.solver}: the axial strain measured over {len(fitted)} probe(s) is {slope:.9e}, "
            f"i.e. zero to within {rel_tol:.1e} of -nu p r / (E t) = {strain:.9e}. An open-ended "
            f"cylinder contracts axially by exactly that, so u_z pinned at zero everywhere means the "
            f"axial direction is restrained along the panel and not only at the one reference node."
        )
    worst_nonlinear = max(abs(value / slope - 1.0) for value in fitted)
    if relative > rel_tol:
        raise AxialRestraintPresent(
            f"{table.solver}: the axial strain measured over {len(fitted)} probe(s) is {slope:.9e} and "
            f"-nu p r / (E t) = {strain:.9e} -- {relative:.3e} away, above {rel_tol:.1e}. A cylinder "
            f"whose ends are free contracts axially by exactly that; a smaller magnitude means an "
            f"axial restraint that the model did not ask for, and zero means u_z is held everywhere."
        )
    if worst_nonlinear > rel_tol:
        raise AxialRestraintPresent(
            f"{table.solver}: u3 / z varies by {worst_nonlinear:.3e} between probes, above "
            f"{rel_tol:.1e}, so the axial displacement is not linear in z. The membrane solution has "
            f"a constant axial strain, so curvature in this field means a restraint acting somewhere "
            f"along the panel rather than at the single reference node."
        )
    return {"slope": slope, "relative": relative, "worst_nonlinearity": worst_nonlinear}


def assert_reaction_total(solve, *, rel_tol: float = curved_hand_check.REACTION_REL_TOL) -> float:
    """Raise :class:`ReactionMismatch` unless a solve reacted ``(-p r L, -p r L, 0)``.

    Returns the worst relative component residual. The one scalar that says the *whole* load
    arrived, and the only one from the solver's own bookkeeping rather than from adapy's. It is a
    sharper check here than it was on the flat strip for two reasons: the two routes are given the
    load in different forms **and**, on a curved face, in a different direction per facet; and the
    ``z`` component being zero is itself a closed form -- an open-ended cylinder reacts nothing
    axially -- so a restraint that crept into the arc ends would show up here as a non-zero
    ``fz`` before it showed up anywhere else.
    """
    expected = curved_model.expected_load_total()
    measured = solve.reaction_total
    scale = max(abs(v) for v in expected)
    # The support reaction opposes the applied load, so it is -(p r L) against a positive applied
    # total. Compared component by component including sign: a reaction with the same sign as the
    # load would mean the supports are pushing the panel outwards.
    worst = max(abs(measured[axis] + expected[axis]) for axis in range(3)) / scale
    if worst > rel_tol:
        raise ReactionMismatch(
            f"{solve.table.solver} (n = {grid_count(solve)}) reacted {measured!r} N against an applied "
            f"{expected!r} N, worst component {worst:.3e} of p r L and above {rel_tol:.1e}. The two "
            f"in-plane components are each p r L = {curved_hand_check.edge_reaction():.6f} N, one per "
            f"symmetry edge, and the axial one is zero because the outward normal of a cylinder has no "
            f"z component -- so a non-zero fz means the arc ends are not free."
        )
    return worst


def assert_reference_node_carries_nothing(reaction, *, rel_tol: float = 1.0e-05) -> float:
    """Raise unless the axial reference node's own reaction is negligible.

    Returns its ``fz`` relative to ``p r L``. The closed form says this node carries **nothing**:
    the exact solution has ``u_z = 0`` everywhere on ``z = 0``, so pinning it at one corner
    removes the rigid-body mode without doing any work. That is the claim
    :data:`curved_model.AXIAL_REFERENCE_SUPPORT` makes about itself, and this is where it is
    checked rather than asserted -- a node that *did* carry load would mean the axial contraction
    was being restrained, which is the same defect :func:`assert_axial_contraction` catches from
    the displacement side. Two independent statements of one thing, which is the point.
    """
    scale = curved_hand_check.edge_reaction()
    relative = abs(float(reaction[2])) / scale
    if relative > rel_tol:
        raise AxialRestraintPresent(
            f"the axial reference node at {curved_model.PROBE_POINTS[0].xyz} reacts {reaction!r} N, "
            f"i.e. {relative:.3e} of p r L in z, above {rel_tol:.1e}. It is meant to remove one "
            f"rigid-body mode and carry nothing: the exact solution has u_z = 0 over the whole z = 0 "
            f"plane, so a real reaction there means the panel is being held axially and the closed "
            f"form -nu p r z / (E t) is not the one it is solving."
        )
    return relative


def assert_meshing_gap_is_still_open(findings: dict[str, str] | None = None) -> dict[str, str]:
    """Raise :class:`MeshingGapClosed` if adapy has learned to mesh a ``PlateCurved``.

    Returns what each of the three refusals said. This is the premise
    :func:`curved_model.shell_grid` rests on, and a premise that stops being true silently is
    how a workaround becomes a second, quieter opinion about the mesh. Measured today:

        ``Part.to_fem_obj``        -> ``0 node(s) and 0 element(s)``
        ``PlateCurved.shell_occ``  -> ``NotImplementedError(no message)``
        ``add_fem_sections``       -> ``NotImplementedError(Unsupported combination of
                                       geom_repr=GeomRepr.SHELL, and <class ...PlateCurved>)``
    """
    findings = curved_model.reproduce_meshing_refusals() if findings is None else findings
    closed = []
    if not findings.get("Part.to_fem_obj", "").startswith("0 node(s)"):
        closed.append(f"Part.to_fem_obj now returns {findings.get('Part.to_fem_obj')!r}")
    if "NotImplementedError" not in findings.get("PlateCurved.shell_occ", ""):
        closed.append(f"PlateCurved.shell_occ now returns {findings.get('PlateCurved.shell_occ')!r}")
    if "NotImplementedError" not in findings.get("add_fem_sections", ""):
        closed.append(f"add_fem_sections now {findings.get('add_fem_sections')!r}")
    if closed:
        raise MeshingGapClosed(
            f"adapy can now mesh a PlateCurved, at least in part: {'; '.join(closed)}. That is good "
            f"news and it makes curved_model.shell_grid the wrong thing to keep: the analytic grid "
            f"exists only because the mesher refused, and leaving it in place beside a working mesher "
            f"would mean this case silently stopped testing the path a user takes. Delete the grid, "
            f"mesh the plate, and re-measure the whole convergence table -- the facets will no longer "
            f"be exact rectangles, so consistent_nodal_loads stops being exact too."
        )
    return findings


def format_closed_forms() -> str:
    """The closed forms and the geometry they came from, printed once at the top of a report."""
    predictions = curved_hand_check.predictions()
    return "\n".join(
        [
            f"quarter cylinder: r = {curved_model.PANEL_RADIUS} m, L = {curved_model.PANEL_LENGTH:.12f} m "
            f"(= pi r / 2, the quarter arc's own length), t = {curved_model.PANEL_THICKNESS} m, "
            f"{curved_model.MATERIAL_NAME}, internal p = {curved_model.PRESSURE} Pa",
            "supports: " + "; ".join(f"{name} {why}" for name, _dofs, why in curved_model.EDGE_SUPPORTS),
            "the two arc ends z = 0 and z = L are FREE, which is what makes sigma_z = 0",
            f"face area    pi r L / 2           = {predictions['face_area']:.12f} m2",
            f"hoop stress  p r / t              = {predictions['hoop_stress']:.6f} Pa",
            f"radial       p r^2 / (E t)        = {predictions['radial_displacement']:.12e} m  (uniform)",
            f"axial strain -nu p r / (E t)      = {predictions['axial_strain']:.12e}",
            f"axial at L   -nu p r L / (E t)    = {predictions['axial_displacement_at_L']:.12e} m",
            f"edge reaction p r L               = {predictions['edge_reaction']:.6f} N  (each symmetry edge)",
        ]
    )
