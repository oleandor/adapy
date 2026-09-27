"""Self-test of the comparison framework. No solver, no licence, runs anywhere.

A comparator that reports "0 vs 0, agree" is the failure this package is built against, and
a guard that has never been seen to fire is not a guard. So these checks run the comparator
against deliberately broken inputs and assert that it *raises*:

* a probe present on one side and absent on the other;
* an all-zero table on either side;
* a table with too few probes;
* a probe point the mesh does not seed;
* two coincident nodes at a probe.

Plus the positive cases: a table that agrees passes, and one perturbed just past
:data:`compare.REL_TOL` fails at the right components.

The **plate** case adds five groups of its own, because a shell comparison goes wrong in ways a
beam comparison cannot (:func:`check_plate_closed_forms`,
:func:`check_plate_convergence`, :func:`check_plate_agreement`,
:func:`check_plate_boundary_semantics`, :func:`check_plate_loud_failures`):

* a table built from a mesh with **no shells in it** -- all the numbers, none of the physics;
* a probe on no node of the plate mesh, and two coincident nodes at one;
* the **stiffener missing from one side**, measured as a stiffness ratio far from
  ``EI_plate / (EI_plate + EI_bar)`` rather than as an element count -- including the case where
  it is present but its section is rotated 90 degrees, which is a ratio of 0.924 against 0.376;
* a strip that is **not in cylindrical bending**, which makes the closed form the wrong one by up
  to 9% while leaving every number plausible;
* a reaction total that is not the load;
* a refinement sequence that is not one: two meshes, mixed variants, out of order, or a component
  that does not converge -- each of which would otherwise be extrapolated into a number and then
  compared;
* the plate probe set compared against the frame's, which must be a hard error rather than an
  empty intersection;
* and a support missing from the model's own ``Bc`` records, or fixing the wrong dofs -- those
  three records are what *both* writers translate now that the CAE writer carries a support along
  a plate edge (adapy PR #405), so they are the one place "simply supported" is stated. The
  writer's own emitted ``DisplacementBC`` calls and edge regions are asserted literally in
  :func:`check_plate_boundary_semantics`, on a script emitted without a licence.

The plate groups are checked against **measured** solver output, carried here as the constants
``_SESTRA_BARE`` .. ``_ABAQUS_STIFF`` and ``_MID_DEFLECTION_SEQUENCES``, so the tolerance the
package ships is pinned against the residuals a correct translation actually leaves -- not only
against multiples of itself. That last distinction is why
:func:`check_plate_convergence` asserts that the *coarsest* mesh pair would **fail** at
:data:`plate_compare.PLATE_REL_TOL`: it is what says the extrapolation is doing the work and the
tolerance is not merely generous.

And the **curved** case adds four more (:func:`check_curved_closed_forms`,
:func:`check_curved_convergence`, :func:`check_curved_boundary_semantics`,
:func:`check_curved_loud_failures`), because a *non-planar* plate goes wrong in ways a flat one
cannot:

* an internal pressure that **contracts** the shell -- the one failure a curved face makes
  possible and a flat plate does not, because "outward" is a direction that turns along the
  surface. A flipped face normal, a positive ``*Dsload`` magnitude or one facet whose node
  ordering reversed all give a perfectly converged, perfectly equilibrated answer of the wrong
  sign;
* a **symmetry dof dropped** from either generator, which leaves a model that still solves, still
  reacts the whole pressure, and is no longer a quarter of a cylinder;
* a curved plate that **meshed to fewer shells than expected**, whose chord error is then not the
  one its place in the refinement sequence assumes -- plus the three geometric clauses that say
  the grid is a faceted cylinder at all: nodes off radius ``r``, a warped facet, and a facet
  normal pointing at the axis;
* an axial restraint that was not released, in three independent forms: ``u_z`` held everywhere,
  ``u_z`` non-zero at the reference plane, and ``u_z`` not linear in ``z``;
* a reaction that is half the load, has the same sign as the load, or has a non-zero axial
  component -- an open-ended cylinder reacts nothing in ``z``;
* and the **premise the whole case rests on**: adapy cannot mesh a ``PlateCurved``, so this
  package builds its own analytic grid, and the day that stops being true the guard says so rather
  than leaving a second quiet opinion about the mesh in place.

The curved groups carry the measured output of **six** real solves (``_CURVED_SESTRA`` ..
``_CURVED_CROSS_SOLVER``) and assert the writer defect the case turned up --
``AnalysisPlan.applied_pressure`` is 15.4% wrong and 7.6% asymmetric on a quarter cylinder --
through the writer's own planner, so that check stops passing the day the writer is fixed.

Run it after any change to :mod:`compare`, :mod:`displacements` or the plate and curved modules::

    python -m verification.genie_vs_abaqus.selftest
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

from . import compare, model
from .displacements import (
    AmbiguousProbe,
    DisplacementTable,
    ProbeNotFound,
    sample_fea_result,
)

#: A plausible Sestra-shaped table: the measured portal frame answer, rounded.
_REFERENCE = {
    "BASE_L": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "BASE_R": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "COL_L_MID": (1.016948e-02, 0.0, 7.335531e-06, 0.0, 5.415976e-03, 0.0),
    "COL_R_MID": (1.016948e-02, 0.0, -7.335531e-06, 0.0, 5.415976e-03, 0.0),
    "TOP_L": (2.468646e-02, 0.0, 1.467106e-05, 0.0, 2.898336e-03, 0.0),
    "TOP_R": (2.468646e-02, 0.0, -1.467106e-05, 0.0, 2.898336e-03, 0.0),
    "GIRDER_QTR": (2.468646e-02, 0.0, -2.154147e-03, 0.0, -3.438878e-04, 0.0),
    "GIRDER_MID": (2.468646e-02, 0.0, 0.0, 0.0, -1.424629e-03, 0.0),
    "GIRDER_3QTR": (2.468646e-02, 0.0, 2.154147e-03, 0.0, -3.438878e-04, 0.0),
}


def _table(solver: str, displacements: dict) -> DisplacementTable:
    return DisplacementTable(
        solver=solver,
        solver_version="selftest",
        model="portal_frame",
        load_case=model.LOAD_CASE,
        displacements={k: tuple(v) for k, v in displacements.items()},
        node_ids={k: i for i, k in enumerate(displacements, start=1)},
        source=f"selftest::{solver}",
    )


def _scaled(factor: float) -> dict:
    return {k: tuple(v * factor for v in values) for k, values in _REFERENCE.items()}


@dataclass
class _FakeNodes:
    identifiers: np.ndarray
    coords: np.ndarray


@dataclass
class _FakeMesh:
    nodes: _FakeNodes


class _FakeField:
    name = "RVNODDIS"
    step = 1
    components = ("U1", "U2", "U3", "U4", "U5", "U6")

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


class _FakeResult:
    """Minimum duck-type that :func:`displacements.sample_fea_result` consumes."""

    results_file_path = "selftest://fake"

    def __init__(self, ids, coords, values):
        self.mesh = _FakeMesh(_FakeNodes(np.asarray(ids), np.asarray(coords, dtype=float)))
        self._field = _FakeField(values)

    def get_results_grouped_by_field_value(self):
        return {"RVNODDIS": [self._field]}


def _expect_raise(what: str, exc_type, fn) -> bool:
    try:
        fn()
    except exc_type as exc:
        first_line = str(exc).strip().splitlines()[0]
        print(f"  PASS  {what}: raised {type(exc).__name__}: {first_line[:110]}")
        return True
    except Exception as exc:  # noqa: BLE001 - reporting an unexpected type is the point
        print(f"  FAIL  {what}: raised {type(exc).__name__}, expected {exc_type.__name__}: {exc}")
        return False
    print(f"  FAIL  {what}: did not raise {exc_type.__name__} -- the guard is not working")
    return False


def _expect(what: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {what}{(': ' + detail) if detail else ''}")
    return condition


def check_loud_failures() -> list[bool]:
    print("loud-failure guards (each must raise):")
    ref = _table("sestra", _REFERENCE)
    results = []

    dropped = dict(_REFERENCE)
    dropped.pop("GIRDER_QTR")
    results.append(
        _expect_raise(
            "a probe missing from one side",
            compare.ProbeSetMismatch,
            lambda: compare.compare(ref, _table("abaqus", dropped)),
        )
    )

    results.append(
        _expect_raise(
            "an all-zero table on side B",
            compare.NoSignal,
            lambda: compare.compare(ref, _table("abaqus", _scaled(0.0))),
        )
    )
    results.append(
        _expect_raise(
            "an all-zero table on side A",
            compare.NoSignal,
            lambda: compare.compare(_table("sestra", _scaled(0.0)), ref),
        )
    )
    results.append(
        _expect_raise(
            "two all-zero tables (the 0-vs-0-agree failure mode)",
            compare.NoSignal,
            lambda: compare.compare(_table("sestra", _scaled(0.0)), _table("abaqus", _scaled(0.0))),
        )
    )

    two = {k: _REFERENCE[k] for k in ("TOP_L", "TOP_R")}
    results.append(
        _expect_raise(
            "fewer probes than MIN_PROBES",
            compare.ProbeSetMismatch,
            lambda: compare.compare(_table("sestra", two), _table("abaqus", two)),
        )
    )

    # Coordinate matching: a probe the mesh does not seed, and a duplicated node.
    ids = [1, 2]
    coords = [(0.0, 0.0, 0.0), (0.0, 0.0, 6.0)]
    values = [[1, 0, 0, 0, 0, 0, 0], [2, 0.0246, 0, 0, 0, 0, 0]]
    unseeded = _FakeResult(ids, coords, values)
    results.append(
        _expect_raise(
            "a probe point with no node",
            ProbeNotFound,
            lambda: sample_fea_result(
                unseeded,
                model.PROBE_POINTS,
                solver="fake",
                solver_version="selftest",
                model_name="portal_frame",
                load_case=model.LOAD_CASE,
            ),
        )
    )

    dup = _FakeResult(
        [1, 2],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        [[1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0]],
    )
    results.append(
        _expect_raise(
            "two coincident nodes at a probe (an unmerged joint)",
            AmbiguousProbe,
            lambda: sample_fea_result(
                dup,
                model.PROBE_POINTS[:1],
                solver="fake",
                solver_version="selftest",
                model_name="portal_frame",
                load_case=model.LOAD_CASE,
            ),
        )
    )
    return results


def check_agreement() -> list[bool]:
    print("\nagreement and disagreement (the comparator must discriminate):")
    ref = _table("sestra", _REFERENCE)
    results = []

    identical = compare.compare(ref, _table("abaqus", _REFERENCE))
    results.append(_expect("identical tables agree", identical.ok, f"{len(identical.diffs)} components"))

    # Just inside the tolerance.
    inside = compare.compare(ref, _table("abaqus", _scaled(1.0 + 0.5 * compare.REL_TOL)))
    results.append(
        _expect(
            f"a {50 * compare.REL_TOL:.1f}% perturbation is inside rel_tol={compare.REL_TOL:.1e}",
            inside.ok,
            f"worst rel {inside.worst.rel_diff:.3e}" if inside.worst else "",
        )
    )

    # Just outside.
    outside = compare.compare(ref, _table("abaqus", _scaled(1.0 + 2.0 * compare.REL_TOL)))
    results.append(
        _expect(
            f"a {200 * compare.REL_TOL:.1f}% perturbation fails",
            not outside.ok,
            (
                f"{len(outside.failures)} failing components, worst rel " f"{outside.worst.rel_diff:.3e}"
                if outside.worst
                else ""
            ),
        )
    )

    # A single localised defect must be caught, and must not drag the other probes down --
    # that is what makes a per-point table diagnostic rather than just a verdict.
    localised = dict(_REFERENCE)
    localised["GIRDER_QTR"] = tuple(v * 1.5 if i == 2 else v for i, v in enumerate(_REFERENCE["GIRDER_QTR"]))
    local_report = compare.compare(ref, _table("abaqus", localised))
    failing_probes = {d.probe for d in local_report.failures}
    results.append(
        _expect(
            "a single perturbed component fails at exactly that probe",
            failing_probes == {"GIRDER_QTR"} and len(local_report.failures) == 1,
            f"failing probes {sorted(failing_probes)}",
        )
    )

    # Zero-by-symmetry components must pass, not divide by zero.
    zero_diffs = [d for d in identical.diffs if d.negligible]
    results.append(
        _expect(
            "zero-by-symmetry components are marked negligible, not divided by zero",
            len(zero_diffs) > 0 and all(d.ok and d.rel_diff == 0.0 for d in zero_diffs),
            f"{len(zero_diffs)} of {len(identical.diffs)} components",
        )
    )

    # The two checks above are written as multiples of compare.REL_TOL, which makes them
    # self-consistent and blind to the one thing they look like they are checking: the *value* of
    # the tolerance. Found by mutation -- raising REL_TOL from 1e-2 to 1e-1 leaves every check in
    # this file passing, while the real comparison would then call a 5% disagreement between two
    # solvers a match. These next three are stated in absolute terms so they cannot drift with it.
    results.append(
        _expect(
            "a flat 2% disagreement fails, whatever REL_TOL has been set to",
            not compare.compare(ref, _table("abaqus", _scaled(1.02))).ok,
        )
    )
    results.append(
        _expect(
            "and a flat 0.1% disagreement passes, so the tolerance is not merely tiny",
            compare.compare(ref, _table("abaqus", _scaled(1.001))).ok,
        )
    )
    # The band REL_TOL has to live in, from both ends and with the measurements that set them.
    # Below 4.945e-03 it rejects the correct Sestra/Abaqus pair, whose worst significant residual
    # is GIRDER_3QTR.r2 at that value -- a shear-rigid/shear-flexible difference at a point where
    # cancellation amplifies it elevenfold, not a defect. Above 2e-2 it starts admitting genuine
    # disagreement: the B31-at-1.0m element error is 7.4e-03 and must stay visible.
    results.append(
        _expect(
            "REL_TOL sits above the measured worst residual and below a defect-admitting value",
            4.945e-03 < compare.REL_TOL <= 2.0e-02,
            f"{compare.REL_TOL:.1e}",
        )
    )

    # And a genuine difference hiding under the floor must still fail if it exceeds it.
    creeping = dict(_REFERENCE)
    creeping["BASE_L"] = (1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
    creep_report = compare.compare(ref, _table("abaqus", creeping))
    results.append(
        _expect(
            "a support that moved 1 mm on one side only is caught",
            not creep_report.ok and any(d.probe == "BASE_L" for d in creep_report.failures),
            f"{len(creep_report.failures)} failing components",
        )
    )
    return results


def check_hand_check() -> list[bool]:
    print("\nhand check (the closed form must agree with itself and reject a wrong value):")
    from . import hand_check as hc

    results = []
    predictions = hc.portal_frame_predictions()
    eb = predictions["euler-bernoulli"]
    timo = predictions["timoshenko"]
    results.append(
        _expect(
            "shear flexibility raises the sway",
            timo.delta > eb.delta,
            f"{100 * (timo.delta / eb.delta - 1):.3f}% higher",
        )
    )

    props = model.section_properties()
    # Limit check: a rigid girder reduces to two fixed-fixed columns in parallel.
    rigid = hc.sway_euler_bernoulli(
        p_total=model.P_TOTAL,
        height=model.HEIGHT,
        span=model.SPAN,
        e_mod=props["E"],
        i_column=props["Iy"],
        i_girder=props["Iy"] * 1.0e9,
    )
    expected_rigid = model.P_TOTAL * model.HEIGHT**3 / (24 * props["E"] * props["Iy"])
    results.append(
        _expect(
            "rigid girder -> P h^3 / (24 E I), two fixed-fixed columns in parallel",
            abs(rigid.delta / expected_rigid - 1) < 1e-6,
            f"{rigid.delta:.6e} vs {expected_rigid:.6e}",
        )
    )

    # Limit check: no girder -> two cantilevers each carrying P/2.
    soft = hc.sway_euler_bernoulli(
        p_total=model.P_TOTAL,
        height=model.HEIGHT,
        span=model.SPAN,
        e_mod=props["E"],
        i_column=props["Iy"],
        i_girder=props["Iy"] * 1.0e-9,
    )
    expected_soft = model.P_TOTAL * model.HEIGHT**3 / (6 * props["E"] * props["Iy"])
    results.append(
        _expect(
            "no girder -> P h^3 / (6 E I), two cantilevers",
            abs(soft.delta / expected_soft - 1) < 1e-6,
            f"{soft.delta:.6e} vs {expected_soft:.6e}",
        )
    )

    good = compare.hand_check(_table("sestra", _REFERENCE))
    results.append(
        _expect(
            "the measured Sestra sway is inside the admissible bracket",
            good.ok,
            f"nearest {good.nearest.formulation} at rel {good.nearest.rel_diff:.3e}",
        )
    )
    results.append(
        _expect(
            "and Sestra is identified as a shear-flexible beam",
            good.nearest.formulation == "timoshenko" and good.nearest.matches,
        )
    )

    # A shear-rigid element (Abaqus B33) lands at the Euler-Bernoulli end. It must PASS the
    # bracket -- policing against the Timoshenko value alone would fail a correct solver,
    # which is the trap the bracket exists to avoid.
    shear_rigid = compare.hand_check(_table("abaqus", _scaled(eb.delta / timo.delta)))
    results.append(
        _expect(
            "a shear-rigid (Euler-Bernoulli) solver passes the bracket",
            shear_rigid.ok,
            f"nearest {shear_rigid.nearest.formulation} at rel {shear_rigid.nearest.rel_diff:.3e}",
        )
    )

    # A pinned-base translation error roughly doubles the sway; the bracket must reject it.
    results.append(
        _expect(
            "a sway twice the closed form is outside the bracket",
            not compare.hand_check(_table("sestra", _scaled(2.0))).ok,
        )
    )
    # And a 2% error -- small, but larger than any beam formulation can explain.
    results.append(
        _expect(
            "a 2% sway error is outside the bracket (formulation cannot explain it)",
            not compare.hand_check(_table("sestra", _scaled(1.02))).ok,
        )
    )
    return results


def check_solver_section_idealisation() -> list[bool]:
    """The section the *solver* integrates is not always the one the model holds.

    Abaqus' ``section=PIPE`` treats the wall as a line. That is 0.276% low on this frame's tube,
    which is larger than the 0.2% of slack at each end of the admissible bracket, so a correct
    Abaqus answer fails the hand check if the closed form is fed adapy's exact annulus. These
    checks pin both halves: the corrected comparison passes, the uncorrected one does *not*, and
    the override still cannot absorb a genuine error.
    """
    print("\nsolver section idealisation (the closed form must use the solver's own section):")
    from . import hand_check as hc

    results = []
    adapy_inertia = model.section_properties()["Iy"]
    pipe_inertia = model.abaqus_pipe_inertia()

    # Measured against the kernel on a cantilever under a pure end moment; the licensed
    # test_the_abaqus_pipe_sections_second_moment_is_the_thin_walled_one is what holds it there.
    results.append(
        _expect(
            "the thin-walled formula reproduces Abaqus' measured effective I (2.693525e-05)",
            abs(pipe_inertia / 2.693525e-05 - 1) < 1e-05,
            f"{pipe_inertia:.6e}, rel {abs(pipe_inertia / 2.693525e-05 - 1):.2e}",
        )
    )
    results.append(
        _expect(
            "and it is 0.276% below the model's exact annulus",
            abs(pipe_inertia / adapy_inertia - 0.99724) < 1e-05,
            f"{pipe_inertia / adapy_inertia:.6f} of {adapy_inertia:.6e}",
        )
    )

    # The measured B32 sway. Feeding the closed form the section Abaqus integrates admits it;
    # feeding the model's own section does not. Without the override there is no way to pass
    # both this and the "2% error is rejected" check below, which is the point.
    measured_b32 = _table("abaqus", _scaled(2.474589e-02 / 2.468646e-02))
    corrected = compare.hand_check(measured_b32, inertia=pipe_inertia)
    results.append(
        _expect(
            "the measured B32 sway is inside the bracket for the section Abaqus integrates",
            corrected.ok,
            f"nearest {corrected.nearest.formulation} at rel {corrected.nearest.rel_diff:.3e}",
        )
    )
    results.append(
        _expect(
            "and it is identified as shear-flexible, at a far tighter rel than the bracket",
            corrected.nearest.formulation == "timoshenko" and corrected.nearest.rel_diff < 1e-04,
            f"rel {corrected.nearest.rel_diff:.3e}",
        )
    )
    uncorrected = compare.hand_check(measured_b32)
    results.append(
        _expect(
            "the same sway against the model's own section falls outside it -- the 0.077% miss",
            not uncorrected.ok,
            f"bracket top {uncorrected.bracket[1]:.6e} vs {uncorrected.solver_value:.6e}",
        )
    )

    # The override shifts which section is being checked against; it must not loosen the check.
    # A 2% error is beyond any beam theory and stays rejected with the pipe section in hand.
    results.append(
        _expect(
            "a 2% sway error is still rejected with the solver's own section",
            not compare.hand_check(_table("abaqus", _scaled(1.02)), inertia=pipe_inertia).ok,
        )
    )
    # The override must not be a loosening device. The bracket is in fact very slightly *narrower*
    # for the thin-walled section, and for a reason worth recording: the gap between the two beam
    # theories is set by phi = 12 E I / (G As L^2), which is proportional to I, so a smaller I
    # means shear matters slightly less and the two theories sit closer -- 0.6345% apart instead
    # of 0.6363%. Anyone "fixing" this class of miss by widening a tolerance instead would make
    # the bracket wider, and this check is what fires.
    pipe_width = corrected.bracket[1] / corrected.bracket[0]
    exact_width = uncorrected.bracket[1] / uncorrected.bracket[0]
    results.append(
        _expect(
            "the bracket is no wider for the pipe section than for the exact one",
            pipe_width <= exact_width,
            f"{pipe_width:.9f} vs {exact_width:.9f} (narrower by {exact_width - pipe_width:.2e}, "
            f"because phi scales with I)",
        )
    )
    # The report must say which section it used, or a pass hides what it was a pass against.
    results.append(
        _expect(
            "the printed report names the section the closed forms were evaluated with",
            f"{pipe_inertia:.6e}" in compare.format_hand_check(corrected),
        )
    )

    # Sestra needs no override: adapy writes its exact Iy straight into GBEAMG.
    results.append(
        _expect(
            "Sestra's default is the model's own section, unchanged",
            compare.hand_check(_table("sestra", _REFERENCE)).inertia == adapy_inertia,
        )
    )
    # Only bending differs between the two section definitions: 2 pi rm t and pi (ro^2 - ri^2) are
    # the same number algebraically, so the areas -- and with them the axial term -- are identical.
    ro, t = 0.1, 0.01
    results.append(
        _expect(
            "the thin-walled and exact sections have the same area, so only bending moves",
            abs(2 * math.pi * (ro - t / 2) * t / model.section_properties()["area"] - 1) < 1e-09,
            f"thin-wall {2 * math.pi * (ro - t / 2) * t:.9e} vs model {model.section_properties()['area']:.9e}",
        )
    )
    results.append(
        _expect(
            "and the formula is being asked about this model's actual section",
            abs(hc.thin_walled_pipe_inertia(radius=ro, thickness=t) / pipe_inertia - 1) < 1e-12,
            f"OD{2 * ro * 1000:.0f}x{t * 1000:.0f}",
        )
    )
    return results


# --------------------------------------------------------------------------------- the plate case
#
# Measured, on this branch, at the three seeds of plate_model.MESH_SIZES: Sestra V11.3-00 (FQUS,
# 128 / 512 / 2048 elements) against Abaqus 2025 (S4R, the same counts on the same structured grid,
# 165 / 585 / 2193 nodes both sides). The tables below are the *Richardson extrapolants* of those
# three solves, per probe, and only the two components that carry signal: u3 and the in-plane
# bending rotation ur2. The other four are zero by construction -- u1 and u2 on the mid-surface of a
# plate in pure bending, ur1 and ur3 on a flat strip in cylindrical bending -- and were measured at
# 1e-20 or smaller on the Abaqus side and exactly 0.0 on the Sestra one, both far below
# compare.ABS_FLOOR. They are carried as 0.0 here and still compared, by compare's absolute rule.

_SESTRA_BARE = {
    "X0_MID": (0.0, 0.13866667449474335),
    "QTR": (-0.12350000441074371, 0.0953333444106019),
    "MID": (-0.1733333319425583, 0.0),
    "TQTR": (-0.12350000441074371, -0.0953333444106019),
    "X1_MID": (0.0, -0.13866667449474335),
    "MID_Y0": (-0.1733333319425583, 0.0),
    "MID_YB": (-0.1733333319425583, 0.0),
}

_ABAQUS_BARE = {
    "X0_MID": (0.0, 0.13866664800897122),
    "QTR": (-0.12350223379024272, 0.09533332454092741),
    "MID": (-0.1733363138306436, 0.0),
    "TQTR": (-0.12350223379024272, -0.09533332454092741),
    "X1_MID": (0.0, -0.13866664800897122),
    "MID_Y0": (-0.1733363138306436, 0.0),
    "MID_YB": (-0.1733363138306436, 0.0),
}

_SESTRA_STIFF = {
    "X0_MID": (0.0, 0.052147124384695605),
    "QTR": (-0.046463328796517725, 0.035856595095737465),
    "MID": (-0.06521147187610148, 0.0),
    "TQTR": (-0.046463328796517725, -0.035856595095737465),
    "X1_MID": (0.0, -0.052147124384695605),
    "MID_Y0": (-0.06521675048214572, 0.0),
    "MID_YB": (-0.06521675048214572, 0.0),
}

_ABAQUS_STIFF = {
    "X0_MID": (0.0, 0.052148039418603236),
    "QTR": (-0.04646270586575523, 0.03585710876921818),
    "MID": (-0.06521064275691632, 0.0),
    "TQTR": (-0.04646270586575523, -0.03585710876921818),
    "X1_MID": (0.0, -0.052148039418603236),
    "MID_Y0": (-0.06521595234073391, 0.0),
    "MID_YB": (-0.06521595234073391, 0.0),
}

#: Mid-span ``u3`` at seeds 0.125 / 0.0625 / 0.03125, coarse to fine. The sequences the order and
#: the extrapolants are read off, carried at full precision so the arithmetic in
#: :mod:`plate_hand_check` is checked against the numbers it was budgeted from.
_MID_DEFLECTION_SEQUENCES = {
    ("sestra", False): (-0.1731979101896286, -0.17329947650432587, -0.17332486808300018),
    ("abaqus", False): (-0.17306548357009888, -0.17326860129833221, -0.17331938445568085),
    ("sestra", True): (-0.06516356766223907, -0.06520097702741623, -0.06520917266607285),
    ("abaqus", True): (-0.06511149555444717, -0.06518637388944626, -0.06520470231771469),
}

#: The measured cross-solver residuals, so a tolerance can be pinned against them absolutely.
#: ``extrapolated`` is the worst significant relative difference between the two extrapolated
#: tables, ``coarsest`` the worst at the coarsest of the three meshes. Both are ``QTR.u3`` on the
#: bare strip; on the stiffened one the extrapolated worst is ``X0_MID.r2`` at 1.7547e-05.
_MEASURED_CROSS_SOLVER = {"extrapolated": 1.8051e-05, "finest": 7.2082e-05, "coarsest": 8.3946e-04}


def _plate_table(solver: str, values: dict, *, stiffened: bool = False) -> DisplacementTable:
    """A :class:`DisplacementTable` from ``{probe: (u3, r2)}``, zeros elsewhere.

    ``model`` names the variant, as :func:`plate_sestra_runner.model_name` does, so a check that
    accidentally compared the bare table against the stiffened one would be visible in the report
    rather than only in the numbers.
    """
    from . import plate_model, plate_sestra_runner

    return DisplacementTable(
        solver=solver,
        solver_version="selftest",
        model=plate_sestra_runner.model_name(stiffened),
        load_case=plate_model.LOAD_CASE,
        displacements={name: (0.0, 0.0, u3, 0.0, r2, 0.0) for name, (u3, r2) in values.items()},
        node_ids={name: index for index, name in enumerate(sorted(values), start=1)},
        source=f"selftest::plate::{solver}",
    )


def _plate_solve(table: DisplacementTable, *, mesh_size: float, stiffened: bool, reaction: float | None = None):
    """A :class:`plate_sestra_runner.PlateSolve` around a table, with the right reaction by default."""
    from . import plate_model, plate_sestra_runner

    total = -plate_model.expected_load_total() if reaction is None else reaction
    return plate_sestra_runner.PlateSolve(
        table=table,
        mesh_size=mesh_size,
        stiffened=stiffened,
        reaction_total=(0.0, 0.0, total),
        node_count=0,
        element_counts={},
        source=table.source,
    )


def _plate_sequence(solver: str, stiffened: bool, *, sizes=None) -> list:
    """The three measured solves for one solver and one variant, coarse to fine.

    Only ``MID.u3`` varies with the mesh in this reconstruction; every other component is held at
    its extrapolated value. That is enough for what the sequence guards check -- the order, the
    extrapolation, the ordering rules -- and it keeps the constants above to the two components
    that carry signal.
    """
    from . import plate_model

    reference = {
        ("sestra", False): _SESTRA_BARE,
        ("abaqus", False): _ABAQUS_BARE,
        ("sestra", True): _SESTRA_STIFF,
        ("abaqus", True): _ABAQUS_STIFF,
    }[(solver, stiffened)]
    sequence = _MID_DEFLECTION_SEQUENCES[(solver, stiffened)]
    sizes = plate_model.MESH_SIZES if sizes is None else sizes
    solves = []
    for size, mid in zip(sizes, sequence):
        values = dict(reference)
        values["MID"] = (mid, reference["MID"][1])
        solves.append(
            _plate_solve(_plate_table(solver, values, stiffened=stiffened), mesh_size=size, stiffened=stiffened)
        )
    return solves


@dataclass
class _FakeNode:
    id: int
    p: tuple[float, float, float]
    refs: tuple = ()

    @property
    def x(self) -> float:
        return self.p[0]

    @property
    def y(self) -> float:
        return self.p[1]

    @property
    def z(self) -> float:
        return self.p[2]


@dataclass
class _FakeElem:
    id: int
    type: str
    nodes: list


class _FakeNodeStore(list):
    """``fem.nodes`` with the one method :func:`plate_model.assert_probes_are_seeded` calls."""

    def get_by_volume(self, p, tol=1e-6):
        return [n for n in self if max(abs(a - b) for a, b in zip(n.p, p)) <= tol]


@dataclass
class _FakeBc:
    """``fem.bcs``' two attributes :func:`plate_model.assert_edge_supports_declared` reads."""

    name: str
    dofs: list


@dataclass
class _FakeFem:
    nodes: _FakeNodeStore
    elements: list
    bcs: list = field(default_factory=list)


def _unit_grid(nx: int, ny: int, *, kind: str = "ShellShapes.QUAD", dx: float = 1.0, dy: float = 1.0) -> _FakeFem:
    """An ``nx`` x ``ny`` grid of ``dx`` x ``dy`` quads, for the consistent-load arithmetic.

    Unit cells by default, so the tributary areas come out 0.25 at a corner, 0.5 on an edge and
    1.0 in the interior -- the three numbers ``integral(N_i) dA = A / 4`` predicts, checkable by
    eye. ``dx`` and ``dy`` exist because unit cells are *also* the one case in which an
    implementation that assumed ``mesh_size**2`` instead of measuring the element would give the
    right answer. Found by mutation: replacing the shoelace area with ``1.0`` left every check on
    a unit grid passing.
    """
    nodes = _FakeNodeStore()
    index = {}
    node_id = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            node = _FakeNode(node_id, (i * dx, j * dy, 0.0))
            index[(i, j)] = node
            nodes.append(node)
            node_id += 1
    elements = []
    element_id = 1
    for i in range(nx):
        for j in range(ny):
            corners = [index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]]
            element = _FakeElem(element_id, kind, corners)
            for corner in corners:
                corner.refs = tuple(corner.refs) + (element,)
            elements.append(element)
            element_id += 1
    return _FakeFem(nodes, elements)


def check_plate_closed_forms() -> list[bool]:
    """The three plate closed forms, and the convergence arithmetic, against independent statements.

    Every check here is an identity or a limit, computed from the literals of the problem rather
    than from the functions being checked -- which is what makes this something other than the code
    agreeing with itself.
    """
    print("\nplate closed forms and convergence arithmetic:")
    from . import plate_hand_check as phc
    from . import plate_model as pm

    results = []
    props = pm.section_properties()
    d_by_hand = props["E"] * pm.PLATE_THICKNESS**3 / (12.0 * (1.0 - props["nu"] ** 2))
    results.append(
        _expect(
            "D = E t^3 / (12 (1 - nu^2)) is 19230.769230769 N m for this strip",
            abs(phc.plate_stiffness() / d_by_hand - 1) < 1e-15
            and abs(phc.plate_stiffness() - 19230.769230769234) < 1e-9,
            f"{phc.plate_stiffness():.9f}",
        )
    )
    results.append(
        _expect(
            "5 q L^4 / (384 D) is 0.1733333333333333 m",
            abs(phc.bare_deflection() - 0.1733333333333333) < 1e-15,
            f"{phc.bare_deflection():.15f}",
        )
    )
    # An identity between the two bare closed forms: theta / w = (q L^3 / 24 D) / (5 q L^4 / 384 D)
    # = 3.2 / L. Neither function knows about the other, so this catches a typo in either.
    results.append(
        _expect(
            "the support rotation and the deflection are in the ratio 3.2 / L, as the two forms require",
            abs(phc.support_rotation() / phc.bare_deflection() - 3.2 / pm.STRIP_LENGTH) < 1e-14,
            f"{phc.support_rotation() / phc.bare_deflection():.12f} vs {3.2 / pm.STRIP_LENGTH:.12f}",
        )
    )
    # The bar's second moment must be adapy's own, not a retyped one: that is the whole reason
    # section_properties reads it off a Section.
    by_hand = props["E"] * pm.BAR_WIDTH * pm.BAR_HEIGHT**3 / 12.0
    results.append(
        _expect(
            "EI_bar is E a b^3 / 12 from adapy's own FB section properties",
            abs(phc.bar_ei() / by_hand - 1) < 1e-12 and abs(phc.bar_ei() - 15946.874999999998) < 1e-9,
            f"{phc.bar_ei():.6f}, adapy Iy {props['bar_Iy']:.9e}",
        )
    )
    # The parallel-spring identity: w_stiff / w_bare must be exactly EI_plate / sum EI.
    results.append(
        _expect(
            "w_stiff / w_bare is exactly EI_plate / (EI_plate + EI_bar)",
            abs(phc.stiffened_deflection() / phc.bare_deflection() / phc.stiffness_ratio() - 1) < 1e-14,
            f"{phc.stiffened_deflection() / phc.bare_deflection():.12f} vs {phc.stiffness_ratio():.12f}",
        )
    )
    results.append(
        _expect(
            "and the bar makes the strip 2.658x stiffer",
            abs(1.0 / phc.stiffness_ratio() - 2.6584) < 1e-3,
            f"{1.0 / phc.stiffness_ratio():.4f}x",
        )
    )

    # An exactly second-order sequence: v(h) = L + C h^2 on h, h/2, h/4. The order must come back
    # 2.0 and the extrapolant must be L to machine precision. This is the arithmetic the whole
    # comparison rests on, checked where the answer is known rather than only on solver output.
    limit, coefficient = 0.5, 3.0
    synthetic = tuple(limit + coefficient * (1.0 / 2**k) ** 2 for k in range(3))
    conv = phc.richardson(synthetic)
    results.append(
        _expect(
            "an exact h^2 sequence reads back order 2.0 and extrapolates to its own limit",
            abs(conv.order - 2.0) < 1e-12 and abs(conv.extrapolated - limit) < 1e-14,
            f"order {conv.order:.12f}, extrapolant {conv.extrapolated:.15f} vs {limit}",
        )
    )
    cubic = tuple(limit + coefficient * (1.0 / 2**k) ** 3 for k in range(3))
    results.append(
        _expect(
            "an exact h^3 sequence reads back order 3.0, so the order is measured and not assumed",
            abs(phc.observed_order(cubic) - 3.0) < 1e-12,
            f"order {phc.observed_order(cubic):.12f}",
        )
    )
    # And the order is *used*: extrapolating the cubic sequence with the quadratic denominator
    # would leave a residual, so a richardson() that ignored the measured order would fail here.
    results.append(
        _expect(
            "the measured order is the one the extrapolation divides by",
            abs(phc.richardson(cubic).extrapolated - limit) < 1e-14,
            f"{phc.richardson(cubic).extrapolated:.15f} vs {limit}",
        )
    )
    # A first-order sequence: v = L + C h on h, h/2, h/4. Its error ratio is 2, so it is a
    # perfectly good sequence that simply is not second order -- and ORDER_BAND is what refuses
    # it. Both differences are positive and neither is zero, so no other clause can catch it.
    linear = tuple(limit + coefficient * (1.0 / 2**k) for k in range(3))
    results.append(
        _expect_raise(
            "a first-order sequence is refused rather than extrapolated as though it were h^2",
            phc.NotConverging,
            lambda: phc.richardson(linear),
        )
    )
    # And a sequence that turns around whose |d1/d2| is 4.0 -- inside ORDER_BAND, so only the
    # sign clause can reject it. Found by mutation: a turn-around at |d1/d2| = 2 is caught by the
    # band instead, and the check then passed with the sign clause deleted.
    turning = (1.0, 1.05, 1.0375)
    results.append(
        _expect_raise(
            "a sequence that turns around is refused even when its error ratio looks like h^2",
            phc.NotConverging,
            lambda: phc.richardson(turning),
        )
    )

    # The consistent nodal load, on a grid whose answer is arithmetic: unit cells, so q A / 4 gives
    # 0.25 at a corner, 0.5 on an edge, 1.0 inside.
    groups = pm.consistent_nodal_loads(_unit_grid(2, 2))
    areas = [group.area for group in groups]
    results.append(
        _expect(
            "the consistent nodal load of a uniform pressure is q A / 4 per node: 0.25 / 0.5 / 1.0",
            areas == [0.25, 0.5, 1.0],
            f"{areas}",
        )
    )
    total_area = sum(group.area * len(group.node_ids) for group in groups)
    results.append(
        _expect(
            "and those tributary areas sum to the loaded area exactly",
            abs(total_area - 4.0) < 1e-15,
            f"{total_area!r} over a 2 x 2 grid of unit cells",
        )
    )
    # On NON-unit cells, which is the case that separates measuring each element from assuming
    # mesh_size**2. 0.5 x 0.25 cells: A = 0.125, so 0.03125 / 0.0625 / 0.125 and a total of 0.5.
    rectangular = pm.consistent_nodal_loads(_unit_grid(2, 2, dx=0.5, dy=0.25))
    rect_areas = [group.area for group in rectangular]
    rect_total = sum(group.area * len(group.node_ids) for group in rectangular)
    results.append(
        _expect(
            "the element area is measured, not assumed: 0.5 x 0.25 cells give 0.03125 / 0.0625 / 0.125",
            rect_areas == [0.03125, 0.0625, 0.125] and abs(rect_total - 0.5) < 1e-15,
            f"{rect_areas}, total {rect_total!r} against a 1.0 x 0.5 m loaded area",
        )
    )
    results.append(
        _expect(
            "a force is negative, so the pressure pushes against the plate's +z normal",
            all(group.force < 0.0 for group in groups),
            f"{[round(group.force, 6) for group in groups]}",
        )
    )
    return results


def check_plate_convergence() -> list[bool]:
    """The measured sequences: their order, their extrapolants, and the tolerance they set.

    This is the group that makes :data:`plate_compare.PLATE_REL_TOL` a measurement. Everything in
    it is the output of twelve real solves, carried in the constants above.
    """
    print("\nplate mesh convergence (measured, Sestra FQUS vs Abaqus S4R at three seeds):")
    from . import plate_compare as pcmp
    from . import plate_hand_check as phc

    results = []
    extrapolants = {}
    for solver in ("sestra", "abaqus"):
        for stiffened in (False, True):
            sequence = _MID_DEFLECTION_SEQUENCES[(solver, stiffened)]
            conv = phc.richardson(sequence)
            extrapolants[(solver, stiffened)] = conv
            low, high = phc.ORDER_BAND
            results.append(
                _expect(
                    f"{solver} {'stiffened' if stiffened else 'bare'} converges at second order",
                    low <= conv.order <= high and abs(conv.order - 2.0) < 0.25,
                    f"order {conv.order:.4f}, extrapolant {conv.extrapolated:.12e}",
                )
            )
            closed = phc.stiffened_deflection() if stiffened else phc.bare_deflection()
            tol = phc.STIFFENED_REL_TOL if stiffened else phc.BARE_REL_TOL
            rel = abs(abs(conv.extrapolated) / closed - 1.0)
            results.append(
                _expect(
                    f"and its extrapolant is within {tol:.0e} of the closed form",
                    rel <= tol,
                    f"rel {rel:.3e} of {closed:.12f}",
                )
            )

    # Every sequence approaches the closed form from below -- both elements are too soft at a
    # finite mesh -- and the finest mesh is the nearest of the three. A sequence that got worse
    # with refinement would still extrapolate, so this is checked separately.
    for solver in ("sestra", "abaqus"):
        for stiffened in (False, True):
            sequence = _MID_DEFLECTION_SEQUENCES[(solver, stiffened)]
            magnitudes = [abs(v) for v in sequence]
            results.append(
                _expect(
                    f"{solver} {'stiffened' if stiffened else 'bare'}: refining the mesh softens it "
                    f"monotonically, so both elements are approaching from below",
                    magnitudes[0] < magnitudes[1] < magnitudes[2],
                    f"{[f'{v:.10f}' for v in magnitudes]}",
                )
            )

    # The bare extrapolation buys two orders of magnitude on Sestra and one on Abaqus. Asserted only
    # for the bare variant: on the stiffened one the closed form itself is 1.6e-04 out (the
    # transverse span between the bar and the long edges), so the extrapolant is legitimately
    # *further* from it than the finest mesh -- see plate_hand_check's docstring.
    for solver in ("sestra", "abaqus"):
        conv = extrapolants[(solver, False)]
        closed = phc.bare_deflection()
        finest = abs(abs(conv.values[-1]) / closed - 1.0)
        extrapolated = abs(abs(conv.extrapolated) / closed - 1.0)
        results.append(
            _expect(
                f"{solver} bare: extrapolating gets closer to the closed form than the finest mesh",
                extrapolated < finest,
                f"{extrapolated:.3e} against {finest:.3e}",
            )
        )

    # The tolerance, from both ends, in absolute terms -- so it cannot drift with itself. Below
    # 1.81e-05 it rejects the correct pair; at or above the coarsest mesh's 8.39e-04 it would admit
    # two solvers that disagree by their whole discretisation error.
    results.append(
        _expect(
            "PLATE_REL_TOL sits above the measured worst extrapolated residual and well below the "
            "coarse-mesh disagreement",
            _MEASURED_CROSS_SOLVER["extrapolated"] < pcmp.PLATE_REL_TOL < _MEASURED_CROSS_SOLVER["coarsest"],
            f"{_MEASURED_CROSS_SOLVER['extrapolated']:.3e} < {pcmp.PLATE_REL_TOL:.1e} < "
            f"{_MEASURED_CROSS_SOLVER['coarsest']:.3e}",
        )
    )
    results.append(
        _expect(
            "and it is at least 5x the worst residual a correct translation leaves",
            pcmp.PLATE_REL_TOL >= 5.0 * _MEASURED_CROSS_SOLVER["extrapolated"],
            f"{pcmp.PLATE_REL_TOL / _MEASURED_CROSS_SOLVER['extrapolated']:.1f}x",
        )
    )
    # The point of the whole design: at the coarsest mesh the two solvers do NOT agree to
    # PLATE_REL_TOL. If they did, the convergence study would be decoration.
    coarse_sestra = _plate_sequence("sestra", False)[0].table
    coarse_abaqus = _plate_sequence("abaqus", False)[0].table
    results.append(
        _expect(
            "the coarsest mesh pair FAILS at PLATE_REL_TOL -- the extrapolation is what closes it",
            not compare.compare(coarse_sestra, coarse_abaqus, rel_tol=pcmp.PLATE_REL_TOL).ok,
            f"worst rel {compare.compare(coarse_sestra, coarse_abaqus, rel_tol=pcmp.PLATE_REL_TOL).worst.rel_diff:.3e}",
        )
    )
    results.append(
        _expect(
            "and PLATE_MESH_REL_TOL admits it, which is why it is diagnostic and not the verdict",
            compare.compare(coarse_sestra, coarse_abaqus, rel_tol=pcmp.PLATE_MESH_REL_TOL).ok,
        )
    )

    # The support rotation: a second closed form, from the same three solves, that nothing was
    # tuned against.
    for solver in ("sestra", "abaqus"):
        table = _plate_table(solver, _SESTRA_BARE if solver == "sestra" else _ABAQUS_BARE)
        measured = abs(table.component("X0_MID", "r2"))
        rel = abs(measured / phc.support_rotation() - 1.0)
        results.append(
            _expect(
                f"{solver}'s extrapolated support rotation is q L^3 / (24 D) to better than 1e-06",
                rel < 1.0e-06,
                f"{measured:.12f} vs {phc.support_rotation():.12f}, rel {rel:.3e}",
            )
        )
    return results


def check_plate_agreement() -> list[bool]:
    """The two solvers' extrapolated tables against each other, and a defect that must not pass."""
    print("\nplate cross-solver agreement (the comparator must discriminate on shells too):")
    from . import plate_compare as pcmp
    from . import plate_hand_check as phc

    results = []
    for stiffened in (False, True):
        variant = "stiffened" if stiffened else "bare"
        sestra = _plate_table("sestra", _SESTRA_STIFF if stiffened else _SESTRA_BARE, stiffened=stiffened)
        abaqus = _plate_table("abaqus", _ABAQUS_STIFF if stiffened else _ABAQUS_BARE, stiffened=stiffened)
        report = compare.compare(sestra, abaqus, rel_tol=pcmp.PLATE_REL_TOL)
        results.append(
            _expect(
                f"the {variant} extrapolated tables agree at rel {pcmp.PLATE_REL_TOL:.1e}",
                report.ok,
                f"worst {report.worst.probe}.{report.worst.component} rel {report.worst.rel_diff:.3e}",
            )
        )
        # A 0.1% error at one probe: ten times the tolerance, a thousand times below anything the
        # case's headline defects do, and it must still be caught at exactly that probe.
        perturbed = dict(_SESTRA_STIFF if stiffened else _SESTRA_BARE)
        u3, r2 = perturbed["QTR"]
        perturbed["QTR"] = (u3 * 1.001, r2)
        local = compare.compare(
            sestra, _plate_table("abaqus", perturbed, stiffened=stiffened), rel_tol=pcmp.PLATE_REL_TOL
        )
        results.append(
            _expect(
                f"a 0.1% error at one {variant} probe fails at exactly that probe",
                {d.probe for d in local.failures} == {"QTR"} and len(local.failures) == 1,
                f"failing {sorted({d.probe for d in local.failures})}",
            )
        )
        results.append(
            _expect(
                f"the {variant} strip is in cylindrical bending: its width spread is within "
                f"{pcmp.CYLINDRICAL_REL_TOL:.0e}",
                pcmp.assert_cylindrical(sestra) <= pcmp.CYLINDRICAL_REL_TOL,
                f"spread {pcmp.assert_cylindrical(sestra):.3e} (Sestra), "
                f"{pcmp.assert_cylindrical(abaqus):.3e} (Abaqus)",
            )
        )

    for solver in ("sestra", "abaqus"):
        bare = _plate_table(solver, _SESTRA_BARE if solver == "sestra" else _ABAQUS_BARE)
        stiff = _plate_table(solver, _SESTRA_STIFF if solver == "sestra" else _ABAQUS_STIFF, stiffened=True)
        ratio = pcmp.assert_stiffener_present(bare, stiff)
        results.append(
            _expect(
                f"{solver}: the measured stiffness ratio is EI_plate / sum EI within " f"{phc.RATIO_REL_TOL:.0e}",
                abs(ratio / phc.stiffness_ratio() - 1.0) <= phc.RATIO_REL_TOL,
                f"{ratio:.9f} vs {phc.stiffness_ratio():.9f}, rel " f"{abs(ratio / phc.stiffness_ratio() - 1.0):.3e}",
            )
        )
    return results


#: One ``_analysis_edge_region`` call in an emitted script: the region's name, and the edge count
#: and total length adapy predicted for it and the kernel then checks.
_EDGE_REGION_CALL = re.compile(
    r"_analysis_edge_region\(assembly, '(?P<name>[^']+)', '[^']+', \(.*\), "
    r"(?P<edges>\d+), (?P<length>[-+0-9.eE]+), '[^']+'\)"
)


def _emitted_plate_script(stiffened: bool, mesh_size: float = 0.125) -> str:
    """The CAE script the **writer** produces for one variant of the strip, as text.

    No licence and no Abaqus: the supports, the regions, the step and the job are all written at
    plan time, so the text is checkable here. It is emitted rather than reconstructed because it
    *is* the deck now -- asserting against anything this package built itself would be asserting
    against a second opinion about the supports, which is the thing that no longer exists.
    """
    from . import plate_abaqus_runner as par
    from . import plate_model as pm

    assembly = pm.build_strip(mesh_size, stiffened=stiffened, route="abaqus")
    with tempfile.TemporaryDirectory() as tmp:
        script = pathlib.Path(tmp) / f"{par.SCRIPT_STEM}.py"
        assembly.to_abaqus_cae_script(
            script,
            mesh_size=mesh_size,
            shell_element_type=par.DEFAULT_SHELL_ELEMENT,
            job_name=par.JOB_NAME,
            submit=True,
        )
        return script.read_text(encoding="utf-8")


def check_plate_boundary_semantics() -> list[bool]:
    """ "Simply supported" must mean the same thing in both decks. This is where it is written down.

    Both decks are now generated from :data:`plate_model.EDGE_SUPPORTS` by a *writer*: ``write_bcs``
    turns the three ``Bc`` records into ``BNBCD`` FIX codes, and the CAE writer -- since adapy PR
    #405 taught it edge and face regions -- into three ``DisplacementBC`` on assembly ``Set``s of
    geometry edges. Until that PR the CAE side resolved a support to a geometric *vertex* only, so
    all three were refused and this package appended a driver of its own; the driver is gone and
    what is checked here is the writer's own emitted text.

    The expected ``DisplacementBC`` lines are asserted **literally** rather than rebuilt from
    ``EDGE_SUPPORTS``. Rebuilding them would make this check move with the very data it is checking:
    found by mutation, dropping ``ur1`` from the ``CYL`` entry -- which frees the long edges and makes
    ``D = E t^3 / (12 (1 - nu^2))`` the wrong stiffness by up to 9% -- was caught by nothing at all.

    The negative half matters as much as the positive: **no rotation is fixed on either supported
    edge**. ``ur2`` is what "simply supported" leaves free, and fixing it makes the strip 5x stiffer.
    The writer states it rather than omitting it -- every one of the six dofs appears in every call,
    ``UNSET`` where the record leaves it free -- so the negative half is now checked positively.

    And the regions are **edges**, whole ones. A support on the two corner vertices of a supported
    end would solve, and come out about 10% too soft; on the stiffened strip the bar splits each end
    into two collinear edges, which is why the writer locates them by bounding box and not by
    ``findAt`` -- ``findAt`` on that boundary returns one of the two. Both counts are asserted below,
    off the script's own calls.
    """
    print("\nplate boundary-condition semantics (both decks must mean one thing):")
    from ada.cadit.cae.analysis import BC_KEYWORDS

    from . import plate_abaqus_runner as par
    from . import plate_model as pm

    results = []
    results.append(
        _expect(
            "BC_KEYWORDS is adapy's own dof 1..6 order, which the writer renders each Bc's dofs through",
            tuple(BC_KEYWORDS) == ("u1", "u2", "u3", "ur1", "ur2", "ur3"),
            f"{tuple(BC_KEYWORDS)}",
        )
    )
    bare = _emitted_plate_script(stiffened=False)
    expected = {
        "SS_X0": "u1=0.0, u2=0.0, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET",
        "SS_X1": "u1=UNSET, u2=0.0, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET",
        "CYL": "u1=UNSET, u2=0.0, u3=UNSET, ur1=0.0, ur2=UNSET, ur3=UNSET",
    }
    for name, keywords in sorted(expected.items()):
        line = (
            f"    model.DisplacementBC(name={name!r}, createStepName='Initial',\n"
            f"                         region=assembly.sets[{name!r}], {keywords})"
        )
        results.append(
            _expect(
                f"the writer emits {name} as exactly '{keywords}'",
                line in bare,
                "found" if line in bare else f"MISSING: {line}",
            )
        )
    results.append(
        _expect(
            "no rotation about the width axis is fixed anywhere -- that is what makes it simply supported",
            bare.count("ur2=UNSET") == len(pm.EDGE_SUPPORTS) and "ur2=0.0" not in bare and "ur3=0.0" not in bare,
            f"ur2 and ur3 UNSET in all {bare.count('ur2=UNSET')} supports",
        )
    )
    results.append(
        _expect(
            "the three supports named in the model are the three the writer emits, and no more",
            bare.count("model.DisplacementBC(") == len(pm.EDGE_SUPPORTS) == 3,
            f"{bare.count('model.DisplacementBC(')} DisplacementBC calls for {len(pm.EDGE_SUPPORTS)} records",
        )
    )
    # The regions. Every support is a set of whole geometry EDGES, located by bounding box, and the
    # kernel checks each one's count and length against what adapy computed from its own body -- so
    # one statement of the support serves all three mesh densities. A vertex region here would be a
    # support on the corners only, which solves and is about 10% too soft.
    stiffened = _emitted_plate_script(stiffened=True)
    for variant, text, ends in (("bare", bare, 1), ("stiffened", stiffened, 2)):
        regions = {
            m.group("name"): (int(m.group("edges")), float(m.group("length"))) for m in _EDGE_REGION_CALL.finditer(text)
        }
        wanted = {
            "SS_X0": (ends, pm.STRIP_WIDTH),
            "SS_X1": (ends, pm.STRIP_WIDTH),
            "CYL": (2, 2 * pm.STRIP_LENGTH),
        }
        results.append(
            _expect(
                f"the {variant} strip's three supports are edge regions of "
                f"{ends}/{ends}/2 edge(s) and 0.5/0.5/8.0 in length",
                regions == wanted,
                f"{regions}",
            )
        )
        # The calls, not the helper definitions the writer emits alongside them: every region call
        # sits indented inside build(), and _analysis_region is the vertex form.
        vertex_calls = text.count("\n    _analysis_region(assembly,")
        edge_calls = text.count("\n    _analysis_edge_region(assembly,")
        results.append(
            _expect(
                f"and the {variant} strip carries no vertex region at all -- no support sits on corners",
                vertex_calls == 0 and edge_calls == len(pm.EDGE_SUPPORTS),
                f"{edge_calls} edge region call(s), {vertex_calls} vertex",
            )
        )
    # Nothing is appended to that script: the job it submits and the sidecar this package reads are
    # the writer's own, which is what says the supports and the answer come from one deck.
    results.append(
        _expect(
            "the writer's own script submits the job and writes the sidecar the runner reads",
            f"DISPLACEMENTS_NAME = {par.DISPLACEMENTS_NAME!r}" in bare and f"JOB_NAME = {par.JOB_NAME!r}" in bare,
            f"{par.JOB_NAME} -> {par.DISPLACEMENTS_NAME}",
        )
    )
    results.append(
        _expect(
            "and this package has no support driver left to append to it",
            not hasattr(par, "support_driver") and not hasattr(par, "reproduce_edge_support_refusal"),
            "the CAE writer carries the supports (adapy PR #405)",
        )
    )
    # And the load: the Sestra route gets the nodal vector, the Abaqus route the pressure. Neither
    # choice is free, and a route that silently changed would change what is being compared.
    results.append(
        _expect(
            "the route decides the load form and nothing else -- the supports are the same records on both",
            pm.LOAD_STYLES == {"sestra": "nodal", "abaqus": "pressure"} and pm.ROUTES == ("sestra", "abaqus"),
            f"{pm.LOAD_STYLES}",
        )
    )
    results.append(
        _expect_raise(
            "an unknown route is refused rather than defaulted",
            pm.PlateModelInvalid,
            lambda: pm.build_strip(0.125, stiffened=False, route="ansys"),
        )
    )
    return results


def check_plate_loud_failures() -> list[bool]:
    """Every plate guard, run against the input it exists for. Each must raise."""
    print("\nplate loud-failure guards (each must raise):")
    from . import plate_abaqus_runner  # noqa: F401 - imported to prove the module loads
    from . import plate_compare as pcmp
    from . import plate_hand_check as phc
    from . import plate_model as pm

    results = []

    # 1. A mesh with no shells in it. The whole case is plate bending; a beam mesh would produce a
    # full table of plausible numbers and a zero pressure load.
    beams_only = _unit_grid(2, 2, kind="LineShapes.LINE")
    results.append(
        _expect_raise(
            "a FEM with no shell elements",
            pm.PlateModelInvalid,
            lambda: pm.assert_has_shells(beams_only, mesh_size=0.125, stiffened=False),
        )
    )
    results.append(
        _expect(
            "and the same FEM yields no nodal load at all, rather than a small one",
            pm.consistent_nodal_loads(beams_only) == (),
            "a uniform pressure resolved onto no shells is a zero load vector",
        )
    )

    # 2. A probe the plate mesh does not seed, and two coincident nodes at one.
    sparse = _FakeResult([1, 2], [(0.0, 0.25, 0.0), (4.0, 0.25, 0.0)], [[1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0.1, 0, 0, 0]])
    results.append(
        _expect_raise(
            "a plate probe with no node",
            ProbeNotFound,
            lambda: sample_fea_result(
                sparse,
                pm.PROBE_POINTS,
                solver="fake",
                solver_version="selftest",
                model_name="plate_strip",
                load_case=pm.LOAD_CASE,
            ),
        )
    )
    doubled = _FakeResult(
        [1, 2],
        [(2.0, 0.25, 0.0), (2.0, 0.25, 0.0)],
        [[1, 0, 0, -0.17, 0, 0, 0], [2, 0, 0, -0.17, 0, 0, 0]],
    )
    results.append(
        _expect_raise(
            "two coincident nodes at a plate probe (an unmerged shell mesh)",
            AmbiguousProbe,
            lambda: sample_fea_result(
                doubled,
                (next(p for p in pm.PROBE_POINTS if p.name == "MID"),),
                solver="fake",
                solver_version="selftest",
                model_name="plate_strip",
                load_case=pm.LOAD_CASE,
            ),
        )
    )
    grid = _unit_grid(2, 2)
    results.append(
        _expect_raise(
            "a mesh that does not seed the strip's probe points",
            pm.PlateModelInvalid,
            lambda: pm.assert_probes_are_seeded(grid, mesh_size=1.0),
        )
    )

    # 3. The stiffener, measured as stiffness rather than counted. Three ways it goes wrong.
    bare = _plate_table("sestra", _SESTRA_BARE)
    results.append(
        _expect_raise(
            "the stiffener missing from one side (the stiffened run is the bare model)",
            pcmp.StiffenerMissing,
            lambda: pcmp.assert_stiffener_present(bare, _plate_table("abaqus", _SESTRA_BARE, stiffened=True)),
        )
    )
    # A bar whose section is rotated 90 degrees: I falls from a b^3 / 12 to b a^3 / 12, 20x, so
    # EI_bar goes from 15947 to 788 N m2 and the ratio from 0.376 to 0.924.
    rotated_ei = pm.section_properties()["E"] * pm.BAR_HEIGHT * pm.BAR_WIDTH**3 / 12.0
    rotated_ratio = phc.plate_ei() / (phc.plate_ei() + rotated_ei)
    rotated = {name: (u3 * rotated_ratio, r2 * rotated_ratio) for name, (u3, r2) in _SESTRA_BARE.items()}
    results.append(
        _expect_raise(
            f"a stiffener whose section is rotated 90 degrees (ratio {rotated_ratio:.4f}, not 0.3762)",
            pcmp.StiffenerMissing,
            lambda: pcmp.assert_stiffener_present(bare, _plate_table("abaqus", rotated, stiffened=True)),
        )
    )
    zeroed = {name: (0.0, 0.0) for name in _SESTRA_BARE}
    results.append(
        _expect_raise(
            "an all-zero bare table, from which no ratio can be taken",
            pcmp.StiffenerMissing,
            lambda: pcmp.assert_stiffener_present(
                _plate_table("sestra", zeroed), _plate_table("abaqus", _SESTRA_STIFF, stiffened=True)
            ),
        )
    )

    # 4. Cylindrical bending lost. The closed form is then the wrong one by up to 9% and every
    # number still looks like a plate under pressure.
    fanned = dict(_SESTRA_BARE)
    fanned["MID_Y0"] = (_SESTRA_BARE["MID_Y0"][0] * 1.01, 0.0)
    results.append(
        _expect_raise(
            "a 1% fan-out across the strip's width (the long-edge constraint lost)",
            pcmp.NotCylindrical,
            lambda: pcmp.assert_cylindrical(_plate_table("sestra", fanned)),
        )
    )
    results.append(
        _expect_raise(
            "three mid-span probes that are all zero -- an unloaded plate is uniform too",
            pcmp.NotCylindrical,
            lambda: pcmp.assert_cylindrical(_plate_table("sestra", zeroed)),
        )
    )

    # 5. The reaction total: the one number that comes from the solver's own bookkeeping.
    solve = _plate_solve(bare, mesh_size=0.03125, stiffened=False)
    results.append(
        _expect(
            "the correct reaction total passes",
            pcmp.assert_reaction_total(solve) <= pcmp.REACTION_REL_TOL,
            f"{solve.reaction_total!r} against {pm.expected_load_total()!r} N",
        )
    )
    results.append(
        _expect_raise(
            "half the load reacted",
            pcmp.ReactionMismatch,
            lambda: pcmp.assert_reaction_total(_plate_solve(bare, mesh_size=0.03125, stiffened=False, reaction=1000.0)),
        )
    )
    results.append(
        _expect_raise(
            "the whole load reacted with the wrong sign, which magnitude alone would admit",
            pcmp.ReactionMismatch,
            lambda: pcmp.assert_reaction_total(
                _plate_solve(bare, mesh_size=0.03125, stiffened=False, reaction=-2000.0)
            ),
        )
    )

    # 6. A refinement sequence that is not one. Each of these would otherwise be extrapolated into
    # a number and then compared against another solver's.
    # NotASequence and not NotConverging, deliberately: a list that mixes two variants also
    # happens to turn around, so a check accepting either could not tell whether the structural
    # guard was still there. Found by mutation -- deleting the _assert_one_sequence call left all
    # four of these passing while they expected NotConverging.
    sestra_bare_seq = _plate_sequence("sestra", False)
    results.append(
        _expect_raise(
            "an extrapolation over two meshes",
            pcmp.NotASequence,
            lambda: pcmp.extrapolate(sestra_bare_seq[:2]),
        )
    )
    results.append(
        _expect_raise(
            "a sequence in the wrong order (fine to coarse)",
            pcmp.NotASequence,
            lambda: pcmp.extrapolate(list(reversed(sestra_bare_seq))),
        )
    )
    results.append(
        _expect_raise(
            "a sequence mixing the bare and stiffened variants",
            pcmp.NotASequence,
            lambda: pcmp.extrapolate(sestra_bare_seq[:2] + _plate_sequence("sestra", True)[2:]),
        )
    )
    results.append(
        _expect_raise(
            "a sequence mixing two solvers",
            pcmp.NotASequence,
            lambda: pcmp.extrapolate(sestra_bare_seq[:2] + _plate_sequence("abaqus", False)[2:]),
        )
    )
    # A significant component that does not converge: the same value at every mesh means the mesh
    # never changed, which is exactly what a seeding bug looks like.
    flat = [
        _plate_solve(_plate_table("sestra", _SESTRA_BARE), mesh_size=size, stiffened=False) for size in pm.MESH_SIZES
    ]
    results.append(
        _expect_raise(
            "a significant component identical at all three meshes (the mesh never changed)",
            phc.NotConverging,
            lambda: pcmp.extrapolate(flat),
        )
    )
    # And it must say WHICH component, out of 42. Found by mutation: a bare re-raise loses the
    # address and the message becomes "the two finest values are identical" with nothing to act on.
    try:
        pcmp.extrapolate(flat)
        message = ""
    except phc.NotConverging as exc:
        message = str(exc)
    results.append(
        _expect(
            "and the failure names the probe and the component it was about",
            "MID.u3" in message and "sestra" in message,
            message.splitlines()[0][:110] if message else "(nothing raised)",
        )
    )

    # 7. The two cases must never be cross-compared: their probe sets are different structures.
    results.append(
        _expect_raise(
            "the plate probe set compared against the portal frame's",
            compare.ProbeSetMismatch,
            lambda: compare.compare(bare, _table("abaqus", _REFERENCE)),
        )
    )

    # 8. An unattached stiffener in adapy's own mesh, before a deck is written.
    detached = _unit_grid(4, 2)
    for node in detached.nodes:
        node.p = (node.p[0], node.p[1] * pm.STRIP_WIDTH / 2.0, 0.0)
    results.append(
        _expect_raise(
            "a stiffener line whose nodes are not shared by a shell and a beam element",
            pm.PlateModelInvalid,
            lambda: pm.assert_stiffener_shares_nodes(detached),
        )
    )

    # 9. The supports themselves, as records. They are now the ONLY route by which either solver
    # hears about them -- both writers translate these three and nothing else -- so a record lost on
    # the way into the model is a support lost from both decks. Abaqus would refuse an unsupported
    # strip as a singular system, but two of three supports still solves and is simply wrong.
    supported = _unit_grid(4, 2, dx=pm.STRIP_LENGTH / 4.0, dy=pm.STRIP_WIDTH / 2.0)
    supported.bcs = [_FakeBc(name, list(dofs)) for name, dofs, _why in pm.EDGE_SUPPORTS]
    results.append(
        _expect(
            "the three EDGE_SUPPORTS records pass the guard that says they are all there",
            pm.assert_edge_supports_declared(supported) is None,
            f"{[bc.name for bc in supported.bcs]}",
        )
    )
    dropped = _unit_grid(4, 2, dx=pm.STRIP_LENGTH / 4.0, dy=pm.STRIP_WIDTH / 2.0)
    dropped.bcs = [_FakeBc(name, list(dofs)) for name, dofs, _why in pm.EDGE_SUPPORTS[:-1]]
    results.append(
        _expect_raise(
            "a model whose long-edge support was never added as a Bc record",
            pm.PlateModelInvalid,
            lambda: pm.assert_edge_supports_declared(dropped),
        )
    )
    # A support that is present but fixes the wrong dofs: ur1 dropped from CYL frees the long edges,
    # which makes D = E t^3 / (12 (1 - nu^2)) the wrong stiffness by up to 9%.
    loosened = _unit_grid(4, 2, dx=pm.STRIP_LENGTH / 4.0, dy=pm.STRIP_WIDTH / 2.0)
    loosened.bcs = [_FakeBc(name, [d for d in dofs if d != 4]) for name, dofs, _why in pm.EDGE_SUPPORTS]
    results.append(
        _expect_raise(
            "a support present in the model but fixing fewer dofs than EDGE_SUPPORTS says",
            pm.PlateModelInvalid,
            lambda: pm.assert_edge_supports_declared(loosened),
        )
    )
    return results


# --- the curved case -------------------------------------------------------------------------
#
# Measured, on this branch, at the three grids of curved_model.MESH_COUNTS: Sestra V11.3-00
# (FQUS, 64 / 256 / 1024 elements) against Abaqus 2025 (S4R, the same counts on the same
# structured grid, 81 / 289 / 1089 nodes both sides). The two tables below are the *Richardson
# extrapolants* of those three solves, per probe, and only the three components that carry
# signal: u1, u2 and u3. The rotations are zero by construction -- a pure membrane hoop state
# has no bending anywhere -- and were measured at 5.3e-10 (Sestra) and 3.5e-08 (Abaqus), both
# far below compare.ABS_FLOOR. They are carried as 0.0 here and still compared, by compare's
# absolute rule.
#
# Read the two tables side by side at T0 and T90: Sestra's are *identical* mirror images
# (T0_MID.u1 == T90_MID.u2 to the last digit) and Abaqus' are not. That asymmetry is the whole
# of this case's cross-solver residual, and it is an S4R property rather than a translation
# defect -- see step 3 of curved_compare's docstring for the measurement that settles it.

_CURVED_SESTRA = {
    "T0_Z0": (0.000190476989212573, 0.0, 0.0),
    "T0_Q": (0.000190476881806862, 0.0, -2.2439955508728e-05),
    "T0_MID": (0.000190476856224673, 0.0, -4.4879911017457e-05),
    "T0_3Q": (0.000190476883352593, 0.0, -6.731985681997e-05),
    "T0_ZL": (0.000190476976631253, 0.0, -8.9759822034914e-05),
    "T90_Z0": (0.0, 0.000190476989212573, -1.5602e-14),
    "T90_Q": (0.0, 0.000190476881806862, -2.2439955508728e-05),
    "T90_MID": (0.0, 0.000190476856224673, -4.4879911017457e-05),
    "T90_3Q": (0.0, 0.000190476883352593, -6.731985681997e-05),
    "T90_ZL": (0.0, 0.000190476976631253, -8.9759822034914e-05),
    "ARC0_MID": (0.000134688712527243, 0.000134688712527243, -3.07636e-13),
    "ARCL_MID": (0.000134688712527243, 0.000134688712527243, -8.9759822845121e-05),
}

_CURVED_ABAQUS = {
    "T0_Z0": (0.000190499374429798, 0.0, -0.0),
    "T0_Q": (0.000190477403385621, 0.0, -2.2440046134648e-05),
    "T0_MID": (0.000190460151919407, 0.0, -4.4880055999935e-05),
    "T0_3Q": (0.000190452197980589, 0.0, -6.7320017706179e-05),
    "T0_ZL": (0.000190457146184144, 0.0, -8.9760102664641e-05),
    "T90_Z0": (0.0, 0.000190530045094939, -2.539290378e-08),
    "T90_Q": (0.0, 0.000190526397820213, -2.2462073612874e-05),
    "T90_MID": (0.0, 0.000190529757134511, -4.4900568323265e-05),
    "T90_3Q": (0.0, 0.000190532342615437, -6.7339593255515e-05),
    "T90_ZL": (0.0, 0.000190535686388942, -8.9778993149868e-05),
    "ARC0_MID": (0.000134682602994648, 0.000134675563265381, -1.2858145659e-08),
    "ARCL_MID": (0.000134663540398779, 0.000134699953879219, -8.976938276247e-05),
}

#: Radial displacement at ``T0_MID`` at 8 / 16 / 32 elements a side, coarse to fine, and the
#: axial ``u3`` at ``T0_ZL`` over the same three. The sequences the orders and the extrapolants
#: are read off, at full precision so the arithmetic in :mod:`plate_hand_check` is checked
#: against the numbers the tolerances were budgeted from.
_CURVED_RADIAL_SEQUENCES = {
    "sestra": (0.00018955643463414162, 0.00019025063375011086, 0.00019042125495616347),
    "abaqus": (0.00018967386859003454, 0.00019026704831048846, 0.00019041272753383964),
}
_CURVED_AXIAL_SEQUENCES = {
    "sestra": (-8.932757191359997e-05, -8.965167216956615e-05, -8.973276271717623e-05),
    "abaqus": (-8.932747732615098e-05, -8.965176675701514e-05, -8.973297371994704e-05),
}

#: The hoop-uniformity spread at the three grids and then on the extrapolant, per solver, and the
#: observed-order range over each solver's 23 significant components. All measured; these are the
#: numbers :data:`curved_hand_check.HOOP_REL_TOL` and
#: :data:`curved_compare.CURVED_ORDER_BAND` are set from.
_CURVED_HOOP_SPREAD = {
    "sestra": (4.5184706917136475e-05, 3.0538398152807766e-05, 1.2984411685847413e-05, 9.175465849968875e-06),
    "abaqus": (0.0013246599028147988, 0.00032910002958305636, 0.00011255843410357026, 0.0003653246408884538),
}
_CURVED_ORDER_RANGE = {
    "sestra": (1.966416317312324, 2.024552561460075),
    "abaqus": (1.457047966478184, 2.9551043963215875),
}

#: The measured cross-solver residuals, so the tolerance can be pinned against them absolutely.
#: All three are ``T90_Q.u3``, which is the probe the S4R asymmetry hits hardest.
_CURVED_CROSS_SOLVER = {
    "extrapolated": 0.0009846866556865203,
    "finest": 0.0011330901441648652,
    "coarsest": 0.0034215768429777483,
}

#: Each solver's own summed reaction at the finest grid, newtons. Against
#: ``(-p r L, -p r L, 0) = (-628318.530718, -628318.530718, 0)``.
_CURVED_REACTIONS = {
    "sestra": (-628318.529296875, -628318.529296875, 5.255969881545752e-09),
    "abaqus": (-628318.546875, -628318.548828125, 0.003337685950100422),
}

#: What ``AnalysisPlan.applied_pressure`` computes for this panel, and the exact resultant. The
#: defect gap 2 of :mod:`curved_model` reports, carried here so it is checked without a licence.
_CURVED_WRITER_PRESSURE = (724863.7521754879, 669823.4477113917, 0.0)
_CURVED_EXACT_PRESSURE = (628318.5307179586, 628318.5307179586, 0.0)


def _curved_table(solver: str, values: dict) -> DisplacementTable:
    """A :class:`DisplacementTable` from ``{probe: (u1, u2, u3)}``, zero rotations."""
    from . import curved_model, curved_sestra_runner

    return DisplacementTable(
        solver=solver,
        solver_version="selftest",
        model=curved_sestra_runner.model_name(),
        load_case=curved_model.LOAD_CASE,
        displacements={name: (u1, u2, u3, 0.0, 0.0, 0.0) for name, (u1, u2, u3) in values.items()},
        node_ids={name: index for index, name in enumerate(sorted(values), start=1)},
        source=f"selftest::curved::{solver}",
    )


def _curved_solve(table: DisplacementTable, *, count: int, reaction=None):
    """A :class:`plate_sestra_runner.PlateSolve` around a curved table, with the right reaction."""
    from . import curved_model, plate_sestra_runner

    if reaction is None:
        reaction = tuple(-v for v in curved_model.expected_load_total())
    return plate_sestra_runner.PlateSolve(
        table=table,
        mesh_size=curved_model.PANEL_ARC / count,
        stiffened=False,
        reaction_total=tuple(float(v) for v in reaction),
        node_count=0,
        element_counts={},
        source=table.source,
    )


def _curved_sequence(solver: str, *, counts=None) -> list:
    """The three measured solves for one solver, coarse to fine.

    The **two** components whose sequences are carried in full -- the radial at
    ``T0_MID``/``T90_MID`` and the axial ``u3`` at ``T0_ZL`` -- take their measured values. Every
    other component is given the *same relative* sequence as the radial one, scaled onto its own
    extrapolated value.

    That is not padding, and the reason is arithmetic: Richardson is an affine combination of the
    three values, so a component whose sequence is ``V (1 + d_i)`` with ``d`` the radial sequence's
    own relative approach extrapolates to exactly ``V``. So this reconstruction reproduces the
    measured extrapolated table component for component, and every component forms a genuine
    converging sequence -- which the alternative (holding the minor components constant) does not:
    three identical values have no rate to read and :func:`plate_hand_check.observed_order` refuses
    them by name, which is how this shape announced itself.
    """
    from . import curved_model

    reference = {"sestra": _CURVED_SESTRA, "abaqus": _CURVED_ABAQUS}[solver]
    radial = _CURVED_RADIAL_SEQUENCES[solver]
    axial = _CURVED_AXIAL_SEQUENCES[solver]
    limit = reference["T0_MID"][0]
    counts = curved_model.MESH_COUNTS if counts is None else counts
    solves = []
    for count, radial_value, axial_value in zip(counts, radial, axial):
        factor = radial_value / limit
        values = {name: tuple(v * factor for v in row) for name, row in reference.items()}
        values["T0_MID"] = (radial_value, 0.0, values["T0_MID"][2])
        values["T90_MID"] = (0.0, radial_value, values["T90_MID"][2])
        values["T0_ZL"] = (values["T0_ZL"][0], 0.0, axial_value)
        solves.append(_curved_solve(_curved_table(solver, values), count=count))
    return solves


def check_curved_closed_forms() -> list[bool]:
    """The curved case's closed forms and its grid, against independent statements.

    Every check here is an identity or a limit computed from the literals of the problem rather
    than from the functions being checked -- which is what makes this something other than the
    code agreeing with itself.
    """
    print("\ncurved closed forms and the analytic grid:")
    from . import curved_compare as ccmp
    from . import curved_hand_check as chc
    from . import curved_model as cvm

    results = []
    E, nu = 2.1e11, 0.3
    r, length, thickness, pressure = 2.0, math.pi, 0.010, 1.0e5

    props = cvm.section_properties()
    results.append(
        _expect(
            "the model's material is S355 at E = 210 GPa and nu = 0.3, read off adapy and not retyped",
            props["E"] == E and props["nu"] == nu,
            f"E={props['E']:.6g}, nu={props['nu']}",
        )
    )
    results.append(
        _expect(
            "the axial length is the quarter arc's own length, pi r / 2 -- which is what puts one "
            "seed on both directions",
            abs(cvm.PANEL_LENGTH - 0.5 * math.pi * cvm.PANEL_RADIUS) < 1e-15 and abs(cvm.PANEL_LENGTH - length) < 1e-15,
            f"L={cvm.PANEL_LENGTH!r}, pi r / 2={0.5 * math.pi * cvm.PANEL_RADIUS!r}",
        )
    )
    results.append(
        _expect(
            "w = p r^2 / (E t), computed from the literals of the problem",
            abs(chc.radial_displacement() - pressure * r * r / (E * thickness)) < 1e-18,
            f"{chc.radial_displacement():.12e} against {pressure * r * r / (E * thickness):.12e}",
        )
    )
    results.append(
        _expect(
            "and w carries NO nu -- the ends are open, so sigma_z = 0 and nothing is subtracted",
            abs(chc.radial_displacement() * (1.0 - nu**2) / chc.radial_displacement() - (1.0 - nu**2)) < 1e-15
            and abs(chc.radial_displacement() - pressure * r * r / (E * thickness)) < 1e-18,
            f"a 1 - nu^2 would make it {pressure * r * r / (E * thickness) * (1 - nu**2):.12e}, 9% away",
        )
    )
    results.append(
        _expect(
            "the hoop stress is p r / t, and w / r is exactly that over E -- so the radial check is "
            "the hoop-stress check with a modulus in it",
            abs(chc.hoop_stress() - pressure * r / thickness) < 1e-6
            and abs(chc.radial_displacement() / r - chc.hoop_stress() / E) < 1e-18,
            f"sigma_theta={chc.hoop_stress():.6g} Pa, w/r={chc.radial_displacement() / r:.12e}",
        )
    )
    results.append(
        _expect(
            "the axial strain is -nu p r / (E t) -- negative, so an internally pressurised open "
            "cylinder gets SHORTER as it swells",
            abs(chc.axial_strain() + nu * pressure * r / (E * thickness)) < 1e-18 and chc.axial_strain() < 0.0,
            f"{chc.axial_strain():.12e}",
        )
    )
    results.append(
        _expect(
            "the axial displacement is linear in z and zero at z = 0",
            chc.axial_displacement(0.0) == 0.0
            and abs(chc.axial_displacement(length) - 2.0 * chc.axial_displacement(0.5 * length)) < 1e-20,
            f"u_z(L)={chc.axial_displacement(length):.12e}, 2 u_z(L/2)="
            f"{2 * chc.axial_displacement(0.5 * length):.12e}",
        )
    )
    # The edge reaction, derived a second way: integral(p sin(theta) r L dtheta) over 0..pi/2 by
    # quadrature rather than by the antiderivative the closed form used.
    steps = 200_000
    quadrature = math.fsum(
        pressure * math.sin((index + 0.5) * 0.5 * math.pi / steps) * r * length * (0.5 * math.pi / steps)
        for index in range(steps)
    )
    results.append(
        _expect(
            "the edge reaction p r L is integral(p n_y dA) over the quarter, by quadrature",
            abs(chc.edge_reaction() / quadrature - 1.0) < 1e-09,
            f"{chc.edge_reaction():.6f} against {quadrature:.6f}, rel {abs(chc.edge_reaction() / quadrature - 1.0):.3e}",
        )
    )
    results.append(
        _expect(
            "the face area is pi r L / 2, and the inscribed polygon's is always BELOW it, " "approaching as O(h^2)",
            abs(cvm.face_area() - 0.5 * math.pi * r * length) < 1e-12
            and all(chc.faceted_area(n) < cvm.face_area() for n in cvm.MESH_COUNTS)
            and abs((cvm.face_area() - chc.faceted_area(8)) / (cvm.face_area() - chc.faceted_area(16)) - 4.0) < 0.02,
            f"exact {cvm.face_area():.9f}, faceted "
            f"{[round(chc.faceted_area(n), 9) for n in cvm.MESH_COUNTS]}, deficit ratio "
            f"{(cvm.face_area() - chc.faceted_area(8)) / (cvm.face_area() - chc.faceted_area(16)):.4f}",
        )
    )
    # Every probe's exact displacement, component by component, against the two closed forms.
    expected = {p.name: chc.expected_components(p.name) for p in cvm.PROBE_POINTS}
    radial_ok = all(
        abs(math.hypot(expected[name]["u1"], expected[name]["u2"]) / chc.radial_displacement() - 1.0) < 1e-12
        for name in expected
    )
    axial_ok = all(abs(expected[p.name]["u3"] - chc.axial_displacement(p.xyz[2])) < 1e-20 for p in cvm.PROBE_POINTS)
    rotations_ok = all(expected[name][c] == 0.0 for name in expected for c in ("r1", "r2", "r3"))
    results.append(
        _expect(
            f"all {len(expected)} probes have an exact displacement: |(u1, u2)| = w, u3 = the axial "
            f"form, and every rotation exactly zero",
            radial_ok and axial_ok and rotations_ok,
            f"radial {radial_ok}, axial {axial_ok}, rotations {rotations_ok}",
        )
    )
    results.append(
        _expect(
            "at theta = 45 the radial displacement is split between u1 and u2, so reading u1 alone "
            "would be 29% low -- which is why RADIAL_BASIS is data",
            abs(expected["ARC0_MID"]["u1"] / chc.radial_displacement() - math.cos(math.radians(45))) < 1e-12
            and abs(expected["ARC0_MID"]["u1"] / chc.radial_displacement() - 1.0) > 0.29,
            f"u1/w = {expected['ARC0_MID']['u1'] / chc.radial_displacement():.9f}",
        )
    )

    # The grid. Every clause of assert_shell_grid, measured positively on the real thing.
    for count in cvm.MESH_COUNTS:
        grid, nodes = cvm.shell_grid(count)
        radii = [math.hypot(node.x, node.y) for node in nodes]
        results.append(
            _expect(
                f"the {count} x {count} grid has ({count} + 1)^2 nodes, all on radius r to 1e-15",
                len(nodes) == (count + 1) ** 2 and max(abs(v - r) for v in radii) < 1e-15,
                f"{len(nodes)} nodes, worst radius error {max(abs(v - r) for v in radii):.3e}",
            )
        )
    # ... and the facets are exact rectangles, which is what makes the nodal load exact.
    assembly = cvm.build_panel(cvm.MESH_COUNTS[0], route="sestra")
    fem = _curved_part(assembly).fem
    worst_area, worst_dot = 0.0, 0.0
    for element in cvm.shell_elements(fem):
        points = [np.asarray(node.p, dtype=float) for node in element.nodes]
        e1, e2 = points[1] - points[0], points[3] - points[0]
        worst_dot = max(worst_dot, abs(float(np.dot(e1, e2))))
        worst_area = max(
            worst_area,
            abs(cvm.element_area(element) - float(np.linalg.norm(e1)) * float(np.linalg.norm(e2))),
        )
    results.append(
        _expect(
            "every facet is a RECTANGLE -- its two edge vectors are orthogonal and its area is "
            "their product -- which is what makes integral(N_i) dA = A / 4 exact",
            worst_dot < 1e-15 and worst_area < 1e-15,
            f"worst edge dot product {worst_dot:.3e}, worst area mismatch {worst_area:.3e}",
        )
    )
    groups = cvm.consistent_nodal_loads(fem)
    total = cvm.applied_load_total(groups)
    results.append(
        _expect(
            "the consistent nodal loads sum to exactly (p r L, p r L, 0) -- the closed form, not a "
            "lumping of the pressure",
            all(
                abs(total[axis] / v - 1.0) < 1e-12 if v else abs(total[axis]) < 1e-06
                for axis, v in enumerate(cvm.expected_load_total())
            ),
            f"{tuple(round(v, 6) for v in total)} against {cvm.expected_load_total()} over "
            f"{len(groups)} distinct vector(s)",
        )
    )
    results.append(
        _expect(
            "every facet normal points AWAY from the cylinder axis, so the internal pressure pushes "
            "outward on all of them",
            all(
                float(
                    np.dot(
                        cvm.element_normal(element),
                        np.asarray(
                            [
                                np.mean([n.p[0] for n in element.nodes]),
                                np.mean([n.p[1] for n in element.nodes]),
                                0.0,
                            ]
                        ),
                    )
                )
                > 0.0
                for element in cvm.shell_elements(fem)
            ),
            f"{len(cvm.shell_elements(fem))} facet(s)",
        )
    )
    results.append(
        _expect(
            "the probe-match tolerance is 1e-05 and the hoop node spacing at the FINEST grid is "
            "0.098 m, so it is 4 orders below the nearest node it could confuse a probe with",
            cvm.PROBE_MATCH_TOL == 1.0e-05
            and 2.0 * r * math.sin(0.25 * math.pi / cvm.MESH_COUNTS[-1]) / cvm.PROBE_MATCH_TOL > 1000.0,
            f"tol {cvm.PROBE_MATCH_TOL:.1e}, spacing "
            f"{2.0 * r * math.sin(0.25 * math.pi / cvm.MESH_COUNTS[-1]):.6f} m",
        )
    )
    results.append(
        _expect(
            "the writer's flat-plate pressure resultant is WRONG on this face, and asymmetric where "
            "the panel is symmetric -- the defect this case reports",
            abs(_CURVED_WRITER_PRESSURE[0] / _CURVED_EXACT_PRESSURE[0] - 1.0) > 0.15
            and abs(_CURVED_WRITER_PRESSURE[0] - _CURVED_WRITER_PRESSURE[1]) > 0.07 * _CURVED_WRITER_PRESSURE[0]
            and _CURVED_EXACT_PRESSURE[0] == _CURVED_EXACT_PRESSURE[1],
            f"writer {_CURVED_WRITER_PRESSURE[:2]} against exact {_CURVED_EXACT_PRESSURE[:2]}",
        )
    )
    results.append(
        _expect(
            "the curved order band is wider than the plate's, and both measured ranges sit inside it",
            ccmp.CURVED_ORDER_BAND[0] < plate_hand_check_order_band()[0]
            and ccmp.CURVED_ORDER_BAND[1] > plate_hand_check_order_band()[1]
            and all(
                ccmp.CURVED_ORDER_BAND[0] <= low and high <= ccmp.CURVED_ORDER_BAND[1]
                for low, high in _CURVED_ORDER_RANGE.values()
            ),
            f"{ccmp.CURVED_ORDER_BAND} against measured {_CURVED_ORDER_RANGE}",
        )
    )
    return results


def plate_hand_check_order_band():
    from . import plate_hand_check

    return plate_hand_check.ORDER_BAND


def _curved_part(assembly):
    from . import curved_model

    for part in assembly.get_all_parts_in_assembly(include_self=True):
        if part.name == curved_model.PART_NAME:
            return part
    raise AssertionError("the curved assembly holds no Panel part")


def check_curved_convergence() -> list[bool]:
    """The measured sequences, their orders, their extrapolants, and the tolerance they set.

    Everything here is arithmetic on the twelve real solves whose numbers are the constants above,
    so it runs with no solver and still pins :data:`curved_compare.CURVED_REL_TOL` against the
    residual a correct translation actually leaves -- not merely against a multiple of itself.
    """
    print("\ncurved convergence and the tolerance it sets:")
    from . import compare as cmp_mod
    from . import curved_compare as ccmp
    from . import curved_hand_check as chc
    from . import curved_model as cvm

    results = []
    closed_radial = chc.radial_displacement()
    closed_axial = chc.axial_displacement(cvm.PANEL_LENGTH)
    extrapolants = {}
    for solver in ("abaqus", "sestra"):
        sequence = _CURVED_RADIAL_SEQUENCES[solver]
        residuals = [abs(v / closed_radial - 1.0) for v in sequence]
        results.append(
            _expect(
                f"{solver}: the radial sequence approaches the closed form monotonically from BELOW "
                f"-- a polygon inscribed in the cylinder is stiffer in hoop than the cylinder",
                all(v < closed_radial for v in sequence) and residuals[0] > residuals[1] > residuals[2],
                f"rel {[f'{v:.3e}' for v in residuals]}",
            )
        )
        conv = chc.richardson(sequence, order_band=ccmp.CURVED_ORDER_BAND)
        extrapolants[solver] = conv.extrapolated
        results.append(
            _expect(
                f"{solver}: the radial order is 2.02, measured off the three values and not assumed",
                abs(conv.order - 2.02) < 0.02,
                f"order {conv.order:.4f}, error ratio "
                f"{(sequence[1] - sequence[0]) / (sequence[2] - sequence[1]):.4f}",
            )
        )
        rel = abs(conv.extrapolated / closed_radial - 1.0)
        results.append(
            _expect(
                f"{solver}: the radial extrapolant is within RADIAL_REL_TOL of p r^2 / (E t)",
                rel <= chc.RADIAL_REL_TOL,
                f"{conv.extrapolated:.12e} against {closed_radial:.12e}, rel {rel:.3e} at tol "
                f"{chc.RADIAL_REL_TOL:.1e}",
            )
        )
        axial = chc.richardson(_CURVED_AXIAL_SEQUENCES[solver], order_band=ccmp.CURVED_ORDER_BAND)
        axial_rel = abs(axial.extrapolated / closed_axial - 1.0)
        results.append(
            _expect(
                f"{solver}: the axial order is 2.00 and its extrapolant is within AXIAL_REL_TOL of "
                f"-nu p r L / (E t)",
                abs(axial.order - 2.0) < 0.01 and axial_rel <= chc.AXIAL_REL_TOL,
                f"order {axial.order:.4f}, {axial.extrapolated:.12e} against {closed_axial:.12e}, "
                f"rel {axial_rel:.3e}",
            )
        )
        low, high = _CURVED_ORDER_RANGE[solver]
        results.append(
            _expect(
                f"{solver}: all 23 significant components converge inside CURVED_ORDER_BAND",
                ccmp.CURVED_ORDER_BAND[0] <= low and high <= ccmp.CURVED_ORDER_BAND[1],
                f"{low:.4f} .. {high:.4f} in {ccmp.CURVED_ORDER_BAND}",
            )
        )
        spread = _CURVED_HOOP_SPREAD[solver]
        results.append(
            _expect(
                f"{solver}: the hoop spread shrinks with the grid, and the finest and the "
                f"extrapolant are inside HOOP_REL_TOL",
                spread[0] > spread[1] > spread[2] and spread[2] <= chc.HOOP_REL_TOL and spread[3] <= chc.HOOP_REL_TOL,
                f"{[f'{v:.3e}' for v in spread[:3]]} then extrapolant {spread[3]:.3e} at tol "
                f"{chc.HOOP_REL_TOL:.1e}",
            )
        )
        worst = max(
            abs(_CURVED_REACTIONS[solver][axis] + v) / max(abs(x) for x in cvm.expected_load_total())
            for axis, v in enumerate(cvm.expected_load_total())
        )
        results.append(
            _expect(
                f"{solver}: the finest grid's reaction total is (-p r L, -p r L, 0) within " f"REACTION_REL_TOL",
                worst <= chc.REACTION_REL_TOL,
                f"{tuple(round(v, 4) for v in _CURVED_REACTIONS[solver])}, worst {worst:.3e} at tol "
                f"{chc.REACTION_REL_TOL:.1e}",
            )
        )

    results.append(
        _expect(
            "Sestra answers SYMMETRICALLY -- its two mirror-image probes carry the same radial "
            "displacement to the last digit -- and Abaqus does not; that is the S4R asymmetry",
            _CURVED_SESTRA["T0_MID"][0] == _CURVED_SESTRA["T90_MID"][1]
            and _CURVED_ABAQUS["T0_MID"][0] != _CURVED_ABAQUS["T90_MID"][1],
            f"sestra {_CURVED_SESTRA['T0_MID'][0]:.12e} both sides; abaqus "
            f"{_CURVED_ABAQUS['T0_MID'][0]:.12e} vs {_CURVED_ABAQUS['T90_MID'][1]:.12e}, rel "
            f"{abs(_CURVED_ABAQUS['T90_MID'][1] / _CURVED_ABAQUS['T0_MID'][0] - 1.0):.3e}",
        )
    )
    results.append(
        _expect(
            "the two extrapolants agree with each other on the headline radial quantity to 8.8e-05",
            abs(extrapolants["abaqus"] / extrapolants["sestra"] - 1.0) < 1.0e-04,
            f"rel {abs(extrapolants['abaqus'] / extrapolants['sestra'] - 1.0):.4e}",
        )
    )
    results.append(
        _expect(
            "CURVED_REL_TOL is at least twice the measured worst extrapolated component",
            ccmp.CURVED_REL_TOL >= 2.0 * _CURVED_CROSS_SOLVER["extrapolated"],
            f"{ccmp.CURVED_REL_TOL:.1e} against a measured {_CURVED_CROSS_SOLVER['extrapolated']:.4e}",
        )
    )
    results.append(
        _expect(
            "the extrapolated and single-grid residuals are the same size -- the extrapolation does "
            "NOT tighten the cross-solver number here, which is this case's finding",
            0.5 < _CURVED_CROSS_SOLVER["extrapolated"] / _CURVED_CROSS_SOLVER["finest"] < 2.0,
            f"extrapolated {_CURVED_CROSS_SOLVER['extrapolated']:.4e}, finest " f"{_CURVED_CROSS_SOLVER['finest']:.4e}",
        )
    )
    results.append(
        _expect(
            "and the COARSEST grid pair would FAIL CURVED_REL_TOL -- so the tolerance is not one a "
            "coarse mesh would pass",
            _CURVED_CROSS_SOLVER["coarsest"] > ccmp.CURVED_REL_TOL,
            f"coarsest {_CURVED_CROSS_SOLVER['coarsest']:.4e} against tol {ccmp.CURVED_REL_TOL:.1e}",
        )
    )
    # And the comparator itself, on the two measured extrapolated tables.
    sestra = _curved_table("sestra", _CURVED_SESTRA)
    abaqus = _curved_table("abaqus", _CURVED_ABAQUS)
    report = cmp_mod.compare(sestra, abaqus, rel_tol=ccmp.CURVED_REL_TOL)
    results.append(
        _expect(
            "the two measured extrapolated tables agree at CURVED_REL_TOL, all 72 components",
            report.ok and len(report.diffs) == 6 * len(cvm.PROBE_POINTS),
            f"{len(report.diffs)} components, {len(report.failures)} failed, worst "
            f"{report.worst.probe}.{report.worst.component} rel {report.worst.rel_diff:.4e}",
        )
    )
    results.append(
        _expect(
            "the worst component is T90_Q.u3, the probe the S4R asymmetry hits hardest",
            f"{report.worst.probe}.{report.worst.component}" == "T90_Q.u3",
            f"{report.worst.probe}.{report.worst.component}",
        )
    )
    # The extrapolation machinery on a real sequence, end to end.
    for solver in ("abaqus", "sestra"):
        sequence = _curved_sequence(solver)
        table = ccmp.extrapolate(sequence)
        measured = ccmp.radial_displacement(table, cvm.RADIAL_PROBE)
        results.append(
            _expect(
                f"{solver}: extrapolate() over the three reconstructed solves reproduces the radial " f"extrapolant",
                abs(measured / extrapolants[solver] - 1.0) < 1e-09,
                f"{measured:.12e} against {extrapolants[solver]:.12e}",
            )
        )
        report = ccmp.convergence_report(sequence)
        results.append(
            _expect(
                f"{solver}: convergence_report names the grids by count and prints the order range",
                report.counts == cvm.MESH_COUNTS and report.radial is not None and report.axial is not None,
                f"counts {report.counts}, order range {tuple(round(v, 4) for v in report.order_range)}",
            )
        )
    return results


def _emitted_curved_script(count: int = 8) -> str:
    """The CAE script the **writer** produces for the panel, plus the appended driver, as text.

    No licence and no Abaqus: the supports, the regions, the step, the pressure and the driver's
    own constants are all written at plan time, so the text is checkable here. It is emitted
    rather than reconstructed because it *is* the deck -- asserting against anything this package
    built itself would be asserting against a second opinion about the supports.
    """
    from . import curved_abaqus_runner as car
    from . import curved_model as cvm

    assembly = cvm.build_panel(count, route="abaqus")
    with tempfile.TemporaryDirectory() as tmp:
        script = pathlib.Path(tmp) / f"{car.SCRIPT_STEM}.py"
        assembly.to_abaqus_cae_script(
            script,
            mesh_size=cvm.PANEL_ARC / count,
            shell_element_type=car.DEFAULT_SHELL_ELEMENT,
            job_name=car.JOB_NAME,
            # The runner's own flag, not a literal: this helper has to emit the deck the production
            # path emits, or the checks below are about a script nobody runs.
            submit=car.WRITER_SUBMITS,
        )
        return script.read_text(encoding="utf-8") + car.driver_source(count)


def check_curved_boundary_semantics() -> list[bool]:
    """ "Symmetry on two straight generators, free arcs, one axial reference" must mean one thing.

    Both decks are generated from :data:`curved_model.EDGE_SUPPORTS` by a *writer*: ``write_bcs``
    turns the three ``Bc`` records into ``BNBCD`` FIX codes and the CAE writer into three
    ``DisplacementBC`` on assembly ``Set``s -- two of geometry **edges** and one of a **vertex**.
    That split is the whole reason the case was designed with straight generators and a corner
    reference, so it is asserted off the writer's own emitted calls.

    The expected ``DisplacementBC`` lines are asserted **literally** rather than rebuilt from
    ``EDGE_SUPPORTS``. Rebuilding them would make this check move with the very data it is
    checking: found by mutation on the plate case, dropping a dof from a symmetry entry was caught
    by nothing at all.

    The negative half matters as much as the positive. On the ``theta = 0`` generator ``ur_y`` is
    **UNSET** -- the rotation about the hoop tangent, which is the one a symmetry plane leaves free
    -- and on the ``theta = 90`` generator ``ur_x`` is. Fixing either would make the generator a
    clamp rather than a symmetry plane. And ``AXIAL_REF`` fixes ``u3`` and **nothing else**: it
    exists to remove one rigid-body mode, and every other dof it touched would be a restraint the
    closed form does not have.
    """
    print("\ncurved boundary-condition semantics (both decks must mean one thing):")
    from ada.cadit.cae.analysis import BC_KEYWORDS

    from . import curved_abaqus_runner as car
    from . import curved_model as cvm

    results = []
    results.append(
        _expect(
            "BC_KEYWORDS is adapy's own dof 1..6 order, which the writer renders each Bc's dofs through",
            tuple(BC_KEYWORDS) == ("u1", "u2", "u3", "ur1", "ur2", "ur3"),
            f"{tuple(BC_KEYWORDS)}",
        )
    )
    text = _emitted_curved_script()
    expected = {
        "AXIAL_REF": "u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET",
        "SYM_T0": "u1=UNSET, u2=0.0, u3=UNSET, ur1=0.0, ur2=UNSET, ur3=0.0",
        "SYM_T90": "u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=0.0, ur3=0.0",
    }
    for name, keywords in sorted(expected.items()):
        line = (
            f"    model.DisplacementBC(name={name!r}, createStepName='Initial',\n"
            f"                         region=assembly.sets[{name!r}], {keywords})"
        )
        results.append(
            _expect(
                f"the writer emits {name} as exactly '{keywords}'",
                line in text,
                "found" if line in text else f"MISSING: {line}",
            )
        )
    results.append(
        _expect(
            "the three supports named in the model are the three the writer emits, and no more",
            text.count("model.DisplacementBC(") == len(cvm.EDGE_SUPPORTS) == 3,
            f"{text.count('model.DisplacementBC(')} DisplacementBC calls for {len(cvm.EDGE_SUPPORTS)} records",
        )
    )
    # The regions: TWO whole straight generator edges and ONE vertex. A curved boundary edge
    # cannot be a region at all -- a bounding box round a 90-degree arc contains the whole panel --
    # which is why the supported edges are the generators and the arcs are free.
    edge_regions = {
        match.group("name"): (int(match.group("edges")), float(match.group("length")))
        for match in _EDGE_REGION_CALL.finditer(text)
    }
    results.append(
        _expect(
            "both symmetry supports are edge regions of exactly one edge, pi m long -- the whole " "generator",
            edge_regions == {"SYM_T0": (1, cvm.PANEL_LENGTH), "SYM_T90": (1, cvm.PANEL_LENGTH)},
            f"{edge_regions}",
        )
    )
    vertex_calls = [line.strip() for line in text.splitlines() if line.startswith("    _analysis_region(assembly,")]
    results.append(
        _expect(
            "the axial reference is a VERTEX region at the corner (r, 0, 0), and it is the only one",
            len(vertex_calls) == 1 and "'AXIAL_REF'" in vertex_calls[0] and "((2.0, 0.0, 0.0),)" in vertex_calls[0],
            f"{vertex_calls}",
        )
    )
    # ...and what the runner refuses a run for must be the same three kinds the writer emitted. The
    # two have to agree or read_checks would reject a correct build, or accept a wrong one.
    emitted_kinds = {name: "edge" for name in edge_regions}
    emitted_kinds.update({"AXIAL_REF": "vertex"})
    results.append(
        _expect(
            "the kinds the runner insists on are exactly the kinds the writer emitted: two edges " "and one vertex",
            dict(car.EXPECTED_REGION_KINDS) == emitted_kinds,
            f"runner {dict(car.EXPECTED_REGION_KINDS)} against emitted {emitted_kinds}",
        )
    )
    # The calls, not the helper definitions the writer emits alongside them: every region call
    # sits indented inside build(), and the writer emits all three helpers whether or not it uses
    # them -- a check that matched the def would pass on any script at all.
    face_calls = text.count("\n    _analysis_face_region(assembly,")
    edge_calls = text.count("\n    _analysis_edge_region(assembly,")
    results.append(
        _expect(
            "no support is a FACE region -- a face region would hold every node of the panel -- and "
            "exactly two are edges",
            face_calls == 0 and edge_calls == 2,
            f"{edge_calls} edge region call(s), {face_calls} face, {len(vertex_calls)} vertex",
        )
    )
    # The load. side1Faces and a NEGATIVE magnitude: the face's normal is outward radial, and
    # Abaqus takes a positive magnitude as acting into side1, which would be external pressure.
    results.append(
        _expect(
            "the pressure is emitted with the model's own NEGATIVE magnitude, over a side1Faces "
            "Surface -- which is what makes an internal pressure expand the shell",
            f"magnitude={cvm.SIGNED_PRESSURE_MAGNITUDE}" in text
            and cvm.SIGNED_PRESSURE_MAGNITUDE < 0.0
            and "side1Faces=faces" in text,
            f"magnitude={cvm.SIGNED_PRESSURE_MAGNITUDE}, one Pressure call: " f"{text.count('model.Pressure(') == 1}",
        )
    )
    results.append(
        _expect(
            "the writer is asked NOT to submit, and the appended driver carries the job and the "
            "sidecar instead -- because the writer's own equilibrium guard cannot check a pressure "
            "on a curved face",
            car.WRITER_SUBMITS is False
            and "def solve(" not in text
            and f"CURVED_JOB = {car.JOB_NAME!r}" in text
            and f"CURVED_DISPLACEMENTS = {car.DISPLACEMENTS_NAME!r}" in text
            and "def _curved_solve():" in text,
            f"WRITER_SUBMITS={car.WRITER_SUBMITS}, {car.JOB_NAME} -> {car.DISPLACEMENTS_NAME}",
        )
    )
    results.append(
        _expect(
            "the driver carries BOTH pressure resultants -- the exact one it checks against and the "
            "writer's flat-plate one -- so the defect is measured on every run",
            f"CURVED_APPLIED_PRESSURE = {car.exact_pressure_resultant()!r}" in text
            and "CURVED_WRITER_PRESSURE = " in text
            and "residual_with_writer_formula" in text,
            f"exact {car.exact_pressure_resultant()}",
        )
    )
    # The defect itself, reproduced through the writer's own planner rather than by reimplementing
    # its formula -- so this check stops passing the day the writer is fixed.
    error = car.writer_pressure_error()
    results.append(
        _expect(
            "AnalysisPlan.applied_pressure is over 15% wrong on this face and 7.6% asymmetric, "
            "against a guard tolerance of 1e-04",
            error["relative"] > 0.15 and error["skew"] > 0.07,
            f"worst component off by {error['worst_component_error']:.6g} N ({error['relative']:.3e}), "
            f"skew {error['skew']:.3e}",
        )
    )
    results.append(
        _expect(
            "the route decides the load form and nothing else -- the supports are the same records " "on both",
            cvm.LOAD_STYLES == {"sestra": "nodal", "abaqus": "pressure"} and cvm.ROUTES == ("sestra", "abaqus"),
            f"{cvm.LOAD_STYLES}",
        )
    )
    results.append(
        _expect_raise(
            "an unknown route is refused rather than defaulted",
            cvm.CurvedModelInvalid,
            lambda: cvm.build_panel(8, route="ansys"),
        )
    )
    return results


def check_curved_loud_failures() -> list[bool]:
    """Every curved guard, run against the input it exists for. Each must raise."""
    print("\ncurved loud-failure guards (each must raise):")
    from . import compare as cmp_mod
    from . import curved_compare as ccmp
    from . import curved_hand_check as chc
    from . import curved_model as cvm

    results = []
    sestra = _curved_table("sestra", _CURVED_SESTRA)

    # 1. THE curved-case failure: an internal pressure that contracts the shell. A flipped face
    # normal, a positive *Dsload magnitude or a facet whose node ordering reversed all produce a
    # perfectly converged, perfectly equilibrated answer of the wrong sign.
    inward = _curved_table("sestra", {name: (-a, -b, c) for name, (a, b, c) in _CURVED_SESTRA.items()})
    results.append(
        _expect_raise(
            "a pressure with the wrong sign -- the panel contracts under an internal pressure",
            ccmp.ContractsUnderInternalPressure,
            lambda: ccmp.assert_pressure_expands(inward),
        )
    )
    results.append(
        _expect(
            "...and the correct table passes it, returning the smallest radial displacement",
            abs(ccmp.assert_pressure_expands(sestra) / chc.radial_displacement() - 1.0) < 1.0e-03,
            f"smallest radial {ccmp.assert_pressure_expands(sestra):.9e}",
        )
    )
    results.append(
        _expect_raise(
            "an unloaded panel, which neither expands nor contracts -- not the same as a pressure " "that arrived",
            ccmp.ContractsUnderInternalPressure,
            lambda: ccmp.assert_pressure_expands(_curved_table("sestra", {k: (0.0, 0.0, 0.0) for k in _CURVED_SESTRA})),
        )
    )

    # 2. A symmetry dof dropped. These three records are the only route by which either solver
    # hears about the supports, so a record lost on the way in is a support lost from both decks.
    supported = _curved_fem_stub()
    supported.bcs = [_FakeBc(name, list(dofs)) for name, dofs, _why in cvm.EDGE_SUPPORTS]
    results.append(
        _expect(
            "the three EDGE_SUPPORTS records pass the guard that says they are all there",
            cvm.assert_supports_declared(supported) is None,
            f"{[bc.name for bc in supported.bcs]}",
        )
    )
    dropped = _curved_fem_stub()
    dropped.bcs = [_FakeBc(name, list(dofs)) for name, dofs, _why in cvm.EDGE_SUPPORTS[:-1]]
    results.append(
        _expect_raise(
            "the axial reference never added as a Bc -- which leaves the one singular direction",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_supports_declared(dropped),
        )
    )
    loosened = _curved_fem_stub()
    loosened.bcs = [_FakeBc(name, [d for d in dofs if d != 6]) for name, dofs, _why in cvm.EDGE_SUPPORTS]
    results.append(
        _expect_raise(
            "ur_z dropped from both symmetry records -- the panel is then not a quarter of a cylinder",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_supports_declared(loosened),
        )
    )

    # 3. A curved plate that meshed to fewer shells than expected. Its chord error is then not the
    # one its place in the refinement sequence assumes, so the extrapolation is of the wrong thing.
    assembly = cvm.build_panel(8, route="sestra")
    fem = _curved_part(assembly).fem
    results.append(
        _expect(
            "the 8 x 8 grid passes assert_shell_grid at count=8",
            cvm.assert_shell_grid(fem, count=8) is None,
            f"{len(cvm.shell_elements(fem))} shell(s)",
        )
    )
    results.append(
        _expect_raise(
            "a panel that meshed to fewer shells than the model asked for",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_shell_grid(fem, count=16),
        )
    )
    beams_only = _unit_grid(2, 2, kind="LineShapes.LINE")
    results.append(
        _expect_raise(
            "a FEM with no shell elements at all -- every number here comes from a membrane state",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_shell_grid(beams_only, count=2),
        )
    )
    # ...and the three geometric clauses, each on a grid broken in exactly one way.
    off_radius = cvm.build_panel(8, route="sestra")
    off_fem = _curved_part(off_radius).fem
    for node in off_fem.nodes:
        if abs(node.z - cvm.PANEL_LENGTH / 2.0) < 1e-12:
            node.p = (node.p[0] * 1.001, node.p[1] * 1.001, node.p[2])
    results.append(
        _expect_raise(
            "a grid whose nodes are off radius r -- w goes as r^2, so 1% of radius is 2% of answer",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_shell_grid(off_fem, count=8),
        )
    )
    # One corner slid ALONG the cylinder to a slightly different theta, keeping its radius and its
    # z. That is the only way to warp a facet of this grid without also moving a node off radius r,
    # and it matters: lifting the corner along the facet normal warps it too, but it is then the
    # on-the-cylinder clause that raises and the planarity clause is never reached -- found by
    # mutation, which is exactly what deleting the planarity clause failed to be caught by.
    warped = cvm.build_panel(8, route="sestra")
    warped_fem = _curved_part(warped).fem
    element = cvm.shell_elements(warped_fem)[0]
    node = element.nodes[2]
    angle = math.atan2(node.p[1], node.p[0]) + 1.0e-04
    node.p = (cvm.PANEL_RADIUS * math.cos(angle), cvm.PANEL_RADIUS * math.sin(angle), node.p[2])
    results.append(
        _expect_raise(
            "a WARPED facet -- integral(N_i) dA = A / 4 holds only on a rectangle, so the "
            "consistent nodal load would stop being the pressure's own vector",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_shell_grid(warped_fem, count=8),
        )
    )
    flipped = cvm.build_panel(8, route="sestra")
    flipped_fem = _curved_part(flipped).fem
    element = cvm.shell_elements(flipped_fem)[0]
    element._nodes = list(reversed(element.nodes))
    results.append(
        _expect_raise(
            "a facet whose normal points at the cylinder axis -- it would carry the internal "
            "pressure inwards while the rest expands",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_shell_grid(flipped_fem, count=8),
        )
    )
    results.append(
        _expect_raise(
            "a grid count that is not a multiple of four, so z = L/4 is not a node",
            cvm.CurvedModelInvalid,
            lambda: cvm.shell_grid(6),
        )
    )
    unseeded = _curved_fem_stub()
    results.append(
        _expect_raise(
            "a mesh with no node at a probe point, which would sample the wrong place",
            cvm.CurvedModelInvalid,
            lambda: cvm.assert_probes_are_seeded(unseeded, count=8),
        )
    )

    # 4. The hoop state, the axial contraction, and the reaction -- the three physical guards.
    fanned = dict(_CURVED_SESTRA)
    fanned["ARC0_MID"] = tuple(v * 0.99 for v in _CURVED_SESTRA["ARC0_MID"])
    results.append(
        _expect_raise(
            "a panel whose radial expansion is not uniform round its hoop -- so p r^2 / (E t) with "
            "no nu in it is the wrong closed form",
            ccmp.NotUniform,
            lambda: ccmp.assert_hoop_uniform(_curved_table("sestra", fanned)),
        )
    )
    results.append(
        _expect_raise(
            "an unloaded panel, which is uniform round its hoop too",
            ccmp.NotUniform,
            lambda: ccmp.assert_hoop_uniform(_curved_table("sestra", {k: (0.0, 0.0, 0.0) for k in _CURVED_SESTRA})),
        )
    )
    held = {name: (a, b, 0.0) for name, (a, b, _c) in _CURVED_SESTRA.items()}
    results.append(
        _expect_raise(
            "u_z held everywhere -- the axial Poisson contraction is restrained and the second "
            "closed form is the wrong one",
            ccmp.AxialRestraintPresent,
            lambda: ccmp.assert_axial_contraction(_curved_table("sestra", held)),
        )
    )
    shifted = {name: (a, b, c - 1.0e-06) for name, (a, b, c) in _CURVED_SESTRA.items()}
    results.append(
        _expect_raise(
            "u_z non-zero at z = 0 -- something other than the one reference node is restraining "
            "the axial direction",
            ccmp.AxialRestraintPresent,
            lambda: ccmp.assert_axial_contraction(_curved_table("sestra", shifted)),
        )
    )
    # The slope clause alone: every u3 5% too small, so the field is still perfectly LINEAR and
    # only its gradient is wrong. That is an axial restraint somewhere, and nothing else in this
    # module would notice it.
    slack = {name: (a, b, 0.95 * c) for name, (a, b, c) in _CURVED_SESTRA.items()}
    results.append(
        _expect_raise(
            "u_z linear but 5% short -- the axial contraction is being partly held",
            ccmp.AxialRestraintPresent,
            lambda: ccmp.assert_axial_contraction(_curved_table("sestra", slack)),
        )
    )
    # The linearity clause alone: ONE probe's u3 1% out. Over ten probes that moves the mean slope
    # by 0.1%, which is inside AXIAL_REL_TOL, while the nonlinearity at that probe is the full 1%
    # and is not -- so this input isolates the second clause from the first. The two mutations that
    # delete them are each caught by exactly one of these two checks.
    bent = dict(_CURVED_SESTRA)
    bent["T0_MID"] = (
        _CURVED_SESTRA["T0_MID"][0],
        0.0,
        _CURVED_SESTRA["T0_MID"][2] * 1.01,
    )
    results.append(
        _expect_raise(
            "u_z not linear in z -- one probe out by 1%, which leaves the mean slope inside its own "
            "tolerance and the linearity outside",
            ccmp.AxialRestraintPresent,
            lambda: ccmp.assert_axial_contraction(_curved_table("sestra", bent)),
        )
    )
    results.append(
        _expect(
            "...and the measured Sestra extrapolant passes all three axial clauses",
            ccmp.assert_axial_contraction(sestra)["relative"] <= chc.AXIAL_REL_TOL,
            f"{ {k: f'{v:.3e}' for k, v in ccmp.assert_axial_contraction(sestra).items()} }",
        )
    )
    results.append(
        _expect_raise(
            "a solve that reacted half the pressure",
            ccmp.ReactionMismatch,
            lambda: ccmp.assert_reaction_total(
                _curved_solve(sestra, count=8, reaction=tuple(-0.5 * v for v in cvm.expected_load_total()))
            ),
        )
    )
    results.append(
        _expect_raise(
            "a solve whose reaction has the SAME sign as the load -- the supports pushing the panel " "outwards",
            ccmp.ReactionMismatch,
            lambda: ccmp.assert_reaction_total(_curved_solve(sestra, count=8, reaction=cvm.expected_load_total())),
        )
    )
    results.append(
        _expect_raise(
            "a non-zero axial reaction -- an open-ended cylinder reacts nothing in z, so the arc " "ends are not free",
            ccmp.ReactionMismatch,
            lambda: ccmp.assert_reaction_total(
                _curved_solve(
                    sestra,
                    count=8,
                    reaction=(-cvm.expected_load_total()[0], -cvm.expected_load_total()[1], 1.0e3),
                )
            ),
        )
    )
    results.append(
        _expect_raise(
            "the axial reference node carrying real load, which is the same defect from the " "reaction side",
            ccmp.AxialRestraintPresent,
            lambda: ccmp.assert_reference_node_carries_nothing((0.0, 0.0, 1.0e3)),
        )
    )
    results.append(
        _expect(
            "...and a node that carries nothing passes it",
            ccmp.assert_reference_node_carries_nothing((0.0, 0.0, 5.3e-09)) < 1.0e-05,
            "5.3e-09 N in z, which is what Sestra measures there",
        )
    )

    # 5. A refinement sequence that is not one, and a component that does not converge.
    results.append(
        _expect_raise(
            "two grids instead of three -- two can only confirm a rate that was assumed",
            plate_compare_not_a_sequence(),
            lambda: ccmp.extrapolate(_curved_sequence("sestra")[:2]),
        )
    )
    results.append(
        _expect_raise(
            "a sequence out of order, fine to coarse -- it extrapolates away from the limit",
            plate_compare_not_a_sequence(),
            lambda: ccmp.extrapolate(list(reversed(_curved_sequence("sestra")))),
        )
    )
    results.append(
        _expect_raise(
            "two solvers mixed into one sequence",
            plate_compare_not_a_sequence(),
            lambda: ccmp.extrapolate(_curved_sequence("sestra")[:2] + _curved_sequence("abaqus")[2:]),
        )
    )
    turning = _curved_sequence("sestra")
    middle = dict(_CURVED_SESTRA)
    middle["T0_MID"] = (_CURVED_SESTRA["T0_MID"][0] * 1.5, 0.0, _CURVED_SESTRA["T0_MID"][2])
    turning[1] = _curved_solve(_curved_table("sestra", middle), count=cvm.MESH_COUNTS[1])
    results.append(
        _expect_raise(
            "a component whose sequence turns around instead of approaching a limit",
            chc.NotConverging,
            lambda: ccmp.extrapolate(turning),
        )
    )
    # A genuine FIRST-order sequence: the error halves rather than quartering when the grid
    # doubles. That is what a locking element, or a "refinement" that did not change the mesh,
    # actually looks like -- and CURVED_ORDER_BAND excludes it even at its width.
    first_order = []
    for index, factor in enumerate((1.0, 0.5, 0.25)):
        values = {name: tuple(v * (1.0 - 0.01 * factor) for v in row) for name, row in _CURVED_SESTRA.items()}
        first_order.append(_curved_solve(_curved_table("sestra", values), count=cvm.MESH_COUNTS[index]))
    results.append(
        _expect_raise(
            "a FIRST-order sequence, which is what a locking element or a mesh that was not "
            "actually refined gives -- outside CURVED_ORDER_BAND even at its width",
            chc.NotConverging,
            lambda: ccmp.extrapolate(first_order),
        )
    )
    odd_seed = dataclasses.replace(_curved_solve(sestra, count=8), mesh_size=0.3)
    results.append(
        _expect_raise(
            "a seed that is not PANEL_ARC / n for any whole n -- a solve from a different study",
            plate_compare_not_a_sequence(),
            lambda: ccmp.grid_count(odd_seed),
        )
    )

    # 6. The two cases must never be cross-compared: their probe sets are different structures.
    results.append(
        _expect_raise(
            "the curved probe set compared against the portal frame's",
            cmp_mod.ProbeSetMismatch,
            lambda: cmp_mod.compare(sestra, _table("abaqus", _REFERENCE)),
        )
    )

    # 7. The premise the analytic grid rests on. A workaround whose reason has quietly become false
    # is worse than no workaround, so the three refusals are reproduced on every run.
    findings = cvm.reproduce_meshing_refusals()
    results.append(
        _expect(
            "adapy still refuses to mesh a PlateCurved at all three points, so the analytic grid is "
            "still the nearest expressible equivalent",
            ccmp.assert_meshing_gap_is_still_open(findings) == findings,
            "; ".join(f"{k}: {findings[k]}" for k in sorted(findings)),
        )
    )
    results.append(
        _expect_raise(
            "and the day one of them starts working, this package is told to delete its grid",
            ccmp.MeshingGapClosed,
            lambda: ccmp.assert_meshing_gap_is_still_open({**findings, "PlateCurved.shell_occ": "returned a shape"}),
        )
    )
    return results


def plate_compare_not_a_sequence():
    from . import plate_compare

    return plate_compare.NotASequence


def _curved_fem_stub():
    """A ``_FakeFem`` with nodes nowhere near the panel -- for the guards that only read nodes."""
    return _unit_grid(4, 4, dx=0.1, dy=0.1)


def main() -> int:
    results = (
        check_loud_failures()
        + check_agreement()
        + check_hand_check()
        + check_solver_section_idealisation()
        + check_plate_closed_forms()
        + check_plate_convergence()
        + check_plate_agreement()
        + check_plate_boundary_semantics()
        + check_plate_loud_failures()
        + check_curved_closed_forms()
        + check_curved_convergence()
        + check_curved_boundary_semantics()
        + check_curved_loud_failures()
    )
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
