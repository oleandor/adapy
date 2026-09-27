"""The Abaqus half of the curved case: the same panel -> a CAE script -> ``S4R`` -> a table.

Written by ``Part.to_abaqus_cae_script`` from :func:`curved_model.build_panel`, the same
function the Sestra half calls. **The model travels entirely through the writer** -- the ACIS
body, the shell section, the mesh, the element type, the three supports and the pressure
itself. What does *not* is the solve, and the reason is a defect in ``src`` reported here rather
than fixed (gap 2 of :mod:`curved_model`):

    ``ada.cadit.cae.analysis.AnalysisPlan.applied_pressure`` computes a pressure's resultant as
    ``-magnitude * area * normal``. That is the flat-plate formula. On a curved face the normal
    is not constant and the resultant is ``integral(p n dA)``, which is smaller and points
    somewhere else. Measured in the emitted script for this panel at ``n = 8``::

        APPLIED_PRESSURE = (724863.7521754879, 669823.4477113917, 0.0)   <- the writer's
        integral(p n dA) = (628318.5307179586, 628318.5307179586, 0.0)   <- the exact one

    15.4% high in ``x``, 6.6% high in ``y``, and **asymmetric** where the panel is symmetric to
    the last bit -- because the "normal" it uses is whatever the face locator probed at one
    point, which on a quarter cylinder is not 45 degrees. The guard's tolerance is 1e-04
    relative, so the residual is about 1000x it, and ``_guard_equilibrium`` runs **before**
    ``_odb_displacements``: the build fails and no displacement sidecar is written at all.

So the script is emitted with ``submit=False`` and :data:`DRIVER_MARKER` appends a driver that
carries the job, the equilibrium check against the *exact* curved-face resultant, and the
displacement sidecar in the writer's own schema. That is the shape the plate case had before
adapy PR #405, and it is deliberately the minimum: the driver creates no part, no section, no
region, no support, no step and no load. It also records what the writer's own formula *would*
have left as a residual (``residual_with_writer_formula`` in the build-result sidecar), so the
defect is measured on every run rather than only described here.

What the writer translates, verified in the emitted script and its sidecars
==========================================================================

* the panel as the **ACIS body adapy's own SAT writer produces** from the rational NURBS patch
  (:func:`curved_model.panel_face`), imported with ``openAcis`` / ``PartFromGeometryFile``:
  ``faces 1, edges 4, vertices 4``. The emitted ``_guard_plate_faces`` asserts the face's area
  against adapy's own ``9.86960440109`` and its normal, in the kernel, and that is the check
  that makes an arc honest: a face Abaqus considers invalid returns ``getSize() == 0.0`` rather
  than raising, so the comparison is absolute and scaled by the expected area;
* a ``HomogeneousShellSection(thickness=0.01, material='S355')`` at ``offsetType=MIDDLE_SURFACE``
  on that face, which puts the reference surface on the cylinder of radius ``r`` -- the surface
  the closed form is written on;
* ``S4R`` on the face, ``seedPart(size=...)`` at :data:`curved_model.MESH_SIZES`. Both of the
  panel's edge lengths are :data:`curved_model.PANEL_ARC`, and ``seedPart`` gives an edge
  ``round(length / size)`` elements, so one seed divides both directions into the same count;
* the three supports of :data:`curved_model.EDGE_SUPPORTS`, each one ``DisplacementBC`` in the
  ``Initial`` step with all six dofs written out -- ``0.0`` where the record fixes it and
  ``UNSET`` where it does not. **Two edge regions and one vertex**, which is the whole reason
  the case was designed with straight generators:

  =============  ================================================  ==============
  region         located by                                         kind
  =============  ================================================  ==============
  ``SYM_T0``     one box, ``x in [r-e, r+e], y in [-e, e]``          edge, 1, ``pi``
  ``SYM_T90``    one box, ``x in [-e, e], y in [r-e, r+e]``          edge, 1, ``pi``
  ``AXIAL_REF``  the vertex ``(2.0, 0.0, 0.0)``                      vertex, 1 node
  =============  ================================================  ==============

  A **curved** boundary edge cannot be a region at all: ``ada.cadit.cae.plates`` keeps only the
  edges whose own curve record is a ``StraightCurve`` and counts the rest as
  ``curved_plate_edges``, because a bounding box around an arc is not the arc -- the box that
  contains a 90-degree arc contains the whole quarter panel. That is why the two supported
  edges are the straight generators and the two arc ends are free, and why the axial reference
  is a corner **vertex** rather than a point halfway along a generator: a mid-edge node is on
  no vertex, is not a whole edge and is not a plate's whole mesh, so it would be refused by
  name. Both are checked off the emitted script's own calls in
  :func:`selftest.check_curved_boundary_semantics`;
* the ``StaticStep``, a ``FieldOutputRequest`` naming ``U``, ``UR`` and ``RF``, and the
  pressure: ``assembly.Surface(side1Faces=<the panel's face>)`` and
  ``model.Pressure(magnitude=-100000.0)``.

The pressure's sign on a curved face, measured on both routes
============================================================

:data:`PRESSURE_SIGN_MEASUREMENT`. This is the one question a curved face asks that a flat plate
does not, because "outward" is a direction that changes along the surface.

Running it
==========

From the repository root::

    python -m verification.genie_vs_abaqus.run_comparison --case curved --work-dir D:/temp/curved

Three CAE solves, run strictly one at a time and each bounded by :data:`RUN_TIMEOUT`; there are
four CAE tokens on the site server. The launcher search, the timeout and the sanitised child
environment come from ``tests.core.cadit.cae.abaqus_runner`` rather than being reimplemented --
that module is where the measurement lives that a relative ``PYTHONPATH`` inherited by the child
kills ``job.submit()`` with a bare ``Abaqus Error`` and no traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass

from . import curved_model
from .abaqus_runner import AbaqusFailed, AbaqusNotInstalled, read_displacements_sidecar
from .curved_sestra_runner import model_name
from .displacements import sample_fea_result
from .plate_sestra_runner import PlateSolve

#: The shell element the comparison is closed with. ``S4R`` is the like-for-like choice against
#: Sestra's ``FQUS``: both are 4-node bilinear shells with reduced integration on the membrane
#: terms, which is what makes a convergence comparison between them a statement about the
#: translation rather than about two element theories. It is also the only shell code whose
#: answer against a closed form has been measured on this writer's output.
DEFAULT_SHELL_ELEMENT = "S4R"

#: Seconds. The finest grid is 1024 ``S4R`` and solves in a few; this is a stuck-run guard, and
#: it matters because a hung run holds one of the four site CAE tokens.
RUN_TIMEOUT = 1800.0

#: The job the appended driver submits, and the stem of the solver's own files.
JOB_NAME = "curved_job"

#: The emitted script's name, without its suffix.
SCRIPT_STEM = "curved"

#: The displacement sidecar the driver writes and :func:`abaqus_runner.read_displacements_sidecar`
#: reads. Same schema and same name shape as the writer's own, so the reader needs no branch.
DISPLACEMENTS_NAME = "{0}.cae_displacements.json".format(SCRIPT_STEM)

#: The writer's build-result sidecar: every guard's verdict, the mesh it built, the regions it
#: resolved -- plus the ``curved_equilibrium`` block the driver adds.
BUILD_RESULT_NAME = "{0}.cae_build_result.json".format(SCRIPT_STEM)

#: A line the appended driver always contains, so a script that was emitted without it can be
#: told apart from one whose driver simply did not run.
DRIVER_MARKER = "ADAPY-CAE CURVED SOLVE OK"

#: Whether ``to_abaqus_cae_script`` is asked to submit the job. **False**, and it is a named
#: constant rather than a literal at the one call site for a reason: it is the whole of what this
#: case does differently from the plate case's, and it is what the licence-free selftest emits
#: with, so a change to it is caught there rather than only on a run that spends a CAE token. With
#: it True the writer's own ``solve()`` runs its ``_guard_equilibrium``, which computes a curved
#: face's pressure resultant with the flat-plate formula, fails by about 1000x its tolerance and
#: leaves no displacement sidecar -- and the appended driver would then be a second solve of a
#: model that had already failed.
WRITER_SUBMITS = False

#: Relative tolerance the driver's own equilibrium check uses, against the larger of the exact
#: pressure resultant and the largest single nodal reaction. 1e-04, the same figure
#: ``ada.cadit.cae.writer.EQUILIBRIUM_REL_TOL`` uses -- the check is the writer's, only its
#: applied-pressure term is replaced.
EQUILIBRIUM_REL_TOL = 1.0e-04

#: The measured answer to "which way does a positive pressure push on a curved face", on both
#: routes. The one question this case asks that a flat plate does not.
#:
#: ===========================  ====================================================================
#: route                        measurement
#: ===========================  ====================================================================
#: Abaqus, ``*Dsload`` on the   the writer builds the region as ``assembly.Surface(side1Faces=...)``
#: imported curved face         over the panel's one face, whose normal survives the ACIS import as
#:                              the **outward radial** direction (adapy's own
#:                              ``thickness_direction()`` reads ``(0.7071, 0.7071, 0.0)`` at the
#:                              mid-face). A positive magnitude acts *into* side1, i.e. inwards, and
#:                              **contracts** the shell. Measured on this panel at ``n = 8`` by
#:                              solving it twice and reading the **reaction total's sign**:
#:                              ``magnitude=-1e5`` gives ``(-628318.555, -628318.559, -0.005)`` and
#:                              ``magnitude=+1e5`` gives ``(+628318.555, +628318.559, +0.005)``. The
#:                              reactions flip with the magnitude, so a positive one pulls the panel
#:                              in and the negative one
#:                              (:data:`curved_model.SIGNED_PRESSURE_MAGNITUDE`) expands it, which is
#:                              what an internal pressure must do. The reversed run's own
#:                              *displacements* were never read, and that is stated rather than
#:                              glossed: the driver's equilibrium check -- rightly -- fails that run
#:                              by ``2 p r L = 1256637.089`` N and writes no displacement sidecar, so
#:                              :func:`measure_pressure_sign` reads the reaction it recorded on the
#:                              way down.
#: Sesam ``BEUSLO``             not run here, and stated as not run: #404 was not merged (see
#:                              :mod:`curved_sestra_runner`). Its own measurement, quoted from that
#:                              branch, is that a BEUSLO intensity pushes along the element's
#:                              **negative** normal whichever ``SIDE`` is written, and that
#:                              ``*Dsload P, 1000.`` on an ``S4R`` whose normal is ``+z`` puts
#:                              ``U3`` negative -- i.e. the same convention as Abaqus.
#: Sesam ``BNLOAD``, used here  the consistent nodal vector is built along each facet's **own**
#:                              outward normal (:func:`curved_model.consistent_nodal_loads`), so its
#:                              sign is stated in the model rather than inherited from a writer
#:                              convention. Measured: it sums to ``(+p r L, +p r L, 0)``, the panel
#:                              expands (``u1`` at ``T0_MID`` positive at every density), and
#:                              :func:`curved_model.assert_shell_grid` refuses a grid any of whose
#:                              facet normals points at the axis -- which is the defect that would
#:                              otherwise flip one facet's share silently.
#: ===========================  ====================================================================
PRESSURE_SIGN_MEASUREMENT = {
    "abaqus_dsload_positive_magnitude": "contracts (side1Faces is the outward radial normal)",
    "abaqus_dsload_negative_magnitude": "expands -- what curved_model.SIGNED_PRESSURE_MAGNITUDE uses",
    "sesam_bnload_used_here": "expands; the sign is the facet's own outward normal, stated in the model",
    "sesam_beuslo_not_run": "#404 not merged; its own probe records the same against-the-normal convention",
}


@dataclass(frozen=True)
class CurvedRunChecks:
    """What the writer and the driver measured about the panel they solved."""

    reaction_total: tuple[float, float, float]
    node_count: int
    element_counts: dict[str, int]
    #: Per support set, what kind of geometry its region resolved to: two ``'edge'``, one
    #: ``'vertex'``.
    region_kinds: dict[str, str]
    #: Per edge-region support set, ``(edges, total length)`` as CAE itself measured them.
    region_edges: dict[str, tuple[int, float]]
    #: The panel's face area as CAE's ``getSize()`` reported it, against adapy's own.
    face_area: float
    #: The exact curved-face pressure resultant the driver checked equilibrium against.
    applied_pressure_exact: tuple[float, float, float]
    #: What ``AnalysisPlan.applied_pressure`` would have used instead -- the defect, measured.
    applied_pressure_writer_formula: tuple[float, float, float]
    #: The residual the writer's own formula would have left, newtons.
    residual_with_writer_formula: float
    solved: bool


def exact_pressure_resultant() -> tuple[float, float, float]:
    """``integral(p n dA)`` over the quarter panel, ``(p r L, p r L, 0)`` newtons.

    Signed as the **load**, i.e. outward, which is the sign the reaction must cancel. Derived in
    :func:`curved_hand_check.edge_reaction`; taken from the model rather than retyped so a
    changed pressure cannot leave a stale number in the driver.
    """
    return curved_model.expected_load_total()


def writer_pressure_resultant(count: int = curved_model.MESH_COUNTS[0]) -> tuple[float, float, float]:
    """What ``AnalysisPlan.applied_pressure`` computes for this panel -- the defect, reproduced.

    Built by asking the writer's own planner, not by reimplementing its formula: a
    reproduction that recomputed ``-q A n`` itself would still pass the day the writer was
    fixed. Needs no licence and no Abaqus -- the whole plan is made at emit time.

    Measured at ``count = 8``: ``(724863.7521754879, 669823.4477113917, 0.0)`` against the exact
    ``(628318.5307179586, 628318.5307179586, 0.0)``.
    """
    from ada.cadit.cae.writer import build_plan

    assembly = curved_model.build_panel(count, route="abaqus")
    part = _emitted_part(assembly)
    plan = build_plan(part, mesh_size=curved_model.PANEL_ARC / count, shell_element_type=DEFAULT_SHELL_ELEMENT)
    resultant = plan.analysis.applied_pressure()
    # applied_pressure() returns the LOAD, i.e. -q A n. The exact resultant above is the load
    # too, so the two are directly comparable; a sign flip here would compare a load against a
    # reaction and make the defect look twice as large as it is.
    return (float(resultant[0]), float(resultant[1]), float(resultant[2]))


def writer_pressure_error(count: int = curved_model.MESH_COUNTS[0]) -> dict[str, float]:
    """The defect as three numbers: the worst component error, its relative size, and the skew.

    ``skew`` is ``|fx - fy| / max(|fx|, |fy|)`` -- the asymmetry. It is reported separately
    because it is the part that cannot be argued away as a tolerance question: the panel is
    symmetric about ``theta = 45`` to the last bit, so the two components of its pressure
    resultant are *equal*, and a formula that returns them 8% apart is using a normal that is
    not the face's.
    """
    exact = exact_pressure_resultant()
    written = writer_pressure_resultant(count)
    worst = max(abs(written[axis] - exact[axis]) for axis in range(3))
    scale = max(abs(v) for v in exact)
    skew_scale = max(abs(written[0]), abs(written[1]))
    return {
        "worst_component_error": worst,
        "relative": worst / scale,
        "skew": abs(written[0] - written[1]) / skew_scale if skew_scale else 0.0,
    }


def _emitted_part(assembly):
    """The one adapy ``Part`` the CAE script is written from."""
    for part in assembly.get_all_parts_in_assembly(include_self=True):
        if part.name == curved_model.PART_NAME:
            return part
    raise AbaqusFailed(
        f"the assembly holds no part named {curved_model.PART_NAME!r}; it has "
        f"{[p.name for p in assembly.get_all_parts_in_assembly(include_self=True)]}"
    )


def driver_source(count: int) -> str:
    """The text appended to the emitted script: the solve, the equilibrium check, the sidecar.

    Two parts, and the split is deliberate. The **constants** are generated from the model, so a
    changed pressure or radius reaches the kernel without anyone editing kernel code; the
    **body** is a literal, so there is no ``%``/``{}`` substitution inside code that formats its
    own messages at runtime and nothing for a brace to go wrong in.
    """
    exact = exact_pressure_resultant()
    writer = writer_pressure_resultant(count)
    constants = [
        "",
        "",
        "# " + "-" * 86,
        "# Appended by verification.genie_vs_abaqus.curved_abaqus_runner. Everything above is the",
        "# writer's: the ACIS body, the shell section, the mesh, the element type, the three",
        "# supports and the pressure, all of it verified in the kernel by the writer's own guards.",
        "# What follows is ONLY the solve, the equilibrium check and the displacement sidecar.",
        "#",
        "# It is here rather than in the writer because",
        "# ada.cadit.cae.analysis.AnalysisPlan.applied_pressure computes a pressure's resultant as",
        "# -magnitude * area * normal -- the FLAT-plate formula. On this quarter cylinder that is",
        "#     CURVED_WRITER_PRESSURE  = {0!r}".format(tuple(writer)),
        "# against the exact integral(p n dA)",
        "#     CURVED_APPLIED_PRESSURE = {0!r}".format(tuple(exact)),
        "# so the writer's own _guard_equilibrium fails by about 1000x its tolerance and, running",
        "# before _odb_displacements, leaves no result at all. The residual that formula WOULD have",
        "# left is measured below and recorded in the sidecar, so the defect is checked on every run.",
        "CURVED_JOB = {0!r}".format(JOB_NAME),
        "CURVED_DISPLACEMENTS = {0!r}".format(DISPLACEMENTS_NAME),
        "CURVED_SHELL_ELEMENT = {0!r}".format(DEFAULT_SHELL_ELEMENT),
        "CURVED_SOLVER_SUCCESS = 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'",
        "CURVED_COMPONENTS = ('U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3')",
        "CURVED_APPLIED_PRESSURE = {0!r}".format(tuple(exact)),
        "CURVED_WRITER_PRESSURE = {0!r}".format(tuple(writer)),
        "CURVED_EQ_REL_TOL = {0!r}".format(EQUILIBRIUM_REL_TOL),
        "",
    ]
    return "\n".join(constants) + _DRIVER_BODY


#: The appended driver's code, verbatim. Python 2/3 neutral: it runs in Abaqus' own kernel
#: interpreter, where ``print`` lands in ``abaqus.rpy`` prefixed ``#: `` rather than on stdout
#: and where ``sys.exit`` does not reach the process status -- which is why every failure goes
#: through the writer's own ``_fail``, and why nothing here reads an exit code.
_DRIVER_BODY = '''

def _curved_sum_nodal(frame, name):
    """``(resultant, largest nodal magnitude)`` of a three-component nodal field, or None.

    The peak comes back with the sum because it is the scale the cancellation in that sum has to
    be judged against: ODB field data is single precision.
    """
    if name not in frame.fieldOutputs.keys():
        return None
    total = [0.0, 0.0, 0.0]
    peak = 0.0
    for value in frame.fieldOutputs[name].values:
        data = value.data
        magnitude = 0.0
        for axis in range(3):
            component = float(data[axis])
            total[axis] = total[axis] + component
            magnitude = magnitude + component * component
        magnitude = magnitude ** 0.5
        if magnitude > peak:
            peak = magnitude
    return total, peak


def _curved_face_area(model):
    """CAE's own getSize() for the panel's face, summed. Recorded, not checked here: the
    writer's _guard_plate_faces already compared it against adapy's and failed the build if it
    disagreed. This puts the number in the sidecar so a reader sees it without reading the log."""
    total = 0.0
    for part_name in sorted(model.parts.keys()):
        part = model.parts[part_name]
        for face in part.faces:
            total = total + float(face.getSize(printResults=False))
    return total


def _curved_displacements(odb):
    """Every node position and its six components per step, joined out of TWO ODB fields.

    Abaqus does not store six displacement components in one place: 'U' carries ('U1','U2','U3')
    and the rotations are a separate 'UR'. The schema and the component order are the writer's
    own, so verification.genie_vs_abaqus.abaqus_runner.read_displacements_sidecar needs no branch
    for this case.
    """
    payload = {
        'schema': 'ada.cae_displacements/1',
        'job': CURVED_JOB,
        'odb': CURVED_JOB + '.odb',
        'model': MODEL_NAME,
        'element_type': CURVED_SHELL_ELEMENT,
        'components': list(CURVED_COMPONENTS),
        'solver_version': str(odb.jobData.version),
        'instances': [],
    }
    for instance_name in sorted(odb.rootAssembly.instances.keys()):
        instance = odb.rootAssembly.instances[instance_name]
        nodes = []
        for node in instance.nodes:
            point = node.coordinates
            nodes.append([int(node.label), float(point[0]), float(point[1]), float(point[2])])
        nodes.sort()
        steps = []
        for step_name in sorted(odb.steps.keys()):
            step = odb.steps[step_name]
            frame = step.frames[-1]
            rows = {}
            for field_name, offset in (('U', 0), ('UR', 3)):
                if field_name not in frame.fieldOutputs.keys():
                    _fail('step ' + repr(step_name) + ' of the odb carries no ' + repr(field_name)
                          + ' field, so half of every displacement would be reported as zero.')
                subset = frame.fieldOutputs[field_name].getSubset(region=instance)
                for value in subset.values:
                    label = int(value.nodeLabel)
                    row = rows.get(label)
                    if row is None:
                        row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        rows[label] = row
                    for axis in range(3):
                        row[offset + axis] = float(value.data[axis])
            table = []
            for label in sorted(rows.keys()):
                table.append([label] + rows[label])
            steps.append({'name': step_name, 'frame': len(step.frames) - 1,
                          'time': float(frame.frameValue), 'displacements': table})
        payload['instances'].append({'name': instance_name, 'nodes': nodes, 'steps': steps})
    return payload


def _curved_solve():
    here = os.path.dirname(_result_path())
    model = mdb.models[MODEL_NAME]
    _RESULT['curved_face_area'] = _curved_face_area(model)
    job = mdb.Job(name=CURVED_JOB, model=MODEL_NAME)
    _RESULT['job'] = CURVED_JOB
    _write_result()
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    # NOT job.status: measured on Abaqus 2025 it reads None after waitForCompletion under
    # `abaqus cae noGUI=`, so a check against COMPLETED would fail every clean run.
    sta_path = os.path.join(here, CURVED_JOB + '.sta')
    completed = False
    if os.path.isfile(sta_path):
        handle = open(sta_path, 'r')
        try:
            completed = CURVED_SOLVER_SUCCESS in handle.read()
        finally:
            handle.close()
    if not completed:
        _fail('the job ' + repr(CURVED_JOB) + ' never wrote ' + repr(CURVED_SOLVER_SUCCESS)
              + ' into its .sta, so it did not finish. The model was built and every one of the '
              'writer\\'s own guards passed, so the solve is what failed: read ' + CURVED_JOB
              + '.msg and ' + CURVED_JOB + '.dat.')
    odb_path = os.path.join(here, CURVED_JOB + '.odb')
    if not os.path.isfile(odb_path):
        _fail('the job ' + repr(CURVED_JOB) + ' reported success and left no .odb at ' + repr(odb_path))
    odb = session.openOdb(name=odb_path)
    try:
        last = sorted(odb.steps.keys())[-1]
        frame = odb.steps[last].frames[-1]
        summed = _curved_sum_nodal(frame, 'RF')
        if summed is None:
            _fail('the odb carries no RF field, so whether the pressure arrived cannot be checked. '
                  'A pressure appears in no nodal field of its own, so the reactions are the only '
                  'trace of it there is.')
        reaction, peak = summed
        scale = 0.0
        for axis in range(3):
            scale = scale + CURVED_APPLIED_PRESSURE[axis] * CURVED_APPLIED_PRESSURE[axis]
        scale = scale ** 0.5
        if peak > scale:
            scale = peak
        tolerance = scale * CURVED_EQ_REL_TOL
        residual = 0.0
        writer_residual = 0.0
        for axis in range(3):
            term = abs(reaction[axis] + CURVED_APPLIED_PRESSURE[axis])
            if term > residual:
                residual = term
            term = abs(reaction[axis] + CURVED_WRITER_PRESSURE[axis])
            if term > writer_residual:
                writer_residual = term
        _RESULT['curved_equilibrium'] = {
            'applied_pressure_exact': list(CURVED_APPLIED_PRESSURE),
            'applied_pressure_writer_formula': list(CURVED_WRITER_PRESSURE),
            'reaction_force_sum': reaction,
            'largest_nodal_force': peak,
            'tolerance': tolerance,
            'residual': residual,
            'residual_with_writer_formula': writer_residual,
        }
        _write_result()
        if residual > tolerance:
            _fail('the solved panel is not in equilibrium: the exact pressure resultant for a '
                  'quarter cylinder is ' + repr(tuple(CURVED_APPLIED_PRESSURE)) + ' and the odb '
                  'reaction total is ' + repr(tuple(reaction)) + ', residual ' + repr(residual)
                  + ' against a tolerance of ' + repr(tolerance) + '. Either the Surface holds the '
                  'wrong faces, the magnitude has the wrong sign, or CAE built the panel at a '
                  'different size.')
        payload = _curved_displacements(odb)
    finally:
        odb.close()
    handle = open(os.path.join(here, CURVED_DISPLACEMENTS), 'w')
    try:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        handle.close()
    _RESULT['displacements'] = CURVED_DISPLACEMENTS
    _write_result()
    print('ADAPY-CAE CURVED SOLVE OK: residual ' + repr(residual) + ' against '
          + repr(tuple(CURVED_APPLIED_PRESSURE)) + '; the writer\\'s own flat-plate formula would '
          'have left ' + repr(writer_residual))
    sys.stdout.flush()


try:
    # Only if the writer's own guards all passed: a half-built model must not be solved, and a
    # driver that solved one anyway would produce a table of numbers for a different structure.
    if _RESULT.get('ok'):
        _curved_solve()
    else:
        print('ADAPY-CAE CURVED SOLVE SKIPPED: the writer\\'s own build did not pass its guards')
        sys.stdout.flush()
except Exception:
    _fail('unhandled exception during the appended curved solve:\\n' + traceback.format_exc())
'''


def emit_and_run(
    work_dir: str | pathlib.Path,
    *,
    count: int,
    shell_element_type: str = DEFAULT_SHELL_ELEMENT,
    pressure_magnitude: float | None = None,
) -> pathlib.Path:
    """Write the CAE script for one grid, append the driver, run it, return the run directory.

    ``submit=False``: the writer builds and verifies the whole model and stops there; the driver
    solves it. See the module docstring for the defect that forces the split.

    ``pressure_magnitude`` exists for one purpose -- measuring the sign on a curved face
    (:data:`PRESSURE_SIGN_MEASUREMENT`) by running the panel with the magnitude reversed. It is
    not a knob for the production sequence, and passing a positive value there would solve an
    *externally* pressurised panel whose equilibrium check still closes.
    """
    try:
        from tests.core.cadit.cae.abaqus_runner import abaqus_command, run_cae_script
    except ImportError as exc:  # pragma: no cover - depends on how this is invoked
        raise AbaqusNotInstalled(
            "could not import tests.core.cadit.cae.abaqus_runner, which owns the Abaqus launcher "
            "search and the sanitised child environment this needs: {0}. Run this from the "
            "repository root.".format(exc)
        ) from exc

    if abaqus_command() is None:
        raise AbaqusNotInstalled(
            "no Abaqus install found. tests.core.cadit.cae.abaqus_runner looks for "
            "C:/SIMULIA/Commands/abqXXXX.bat and honours ADA_ABAQUS_CMD."
        )

    work_dir = pathlib.Path(work_dir)
    run_dir = work_dir / run_dir_name(count, pressure_magnitude=pressure_magnitude)
    run_dir.mkdir(parents=True, exist_ok=True)

    assembly = curved_model.build_panel(count, route="abaqus")
    if pressure_magnitude is not None:
        _override_pressure(assembly, pressure_magnitude)
    script = run_dir / "{0}.py".format(SCRIPT_STEM)
    # The assembly, not the part: the supports are on the part's FEM while the step carrying the
    # pressure is on the assembly's, so writing the part alone would refuse for want of the step.
    assembly.to_abaqus_cae_script(
        script,
        mesh_size=curved_model.PANEL_ARC / count,
        shell_element_type=shell_element_type,
        job_name=JOB_NAME,
        submit=WRITER_SUBMITS,
    )
    with script.open("a", encoding="utf-8") as handle:
        handle.write(driver_source(count))

    run = run_cae_script(script, run_dir, timeout=RUN_TIMEOUT)
    if run.licence_denied:
        raise AbaqusNotInstalled("Abaqus is installed but no CAE licence was available:\n" + run.stdout)
    if run.failed:
        raise AbaqusFailed(
            "the emitted CAE script did not run cleanly for the {0} x {0} panel, so there is no "
            "result to compare. Signals: {1}\n{2}".format(count, run.failure_signals, run.describe())
        )
    if not (run_dir / DISPLACEMENTS_NAME).is_file():
        raise AbaqusFailed(
            "the build reported success and left no displacement sidecar at {0} for the {1} x {1} panel. "
            "The appended driver writes one only when the job it submitted finished and the equilibrium "
            "check closed, so read {2} for what it did instead.".format(
                run_dir / DISPLACEMENTS_NAME, count, run_dir / BUILD_RESULT_NAME
            )
        )
    return run_dir


def run_dir_name(count: int, *, pressure_magnitude: float | None = None) -> str:
    """Where one run's artefacts live, encoding the grid and any sign override."""
    name = "abaqus_n{0}".format(count)
    if pressure_magnitude is not None and pressure_magnitude != curved_model.SIGNED_PRESSURE_MAGNITUDE:
        name += "_p{0}".format("pos" if pressure_magnitude > 0 else "neg")
    return name


def _override_pressure(assembly, magnitude: float) -> None:
    """Replace the pressure's magnitude in place, for the sign measurement only."""
    found = 0
    for step in assembly.fem.steps:
        for load in step.loads:
            if str(load.type).lower().endswith("pressure"):
                load._magnitude = magnitude  # noqa: SLF001 - Load has no magnitude setter
                found += 1
    if found != 1:
        raise AbaqusFailed(
            f"expected exactly one pressure load to override and found {found}. The sign measurement "
            f"only means anything if it is the panel's own pressure that was reversed."
        )


def read_checks(run_dir: str | pathlib.Path) -> CurvedRunChecks:
    """Read this run's provenance out of :data:`BUILD_RESULT_NAME`.

    Refused rather than defaulted at every step: the reaction total is the check that the whole
    pressure arrived, the region kinds are the check that the two symmetry supports ran along
    their generators rather than sitting on the four corners a vertex region would have caught,
    and the ``curved_equilibrium`` block is the check that the driver ran at all rather than the
    script having been emitted without it.
    """
    run_dir = pathlib.Path(run_dir)
    path = run_dir / BUILD_RESULT_NAME
    if not path.is_file():
        raise AbaqusFailed(
            "no {0} in {1}. The emitted script writes it whenever CAE ran it at all, so its absence "
            "means the script never started.".format(BUILD_RESULT_NAME, run_dir)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = str(payload.get("schema", ""))
    if not schema.startswith("ada.cae_build_result/"):
        raise AbaqusFailed("{0} is not a CAE build-result sidecar (schema {1!r})".format(path, schema))
    if not payload.get("ok"):
        raise AbaqusFailed(
            "the emitted script's own guards failed for {0}, so whatever it solved is not the model "
            "adapy described: {1}".format(run_dir, payload.get("errors") or "(no errors recorded)")
        )

    equilibrium = payload.get("curved_equilibrium") or {}
    missing = sorted(
        {"reaction_force_sum", "applied_pressure_exact", "residual_with_writer_formula"} - set(equilibrium)
    )
    if missing:
        raise AbaqusFailed(
            "{0} carries no {1} in its curved_equilibrium block (it holds {2}), which means the "
            "appended driver did not run: the writer's own script emits no solve at all when it is "
            "asked not to submit, so without the driver there is no equilibrium check and no "
            "result.".format(path, missing, sorted(equilibrium) or "nothing")
        )
    total = [float(v) for v in equilibrium["reaction_force_sum"]]
    exact = [float(v) for v in equilibrium["applied_pressure_exact"]]
    written = [float(v) for v in equilibrium.get("applied_pressure_writer_formula", (0.0, 0.0, 0.0))]

    mesh = payload.get("mesh") or {}
    if len(mesh) != 1:
        raise AbaqusFailed(
            "{0} reports {1} meshed part(s) ({2}) and this model is one part: a node count summed over "
            "several would not be the panel's.".format(path, len(mesh), sorted(mesh))
        )
    meshed = mesh[sorted(mesh)[0]]

    kinds = {str(k): str(v) for k, v in sorted((payload.get("analysis") or {}).get("region_kinds", {}).items())}
    edges = {str(k): (int(v[0]), float(v[1])) for k, v in sorted((payload.get("region_edges") or {}).items())}
    expected_kinds = dict(EXPECTED_REGION_KINDS)
    wrong = [
        "{0!r} is {1} and should be {2}".format(name, kinds.get(name, "absent"), expected_kinds[name])
        for name in sorted(expected_kinds)
        if kinds.get(name) != expected_kinds[name]
    ]
    if wrong:
        raise AbaqusFailed(
            "the supports did not all reach CAE as the regions this model is built around -- {0}. The "
            "two symmetry supports have to be whole straight generator EDGES (a support on the two end "
            "vertices of a generator holds two nodes instead of all of them) and the axial reference "
            "has to be the one corner VERTEX (a whole edge there would restrain the axial Poisson "
            "contraction and make the second closed form the wrong one). The regions read back as "
            "{1}.".format(", ".join(wrong), kinds or "(none)")
        )

    return CurvedRunChecks(
        reaction_total=(total[0], total[1], total[2]),
        node_count=int(meshed["nodes"]),
        element_counts={str(k): int(v) for k, v in sorted(meshed["elements_by_type"].items())},
        region_kinds=kinds,
        region_edges=edges,
        face_area=float(payload.get("curved_face_area", 0.0)),
        applied_pressure_exact=(exact[0], exact[1], exact[2]),
        applied_pressure_writer_formula=(written[0], written[1], written[2]),
        residual_with_writer_formula=float(equilibrium["residual_with_writer_formula"]),
        solved=bool(payload.get("displacements")),
    )


#: What each support must resolve to in the kernel. Written out rather than derived from
#: :data:`curved_model.EDGE_SUPPORTS`, because a check that moves with the data it checks cannot
#: catch the data changing: the whole design of this case is that the two symmetry supports are
#: on *straight* edges and the axial reference is on a *vertex*.
EXPECTED_REGION_KINDS = (("SYM_T0", "edge"), ("SYM_T90", "edge"), ("AXIAL_REF", "vertex"))


def abaqus_displacements(
    run_dir: str | pathlib.Path,
    *,
    count: int,
    step: int | None = None,
) -> PlateSolve:
    """Read one run directory's sidecars and sample the probes.

    The Abaqus counterpart of :func:`curved_sestra_runner.sestra_displacements`, and symmetric
    with it: both end in :func:`displacements.sample_fea_result`, so probe correspondence is
    solved once for both solvers. It is that shared sampler that refuses -- rather than
    settling for the nearest node -- if CAE's own mesh has no node at a probe, which is the
    check that the two meshers put nodes at the same places on the boundary edges.
    """
    run_dir = pathlib.Path(run_dir)
    checks = read_checks(run_dir)
    result = read_displacements_sidecar(run_dir / DISPLACEMENTS_NAME)
    version = result.solver_version or "unknown"
    if result.element_type:
        version = "{0} {1}".format(version, result.element_type)
    table = sample_fea_result(
        result,
        curved_model.PROBE_POINTS,
        solver="abaqus",
        solver_version=version,
        model_name=model_name(),
        load_case=curved_model.LOAD_CASE,
        step=step,
        match_tol=curved_model.PROBE_MATCH_TOL,
    )
    return PlateSolve(
        table=table,
        mesh_size=curved_model.PANEL_ARC / count,
        stiffened=False,
        reaction_total=checks.reaction_total,
        node_count=checks.node_count,
        element_counts=checks.element_counts,
        source=str(run_dir / DISPLACEMENTS_NAME),
    )


def run_and_sample(work_dir: str | pathlib.Path, *, count: int) -> PlateSolve:
    """:func:`emit_and_run` then :func:`abaqus_displacements`. One grid."""
    return abaqus_displacements(emit_and_run(work_dir, count=count), count=count)


def run_sequence(
    work_dir: str | pathlib.Path,
    *,
    counts: tuple[int, ...] = curved_model.MESH_COUNTS,
) -> list[PlateSolve]:
    """Every grid, coarse to fine. Strictly sequential: four CAE tokens exist."""
    return [run_and_sample(work_dir, count=count) for count in counts]


def read_sequence(
    work_dir: str | pathlib.Path,
    *,
    counts: tuple[int, ...] = curved_model.MESH_COUNTS,
) -> list[PlateSolve]:
    """Re-read the sidecars already in ``work_dir`` rather than spending three CAE tokens."""
    sequence = []
    for count in counts:
        run_dir = pathlib.Path(work_dir) / run_dir_name(count)
        if not (run_dir / DISPLACEMENTS_NAME).is_file():
            raise OSError(
                f"there is no Abaqus sidecar at {run_dir / DISPLACEMENTS_NAME}. Run the case once " f"without --reuse."
            )
        sequence.append(abaqus_displacements(run_dir, count=count))
    return sequence


def measure_pressure_sign(
    work_dir: str | pathlib.Path, *, count: int = curved_model.MESH_COUNTS[0]
) -> dict[str, float]:
    """Solve the panel with the pressure magnitude **reversed**, and report which way it pushed.

    One extra CAE solve, on the coarsest grid, for the one question a curved face asks that a flat
    plate does not: which way does ``side1Faces`` point once the face's own normal turns through
    90 degrees. Reported rather than asserted in the production sequence, because the answer is a
    property of Abaqus and of this writer and not of the model.

    The **reaction total's sign** is the measurement, and it is read out of the build-result
    sidecar rather than out of a displacement table -- for a reason worth stating, because it is
    what makes this function work at all. The appended driver checks equilibrium against the
    *model's own* (outward) pressure resultant, so a run with the magnitude reversed is **meant**
    to fail that check: it comes back with a residual of ``2 p r L`` -- the whole load twice over --
    and writes no displacement sidecar. That failure is the answer rather than an obstacle, so this
    reads the reaction the driver recorded on its way down. Measured on this panel at ``n = 8``:

        magnitude -1e5 (the model's)  reaction total  (-628318.555, -628318.559, -0.005)
        magnitude +1e5 (reversed)     reaction total  (+628318.555, +628318.559, +0.005)
                                      residual against the outward resultant 1256637.089 = 2 p r L

    -- the reactions flip with the magnitude, so a **positive** ``Pressure`` magnitude on this face
    pulls the panel inwards, and an internal pressure needs the negative one
    (:data:`curved_model.SIGNED_PRESSURE_MAGNITUDE`).
    """
    reversed_magnitude = -curved_model.SIGNED_PRESSURE_MAGNITUDE
    try:
        emit_and_run(work_dir, count=count, pressure_magnitude=reversed_magnitude)
    except AbaqusFailed:
        # Expected: the driver's equilibrium term is the model's own outward resultant, so a
        # reversed pressure is out of balance by 2 p r L. The reaction it recorded is the answer.
        pass
    run_dir = pathlib.Path(work_dir) / run_dir_name(count, pressure_magnitude=reversed_magnitude)
    path = run_dir / BUILD_RESULT_NAME
    if not path.is_file():
        raise AbaqusFailed(
            "no {0} in {1}, so the reversed-pressure run left nothing to read the sign off. The "
            "measurement is the reaction total's sign, which the appended driver records before it "
            "fails the equilibrium check.".format(BUILD_RESULT_NAME, run_dir)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    equilibrium = payload.get("curved_equilibrium") or {}
    if "reaction_force_sum" not in equilibrium:
        raise AbaqusFailed(
            "{0} carries no reaction total (it holds {1}), which means the solve never got far "
            "enough to report one -- so whether the pressure pushed in or out is unanswered.".format(
                path, sorted(equilibrium) or "nothing"
            )
        )
    reaction = [float(v) for v in equilibrium["reaction_force_sum"]]
    outward = curved_model.expected_load_total()
    return {
        "magnitude": reversed_magnitude,
        "reaction_x": reaction[0],
        "reaction_y": reaction[1],
        "reaction_z": reaction[2],
        # +1 when the reaction opposes an outward load (an internal pressure), -1 when it opposes
        # an inward one. The sign as one number, so a report cannot misread three.
        "load_direction": -1.0 if reaction[0] * outward[0] > 0.0 else 1.0,
        "residual_against_outward": float(equilibrium.get("residual", 0.0)),
    }


def radial_displacement(table, probe_name: str) -> float:
    """The radial displacement at one probe, recovered through :data:`curved_model.RADIAL_BASIS`.

    Positive is outward. A single function, used by every check and every report, because the
    recovery differs by probe -- ``u1`` on one generator, ``u2`` on the other, and
    ``(u1 + u2) / sqrt(2)`` at ``theta = 45`` -- and a caller that read ``u1`` everywhere would
    report the mid-hoop probe 29% low and call it a translation error.
    """
    basis = curved_model.RADIAL_BASIS.get(probe_name)
    if basis is None:
        raise KeyError(
            f"{probe_name!r} has no radial basis, so its radial displacement is not defined; "
            f"curved_model.RADIAL_BASIS covers {sorted(curved_model.RADIAL_BASIS)}"
        )
    return math.fsum(table.component(probe_name, component) * direction for component, direction in basis)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curved_abaqus_runner",
        description="Solve the curved panel in Abaqus at every grid density and write the tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--work-dir", default="temp/curved_verification", help="where to emit and run")
    parser.add_argument(
        "--count", type=int, default=None, help="one grid only (default: every one of curved_model.MESH_COUNTS)"
    )
    parser.add_argument(
        "--measure-pressure-sign",
        action="store_true",
        help="also solve the coarsest grid with the pressure reversed, and report which way it moved",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Solve the Abaqus side and print its tables. ``0`` written, ``2`` could not be run."""
    args = build_parser().parse_args(argv)
    counts = (args.count,) if args.count else curved_model.MESH_COUNTS
    try:
        solves = [run_and_sample(args.work_dir, count=count) for count in counts]
    except (AbaqusNotInstalled, AbaqusFailed, OSError, ValueError) as exc:
        print("ERROR: the Abaqus side could not be run: {0}".format(exc), file=sys.stderr)
        return 2

    for count, solve in zip(counts, solves):
        print(
            "n={0:<4} nodes {1:>5}  {2}  reaction {3}  radial({4}) {5:.12e}".format(
                count,
                solve.node_count,
                solve.element_counts,
                tuple(round(v, 3) for v in solve.reaction_total),
                curved_model.RADIAL_PROBE,
                radial_displacement(solve.table, curved_model.RADIAL_PROBE),
            )
        )
    if args.measure_pressure_sign:
        print("pressure sign, reversed magnitude: {0}".format(measure_pressure_sign(args.work_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
