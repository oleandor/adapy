"""The Sestra half of the curved case: the panel -> a Sesam deck -> ``.SIN`` -> a table.

Same locator, same success check and same sampler as :mod:`sestra_runner` and
:mod:`plate_sestra_runner`; this module adds only what a **curved** panel needed, which turned
out to be nothing on the Sesam side at all. That is worth stating rather than passing over: the
Sesam interface file carries a *mesh*, not geometry, so a faceted cylinder reaches Sestra as
2048 ordinary ``FQUS`` quadrilaterals and there is no curved-surface record for anything to be
lost from. Everything difficult about this case on this route is upstream of the writer, in
adapy's inability to mesh a ``PlateCurved`` at all (gap 1 of :mod:`curved_model`).

What reaches Sestra, measured on this model's own deck
=====================================================

**The facets arrive as ``FQUS``.** ``write_elements.eltype_2_sesam`` maps a quadrilateral
shell to Sesam element type ``24``; 64 / 256 / 1024 of them at the three counts, against the
same counts on the Abaqus side. Sestra reports ``Execution completed successfully`` at every
density and its message log carries no singularity -- which is the check that
:data:`curved_model.AXIAL_REFERENCE_SUPPORT` did its job, because without it the panel is free
to slide along ``z``.

**The load is the exact consistent nodal vector, and the substitution is exact.** adapy's
Sesam writer still emits no distributed load (``write_loads.load_str`` reports
``[OMITTED] a "pressure" load is not written by the Sesam writer``), so this side is given
:func:`curved_model.consistent_nodal_loads` instead. On a **rectangular** facet
``integral(N_i) dA = A / 4`` exactly, and every facet of :func:`curved_model.shell_grid` is a
rectangle to 1e-09 m -- so ``q A / 4`` along that facet's own normal at each of its four nodes
*is* the pressure's load vector, which is what Abaqus' ``*Dsload`` assembles element by
element. The evidence that nothing was lost in the swap is measured on both sides:

* the nodal vectors sum to ``(628318.530717959, 628318.530717959, 0.0)`` N at ``n = 8`` and
  ``(628318.5307179524, ...)`` at ``n = 32``, against ``(p r L, p r L, 0)`` =
  ``(628318.5307179586, ...)`` -- 1e-14 relative, i.e. float noise;
* Sestra's own reaction total comes back ``(-628318.508, -628318.508, -3.9e-10)`` at ``n = 8``
  and ``(-628318.529, -628318.529, 6.4e-09)`` at ``n = 32`` -- :func:`reaction_total`, and
  2.7e-09 relative at the finest;
* and the Abaqus side, given the *pressure* rather than these forces, reacts the same
  ``p r L`` per axis (:mod:`curved_abaqus_runner`).

``BEUSLO`` exists on ``origin/feat/sesam-pressure-loads`` (#404) and was deliberately **not**
merged. Two reasons, both stated so the choice can be argued with: that branch is cut from
0.91.1 and has diverged from this stack (``git merge-base --is-ancestor`` says so), and the
Sesam side does not need it -- the vector above is not an approximation of the pressure, it is
the pressure. What the branch does carry is a measurement worth quoting, because it settles
half of this case's sign question without a run: its own probe records that a BEUSLO intensity
pushes along the element's **negative** normal whichever ``SIDE`` is written, and that
``*Dsload P, 1000.`` on an ``S4R`` whose normal is ``+z`` puts ``U3`` negative. Both routes
therefore push *against* the element normal for a positive magnitude, which is why
:data:`curved_model.SIGNED_PRESSURE_MAGNITUDE` is negative and why these nodal forces point
outward. The solved numbers are in
:data:`curved_abaqus_runner.PRESSURE_SIGN_MEASUREMENT`.

**The single-precision floor.** A ``.SIN`` stores single-precision floats, so the 1.9e-04 m
radial displacement carries about seven significant digits -- an absolute resolution near
1e-11 m, 6e-08 relative. That is 4800x *below* the 2.9e-04 relative residual the finest mesh
still has, so it does not limit the convergence study; it does set a floor on the Richardson
extrapolant of order 1e-07 relative, and the measured 3.5e-06 residual is thirty times above
it, so that residual is the next-order term and not the storage.
"""

from __future__ import annotations

import collections
import pathlib
import shutil

import numpy as np

from ada.fem.formats.sesam.results.read_sin import read_sin_file
from ada.fem.formats.sesam.sesam_exe_locator import get_sestra_version

from . import curved_model
from .displacements import DisplacementTable, sample_fea_result
from .plate_sestra_runner import REACTION_FIELD, PlateSolve
from .sestra_runner import SUCCESS_MARKER, SestraFailed, sestra_exe

#: The ``model`` field every table of this case carries. Distinct from the plate case's on
#: purpose: :func:`compare.compare` would otherwise happily compare a strip against a panel.
MODEL_NAME = "curved_panel"


def model_name() -> str:
    """``"curved_panel"``. A function for symmetry with :func:`plate_sestra_runner.model_name`."""
    return MODEL_NAME


def default_case_name(count: int) -> str:
    """A deck name that encodes the grid, so three runs can share a work dir."""
    return f"panel_n{count}"


def run_sestra(
    work_dir: str | pathlib.Path,
    *,
    count: int,
    case_name: str | None = None,
    clean: bool = True,
) -> pathlib.Path:
    """Write the Sesam deck for one grid, solve it, and return the ``.SIN``.

    Verified rather than trusted, for the reason :func:`sestra_runner.run_sestra` gives: adapy's
    ``LocalExecute`` shells out through a batch file and does not surface a non-zero solver
    exit, so the ``.SIN`` and the ``SESTRA.MLG`` success marker are both checked.
    """
    exe = sestra_exe()
    version = get_sestra_version(exe)

    work_dir = pathlib.Path(work_dir)
    case_name = case_name or default_case_name(count)
    deck_dir = work_dir / case_name
    if clean and deck_dir.exists():
        shutil.rmtree(deck_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # "sestra": the nodal form, because the Sesam writer has no distributed-load record --
    # see the module docstring. build_panel's own guards check the grid, the probes and the
    # supports before a solve is spent on it.
    assembly = curved_model.build_panel(count, route="sestra")
    assembly.to_fem(case_name, "sesam", scratch_dir=work_dir, overwrite=True, execute=True)

    sin_path = deck_dir / f"{case_name}R1.SIN"
    mlg_path = deck_dir / "SESTRA.MLG"
    if not sin_path.is_file():
        raise SestraFailed(
            f"Sestra ({version}, {exe}) produced no result file at {sin_path} for the {count} x {count} "
            f"panel. Message log: {mlg_path if mlg_path.is_file() else '(none written)'}"
        )
    if mlg_path.is_file():
        mlg = mlg_path.read_text(encoding="utf-8", errors="replace")
        if SUCCESS_MARKER not in mlg:
            raise SestraFailed(
                f"Sestra ({version}) wrote {sin_path.name} but its message log does not contain "
                f"'{SUCCESS_MARKER}'. A partial .SIN from an aborted run must not be read as a result. "
                f"Log tail:\n{mlg[-2000:]}"
            )
    return sin_path


def sestra_displacements(
    sin_path: str | pathlib.Path,
    *,
    count: int,
    step: int | None = None,
) -> PlateSolve:
    """Read a ``.SIN``, sample it at :data:`curved_model.PROBE_POINTS`, read its reactions.

    Returns a :class:`plate_sestra_runner.PlateSolve` -- the same carrier the plate case uses,
    with ``mesh_size`` holding this grid's seed and ``stiffened`` always ``False``. Reused
    rather than re-declared so :func:`curved_compare.extrapolate` and
    :func:`plate_compare._assert_one_sequence` need no second shape to understand; the grid
    count travels in the label through the seed, which is one-to-one with it
    (:data:`curved_model.MESH_SIZES`).
    """
    sin_path = pathlib.Path(sin_path)
    result = read_sin_file(sin_path, step=step)
    table = sample_fea_result(
        result,
        curved_model.PROBE_POINTS,
        solver="sestra",
        solver_version=get_sestra_version(sestra_exe()),
        model_name=model_name(),
        load_case=curved_model.LOAD_CASE,
        step=step,
        match_tol=curved_model.PROBE_MATCH_TOL,
    )
    counts: dict[str, int] = collections.Counter()
    for block in getattr(result.mesh, "elements", ()):
        identifiers = getattr(block, "identifiers", None)
        if identifiers is None:
            continue
        # ``ElementBlock.elem_info.type``, not ``block.type``: a block has no ``type`` of its
        # own, and reading one through getattr's default reported every element as "unknown".
        info = getattr(block, "elem_info", None)
        counts[str(getattr(info, "type", "unknown"))] += int(np.asarray(identifiers).size)
    return PlateSolve(
        table=table,
        mesh_size=curved_model.PANEL_ARC / count,
        stiffened=False,
        reaction_total=reaction_total(result),
        node_count=int(np.asarray(result.mesh.nodes.identifiers).size),
        element_counts=dict(sorted(counts.items())),
        source=str(sin_path),
    )


def reaction_total(result) -> tuple[float, float, float]:
    """The summed support reaction from a Sestra result, ``(fx, fy, fz)`` newtons.

    Checked against ``(-p r L, -p r L, 0)`` by
    :func:`curved_compare.assert_reaction_total`. It is the one scalar that says the *whole*
    load arrived, and the only one that comes from the solver's own bookkeeping rather than
    from adapy's -- comparing adapy's applied load against adapy's own arithmetic would prove
    nothing. It matters more here than it did for the flat strip because the two sides are
    given the load in different forms and, on a curved face, in different *directions* per
    facet: if those two were not the same load this is where it would show.
    """
    grouped = result.get_results_grouped_by_field_value()
    if REACTION_FIELD not in grouped:
        raise SestraFailed(
            f"the result carries no {REACTION_FIELD!r} field, so the load Sestra actually reacted cannot "
            f"be read; it holds {sorted(grouped)}. Without it a deck that lost part of its load would "
            f"still produce a full displacement table."
        )
    values = np.asarray(grouped[REACTION_FIELD][0].values, dtype=float)
    # Column 0 is the node label; the next three are X/Y/Z-FORCE.
    total = values[:, 1:4].sum(axis=0)
    return (float(total[0]), float(total[1]), float(total[2]))


def node_reaction(result, point, *, tol: float = 1.0e-06) -> tuple[float, float, float]:
    """The reaction at the one node nearest ``point``, ``(fx, fy, fz)`` newtons.

    Exists for one check only: :data:`curved_model.AXIAL_REFERENCE_SUPPORT` must carry no load
    (:func:`curved_compare.assert_reference_node_carries_nothing`). A node whose reaction had to
    be singled out for any other reason would be a sign the supports were doing something the
    closed form does not describe.
    """
    grouped = result.get_results_grouped_by_field_value()
    if REACTION_FIELD not in grouped:
        raise SestraFailed(f"the result carries no {REACTION_FIELD!r} field; it holds {sorted(grouped)}")
    values = np.asarray(grouped[REACTION_FIELD][0].values, dtype=float)
    ids = np.asarray(result.mesh.nodes.identifiers)
    coords = np.asarray(result.mesh.nodes.coords, dtype=float)
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    distances = np.abs(coords - np.asarray(point, dtype=float)).max(axis=1)
    index = int(distances.argmin())
    if distances[index] > tol:
        raise SestraFailed(
            f"no node within {tol} m of {tuple(point)}; the nearest is {distances[index]:.3e} away. The "
            f"axial reference node is a corner of the panel, so a mesh that has no node there is not the "
            f"mesh this model described."
        )
    label = int(ids[index])
    rows = values[values[:, 0] == label]
    if rows.shape[0] == 0:
        # A node with no reaction record is a node Sestra reports as carrying nothing, which is
        # exactly the expected answer here -- so it is reported as zero rather than refused.
        return (0.0, 0.0, 0.0)
    return (float(rows[0, 1]), float(rows[0, 2]), float(rows[0, 3]))


def run_and_sample(work_dir: str | pathlib.Path, *, count: int) -> PlateSolve:
    """:func:`run_sestra` then :func:`sestra_displacements`. One grid."""
    return sestra_displacements(run_sestra(work_dir, count=count), count=count)


def run_sequence(
    work_dir: str | pathlib.Path,
    *,
    counts: tuple[int, ...] = curved_model.MESH_COUNTS,
) -> list[PlateSolve]:
    """Every grid, coarse to fine -- the sequence the extrapolation consumes."""
    return [run_and_sample(work_dir, count=count) for count in counts]


def read_sequence(
    work_dir: str | pathlib.Path,
    *,
    counts: tuple[int, ...] = curved_model.MESH_COUNTS,
) -> list[PlateSolve]:
    """Re-read the decks already in ``work_dir`` rather than solving again."""
    sequence = []
    for count in counts:
        case_name = default_case_name(count)
        sin_path = pathlib.Path(work_dir) / case_name / f"{case_name}R1.SIN"
        if not sin_path.is_file():
            raise OSError(f"there is no Sestra result at {sin_path}. Run the case once without --reuse to produce it.")
        sequence.append(sestra_displacements(sin_path, count=count))
    return sequence


def read_result(work_dir: str | pathlib.Path, *, count: int):
    """The raw ``FEAResult`` for one grid -- what the per-node reaction check needs."""
    case_name = default_case_name(count)
    sin_path = pathlib.Path(work_dir) / case_name / f"{case_name}R1.SIN"
    if not sin_path.is_file():
        raise OSError(f"there is no Sestra result at {sin_path}")
    return read_sin_file(sin_path)


def table_source(table: DisplacementTable) -> str:
    """``table.source``, or its solver name when the table came from a reconstruction."""
    return table.source or f"{table.solver}:{table.model}"
