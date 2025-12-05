# work in progress for creating sat info for beams in a Genie xml file
import pathlib

from ada import Beam, Assembly, Part

name = "bm_sat_test"
workspace_path = pathlib.Path(f"temp/{name}")

from_fem = False

if from_fem:
    my_fem_file = workspace_path / f"{name}_T1.FEM"
    a = Assembly()
    a.read_fem(fem_file=my_fem_file)
else:
    blist = []
    ypos = 0

    for sec in ["HP200x10", "OD1000x10"]:
        bm = Beam(sec, (0, 0, 0), (0, ypos, 1), sec)
        blist += [Part(sec + "_Z") / [bm]]
        ypos += 1

    bm = Beam(sec, (0, 0, 1), (0, 1, 1), sec)
    blist += [Part(sec + "_Z_2") / [bm]]

    a = Assembly() / blist


gxml = workspace_path / f"{name}_model.xml"


a.to_genie_xml(destination_xml=gxml, writer_postprocessor=None, embed_sat=True)