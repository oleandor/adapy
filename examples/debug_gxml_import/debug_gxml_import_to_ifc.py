import xml.etree.ElementTree as ET
import ada


def print_plate_xml(xml_path, target_names):
    root = ET.parse(xml_path).getroot()
    for tag in [".//flat_plate", ".//curved_shell"]:
        for el in root.iterfind(tag):
            name = el.attrib.get("name")
            if name in target_names:
                print(f"\n===== XML FOR {name} ({el.tag}) =====")
                print(ET.tostring(el, encoding="unicode"))


def convertXMLtoIFC(pathAndFileName):
    print("convertXMLtoIFC(): start")
    print(pathAndFileName)

    """
    print_plate_xml(
        pathAndFileName,
        {
            #"pon2yskin_elev5plate4",
            #"pon2yskin_elev5plate5",
            #"pon1xskin_elev1plate12",
            #"pon2y_rs_3_H19__pl0",
            #"pon1xskin_elev1plate11_1_2",
            "pon1xskin_elev1plate11",
        },
    )
    """

    a = ada.from_genie_xml(pathAndFileName)

    # matches = [pl for pl in a.get_all_physical_objects() if getattr(pl, "name", "") == "pon1xskin_elev1plate11"]
    # print("matching exact plates:", [pl.name for pl in matches])
    # print("count:", len(matches))

    a.to_ifc(pathAndFileName + ".ifc")

    print("convertXMLtoIFC(): end")


#pathAndFileName = r"c:\Aibelprogs\projects\Utror_concept\XML_to_IFC\Utror_6k_pl_v2005_FAL_noCage_ULST11.xml"
#pathAndFileName = r"c:\AibelProgs\Downloads\xmlexp 1.xml"
pathAndFileName = r"c:\Aibelprogs\tmp\miguel\beams_constant_offset.xml"

convertXMLtoIFC(pathAndFileName)
