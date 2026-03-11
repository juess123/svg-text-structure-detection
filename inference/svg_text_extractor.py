import xml.etree.ElementTree as ET


def extract_svg_paths(svg_file):

    tree = ET.parse(svg_file)
    root = tree.getroot()

    paths = []

    for elem in root.iter():

        if elem.tag.endswith("path"):

            d = elem.attrib.get("d")

            if d:
                paths.append(d)

    return paths