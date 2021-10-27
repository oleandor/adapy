# GMSH Node Ordering
# """
#         Hexahedron:             Hexahedron20:          Hexahedron27:
#
#        v
# 3----------2            3----13----2           3----13----2
# |\     ^   |\           |\         |\          |\         |\
# | \    |   | \          | 15       | 14        |15    24  | 14
# |  \   |   |  \         9  \       11 \        9  \ 20    11 \
# |   7------+---6        |   7----19+---6       |   7----19+---6
# |   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
# 0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
#  \  |    \  \  |         \  17      \  18       \ 17    25 \  18
#   \ |     \  \ |         10 |        12|        10 |  21    12|
#    \|      w  \|           \|         \|          \|         \|
#     4----------5            4----16----5           4----16----5
#         Tetrahedron:                          Tetrahedron10:
#
#                    v
#                  .
#                ,/
#               /
#            2                                     2
#          ,/|`\                                 ,/|`\
#        ,/  |  `\                             ,/  |  `\
#      ,/    '.   `\                         ,6    '.   `5
#    ,/       |     `\                     ,/       8     `\
#  ,/         |       `\                 ,/         |       `\
# 0-----------'.--------1 --> u         0--------4--'.--------1
#  `\.         |      ,/                 `\.         |      ,/
#     `\.      |    ,/                      `\.      |    ,9
#        `\.   '. ,/                           `7.   '. ,/
#           `\. |/                                `\. |/
#              `3                                    `3
#                 `\.
#                    ` w
# """
# tet10 is modified from GMSH to abaqus. See gmsh_to_meshio_ordering for complete overview

volume_edges = dict(
    hexahedron=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 7], [7, 3], [3, 0], [4, 5], [5, 6], [6, 7], [1, 5], [2, 6]],
    hexahedron20=[
        (0, 8),
        (8, 1),
        (1, 17),
        (17, 5),
        (5, 12),
        (12, 4),
        (4, 15),
        (15, 7),
        (0, 11),
        (11, 3),
        (1, 9),
        (9, 2),
        (2, 18),
        (18, 6),
        (6, 14),
        (14, 7),
        (3, 19),
        (19, 7),
        (2, 10),
        (10, 3),
        (5, 13),
        (13, 6),
        (0, 16),
        (16, 4),
    ],
    tetra=[(0, 1), (1, 3), (3, 0), (0, 2), (2, 3), (1, 2)],
    tetra10=[(0, 7), (7, 3), (3, 8), (8, 1), (1, 4), (4, 0), (0, 6), (6, 2), (2, 5), (5, 1), (3, 9), (9, 2)],
    C3D5=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
    wedge=[(0, 1), (1, 2), (2, 0), (0, 3), (3, 5), (5, 4), (4, 3), (3, 0), (5, 2), (4, 1)],
)

volume_faces = dict(
    hexahedron=[
        [0, 2, 3],
        [0, 1, 3],
        [0, 4, 7],
        [0, 7, 3],
        [0, 4, 5],
        [0, 5, 1],
        [2, 7, 6],
        [2, 3, 7],
        [5, 6, 7],
        [5, 7, 4],
        [5, 2, 1],
        [5, 6, 2],
    ],
    tetra=[(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)],
    tetra10=[(0, 1, 2), (0, 2, 3), (1, 2, 3), (0, 2, 3)],
    wedge=[(0, 1, 2), (0, 2, 5), (0, 5, 3), (3, 4, 5), (0, 1, 4), (0, 4, 3)],
)
