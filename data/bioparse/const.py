# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dictionary containing ideal internal coordinates and chi angle assignments
 for building amino acid 3D coordinates"""
from typing import Dict


AA_GEOMETRY: Dict[str, dict] = {
    "ALA": {
        "atoms": ["CB"],
        "chi_indices": [],
        "parents": [["N", "C", "CA"]],
        "types": {"C": "C", "CA": "CT1", "CB": "CT3", "N": "NH1", "O": "O"},
        "z-angles": [111.09],
        "z-dihedrals": [123.23],
        "z-lengths": [1.55],
    },
    "ARG": {
        "atoms": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "chi_indices": [1, 2, 3, 4],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "NE"],
            ["CD", "NE", "CZ"],
            ["NH1", "NE", "CZ"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD": "CT2",
            "CG": "CT2",
            "CZ": "C",
            "N": "NH1",
            "NE": "NC2",
            "NH1": "NC2",
            "NH2": "NC2",
            "O": "O",
        },
        "z-angles": [112.26, 115.95, 114.01, 107.09, 123.05, 118.06, 122.14],
        "z-dihedrals": [123.64, 180.0, 180.0, 180.0, 180.0, 180.0, 178.64],
        "z-lengths": [1.56, 1.55, 1.54, 1.5, 1.34, 1.33, 1.33],
    },
    "ASN": {
        "atoms": ["CB", "CG", "OD1", "ND2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["OD1", "CB", "CG"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CG": "CC",
            "N": "NH1",
            "ND2": "NH2",
            "O": "O",
            "OD1": "O",
        },
        "z-angles": [113.04, 114.3, 122.56, 116.15],
        "z-dihedrals": [121.18, 180.0, 180.0, -179.19],
        "z-lengths": [1.56, 1.53, 1.23, 1.35],
    },
    "ASP": {
        "atoms": ["CB", "CG", "OD1", "OD2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["OD1", "CB", "CG"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2A",
            "CG": "CC",
            "N": "NH1",
            "O": "O",
            "OD1": "OC",
            "OD2": "OC",
        },
        "z-angles": [114.1, 112.6, 117.99, 117.7],
        "z-dihedrals": [122.33, 180.0, 180.0, -170.23],
        "z-lengths": [1.56, 1.52, 1.26, 1.25],
    },
    "CYS": {
        "atoms": ["CB", "SG"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"]],
        "types": {"C": "C", "CA": "CT1", "CB": "CT2", "N": "NH1", "O": "O", "SG": "S"},
        "z-angles": [111.98, 113.87],
        "z-dihedrals": [121.79, 180.0],
        "z-lengths": [1.56, 1.84],
    },
    "GLN": {
        "atoms": ["CB", "CG", "CD", "OE1", "NE2"],
        "chi_indices": [1, 2, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["OE1", "CG", "CD"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD": "CC",
            "CG": "CT2",
            "N": "NH1",
            "NE2": "NH2",
            "O": "O",
            "OE1": "O",
        },
        "z-angles": [111.68, 115.52, 112.5, 121.52, 116.84],
        "z-dihedrals": [121.91, 180.0, 180.0, 180.0, 179.57],
        "z-lengths": [1.55, 1.55, 1.53, 1.23, 1.35],
    },
    "GLU": {
        "atoms": ["CB", "CG", "CD", "OE1", "OE2"],
        "chi_indices": [1, 2, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["OE1", "CG", "CD"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2A",
            "CD": "CC",
            "CG": "CT2",
            "N": "NH1",
            "O": "O",
            "OE1": "OC",
            "OE2": "OC",
        },
        "z-angles": [111.71, 115.69, 115.73, 114.99, 120.08],
        "z-dihedrals": [121.9, 180.0, 180.0, 180.0, -179.1],
        "z-lengths": [1.55, 1.56, 1.53, 1.26, 1.25],
    },
    "GLY": {
        "atoms": [],
        "chi_indices": [],
        "parents": [],
        "types": {"C": "C", "CA": "CT2", "N": "NH1", "O": "O"},
        "z-angles": [],
        "z-dihedrals": [],
        "z-lengths": [],
    },
    "HIS": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR1",
            "NE2": "NR2",
            "O": "O",
        },
        "z-angles": [109.99, 114.05, 124.1, 129.6, 107.03, 110.03],
        "z-dihedrals": [122.46, 180.0, 90.0, -171.29, -173.21, 171.99],
        "z-lengths": [1.55, 1.5, 1.38, 1.36, 1.35, 1.38],
    },
    "HSD": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR1",
            "NE2": "NR2",
            "O": "O",
        },
        "z-angles": [109.99, 114.05, 124.1, 129.6, 107.03, 110.03],
        "z-dihedrals": [122.46, 180.0, 90.0, -171.29, -173.21, 171.99],
        "z-lengths": [1.55, 1.5, 1.38, 1.36, 1.35, 1.38],
    },
    "HSE": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR2",
            "NE2": "NR1",
            "O": "O",
        },
        "z-angles": [111.67, 116.94, 120.17, 129.71, 105.2, 105.8],
        "z-dihedrals": [123.52, 180.0, 90.0, -178.26, -179.2, 178.66],
        "z-lengths": [1.56, 1.51, 1.39, 1.36, 1.32, 1.38],
    },
    "HSP": {
        "atoms": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "chi_indices": [],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["ND1", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CB", "CG", "CD2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2A",
            "CD2": "CPH1",
            "CE1": "CPH2",
            "CG": "CPH1",
            "N": "NH1",
            "ND1": "NR3",
            "NE2": "NR3",
            "O": "O",
        },
        "z-angles": [109.38, 114.18, 122.94, 128.93, 108.9, 106.93],
        "z-dihedrals": [125.13, 180.0, 90.0, -165.26, -167.62, 167.13],
        "z-lengths": [1.55, 1.52, 1.37, 1.35, 1.33, 1.37],
    },
    "ILE": {
        "atoms": ["CB", "CG1", "CG2", "CD1"],
        "chi_indices": [1, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CG1", "CA", "CB"],
            ["CA", "CB", "CG1"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT1",
            "CD1": "CT3",
            "CG1": "CT2",
            "CG2": "CT3",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [112.93, 113.63, 113.93, 114.09],
        "z-dihedrals": [124.22, 180.0, -130.04, 180.0],
        "z-lengths": [1.57, 1.55, 1.55, 1.54],
    },
    "LEU": {
        "atoms": ["CB", "CG", "CD1", "CD2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD1", "CB", "CG"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CT3",
            "CD2": "CT3",
            "CG": "CT1",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [112.12, 117.46, 110.48, 112.57],
        "z-dihedrals": [121.52, 180.0, 180.0, 120.0],
        "z-lengths": [1.55, 1.55, 1.54, 1.54],
    },
    "LYS": {
        "atoms": ["CB", "CG", "CD", "CE", "NZ"],
        "chi_indices": [1, 2, 3, 4],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "CE"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD": "CT2",
            "CE": "CT2",
            "CG": "CT2",
            "N": "NH1",
            "NZ": "NH3",
            "O": "O",
        },
        "z-angles": [111.36, 115.76, 113.28, 112.33, 110.46],
        "z-dihedrals": [122.23, 180.0, 180.0, 180.0, 180.0],
        "z-lengths": [1.56, 1.54, 1.54, 1.53, 1.46],
    },
    "MET": {
        "atoms": ["CB", "CG", "SD", "CE"],
        "chi_indices": [1, 2, 3],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "SD"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CE": "CT3",
            "CG": "CT2",
            "N": "NH1",
            "O": "O",
            "SD": "S",
        },
        "z-angles": [111.88, 115.92, 110.28, 98.94],
        "z-dihedrals": [121.62, 180.0, 180.0, 180.0],
        "z-lengths": [1.55, 1.55, 1.82, 1.82],
    },
    "PHE": {
        "atoms": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD1", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CG", "CD1", "CE1"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CA",
            "CD2": "CA",
            "CE1": "CA",
            "CE2": "CA",
            "CG": "CA",
            "CZ": "CA",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [112.45, 112.76, 120.32, 120.76, 120.63, 120.62, 119.93],
        "z-dihedrals": [122.49, 180.0, 90.0, -177.96, -177.37, 177.2, -0.12],
        "z-lengths": [1.56, 1.51, 1.41, 1.41, 1.4, 1.4, 1.4],
    },
    "PRO": {
        "atoms": ["CB", "CG", "CD"],
        "chi_indices": [1, 2],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"], ["CA", "CB", "CG"]],
        "types": {
            "C": "C",
            "CA": "CP1",
            "CB": "CP2",
            "CD": "CP3",
            "CG": "CP2",
            "N": "N",
            "O": "O",
        },
        "z-angles": [111.74, 104.39, 103.21],
        "z-dihedrals": [113.74, 31.61, -34.59],
        "z-lengths": [1.54, 1.53, 1.53],
    },
    "SER": {
        "atoms": ["CB", "OG"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"]],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "N": "NH1",
            "O": "O",
            "OG": "OH1",
        },
        "z-angles": [111.4, 112.45],
        "z-dihedrals": [124.75, 180.0],
        "z-lengths": [1.56, 1.43],
    },
    "THR": {
        "atoms": ["CB", "OG1", "CG2"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"], ["OG1", "CA", "CB"]],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT1",
            "CG2": "CT3",
            "N": "NH1",
            "O": "O",
            "OG1": "OH1",
        },
        "z-angles": [112.74, 112.16, 115.91],
        "z-dihedrals": [126.46, 180.0, -124.13],
        "z-lengths": [1.57, 1.43, 1.53],
    },
    "TRP": {
        "atoms": ["CB", "CG", "CD2", "CD1", "CE2", "NE1", "CE3", "CZ3", "CH2", "CZ2"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD2", "CB", "CG"],
            ["CD1", "CG", "CD2"],
            ["CG", "CD2", "CE2"],
            ["CE2", "CG", "CD2"],
            ["CE2", "CD2", "CE3"],
            ["CD2", "CE3", "CZ3"],
            ["CE3", "CZ3", "CH2"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CA",
            "CD2": "CPT",
            "CE2": "CPT",
            "CE3": "CAI",
            "CG": "CY",
            "CH2": "CA",
            "CZ2": "CAI",
            "CZ3": "CA",
            "N": "NH1",
            "NE1": "NY",
            "O": "O",
        },
        "z-angles": [
            111.23,
            115.14,
            123.95,
            129.18,
            106.65,
            107.87,
            132.54,
            118.16,
            120.97,
            120.87,
        ],
        "z-dihedrals": [
            122.68,
            180.0,
            90.0,
            -172.81,
            -0.08,
            0.14,
            179.21,
            -0.2,
            0.1,
            0.01,
        ],
        "z-lengths": [1.56, 1.52, 1.44, 1.37, 1.41, 1.37, 1.4, 1.4, 1.4, 1.4],
    },
    "TYR": {
        "atoms": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "chi_indices": [1, 2],
        "parents": [
            ["N", "C", "CA"],
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CD1", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CG", "CD1", "CE1"],
            ["CE1", "CE2", "CZ"],
        ],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT2",
            "CD1": "CA",
            "CD2": "CA",
            "CE1": "CA",
            "CE2": "CA",
            "CG": "CA",
            "CZ": "CA",
            "N": "NH1",
            "O": "O",
            "OH": "OH1",
        },
        "z-angles": [112.34, 112.94, 120.49, 120.46, 120.4, 120.56, 120.09, 120.25],
        "z-dihedrals": [122.27, 180.0, 90.0, -176.46, -175.49, 175.32, -0.19, -178.98],
        "z-lengths": [1.56, 1.51, 1.41, 1.41, 1.4, 1.4, 1.4, 1.41],
    },
    "VAL": {
        "atoms": ["CB", "CG1", "CG2"],
        "chi_indices": [1],
        "parents": [["N", "C", "CA"], ["N", "CA", "CB"], ["CG1", "CA", "CB"]],
        "types": {
            "C": "C",
            "CA": "CT1",
            "CB": "CT1",
            "CG1": "CT3",
            "CG2": "CT3",
            "N": "NH1",
            "O": "O",
        },
        "z-angles": [111.23, 113.97, 112.17],
        "z-dihedrals": [122.95, 180.0, 123.99],
        "z-lengths": [1.57, 1.54, 1.54],
    },
}


# our constants
# elements
periodic_table = [ # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]
#atomic weights
atomic_weights = {
    'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,
    'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,
    'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,
    'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,
    'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,
    'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,
    'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,
    'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,
    'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,
    'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,
    'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,
    'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,
    'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,
    'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,
    'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,
    'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,
    'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,
    'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,
    'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,
    'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,
    'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,
    'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,
    'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,
    'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247
}
#covalent radii from Alvarez (2008)
#DOI: 10.1039/b801115j
covalent_radii = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28,
    'Be': 0.96, 'B': 0.84, 'C': 0.76, 
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 
    'V': 1.53, 'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 
    'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 
    'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95,
    'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39,
    'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
    'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04,
    'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98,
    'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92,
    'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87,
    'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62,
    'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36,
    'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46,
    'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 
    'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
    'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87,
    'Am': 1.80, 'Cm': 1.69
}


protein_atoms = ['C', 'N', 'O', 'S']

# amino acids
aas = [
    ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
    ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
    ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
    ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
    ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO')
]

# backbone atoms
backbone_atoms = ['N', 'CA', 'C', 'O']

# side-chain atoms
sidechain_atoms = { symbol: AA_GEOMETRY[aa]['atoms'] for symbol, aa in aas }
# sidechain_atoms = {
#     'G': [],   # -H
#     'A': ['CB'],  # -CH3
#     'V': ['CB', 'CG1', 'CG2'],  # -CH-(CH3)2
#     'L': ['CB', 'CG', 'CD1', 'CD2'],  # -CH2-CH(CH3)2
#     'I': ['CB', 'CG1', 'CG2', 'CD1'], # -CH(CH3)-CH2-CH3
#     'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # -CH2-C6H5
#     'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # -CH2-C8NH6
#     'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],  # -CH2-C6H4-OH
#     'D': ['CB', 'CG', 'OD1', 'OD2'],  # -CH2-COOH
#     'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # -CH2-C3H3N2
#     'N': ['CB', 'CG', 'OD1', 'ND2'],  # -CH2-CONH2
#     'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],  # -(CH2)2-COOH
#     'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],  # -(CH2)4-NH2
#     'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],  # -(CH2)-CONH2
#     'M': ['CB', 'CG', 'SD', 'CE'],  # -(CH2)2-S-CH3
#     'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],  # -(CH2)3-NHC(NH)NH2
#     'S': ['CB', 'OG'],  # -CH2-OH
#     'T': ['CB', 'OG1', 'CG2'],  # -CH(CH3)-OH
#     'C': ['CB', 'SG'],  # -CH2-SH
#     'P': ['CB', 'CG', 'CD'],  # -C3H6
# }

# bonds
aa_bonds = {
    'G': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2)],
    'A': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1)],
    'V': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG1', 'CB', 1), ('CG2', 'CB', 1)],
    'L': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD1', 'CG', 1), ('CD2', 'CG', 1)],
    'I': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG1', 'CB', 1), ('CG2', 'CB', 1), ('CD1', 'CG1', 1)],
    'F': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD1', 'CG', 2), ('CD2', 'CG', 1), ('CE1', 'CD1', 1), ('CE2', 'CD2', 2), ('CZ', 'CE2', 1), ('CZ', 'CE1', 2)],
    'W': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD2', 'CG', 1), ('CD1', 'CG', 2), ('CE2', 'CD2', 2), ('NE1', 'CD1', 1), ('CE2', 'NE1', 1), ('CE3', 'CD2', 1), ('CZ3', 'CE3', 2), ('CH2', 'CZ3', 1), ('CZ2', 'CE2', 1), ('CH2', 'CZ2', 2)],
    'Y': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD1', 'CG', 2), ('CD2', 'CG', 1), ('CE1', 'CD1', 1), ('CE2', 'CD2', 2), ('CZ', 'CE2', 1), ('CZ', 'CE1', 2), ('OH', 'CZ', 1)],
    'D': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('OD1', 'CG', 2), ('OD2', 'CG', 1)],
    'H': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('ND1', 'CG', 1), ('CD2', 'CG', 2), ('CE1', 'ND1', 2), ('NE2', 'CE1', 1), ('NE2', 'CD2', 1)],
    'N': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('OD1', 'CG', 2), ('ND2', 'CG', 1)],
    'E': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD', 'CG', 1), ('OE1', 'CD', 2), ('OE2', 'CD', 1)],
    'K': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD', 'CG', 1), ('CE', 'CD', 1), ('NZ', 'CE', 1)],
    'Q': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD', 'CG', 1), ('OE1', 'CD', 2), ('NE2', 'CD', 1)],
    'M': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('SD', 'CG', 1), ('CE', 'SD', 1)],
    'R': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD', 'CG', 1), ('NE', 'CD', 1), ('CZ', 'NE', 1), ('NH1', 'CZ', 1), ('NH2', 'CZ', 2)],
    'S': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('OG', 'CB', 1)],
    'T': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('OG1', 'CB', 1), ('CG2', 'CB', 1)],
    'C': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('SG', 'CB', 1)],
    'P': [('CA', 'N', 1), ('C', 'CA', 1), ('O', 'C', 2), ('CB', 'CA', 1), ('CG', 'CB', 1), ('CD', 'CG', 1), ('CD', 'N', 1)]
}

# atoms for defining chi angles on the side chains
chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# amino acid smiles
aa_smiles = {
    'G': 'C(C(=O)O)N',
    'A': 'O=C(O)C(N)C',
    'V': 'CC(C)[C@@H](C(=O)O)N',
    'L': 'CC(C)C[C@@H](C(=O)O)N',
    'I': 'CC[C@H](C)[C@@H](C(=O)O)N',
    'F': 'NC(C(=O)O)Cc1ccccc1',
    'W': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',
    'Y': 'N[C@@H](Cc1ccc(O)cc1)C(O)=O',
    'D': 'O=C(O)CC(N)C(=O)O',
    'H': 'O=C([C@H](CC1=CNC=N1)N)O',
    'N': 'NC(=O)CC(N)C(=O)O',
    'E': 'OC(=O)CCC(N)C(=O)O',
    'K': 'NCCCC(N)C(=O)O',
    'Q': 'O=C(N)CCC(N)C(=O)O',
    'M': 'CSCC[C@H](N)C(=O)O',
    'R': 'NC(=N)NCCCC(N)C(=O)O',
    'S': 'C([C@@H](C(=O)O)N)O',
    'T': 'C[C@H]([C@@H](C(=O)O)N)O',
    'C': 'C([C@@H](C(=O)O)N)S',
    'P': 'OC(=O)C1CCCN1'
}

aa_max_n_atoms = max([len(atoms) for atoms in sidechain_atoms.values()]) + len(backbone_atoms) # plus backbone


# bases for RNA/DNA
bases = [
    ('DA', 'DA'), ('DG', 'DG'), ('DC', 'DC'), ('DT', 'DT'), # DNA
    ('RA', 'A'), ('RG', 'G'), ('RC', 'C'), ('RU', 'U')  # RNA (we add R in symbols to distinguish with those from the amino acid)
]

base_atoms = {
    'DA': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'C8', 'N1', 'N3', 'N6', 'N7', 'N9', "O3'", "O4'", "O5'", 'OP1', 'OP2', 'P'],
    'DG': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'C8', 'N1', 'N2', 'N3', 'N7', 'N9', "O3'", "O4'", "O5'", 'O6', 'OP1', 'OP2', 'P'],
    'DC': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'N1', 'N3', 'N4', 'O2', "O3'", "O4'", "O5'", 'OP1', 'OP2', 'P'],
    'DT': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'C7', 'N1', 'N3', 'O2', "O3'", 'O4', "O4'", "O5'", 'OP1', 'OP2', 'P'],
    'RA': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'C8', 'N1', 'N3', 'N6', 'N7', 'N9', "O2'", "O3'", "O4'", "O5'", 'OP1', 'OP2', 'P'],
    'RG': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'C8', 'N1', 'N2', 'N3', 'N7', 'N9', "O2'", "O3'", "O4'", "O5'", 'O6', 'OP1', 'OP2', 'P'],
    'RC': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'N1', 'N3', 'N4', 'O2', "O2'", "O3'", "O4'", "O5'", 'OP1', 'OP2', 'P'],
    'RU': ["C1'", 'C2', "C2'", "C3'", 'C4', "C4'", 'C5', "C5'", 'C6', 'N1', 'N3', 'O2', "O2'", "O3'", 'O4', "O4'", "O5'", 'OP1', 'OP2', 'P']
}

base_bonds = {
    'DA': [("C3'", "O3'", 1), ('C5', 'N7', 1), ('C2', 'N1', 1), ('C8', 'N7', 2), ("C4'", "C5'", 1), ("C4'", "O4'", 1), ('C5', 'C6', 1), ('C6', 'N6', 1), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ('C4', 'C5', 2), ('C8', 'N9', 1), ('C4', 'N9', 1), ("O5'", 'P', 1), ('C4', 'N3', 1), ('C6', 'N1', 2), ('OP2', 'P', 1), ('OP1', 'P', 2), ("C5'", "O5'", 1), ("C3'", "C4'", 1), ("C1'", 'N9', 1), ('C2', 'N3', 2), ("C1'", "O4'", 1)],
    'DG': [("C3'", "O3'", 1), ('C5', 'N7', 1), ('C2', 'N1', 1), ('C8', 'N7', 2), ('C6', 'N1', 1), ("C4'", "C5'", 1), ("C4'", "O4'", 1), ('C5', 'C6', 1), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ('C4', 'C5', 2), ('C6', 'O6', 2), ('C8', 'N9', 1), ('C4', 'N9', 1), ("O5'", 'P', 1), ('C4', 'N3', 1), ('OP2', 'P', 1), ('OP1', 'P', 2), ("C5'", "O5'", 1), ("C3'", "C4'", 1), ("C1'", 'N9', 1), ('C2', 'N3', 2), ('C2', 'N2', 1), ("C1'", "O4'", 1)],
    'DC': [('C5', 'C6', 2), ("C3'", "O3'", 1), ('C2', 'N1', 1), ('C6', 'N1', 1), ("C4'", "O4'", 1), ("C4'", "C5'", 1), ('C2', 'N3', 1), ('C4', 'N3', 2), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ("O5'", 'P', 1), ('OP2', 'P', 1), ('OP1', 'P', 2), ("C5'", "O5'", 1), ("C3'", "C4'", 1), ("C1'", 'N1', 1), ('C4', 'C5', 1), ('C4', 'N4', 1), ("C1'", "O4'", 1), ('C2', 'O2', 2)],
    'DT': [('C5', 'C6', 2), ("C3'", "O3'", 1), ('C2', 'N1', 1), ('C6', 'N1', 1), ("C4'", "C5'", 1), ("C4'", "O4'", 1), ('C2', 'N3', 1), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ('C4', 'O4', 2), ('C5', 'C7', 1), ("O5'", 'P', 1), ('C4', 'N3', 1), ('OP2', 'P', 1), ('OP1', 'P', 2), ("C5'", "O5'", 1), ("C3'", "C4'", 1), ("C1'", 'N1', 1), ('C4', 'C5', 1), ("C1'", "O4'", 1), ('C2', 'O2', 2)],
    'RA': [('OP2', 'P', 1), ('C2', 'N1', 1), ("C4'", "C5'", 1), ('C4', 'N9', 1), ('C5', 'C6', 1), ('C8', 'N9', 1), ("C1'", "O4'", 1), ('OP1', 'P', 2), ("C3'", "C4'", 1), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ('C6', 'N1', 2), ('C4', 'N3', 1), ("C1'", 'N9', 1), ('C2', 'N3', 2), ("C3'", "O3'", 1), ("C2'", "O2'", 1), ("C4'", "O4'", 1), ('C4', 'C5', 2), ("O5'", 'P', 1), ('C5', 'N7', 1), ("C5'", "O5'", 1), ('C6', 'N6', 1), ('C8', 'N7', 2)],
    'RG': [('OP2', 'P', 1), ('C2', 'N1', 1), ("C4'", "C5'", 1), ('C4', 'N9', 1), ('C5', 'C6', 1), ('C6', 'O6', 2), ('C8', 'N9', 1), ("C1'", "O4'", 1), ('OP1', 'P', 2), ("C3'", "C4'", 1), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ('C4', 'N3', 1), ("C1'", 'N9', 1), ('C2', 'N3', 2), ("C3'", "O3'", 1), ("C2'", "O2'", 1), ("C4'", "O4'", 1), ('C4', 'C5', 2), ("O5'", 'P', 1), ('C6', 'N1', 1), ('C5', 'N7', 1), ("C5'", "O5'", 1), ('C8', 'N7', 2), ('C2', 'N2', 1)],
    'RC': [('OP2', 'P', 1), ("C1'", 'N1', 1), ("C4'", "C5'", 1), ('C2', 'N1', 1), ('C4', 'N3', 2), ("C1'", "O4'", 1), ('OP1', 'P', 2), ('C2', 'O2', 2), ("C3'", "C4'", 1), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ("C3'", "O3'", 1), ("C2'", "O2'", 1), ("C4'", "O4'", 1), ("O5'", 'P', 1), ('C6', 'N1', 1), ('C5', 'C6', 2), ("C5'", "O5'", 1), ('C2', 'N3', 1), ('C4', 'C5', 1), ('C4', 'N4', 1)],
    'RU': [('OP2', 'P', 1), ("C1'", 'N1', 1), ("C4'", "C5'", 1), ('C2', 'N1', 1), ("C1'", "O4'", 1), ('OP1', 'P', 2), ('C2', 'O2', 2), ("C3'", "C4'", 1), ("C1'", "C2'", 1), ("C2'", "C3'", 1), ('C4', 'O4', 2), ('C4', 'N3', 1), ("C3'", "O3'", 1), ("C2'", "O2'", 1), ("C4'", "O4'", 1), ("O5'", 'P', 1), ('C6', 'N1', 1), ('C5', 'C6', 2), ("C5'", "O5'", 1), ('C2', 'N3', 1), ('C4', 'C5', 1)]
}