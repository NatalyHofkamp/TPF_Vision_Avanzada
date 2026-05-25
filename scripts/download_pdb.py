"""
Descargar proteínas desde PDB.
"""

from Bio.PDB import PDBList


def download_example():

    pdbl = PDBList()

    pdbl.retrieve_pdb_file(
        "1ATP",
        pdir="data/raw",
        file_format="pdb"
    )


if __name__ == "__main__":
    download_example()
