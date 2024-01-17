"""Testing single point calculations between pure SPARC and socket mode
"""

import ase
from ase.io import read, write
from ase.build import bulk
from pathlib import Path
from sparc import SPARC
import numpy as np
from ase.calculators.socketio import SocketIOCalculator
from subprocess import Popen, PIPE
import os
import shutil
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize.lbfgs import LBFGS

os.environ["SPARC_PSP_PATH"] = "../../../psps/"

sparc_params = {
    "h": 0.28,
    "PRECOND_KERKER_THRESH": 0,
    "ELEC_TEMP_TYPE": "Fermi-Dirac",
    "ELEC_TEMP": 300,
    "MIXING_PARAMETER": 1.0,
    "TOL_SCF": 1e-3,
    "RELAX_FLAG": 1,
    "CALC_STRESS": 1,
    "PRINT_ATOMS": 1,
    "PRINT_FORCES": 1,
}


al = bulk("Al", cubic=True)
al.rattle(0.2, seed=42)


def sparc_singlepoint():
    shutil.rmtree("sparc_geopt", ignore_errors=True)
    try:
        os.remove("sparc_geopt.extxyz")
    except Exception:
        pass
    atoms = al.copy()
    calc = SPARC(
        directory=f"sparc_geopt",
        command="mpirun -n 2 ../../../../lib/sparc",
        **sparc_params,
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    out_images = read("sparc_geopt", format="sparc", index=":")
    write("sparc_geopt.extxyz", out_images)
    return out_images


def sparc_socket():
    atoms = al.copy()

    inputs = Path("./sparc_geopt")
    copy_to = Path("./socket_test")
    shutil.rmtree(copy_to, ignore_errors=True)
    os.makedirs(copy_to, exist_ok=True)
    shutil.copy(inputs / "SPARC.ion", copy_to)
    shutil.copy(inputs / "SPARC.inpt", copy_to)
    shutil.copy(inputs / "13_Al_3_1.9_1.9_pbe_n_v1.0.psp8", copy_to)

    calc = SocketIOCalculator(port=12345)
    p_ = Popen(
        "mpirun -n 2 ../../../../lib/sparc -socket :12345 -name SPARC > sparc.log",
        shell=True,
        cwd=copy_to,
    )
    # out_images = []
    with calc:
        atoms.calc = calc
        opt = LBFGS(atoms, trajectory="sparc-socket.traj")
        opt.run(fmax=0.02)
    return atoms.copy()


def main():
    images_sparc = sparc_singlepoint()
    final_socket = sparc_socket()
    final_sparc = images_sparc[-1]
    final_socket.wrap()
    final_sparc.wrap()
    positions_change = final_socket.positions - final_sparc.positions
    max_change = np.linalg.norm(positions_change)
    print("Max shift: ", max_change)


if __name__ == "__main__":
    main()
