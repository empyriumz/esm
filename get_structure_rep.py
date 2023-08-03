import esm.inverse_folding
import torch
from Bio import SeqIO
import numpy as np
import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile


def fix_pdb(pdb_path):
    fixer = PDBFixer(filename=pdb_path)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    directory, file = os.path.split(pdb_path)
    name, extension = os.path.splitext(file)
    # Add the postfix to the file name
    new_name = name + "_fixed"

    # Join the new file name with the extension
    new_file = new_name + extension
    new_fpath = os.path.join(directory, new_file)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(new_fpath, "w"))

    return new_fpath


def get_structure(pdb_path, chain_id):
    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
    # extract coordinates and sequences from the structure
    coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    return coords, seq


def embedding(ligand):
    # load fasta file and get ids
    with open(
        "/host/Protein-ligand-binding/data_processing/biolip2/{}/{}_total_30_identity.fasta".format(
            ligand, ligand
        )
    ) as handle:
        recs = list(SeqIO.parse(handle, "fasta"))
    # ids = [rec.id for rec in recs]
    seqs = {}
    for rec in recs:
        seqs[rec.id] = str(rec.seq)
    # with open("total_failed_id.txt") as file:
    #     failed_ids = file.readlines()
    #     failed_ids = [line.rstrip() for line in failed_ids]
    # ids = list(set(ids).intersection(set(failed_ids)))
    # load the model
    device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    model = model.to(device)
    failed_id = []
    target_dir = "/host/Protein-ligand-binding/biolip2_embedding/{}/esm_if".format(ligand)
    os.makedirs(target_dir, exist_ok=True)
    with torch.no_grad():
        for id in seqs:
            fpath = (
                "/host/Protein-ligand-binding/data_processing/original_pdb/"
                + id[:4].upper()
                + ".pdb"
            )  # .cif format is also acceptable
            if not os.path.exists(fpath):
                fpath = fpath.replace(".pdb", ".cif")
                if not os.path.exists(fpath):
                    failed_id.append(id)
                    print("failed to find pdb file for {}".format(id))
                    continue

            chain_id = id[5:]
            # the file can be either in pdb or cif format
            try:
                coords, seq = get_structure(fpath, chain_id)
                if seq != seqs[id]:
                    print("sequence mismatch for {}, try fixing with PDBFixer".format(id))
                    fpath = fix_pdb(fpath)  # fix pdb file and return the new path
                    coords, seq = get_structure(fpath, chain_id)
                    if seq != seqs[id]:
                        print("sequence mismatch after fixer {}".format(id))
                        # print("seq after fixing:", seq)
                        # print("right seq:", seqs[id])
                        failed_id.append(id)
                        continue
                rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
                np.save(target_dir + "/{}".format(id), rep.detach().cpu().numpy())
            except:
                failed_id.append(id)
    if failed_id:
        with open("failed_id_{}.txt".format(ligand), "w") as f:
            for line in failed_id:
                f.write(line + "\n")
        print("failed IDs for {}".format(ligand), failed_id)


if __name__ == "__main__":
    # ligand_list = ["DNA", "RNA", "MG", "FE2", "NI", "NA", "K", "CO", "MN", "CA", "FE", "CU", "ZN"]
    ligand_list = ["MN", "FE2", "NI", "FE", "NA", "CO", "K", "CU"]
    # ligand_list = ["CA"]
    # ligand_list = ["MG"]
    # ligand_list = ["ZN"]
    total_failed_id = []
    for ligand in ligand_list:
        # embedding(ligand)
        try:
            with open("failed_id_{}.txt".format(ligand)) as file:
                ids = file.readlines()
                ids = [line.rstrip() for line in ids]
            total_failed_id.extend(ids)
        except:
            continue
    total_failed_id = list(set(total_failed_id))
    with open("failed_id_mn_and_others.txt", "w") as f:
        for line in total_failed_id:
            f.write(line + "\n")
