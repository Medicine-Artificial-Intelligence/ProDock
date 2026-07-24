#!/usr/bin/env python3
# rename_atoms.py
# Usage: python3 rename_atoms.py input.mol2 output.mol2
import sys

def rename_mol2_atoms(inp, outp):
    with open(inp) as f, open(outp, 'w') as fo:
        atom_idx = 1
        in_atoms = False
        for line in f:
            # Replace any occurrence of "UNL1" with "LIG"
            line = line.replace("UNL1", "LIG")

            # If the line ends with an .sdf filename, replace it with "LIG"
            if line.strip().endswith('.sdf'):
                fo.write('LIG1\n')
                continue

            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                fo.write(line)
                continue
            if line.startswith("@<TRIPOS>"):
                in_atoms = False
                fo.write(line)
                continue
            if in_atoms:
                parts = line.rstrip().split()
                # TRIPOS format: atom_id name x y z type [subst_id [subst_name [charge [status_bit]]]]
                element = parts[5].split('.')[0]  # e.g. "C.3" → "C"
                new_name = f"{element}{atom_idx}"
                parts[1] = new_name
                parts[0] = str(atom_idx)
                atom_idx += 1
                # Write fields separated by a tab
                fo.write("\t".join(parts) + "\n")
            else:
                fo.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: rename_atoms.py input.mol2 output.mol2")
        sys.exit(1)
    rename_mol2_atoms(sys.argv[1], sys.argv[2])