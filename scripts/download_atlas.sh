#!/bin/bash
out_root="./data/pdbxtc"
names=("1a62_A" "1bkp_A")
for name in "${names[@]}"; do
    out_dir="${out_root}/${name}"
    pdb_file="${out_dir}/${name}.pdb"
    zip_file="${out_dir}/${name}_protein.zip"
    url="https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip"
    if [ -f "$pdb_file" ]; then
        echo "Skip ${name}: ${pdb_file} already exists"
        continue
    fi
    mkdir -p "$out_dir"
    echo "Downloading ${name} ..."
    wget "$url" -O "$zip_file"
    unzip "$zip_file" -d "$out_dir"
    rm -f "$zip_file"
done