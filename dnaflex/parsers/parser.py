"""DNA structure parsing and modification handling."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from dnaflex.structure.structure import DnaAtom, DnaChain, DnaResidue, DnaStructure

class DnaParser:
    """Parser for DNA structures from PDB/mmCIF files."""
    
    # Standard DNA residue names
    DNA_RESIDUES = {'DA', 'DG', 'DC', 'DT', 'DN'}
    
    # DNA atom types based on AlphaFold3's implementation
    BACKBONE_ATOMS = {
        'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"
    }
    
    def __init__(self):
        self._structure = DnaStructure()
        self._current_chain: Optional[DnaChain] = None
        self._seen_chains: Set[str] = set()
        
    def parse_pdb(self, file_path: str) -> DnaStructure:
        """Parse a PDB file containing DNA structure."""
        with open(file_path) as f:
            for line in f:
                if line.startswith('ATOM  ') or line.startswith('HETATM'):
                    self._parse_atom_line(line)
        return self._structure
        
    def _parse_atom_line(self, line: str) -> None:
        """Parse a PDB ATOM/HETATM line."""
        # PDB format specification
        atom_id = line[6:11].strip()
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_num = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
        b_factor = float(line[60:66]) if line[60:66].strip() else 0.0
        element = line[76:78].strip()
        
        if residue_name not in self.DNA_RESIDUES:
            return
            
        # Handle chain changes
        if chain_id not in self._seen_chains:
            self._seen_chains.add(chain_id)
            chain = DnaChain(chain_id)
            self._structure.add_chain(chain)
            self._current_chain = chain
            
        # Create atom
        atom = DnaAtom(
            id=atom_id,
            name=atom_name,
            element=element or self._guess_element(atom_name),
            x=x, y=y, z=z,
            occupancy=occupancy,
            b_factor=b_factor
        )
        
        # Add to residue
        if residue := self._current_chain.get_residue(residue_num):
            residue.atoms[atom_name] = atom
        else:
            new_residue = DnaResidue(
                name=residue_name,
                number=residue_num,
                atoms={atom_name: atom},
                chain_id=chain_id
            )
            self._current_chain.add_residue(new_residue)
            
    def _guess_element(self, atom_name: str) -> str:
        """Guess element from atom name."""
        # Strip numbers and quotes
        base = re.sub(r'[0-9\'"]+', '', atom_name)
        if len(base) == 1:
            return base
        return base[0]
        
    def apply_modifications(self, modifications: List[Tuple[str, int]]) -> None:
        """Apply modifications to DNA residues.
        
        Args:
            modifications: List of (modification_type, position) tuples
        """
        for mod_type, position in modifications:
            for chain in self._structure.chains:
                if residue := chain.get_residue(position):
                    # Replace residue name with modification code
                    new_residue = DnaResidue(
                        name=mod_type,
                        number=residue.number,
                        atoms=residue.atoms,
                        chain_id=residue.chain_id
                    )
                    chain._residues[position] = new_residue