"""DNA drug binding prediction module."""

from typing import Dict, List, Any
import numpy as np

class BindingAnalyzer:
    """Analyze and predict DNA-drug binding sites."""
    
    def __init__(self):
        # Common DNA binding motifs
        self.binding_motifs = {
            'minor_groove': ['AATT', 'TTAA'],  # Minor groove binders
            'major_groove': ['GGCC', 'CCGG'],  # Major groove binders
            'intercalation': ['CG', 'GC'],     # Intercalation sites
        }
        
        # Simple scoring weights
        self.weights = {
            'sequence': 0.4,
            'structure': 0.3,
            'flexibility': 0.3
        }
        
    def predict(self, sequence: str) -> Dict[str, Any]:
        """Predict potential binding sites in DNA sequence."""
        # Find binding sites
        sites = self._find_binding_sites(sequence)
        
        # Score and classify sites
        scored_sites = self._score_binding_sites(sequence, sites)
        
        return {
            'binding_sites': scored_sites,
            'num_sites': len(scored_sites),
            'binding_propensity': self._calculate_binding_propensity(scored_sites)
        }
        
    def _find_binding_sites(self, sequence: str) -> List[Dict[str, Any]]:
        """Identify potential binding sites."""
        sites = []
        
        # Search for known binding motifs
        for binding_type, motifs in self.binding_motifs.items():
            for motif in motifs:
                pos = 0
                while True:
                    pos = sequence.find(motif, pos)
                    if pos == -1:
                        break
                        
                    sites.append({
                        'position': pos,
                        'length': len(motif),
                        'sequence': motif,
                        'type': binding_type
                    })
                    pos += 1
                    
        return sites
        
    def _score_binding_sites(self, sequence: str, 
                           sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score identified binding sites."""
        scored_sites = []
        
        for site in sites:
            # Sequence-based score
            seq_score = self._sequence_score(sequence, site)
            
            # Structure-based score (simplified)
            struct_score = self._structure_score(sequence, site)
            
            # Flexibility-based score
            flex_score = self._flexibility_score(sequence, site)
            
            # Calculate total score
            total_score = (
                self.weights['sequence'] * seq_score +
                self.weights['structure'] * struct_score +
                self.weights['flexibility'] * flex_score
            )
            
            site_info = {
                **site,
                'sequence_score': seq_score,
                'structure_score': struct_score,
                'flexibility_score': flex_score,
                'total_score': total_score
            }
            scored_sites.append(site_info)
            
        return sorted(scored_sites, key=lambda x: x['total_score'], reverse=True)
        
    def _sequence_score(self, sequence: str, site: Dict[str, Any]) -> float:
        """Calculate sequence-based binding score."""
        pos = site['position']
        motif = site['sequence']
        
        # Check local sequence composition
        local_seq = sequence[max(0, pos-2):pos+len(motif)+2]
        gc_content = (local_seq.count('G') + local_seq.count('C')) / len(local_seq)
        
        # Score based on binding type
        if site['type'] == 'minor_groove':
            return 0.8 * (1 - gc_content)  # AT-rich preferred
        elif site['type'] == 'major_groove':
            return 0.8 * gc_content  # GC-rich preferred
        else:  # intercalation
            return 0.6 * gc_content + 0.4  # Slightly GC preferred
            
    def _structure_score(self, sequence: str, site: Dict[str, Any]) -> float:
        """Calculate structure-based binding score."""
        # Simplified structure prediction
        pos = site['position']
        length = site['length']
        
        # Consider neighboring sequence influence
        context_seq = sequence[max(0, pos-3):pos+length+3]
        
        # Basic structural properties
        if site['type'] == 'minor_groove':
            # Narrow minor groove preferred
            score = 0.7 + 0.3 * (context_seq.count('A') + context_seq.count('T')) / len(context_seq)
        elif site['type'] == 'major_groove':
            # Wide major groove preferred
            score = 0.6 + 0.4 * (context_seq.count('G') + context_seq.count('C')) / len(context_seq)
        else:  # intercalation
            # Base stacking important
            score = 0.5 + 0.5 * (1 - abs(0.5 - (context_seq.count('G') + context_seq.count('C')) / len(context_seq)))
            
        return score
        
    def _flexibility_score(self, sequence: str, site: Dict[str, Any]) -> float:
        """Calculate flexibility-based binding score."""
        pos = site['position']
        length = site['length']
        
        # Consider sequence-dependent flexibility
        flex_seq = sequence[max(0, pos-2):pos+length+2]
        
        # AT-rich regions are generally more flexible
        at_content = (flex_seq.count('A') + flex_seq.count('T')) / len(flex_seq)
        
        # Different binding modes prefer different flexibility
        if site['type'] == 'minor_groove':
            return 0.4 + 0.6 * at_content  # More flexible preferred
        elif site['type'] == 'major_groove':
            return 0.6 + 0.4 * (1 - at_content)  # Less flexible preferred
        else:  # intercalation
            return 0.5 + 0.5 * at_content  # Moderate flexibility preferred
            
    def _calculate_binding_propensity(self, scored_sites: List[Dict[str, Any]]) -> float:
        """Calculate overall binding propensity."""
        if not scored_sites:
            return 0.0
            
        # Weight scores by site quality
        weights = np.exp([site['total_score'] for site in scored_sites])
        weights = weights / np.sum(weights)
        
        # Calculate weighted average score
        avg_score = np.sum(weights * [site['total_score'] for site in scored_sites])
        
        # Adjust for number of sites
        site_factor = min(len(scored_sites) / 5, 1.0)  # Cap at 5 sites
        
        return avg_score * site_factor

# Create global instance
binding_analysis = BindingAnalyzer()