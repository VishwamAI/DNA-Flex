"""DNA mutation effects analysis module."""

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

class MutationAnalyzer:
    """Analyze effects of DNA mutations."""
    
    def __init__(self):
        # Mutation type weights for impact scoring
        self.mutation_weights = {
            'transition': 0.6,    # A<->G, C<->T
            'transversion': 0.8,  # All other substitutions
            'insertion': 0.9,
            'deletion': 0.9
        }
        
        # Define transition pairs
        self.transitions = {
            'A': 'G', 'G': 'A',
            'C': 'T', 'T': 'C'
        }
        
    def analyze(self, sequence: str) -> Dict[str, Any]:
        """Analyze potential mutation effects in DNA sequence."""
        # Analyze potential mutations at each position
        mutation_effects = self._analyze_position_effects(sequence)
        
        # Identify mutation hotspots
        hotspots = self._identify_hotspots(sequence, mutation_effects)
        
        # Analyze sequence context effects
        context_effects = self._analyze_sequence_context(sequence)
        
        return {
            'position_effects': mutation_effects,
            'hotspots': hotspots,
            'context_effects': context_effects,
            'overall_stability': self._calculate_overall_stability(sequence)
        }
        
    def _analyze_position_effects(self, sequence: str) -> List[Dict[str, Any]]:
        """Analyze mutation effects at each position."""
        effects = []
        
        for i, base in enumerate(sequence):
            pos_effects = {
                'position': i,
                'reference': base,
                'mutations': []
            }
            
            # Analyze each possible mutation
            for mut_base in 'ACGT':
                if mut_base != base:
                    effect = self._score_mutation(sequence, i, mut_base)
                    pos_effects['mutations'].append(effect)
                    
            effects.append(pos_effects)
            
        return effects
        
    def _score_mutation(self, sequence: str, position: int, 
                       mutant_base: str) -> Dict[str, Any]:
        """Score the effect of a specific mutation."""
        ref_base = sequence[position]
        
        # Determine mutation type
        if mutant_base in self.transitions.get(ref_base, ''):
            mut_type = 'transition'
        else:
            mut_type = 'transversion'
            
        # Calculate base impact score
        impact = self.mutation_weights[mut_type]
        
        # Adjust for sequence context
        context_factor = self._context_impact(sequence, position)
        impact *= context_factor
        
        return {
            'mutant': mutant_base,
            'type': mut_type,
            'impact': impact,
            'stability_change': self._estimate_stability_change(sequence, position, mutant_base)
        }
        
    def _context_impact(self, sequence: str, position: int) -> float:
        """Calculate impact factor based on sequence context."""
        # Get local sequence context
        start = max(0, position - 2)
        end = min(len(sequence), position + 3)
        context = sequence[start:end]
        
        # Consider GC content in context
        gc_content = (context.count('G') + context.count('C')) / len(context)
        
        # Higher impact in GC-rich regions (more conserved typically)
        return 0.8 + 0.4 * gc_content
        
    def _identify_hotspots(self, sequence: str, 
                          position_effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify mutation hotspots in sequence."""
        hotspots = []
        window_size = 5
        
        # Calculate average impact scores in sliding windows
        for i in range(len(sequence) - window_size + 1):
            window_effects = position_effects[i:i+window_size]
            avg_impact = np.mean([
                max(mut['impact'] for mut in pos['mutations'])
                for pos in window_effects
            ])
            
            if avg_impact > 0.75:  # Threshold for hotspot identification
                hotspots.append({
                    'start': i,
                    'end': i + window_size,
                    'sequence': sequence[i:i+window_size],
                    'average_impact': avg_impact
                })
                
        return hotspots
        
    def _analyze_sequence_context(self, sequence: str) -> Dict[str, Any]:
        """Analyze sequence context effects on mutation impact."""
        context_effects = defaultdict(list)
        
        # Analyze 3-base contexts
        for i in range(len(sequence)-2):
            context = sequence[i:i+3]
            impact = self._context_impact(sequence, i+1)
            context_effects[context].append(impact)
            
        # Calculate average impact for each context
        return {
            context: np.mean(impacts)
            for context, impacts in context_effects.items()
        }
        
    def _estimate_stability_change(self, sequence: str, 
                                 position: int, mutant: str) -> float:
        """Estimate change in sequence stability due to mutation."""
        # Simple stability model based on base stacking
        stability_change = 0.0
        
        # Consider effects on neighboring base pairs
        for i in range(max(0, position-1), min(len(sequence), position+2)):
            if i == position:
                continue
            
            # Base stacking contribution
            if abs(i - position) == 1:
                ref_pair = sequence[min(i, position)] + sequence[max(i, position)]
                mut_pair = (mutant + sequence[i] if i > position 
                          else sequence[i] + mutant)
                
                # Stronger base stacking for GC pairs
                ref_strength = (ref_pair.count('G') + ref_pair.count('C')) / 2
                mut_strength = (mut_pair.count('G') + mut_pair.count('C')) / 2
                
                stability_change += mut_strength - ref_strength
                
        return np.clip(stability_change, -1.0, 1.0)
        
    def _calculate_overall_stability(self, sequence: str) -> float:
        """Calculate overall sequence stability score."""
        # Based on GC content and repeat sequences
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Look for destabilizing repeat sequences
        repeat_penalty = 0
        for i in range(len(sequence)-3):
            if sequence[i:i+4] in sequence[i+4:]:
                repeat_penalty += 0.1
                
        stability = 0.6 * gc_content + 0.4 * (1 - min(repeat_penalty, 0.8))
        return np.clip(stability, 0, 1)

# Create global instance
mutation_effects = MutationAnalyzer()