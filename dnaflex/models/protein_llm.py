"""BioLLM: JAX-based DNA and Protein Language Model using JAX and Haiku."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force JAX to use CPU

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import haiku as hk
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import numpy as np
from functools import partial

class BioLLM:
    """Biology sequence analysis using JAX-based language models.
    
    This class provides functionality for analyzing DNA and protein sequences
    using transformer models implemented with JAX, Flax, and dm-haiku.
    """
    
    def __init__(
        self, 
        model_type: str = "dna",
        embedding_size: int = 128, 
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-4
    ):
        """Initialize the BioLLM model.
        
        Args:
            model_type: Type of biological sequence model ('dna' or 'protein')
            embedding_size: Size of the embedding vectors
            hidden_size: Size of hidden layers in the transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
        """
        self.model_type = model_type.lower()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Set up vocabularies based on model type
        if self.model_type == "dna":
            self.vocab = list("ACGT")
            self.max_seq_length = 1024
            self.special_tokens = ["<PAD>", "<MASK>", "<CLS>", "<SEP>"]
        elif self.model_type == "protein":
            # Standard amino acids plus some special tokens
            self.vocab = list("ACDEFGHIKLMNPQRSTVWY")
            self.max_seq_length = 512
            self.special_tokens = ["<PAD>", "<MASK>", "<CLS>", "<SEP>", "<UNK>"]
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'dna' or 'protein'")
        
        # Create vocabulary mapping
        self.token_to_id = {token: i for i, token in enumerate(self.special_tokens + self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        
        # Initialize model
        self._init_model()
        
    def _init_model(self):
        """Initialize the transformer model using dm-haiku."""
        def _transformer_fn(sequences):
            """Define the transformer model architecture."""
            # Initial embedding layer
            embedding_layer = hk.Embed(
                vocab_size=self.vocab_size,
                embed_dim=self.embedding_size
            )
            
            # Get embeddings
            embeddings = embedding_layer(sequences)
            
            # Add positional encodings
            max_len = self.max_seq_length
            positions = jnp.arange(max_len)[None, :embeddings.shape[1]]
            position_embedding = hk.Embed(
                vocab_size=max_len,
                embed_dim=self.embedding_size
            )(positions)
            
            # Combine token embeddings and position embeddings
            embeddings = embeddings + position_embedding
            
            # Apply transformer layers
            for _ in range(self.num_layers):
                # Multi-head attention
                attn_output = hk.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_size=self.embedding_size // self.num_heads,
                    w_init_scale=2.0
                )(embeddings, embeddings, embeddings)
                
                # Residual connection and layer norm
                embeddings = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True
                )(embeddings + attn_output)
                
                # Feed-forward network
                ff = hk.Sequential([
                    hk.Linear(self.hidden_size * 4),
                    jax.nn.gelu,
                    hk.Linear(self.embedding_size),
                ])
                ff_output = ff(embeddings)
                ff_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff_output)
                
                # Residual connection and layer norm
                embeddings = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True
                )(embeddings + ff_output)
            
            # Final projection for predictions
            output = hk.Linear(self.vocab_size)(embeddings)
            return output
        
        # Transform the model function to pure functions with explicit parameters
        self.model = hk.transform(_transformer_fn)
        
        # Initialize parameters with a dummy input
        dummy_input = jnp.zeros((1, 64), dtype=jnp.int32)
        self.rng_key = jax.random.PRNGKey(42)
        self.params = self.model.init(self.rng_key, dummy_input)
        
        # Set up optimizer
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def tokenize(self, sequence: str) -> jnp.ndarray:
        """Convert a biological sequence to token IDs.
        
        Args:
            sequence: DNA or protein sequence string
            
        Returns:
            Array of token IDs
        """
        # Add special tokens
        sequence = "<CLS>" + sequence + "<SEP>"
        
        # Convert to token IDs
        token_ids = [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0)) 
                    for token in sequence]
        
        # Pad or truncate to max length
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            padding = [self.token_to_id["<PAD>"]] * (self.max_seq_length - len(token_ids))
            token_ids.extend(padding)
            
        return jnp.array([token_ids], dtype=np.int32)  # Use numpy's int32
    
    def batch_tokenize(self, sequences: List[str]) -> jnp.ndarray:
        """Tokenize a batch of sequences.
        
        Args:
            sequences: List of biological sequences
            
        Returns:
            Batch of token IDs as a 2D array
        """
        return jnp.stack([self.tokenize(seq) for seq in sequences])
    
    def generate_embeddings(self, sequence: str) -> jnp.ndarray:
        """Generate embeddings for a sequence.
        
        Args:
            sequence: DNA or protein sequence
            
        Returns:
            Embedding array
        """
        tokens = self.tokenize(sequence)
        
        # Forward pass through the model without final projection
        def _embed_fn(tokens):
            """Embedding function that takes tokens as input."""
            embedding_layer = hk.Embed(
                vocab_size=self.vocab_size,
                embed_dim=self.embedding_size
            )
            return embedding_layer(tokens)
        
        embed_fn = hk.transform(_embed_fn)
        embeddings = embed_fn.apply(self.params, self.rng_key, tokens)
        
        # Return embeddings for the actual sequence (excluding padding)
        actual_length = min(len(sequence) + 2, self.max_seq_length)  # +2 for <CLS> and <SEP>
        return embeddings[0, :actual_length]
    
    def predict_next_tokens(self, sequence: str, num_predictions: int = 1) -> List[str]:
        """Predict the next tokens in a sequence.
        
        Args:
            sequence: DNA or protein sequence
            num_predictions: Number of next tokens to predict
            
        Returns:
            List of predicted tokens
        """
        tokens = self.tokenize(sequence)
        tokens = tokens.reshape(1, -1)  # Add batch dimension
        
        # Forward pass to get logits
        logits = self.model.apply(self.params, self.rng_key, tokens)
        
        # Get the logits for the last position in the sequence
        seq_length = min(len(sequence) + 2, self.max_seq_length)  # +2 for <CLS> and <SEP>
        last_position_logits = logits[0, seq_length - 1]
        
        # Get top-k predictions
        top_indices = jnp.argsort(-last_position_logits)[:num_predictions]
        return [self.id_to_token[idx.item()] for idx in top_indices]
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(self, params, batch, targets, rng_key):
        """Compute loss for a batch of sequences."""
        logits = self.model.apply(params, rng_key, batch)
        
        # Mask out padding tokens
        mask = (targets != self.token_to_id["<PAD>"])
        
        # Cross-entropy loss
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, self.vocab_size))
        
        # Apply mask and compute mean
        masked_loss = jnp.sum(loss * mask) / jnp.sum(mask)
        return masked_loss
    
    def train_step(self, batch: jnp.ndarray, targets: jnp.ndarray):
        """Perform a single training step.
        
        Args:
            batch: Batch of token IDs
            targets: Target token IDs
            
        Returns:
            Loss value
        """
        # Define loss and grad function
        loss_fn = lambda params: self._compute_loss(params, batch, targets, self.rng_key)
        loss_val, grads = jax.value_and_grad(loss_fn)(self.params)
        
        # Update parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        
        return loss_val
    
    def analyze_dna(self, sequence: str) -> Dict[str, Any]:
        """Analyze DNA sequence using the model.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Dictionary of analysis results
        """
        if self.model_type != "dna":
            raise ValueError("This model is not configured for DNA analysis")
            
        # Validate sequence
        if not all(base in "ACGT" for base in sequence):
            raise ValueError("Invalid DNA sequence")
            
        # Generate embeddings
        embeddings = self.generate_embeddings(sequence)
        
        # Compute sequence properties
        properties = self._analyze_sequence_properties(sequence)
        
        # Analyze patterns
        patterns = self._analyze_patterns(sequence)
        
        # Predict functions
        functions = self._predict_functions(sequence)
        
        return {
            'embeddings': jax.device_get(embeddings),
            'properties': properties,
            'patterns': patterns,
            'predicted_functions': functions
        }
    
    def analyze_protein(self, sequence: str) -> Dict[str, Any]:
        """Analyze protein sequence using the model.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary of analysis results
        """
        if self.model_type != "protein":
            raise ValueError("This model is not configured for protein analysis")
            
        # Validate sequence
        if not sequence:
            raise ValueError("Empty sequence provided")
            
        if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence):
            raise ValueError("Invalid protein sequence")
            
        # Generate embeddings
        embeddings = self.generate_embeddings(sequence)
        
        # Compute sequence properties
        properties = self._analyze_protein_properties(sequence)
        
        # Predict structure elements
        structure = self._predict_protein_structure(sequence)
        
        # Predict functions
        functions = self._predict_protein_functions(sequence)
        
        return {
            'embeddings': jax.device_get(embeddings),
            'properties': properties,
            'structure': structure,
            'predicted_functions': functions
        }
    
    def _analyze_sequence_properties(self, sequence: str) -> Dict[str, Any]:
        """Analyze DNA sequence properties."""
        length = len(sequence)
        
        # Composition
        composition = {
            'A': sequence.count('A') / length,
            'T': sequence.count('T') / length,
            'G': sequence.count('G') / length,
            'C': sequence.count('C') / length,
            'gc_content': (sequence.count('G') + sequence.count('C')) / length
        }
        
        # Calculate sequence complexity
        observed_kmers = set()
        possible_kmers = 0
        
        for k in range(1, 4):
            for i in range(len(sequence) - k + 1):
                observed_kmers.add(sequence[i:i+k])
            possible_kmers += min(4**k, len(sequence) - k + 1)
            
        complexity = len(observed_kmers) / possible_kmers if possible_kmers > 0 else 0
        
        # Analyze periodicity
        periodicities = {}
        for period in [2, 3, 4, 10]:  # Common DNA periods
            score = 0
            for i in range(len(sequence) - period):
                if sequence[i] == sequence[i + period]:
                    score += 1
            periodicities[f'period_{period}'] = score / (len(sequence) - period)
        
        return {
            'composition': composition,
            'complexity': complexity,
            'periodicities': periodicities
        }
    
    def _analyze_patterns(self, sequence: str) -> Dict[str, Any]:
        """Analyze DNA sequence patterns."""
        # Find repeats
        repeats = self._find_sequence_repeats(sequence)
        
        # Identify motifs
        motifs = self._identify_motifs(sequence)
        
        # Predict structural elements
        structural_elements = self._predict_structural_elements(sequence)
        
        return {
            'repeats': repeats,
            'motifs': motifs,
            'structural_elements': structural_elements
        }
    
    def _find_sequence_repeats(self, sequence: str) -> Dict[str, Any]:
        """Find repetitive patterns in DNA sequence."""
        repeats = {}
        
        # Direct repeats
        direct_repeats = []
        for length in range(3, 11):
            for i in range(len(sequence) - length):
                pattern = sequence[i:i+length]
                if sequence.count(pattern) > 1:
                    direct_repeats.append({
                        'pattern': pattern,
                        'length': length,
                        'count': sequence.count(pattern)
                    })
                    
        # Inverted repeats
        inverted_repeats = []
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        for length in range(4, 11):
            for i in range(len(sequence) - length):
                pattern = sequence[i:i+length]
                rev_comp = ''.join(complement[b] for b in reversed(pattern))
                if rev_comp in sequence[i+length:]:
                    inverted_repeats.append({
                        'pattern': pattern,
                        'reverse_complement': rev_comp,
                        'length': length
                    })
                    
        repeats['direct'] = direct_repeats
        repeats['inverted'] = inverted_repeats
        return repeats
    
    def _identify_motifs(self, sequence: str) -> Dict[str, Any]:
        """Identify potential functional motifs in DNA."""
        # Common regulatory elements
        regulatory_motifs = []
        motif_patterns = {
            'TATA_box': 'TATAAA',
            'GC_box': 'GGGCGG',
            'CAAT_box': 'CCAAT'
        }
        
        for motif_name, pattern in motif_patterns.items():
            pos = 0
            while True:
                pos = sequence.find(pattern, pos)
                if pos == -1:
                    break
                    
                regulatory_motifs.append({
                    'type': motif_name,
                    'position': pos,
                    'sequence': pattern,
                    'context': sequence[max(0, pos-5):pos+len(pattern)+5]
                })
                pos += 1
        
        # Structural motifs
        structural_motifs = self._find_structural_motifs(sequence)
        
        return {
            'regulatory': regulatory_motifs,
            'structural': structural_motifs
        }
    
    def _find_structural_motifs(self, sequence: str) -> List[Dict[str, Any]]:
        """Find potential structural motifs in DNA."""
        structural_motifs = []
        
        # Look for potential hairpin structures
        min_stem = 4
        max_loop = 8
        
        for i in range(len(sequence) - 2*min_stem - max_loop):
            for loop_size in range(3, max_loop+1):
                stem_size = min_stem
                while i + 2*stem_size + loop_size <= len(sequence):
                    left_stem = sequence[i:i+stem_size]
                    right_stem = sequence[i+stem_size+loop_size:i+2*stem_size+loop_size]
                    
                    # Check if stems are complementary
                    if self._are_complementary(left_stem, right_stem):
                        structural_motifs.append({
                            'type': 'hairpin',
                            'position': i,
                            'stem_size': stem_size,
                            'loop_size': loop_size,
                            'sequence': sequence[i:i+2*stem_size+loop_size]
                        })
                    stem_size += 1
                    
        return structural_motifs
    
    def _are_complementary(self, seq1: str, seq2: str) -> bool:
        """Check if two DNA sequences are complementary."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        rev_seq2 = seq2[::-1]
        return all(base2 == complement.get(base1, 'X') 
                  for base1, base2 in zip(seq1, rev_seq2))
    
    def _predict_structural_elements(self, sequence: str) -> Dict[str, Any]:
        """Predict potential structural elements in DNA."""
        # Predict bendability
        bendability = self._predict_bendability(sequence)
        
        # Predict stability
        stability = self._predict_stability(sequence)
        
        return {
            'bendability': bendability,
            'stability': stability
        }
    
    def _predict_bendability(self, sequence: str) -> float:
        """Predict DNA sequence bendability."""
        # Simple model based on sequence composition
        dinuc_flex = {
            'AA': 0.7, 'AT': 0.8, 'AG': 0.5, 'AC': 0.5,
            'TA': 1.0, 'TT': 0.7, 'TG': 0.7, 'TC': 0.7,
            'GA': 0.5, 'GT': 0.7, 'GG': 0.3, 'GC': 0.4,
            'CA': 0.5, 'CT': 0.7, 'CG': 0.4, 'CC': 0.3
        }
        
        scores = []
        for i in range(len(sequence)-1):
            dinuc = sequence[i:i+2]
            scores.append(dinuc_flex.get(dinuc, 0.5))
            
        return float(np.mean(scores))
    
    def _predict_stability(self, sequence: str) -> float:
        """Predict DNA sequence stability."""
        # Based on GC content and stacking interactions
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Stacking energy contribution
        stacking = {
            'AA': -1.0, 'AT': -0.9, 'AG': -1.5, 'AC': -1.3,
            'TA': -0.9, 'TT': -1.0, 'TG': -1.3, 'TC': -1.1,
            'GA': -1.5, 'GT': -1.3, 'GG': -2.0, 'GC': -2.3,
            'CA': -1.3, 'CT': -1.1, 'CG': -2.3, 'CC': -2.0
        }
        
        stack_energy = 0
        for i in range(len(sequence)-1):
            dinuc = sequence[i:i+2]
            stack_energy += stacking.get(dinuc, -1.0)
            
        # Combine GC content and stacking energy
        stability = 0.6 * gc_content + 0.4 * (-stack_energy / len(sequence))
        return float(np.clip(stability, 0, 1))
    
    def _predict_functions(self, sequence: str) -> Dict[str, Any]:
        """Predict potential functional roles of DNA sequence."""
        # Assess regulatory potential
        regulatory_potential = self._assess_regulatory_potential(sequence)
        
        # Assess coding potential
        coding_potential = self._assess_coding_potential(sequence)
        
        # Predict structural roles
        structural_roles = self._predict_structural_roles(sequence)
        
        return {
            'regulatory_potential': regulatory_potential,
            'coding_potential': coding_potential,
            'structural_roles': structural_roles
        }
    
    def _assess_regulatory_potential(self, sequence: str) -> float:
        """Assess potential regulatory function of DNA sequence."""
        # Look for regulatory motifs and structural features
        regulatory_score = 0.0
        
        # Check for common regulatory elements
        if 'TATA' in sequence:
            regulatory_score += 0.3
        if 'CAAT' in sequence:
            regulatory_score += 0.2
        if 'GGGCGG' in sequence:  # GC box
            regulatory_score += 0.2
            
        # Consider sequence composition
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        if 0.4 <= gc_content <= 0.6:
            regulatory_score += 0.2
            
        return float(min(regulatory_score, 1.0))
    
    def _assess_coding_potential(self, sequence: str) -> float:
        """Assess potential protein coding function of DNA sequence."""
        # Simple coding potential estimation
        coding_score = 0.0
        
        # Check sequence length is multiple of 3
        if len(sequence) % 3 == 0:
            coding_score += 0.2
            
        # Look for start codon
        if sequence.startswith('ATG'):
            coding_score += 0.3
            
        # Look for stop codons
        if any(sequence.endswith(stop) for stop in ['TAA', 'TAG', 'TGA']):
            coding_score += 0.3
            
        # Consider GC content in third positions
        third_positions = sequence[2::3]
        gc_third = (third_positions.count('G') + third_positions.count('C')) / len(third_positions)
        coding_score += 0.2 * gc_third
        
        return float(min(coding_score, 1.0))
    
    def _predict_structural_roles(self, sequence: str) -> Dict[str, float]:
        """Predict potential structural roles of DNA sequence."""
        return {
            'bendability': self._predict_bendability(sequence),
            'stability': self._predict_stability(sequence),
            'protein_binding': self._predict_protein_binding_potential(sequence)
        }
    
    def _predict_protein_binding_potential(self, sequence: str) -> float:
        """Predict potential for protein binding."""
        # Simple model based on sequence features
        score = 0.0
        
        # Look for common binding motifs
        if 'TATA' in sequence:
            score += 0.2
        if 'CAAT' in sequence:
            score += 0.2
        if 'GC' in sequence:
            score += 0.1
            
        # Consider sequence composition
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        if 0.4 <= gc_content <= 0.6:
            score += 0.2
            
        # Consider sequence complexity
        complexity = self._analyze_sequence_complexity(sequence)
        score += 0.3 * complexity
        
        return float(min(score, 1.0))
    
    def _analyze_sequence_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity."""
        # Use linguistic complexity measure
        observed_kmers = set()
        possible_kmers = 0
        
        max_k = min(4, len(sequence))
        for k in range(1, max_k + 1):
            for i in range(len(sequence) - k + 1):
                observed_kmers.add(sequence[i:i+k])
            possible_kmers += min(4**k, len(sequence) - k + 1)
            
        return len(observed_kmers) / possible_kmers if possible_kmers > 0 else 0
    
    # === Protein-specific methods ===
    
    def _analyze_protein_properties(self, sequence: str) -> Dict[str, Any]:
        """Analyze protein sequence properties."""
        # Amino acid composition
        aa_composition = {aa: sequence.count(aa) / len(sequence) for aa in set(sequence)}
        
        # Physicochemical properties
        hydrophobicity = self._calculate_hydrophobicity(sequence)
        
        # Sequence complexity
        complexity = self._calculate_protein_complexity(sequence)
        
        # Secondary structure propensities
        ss_propensities = self._predict_secondary_structure_propensities(sequence)
        
        return {
            'composition': aa_composition,
            'hydrophobicity': hydrophobicity,
            'complexity': complexity,
            'secondary_structure_propensities': ss_propensities
        }
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate average hydrophobicity of protein sequence."""
        if not sequence:
            return 0.0
            
        # Kyte-Doolittle hydrophobicity scale
        hydropathy = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        total_hydropathy = sum(hydropathy.get(aa, 0) for aa in sequence)
        return total_hydropathy / len(sequence)
    
    def _calculate_protein_complexity(self, sequence: str) -> float:
        """Calculate protein sequence complexity."""
        # Use linguistic complexity measure
        observed_kmers = set()
        possible_kmers = 0
        
        max_k = min(3, len(sequence))
        for k in range(1, max_k + 1):
            for i in range(len(sequence) - k + 1):
                observed_kmers.add(sequence[i:i+k])
            possible_kmers += min(20**k, len(sequence) - k + 1)
            
        return len(observed_kmers) / possible_kmers if possible_kmers > 0 else 0
    
    def _predict_secondary_structure_propensities(self, sequence: str) -> Dict[str, List[float]]:
        """Predict secondary structure propensities along protein sequence."""
        # Simple propensity scales based on amino acid preferences
        helix_propensity = {
            'A': 1.41, 'C': 0.66, 'D': 0.98, 'E': 1.39, 'F': 1.13,
            'G': 0.57, 'H': 1.05, 'I': 1.09, 'K': 1.23, 'L': 1.34,
            'M': 1.30, 'N': 0.76, 'P': 0.52, 'Q': 1.27, 'R': 1.21,
            'S': 0.79, 'T': 0.82, 'V': 0.91, 'W': 1.02, 'Y': 0.72
        }
        
        sheet_propensity = {
            'A': 0.72, 'C': 1.40, 'D': 0.80, 'E': 0.26, 'F': 1.33,
            'G': 0.92, 'H': 0.87, 'I': 1.00, 'K': 0.50, 'L': 0.75,
            'M': 0.60, 'N': 0.40, 'P': 0.30, 'Q': 0.45, 'R': 0.55,
            'S': 0.35, 'T': 0.25, 'V': 0.80, 'W': 0.90, 'Y': 0.65
        }
        
        # Calculate raw propensities
        helix = [helix_propensity.get(aa, 0.5) for aa in sequence]
        sheet = [sheet_propensity.get(aa, 0.5) for aa in sequence]
        
        # Normalize propensities to probabilities
        for i in range(len(sequence)):
            total = helix[i] + sheet[i] + 0.5  # 0.5 for coil
            helix[i] /= total
            sheet[i] /= total
        
        return {
            'helix': helix,
            'sheet': sheet
        }
    
    def _predict_protein_structure(self, sequence: str) -> Dict[str, Any]:
        """Predict protein structural elements."""
        # Secondary structure prediction
        ss_pred = self._predict_secondary_structure_propensities(sequence)
        
        # Domain prediction
        domains = self._predict_protein_domains(sequence)
        
        # Disorder prediction
        disorder = self._predict_protein_disorder(sequence)
        
        # Contact prediction (simplified)
        contacts = self._predict_contacts(sequence)
        
        return {
            'secondary_structure': ss_pred,
            'domains': domains,
            'disorder': disorder,
            'contacts': contacts
        }
    
    def _predict_protein_domains(self, sequence: str) -> List[Dict[str, Any]]:
        """Predict potential protein domains."""
        domains = []
        min_domain_size = 30
        
        # Handle sequences shorter than window size
        if len(sequence) < min_domain_size:
            return [{
                'start': 0,
                'end': len(sequence),
                'sequence': sequence,
                'type': 'unknown'
            }]
        
        # Simple domain prediction based on hydrophobicity patterns
        hydrophobicity = [self._aa_hydrophobicity(aa) for aa in sequence]
        
        # Sliding window analysis
        window_size = min(20, len(sequence))
        scores = []
        
        for i in range(len(sequence) - window_size + 1):
            window_score = sum(hydrophobicity[i:i+window_size]) / window_size
            scores.append(window_score)
        
        if not scores:  # Handle case when no scores were calculated
            return domains
            
        # Find regions with consistent hydrophobicity patterns
        current_domain = {'start': 0, 'score': scores[0]}
        
        for i in range(1, len(scores)):
            if abs(scores[i] - current_domain['score']) > 0.5:
                # End of current domain
                if i - current_domain['start'] >= min_domain_size:
                    domains.append({
                        'start': current_domain['start'],
                        'end': i,
                        'sequence': sequence[current_domain['start']:i],
                        'type': 'hydrophobic' if current_domain['score'] > 0 else 'hydrophilic'
                    })
                current_domain = {'start': i, 'score': scores[i]}
        
        # Add final domain if large enough
        if len(scores) - current_domain['start'] >= min_domain_size:
            domains.append({
                'start': current_domain['start'],
                'end': len(scores),
                'sequence': sequence[current_domain['start']:],
                'type': 'hydrophobic' if current_domain['score'] > 0 else 'hydrophilic'
            })
        
        return domains
    
    def _predict_protein_disorder(self, sequence: str) -> List[float]:
        """Predict protein disorder propensity."""
        # Disorder propensity values (normalized 0-1)
        disorder_propensity = {
            'A': 0.06, 'C': 0.02, 'D': 0.19, 'E': 0.18, 'F': 0.02,
            'G': 0.10, 'H': 0.07, 'I': 0.02, 'K': 0.19, 'L': 0.02,
            'M': 0.03, 'N': 0.14, 'P': 0.17, 'Q': 0.16, 'R': 0.18,
            'S': 0.12, 'T': 0.09, 'V': 0.02, 'W': 0.02, 'Y': 0.03
        }
        
        # Calculate disorder scores
        disorder_scores = [disorder_propensity.get(aa, 0.1) for aa in sequence]
        
        # Smooth scores with sliding window
        window_size = 7
        smoothed_scores = []
        for i in range(len(sequence)):
            start = max(0, i - window_size//2)
            end = min(len(sequence), i + window_size//2 + 1)
            window_scores = disorder_scores[start:end]
            smoothed_scores.append(sum(window_scores) / len(window_scores))
            
        return smoothed_scores
    
    def _predict_contacts(self, sequence: str) -> List[Tuple[int, int, float]]:
        """Predict protein residue contacts."""
        contacts = []
        seq_len = len(sequence)
        
        # Simple contact prediction based on amino acid properties
        for i in range(seq_len):
            for j in range(i + 4, seq_len):  # Minimum separation of 4 residues
                # Calculate contact probability based on amino acid properties
                contact_prob = self._calculate_contact_probability(
                    sequence[i], sequence[j]
                )
                
                if contact_prob > 0.5:  # Only include likely contacts
                    contacts.append((i, j, float(contact_prob)))
                    
        return contacts
    
    def _calculate_contact_probability(self, aa1: str, aa2: str) -> float:
        """Calculate probability of contact between two amino acids."""
        # Contact propensity matrix (simplified)
        contact_propensity = {
            ('C', 'C'): 0.9,  # Disulfide bonds
            ('H', 'H'): 0.7,  # Histidine interactions
            ('K', 'D'): 0.8, ('D', 'K'): 0.8,  # Salt bridges
            ('K', 'E'): 0.8, ('E', 'K'): 0.8,
            ('R', 'D'): 0.8, ('D', 'R'): 0.8,
            ('R', 'E'): 0.8, ('E', 'R'): 0.8,
            ('I', 'I'): 0.6, ('L', 'L'): 0.6,  # Hydrophobic interactions
            ('V', 'V'): 0.6, ('F', 'F'): 0.6,
            ('W', 'W'): 0.6, ('Y', 'Y'): 0.6
        }
        
        return contact_propensity.get((aa1, aa2), 0.3)
    
    def _aa_hydrophobicity(self, aa: str) -> float:
        """Get hydrophobicity value for an amino acid."""
        # Normalized hydrophobicity scale
        hydrophobicity = {
            'A': 0.31, 'C': 0.85, 'D': -0.77, 'E': -0.64, 'F': 1.00,
            'G': 0.00, 'H': -0.11, 'I': 0.99, 'K': -0.76, 'L': 0.97,
            'M': 0.74, 'N': -0.60, 'P': -0.07, 'Q': -0.69, 'R': -0.68,
            'S': -0.26, 'T': -0.18, 'V': 0.76, 'W': 0.97, 'Y': 0.02
        }
        return hydrophobicity.get(aa, 0.0)
    
    def _predict_protein_functions(self, sequence: str) -> Dict[str, Any]:
        """Predict potential protein functions."""
        # Functional site prediction
        sites = self._predict_functional_sites(sequence)
        
        # Structural classification
        structure_class = self._predict_structure_class(sequence)
        
        # Function prediction
        functions = self._predict_protein_function_classes(sequence)
        
        # Localization prediction
        localization = self._predict_localization(sequence)
        
        return {
            'functional_sites': sites,
            'structure_class': structure_class,
            'predicted_functions': functions,
            'localization': localization
        }
    
    def _predict_functional_sites(self, sequence: str) -> Dict[str, List[Dict[str, Any]]]:
        """Predict potential functional sites in protein sequence."""
        sites = {
            'active_sites': [],
            'binding_sites': [],
            'ptm_sites': []
        }
        
        # Active site patterns (simplified)
        active_patterns = {
            'catalytic_triad': ['HDS', 'HSL'],
            'zinc_binding': ['HEXXH'],
            'nucleotide_binding': ['GXGXXG']
        }
        
        # Check for active site patterns
        for site_type, patterns in active_patterns.items():
            for pattern in patterns:
                for i in range(len(sequence) - len(pattern) + 1):
                    if self._match_pattern(sequence[i:i+len(pattern)], pattern):
                        sites['active_sites'].append({
                            'type': site_type,
                            'position': i,
                            'pattern': pattern
                        })
                        
        # Predict binding sites based on surface properties
        for i in range(len(sequence) - 5):
            window = sequence[i:i+6]
            if self._is_potential_binding_site(window):
                sites['binding_sites'].append({
                    'position': i,
                    'sequence': window
                })
                
        # Predict post-translational modification sites
        ptm_patterns = {
            'phosphorylation': ['ST'],  # Ser/Thr phosphorylation
            'glycosylation': ['N[^P][ST]'],  # N-glycosylation
            'sumoylation': ['[VILMF]KX[ED]']  # SUMOylation
        }
        
        for mod_type, patterns in ptm_patterns.items():
            for pattern in patterns:
                for i in range(len(sequence) - len(pattern) + 1):
                    if self._match_pattern(sequence[i:i+len(pattern)], pattern):
                        sites['ptm_sites'].append({
                            'type': mod_type,
                            'position': i,
                            'pattern': pattern
                        })
                        
        return sites
    
    def _match_pattern(self, sequence: str, pattern: str) -> bool:
        """Match sequence against pattern with wildcards and character classes."""
        pos = 0  # Position in sequence
        pat_pos = 0  # Position in pattern
        
        while pos < len(sequence):
            if pat_pos >= len(pattern):
                return False
                
            if pattern[pat_pos] == 'X':
                pos += 1
                pat_pos += 1
                continue
                
            if pattern[pat_pos] == '[':
                end = pattern.find(']', pat_pos)
                if end == -1:
                    return False
                    
                chars = pattern[pat_pos + 1:end]
                negation = chars.startswith('^')
                if negation:
                    chars = chars[1:]
                    
                if negation == (sequence[pos] in chars):
                    return False
                    
                pos += 1
                pat_pos = end + 1
                continue
                
            if sequence[pos] != pattern[pat_pos]:
                return False
                
            pos += 1
            pat_pos += 1
        
        return pat_pos >= len(pattern)
    
    def _is_potential_binding_site(self, window: str) -> bool:
        """Predict if a sequence window could be a binding site."""
        # Check for characteristic properties of binding sites
        hydrophobic_count = sum(1 for aa in window if self._aa_hydrophobicity(aa) > 0.5)
        charged_count = sum(1 for aa in window if aa in 'DEKR')
        
        # Binding sites often have mixed hydrophobic and charged residues
        return hydrophobic_count >= 2 and charged_count >= 1
    
    def _predict_structure_class(self, sequence: str) -> Dict[str, float]:
        """Predict protein structure class probabilities."""
        if not sequence:
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'mixed': 1.0  # Default to mixed for empty sequence
            }
        
        # Calculate propensities for different structure classes
        alpha_propensity = sum(self._predict_secondary_structure_propensities(sequence)['helix']) / len(sequence)
        beta_propensity = sum(self._predict_secondary_structure_propensities(sequence)['sheet']) / len(sequence)
        
        # Normalize scores
        total = alpha_propensity + beta_propensity + 0.2  # 0.2 for other
        
        return {
            'alpha': float(alpha_propensity / total),
            'beta': float(beta_propensity / total),
            'mixed': float(0.2 / total)
        }
    
    def _predict_protein_function_classes(self, sequence: str) -> Dict[str, float]:
        """Predict protein function class probabilities."""
        # Simple function prediction based on sequence properties
        functions = {}
        
        # Enzyme prediction
        enzyme_score = 0.0
        if self._has_catalytic_residues(sequence):
            enzyme_score += 0.5
        if self._has_cofactor_binding_motifs(sequence):
            enzyme_score += 0.3
            
        # Binding protein prediction
        binding_score = 0.0
        if self._has_binding_pockets(sequence):
            binding_score += 0.4
        if self._has_recognition_motifs(sequence):
            binding_score += 0.3
            
        # Structural protein prediction
        structural_score = 0.0
        if self._has_repeat_regions(sequence):
            structural_score += 0.4
        if self._has_structural_motifs(sequence):
            structural_score += 0.3
            
        # Normalize scores
        total = enzyme_score + binding_score + structural_score + 0.1  # 0.1 for other
        
        functions['enzyme'] = float(enzyme_score / total)
        functions['binding_protein'] = float(binding_score / total)
        functions['structural'] = float(structural_score / total)
        functions['other'] = float(0.1 / total)
        
        return functions
    
    def _has_catalytic_residues(self, sequence: str) -> bool:
        """Check for presence of common catalytic residues patterns."""
        catalytic_patterns = ['HIS', 'SER', 'ASP', 'GLU', 'CYS']
        return any(pattern in sequence for pattern in catalytic_patterns)
    
    def _has_cofactor_binding_motifs(self, sequence: str) -> bool:
        """Check for presence of cofactor binding motifs."""
        cofactor_motifs = ['GXGXXG', 'CXXC']  # Examples: NAD binding, metal binding
        return any(self._match_pattern(sequence[i:i+len(motif)], motif)
                  for motif in cofactor_motifs
                  for i in range(len(sequence) - len(motif) + 1))
    
    def _has_binding_pockets(self, sequence: str) -> bool:
        """Predict presence of binding pockets."""
        # Simple heuristic: look for alternating hydrophobic/hydrophilic regions
        hydrophobicity = [self._aa_hydrophobicity(aa) for aa in sequence]
        transitions = sum(1 for i in range(len(hydrophobicity)-1)
                        if (hydrophobicity[i] > 0) != (hydrophobicity[i+1] > 0))
        return transitions >= len(sequence) / 10
    
    def _has_recognition_motifs(self, sequence: str) -> bool:
        """Check for presence of recognition motifs."""
        recognition_motifs = ['RGD', 'KDEL', 'NLS', 'NES']
        return any(motif in sequence for motif in recognition_motifs)
    
    def _has_repeat_regions(self, sequence: str) -> bool:
        """Check for presence of repeat regions."""
        # Look for repeating patterns of length 2-10
        for length in range(2, 11):
            for i in range(len(sequence) - 2*length):
                pattern = sequence[i:i+length]
                if sequence.count(pattern) > 1:
                    return True
        return False
    
    def _has_structural_motifs(self, sequence: str) -> bool:
        """Check for presence of structural motifs."""
        structural_motifs = ['PPII', 'GPGG']  # Polyproline, glycine-rich regions
        return any(motif in sequence for motif in structural_motifs)
    
    def _predict_localization(self, sequence: str) -> Dict[str, float]:
        """Predict protein subcellular localization."""
        # Signal peptide features
        has_signal = self._has_signal_peptide(sequence)
        
        # Nuclear localization features
        has_nls = self._has_nuclear_features(sequence)
        
        # Membrane features
        has_transmembrane = self._has_transmembrane_features(sequence)
        
        # Calculate localization probabilities
        localizations = {
            'cytoplasmic': 0.3,
            'nuclear': 0.2,
            'membrane': 0.2,
            'secreted': 0.1,
            'mitochondrial': 0.1
        }
        
        if has_signal:
            localizations['secreted'] += 0.3
            localizations['cytoplasmic'] -= 0.2
            
        if has_nls:
            localizations['nuclear'] += 0.3
            localizations['cytoplasmic'] -= 0.2
            
        if has_transmembrane:
            localizations['membrane'] += 0.3
            localizations['cytoplasmic'] -= 0.2
            
        # Normalize probabilities
        total = sum(localizations.values())
        return {k: float(v/total) for k, v in localizations.items()}
    
    def _has_signal_peptide(self, sequence: str) -> bool:
        """Predict presence of signal peptide."""
        if len(sequence) < 15:  # Allow shorter sequences
            return False
            
        # Check N-terminal region for signal peptide features
        n_region = sequence[:min(8, len(sequence))]
        h_region = sequence[min(8, len(sequence)):min(20, len(sequence))]
        c_region = sequence[min(20, len(sequence)):min(30, len(sequence))]
        
        # N-region should be positively charged
        n_positive = sum(1 for aa in n_region if aa in 'KR')
        
        # H-region should be hydrophobic
        h_hydrophobic = sum(1 for aa in h_region if self._aa_hydrophobicity(aa) > 0.5)
        
        # C-region should have small, uncharged residues
        c_small = sum(1 for aa in c_region if aa in 'AGSPT')
        
        # More lenient thresholds
        return (n_positive >= 1 and  # Was 2
                (h_hydrophobic / len(h_region) if h_region else 0) >= 0.5 and  # Was 8
                (c_small / len(c_region) if c_region else 0) >= 0.3)  # Was 4
    
    def _has_nuclear_features(self, sequence: str) -> bool:
        """Check for nuclear localization features."""
        # Classic NLS patterns
        nls_patterns = [
            'K[KR][KR]R',  # Classic monopartite NLS
            'KR[KR]',      # Simplified NLS
            'PKKKRKV',     # SV40 large T antigen NLS
            'PAAKRVKLD'    # c-Myc NLS
        ]
        
        # Check for NLS patterns
        for pattern in nls_patterns:
            for i in range(len(sequence) - len(pattern) + 1):
                if self._match_pattern(sequence[i:i+len(pattern)], pattern):
                    return True
                    
        # Check for high basic amino acid content in windows
        window_size = 8
        basic_threshold = 0.5
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            basic_fraction = sum(1 for aa in window if aa in 'KR') / window_size
            if basic_fraction >= basic_threshold:
                return True
                
        return False
    
    def _has_transmembrane_features(self, sequence: str) -> bool:
        """Check for transmembrane features."""
        # Look for hydrophobic stretches characteristic of transmembrane regions
        window_size = 20
        min_hydrophobic = 12  # Reduced from 15 for better sensitivity
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            hydrophobic_count = sum(1 for aa in window if self._aa_hydrophobicity(aa) > 0.5)
            if hydrophobic_count >= min_hydrophobic:
                return True
                
        return False
