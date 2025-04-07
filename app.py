"""DNA-Flex main application module."""

from pathlib import Path
import argparse
from typing import List, Optional, Tuple
from flask import Flask, request, jsonify

from dnaflex.structure.structure import DnaStructure
from dnaflex.parsers.parser import DnaParser
from dnaflex.flexibility import FlexibilityAnalyzer
from dnaflex.models.analysis import analyze as dna_sequence_analysis
from dnaflex.models.dynamics import molecular_dynamics
from dnaflex.models.generative import dna_generation
from dnaflex.models.drug_binding import binding_analysis
from dnaflex.models.mutation_analysis import mutation_effects
from dnaflex.models.nlp_analysis import sequence_nlp
from dnaflex.models.dna_llm import BioLLM

# Initialize models
dna_model = BioLLM(model_type='dna')
protein_model = BioLLM(model_type='protein')

app = Flask(__name__)

def analyze_dna_structure(pdb_file: str,
                        modifications: Optional[List[Tuple[str, int]]] = None) -> dict:
    """Analyze DNA structure and flexibility.
    
    Args:
        pdb_file: Path to PDB file containing DNA structure
        modifications: Optional list of (modification_type, position) tuples
        
    Returns:
        Dictionary containing analysis results
    """
    # Parse structure
    parser = DnaParser()
    structure = parser.parse_pdb(pdb_file)
    
    # Apply any modifications
    if modifications:
        parser.apply_modifications(modifications)
    
    # Analyze flexibility
    analyzer = FlexibilityAnalyzer(structure)
    results = {}
    
    for chain in structure.chains:
        chain_results = {
            'sequence': chain.sequence,
            'length': len(chain),
            'flexibility_scores': analyzer.predict_flexibility(chain).tolist(),
            'flexible_regions': analyzer.identify_flexible_regions(chain),
            'base_step_parameters': {
                k: v.tolist() 
                for k, v in analyzer.calculate_base_step_parameters(chain).items()
            }
        }
        results[chain.chain_id] = chain_results
        
    # Add global structure properties
    results['center_of_mass'] = structure.calculate_center_of_mass().tolist()
    results['radius_of_gyration'] = structure.calculate_radius_of_gyration()
    
    return results

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for DNA sequence analysis and prediction
    """
    data = request.get_json()
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    try:
        # Analyze DNA sequence
        analysis_result = dna_sequence_analysis.analyze(sequence)

        # Predict molecular dynamics
        dynamics_result = molecular_dynamics.simulate(sequence)

        # Generate variations
        variations = dna_generation.generate(sequence)

        # Analyze binding sites
        binding_sites = binding_analysis.predict(sequence)

        # Analyze mutations
        mutation_analysis = mutation_effects.analyze(sequence)

        # NLP analysis of sequence patterns
        nlp_insights = sequence_nlp.analyze(sequence)

        # LLM-based analysis
        llm_analysis = dna_model.analyze(sequence)

        return jsonify({
            'analysis': analysis_result,
            'dynamics': dynamics_result,
            'variations': variations,
            'binding_sites': binding_sites,
            'mutations': mutation_analysis,
            'nlp_insights': nlp_insights,
            'llm_analysis': llm_analysis
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """
    Endpoint for DNA-related questions using LLM
    """
    data = request.get_json()
    question = data.get('question')
    context = data.get('context', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        answer = dna_model.answer_question(question, context)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_protein', methods=['POST'])
def analyze_protein():
    """Endpoint for protein sequence analysis."""
    data = request.get_json()
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    try:
        # Perform protein sequence analysis
        analysis = protein_model.analyze_protein(sequence)
        
        return jsonify({
            'properties': analysis['properties'],
            'structure': analysis['structure'],
            'functions': analysis['predicted_functions'],
            'embeddings': analysis['embeddings'].tolist()
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/protein/predict_structure', methods=['POST'])
def predict_protein_structure():
    """Endpoint for protein structure prediction."""
    data = request.get_json()
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    try:
        structure = protein_model._predict_protein_structure(sequence)
        return jsonify(structure)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/protein/predict_function', methods=['POST'])
def predict_protein_function():
    """Endpoint for protein function prediction."""
    data = request.get_json()
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    try:
        functions = protein_model._predict_protein_functions(sequence)
        return jsonify(functions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/protein/predict_localization', methods=['POST'])
def predict_protein_localization():
    """Endpoint for protein subcellular localization prediction."""
    data = request.get_json()
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    try:
        localization = protein_model._predict_localization(sequence)
        return jsonify({'localization': localization})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/protein/analyze_domains', methods=['POST'])
def analyze_protein_domains():
    """Endpoint for protein domain analysis."""
    data = request.get_json()
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    try:
        domains = protein_model._predict_protein_domains(sequence)
        return jsonify({'domains': domains})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/protein/predict_sites', methods=['POST'])
def predict_protein_sites():
    """Endpoint for protein functional sites prediction."""
    data = request.get_json()
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    try:
        sites = protein_model._predict_functional_sites(sequence)
        return jsonify(sites)
    except Exception as e:
        return jsonify({'error': str(e)}), 500