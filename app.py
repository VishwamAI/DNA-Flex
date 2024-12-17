from flask import Flask, request, jsonify
from models.analysis import dna_sequence_analysis
from models.dynamics import molecular_dynamics
from models.generative import dna_generation
from models.drug_binding import binding_analysis
from models.mutation_analysis import mutation_effects
from models.nlp_analysis import sequence_nlp
from models.protein_llm import dna_llm

app = Flask(__name__)

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
        llm_analysis = dna_llm.analyze(sequence)

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
        answer = dna_llm.answer_question(question, context)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
