from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, validator

from dnaflex.structure.structure import DnaStructure
from dnaflex.parsers.parser import DnaParser
from dnaflex.flexablity.flexibility import FlexibilityAnalyzer
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

app = FastAPI()

class SequenceInput(BaseModel):
    sequence: str
    
    @validator('sequence')
    def validate_sequence(cls, v):
        if not v:
            raise ValueError("Sequence cannot be empty")
        # Check for valid amino acids (you can adjust the valid chars as needed)
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(c in valid_chars for c in v.upper()):
            raise ValueError("Sequence contains invalid amino acids")
        return v

class QuestionInput(BaseModel):
    question: str
    context: Optional[str] = ""

def analyze_dna_structure(pdb_file: str, modifications: Optional[List[Tuple[str, int]]] = None) -> dict:
    parser = DnaParser()
    structure = parser.parse_pdb(pdb_file)

    if modifications:
        parser.apply_modifications(modifications)

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

    results['center_of_mass'] = structure.calculate_center_of_mass().tolist()
    results['radius_of_gyration'] = structure.calculate_radius_of_gyration()

    return results

@app.post("/predict")
async def predict(data: SequenceInput):
    try:
        sequence = data.sequence
        analysis_result = dna_sequence_analysis.analyze(sequence)
        dynamics_result = molecular_dynamics.simulate(sequence)
        variations = dna_generation.generate(sequence)
        binding_sites = binding_analysis.predict(sequence)
        mutation_analysis_result = mutation_effects.analyze(sequence)
        nlp_insights = sequence_nlp.analyze(sequence)
        llm_analysis = dna_model.analyze(sequence)

        return {
            'analysis': analysis_result,
            'dynamics': dynamics_result,
            'variations': variations,
            'binding_sites': binding_sites,
            'mutations': mutation_analysis_result,
            'nlp_insights': nlp_insights,
            'llm_analysis': llm_analysis
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_question")
async def ask_question(data: QuestionInput):
    try:
        answer = dna_model.answer_question(data.question, data.context)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_protein")
async def analyze_protein(data: SequenceInput):
    try:
        analysis = protein_model.analyze_protein(data.sequence)
        return {
            'properties': analysis['properties'],
            'structure': analysis['structure'],
            'functions': analysis['predicted_functions'],
            'embeddings': analysis['embeddings'].tolist()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/predict_structure")
async def predict_protein_structure(data: SequenceInput):
    try:
        structure = protein_model._predict_protein_structure(data.sequence)
        return structure
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/predict_function")
async def predict_protein_function(data: SequenceInput):
    try:
        functions = protein_model._predict_protein_functions(data.sequence)
        return functions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/predict_localization")
async def predict_protein_localization(data: SequenceInput):
    try:
        localization = protein_model._predict_localization(data.sequence)
        return {"localization": localization}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/analyze_domains")
async def analyze_protein_domains(data: SequenceInput):
    try:
        domains = protein_model._predict_protein_domains(data.sequence)
        return {"domains": domains}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/predict_sites")
async def predict_protein_sites(data: SequenceInput):
    try:
        sites = protein_model._predict_functional_sites(data.sequence)
        return sites
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
