from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Annotated
from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator, EmailStr, ConfigDict
import jwt
from datetime import datetime, timedelta, UTC
import time
import logging
import uuid
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np
import asyncio
from contextlib import asynccontextmanager
import jax
import jax.numpy as jnp

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Cache for background tasks
task_cache = {}

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state."""
    try:
        # Configure protein model
        logger.info("Initializing protein model...")
        protein_model = BioLLM(
            model_type="protein",
            embedding_size=256,
            hidden_size=512,
            num_heads=8,
            num_layers=6
        )
        app.state.protein_model = protein_model
        logger.info("Protein model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize protein model: {str(e)}")
        # Initialize with minimal model to prevent crashes
        app.state.protein_model = BioLLM(
            model_type="protein",
            embedding_size=64,
            hidden_size=128,
            num_heads=4,
            num_layers=2
        )
    yield
    # Cleanup (if needed)
    logger.info("Shutting down protein model...")
    if hasattr(app.state, 'protein_model'):
        del app.state.protein_model

# Initialize FastAPI with metadata
app = FastAPI(
    title="DNA-Flex API",
    description="Advanced DNA sequence analysis and flexibility prediction API",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models with updated Pydantic V2 configuration
class BaseModelWithConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "sequence": "ATGCATGCATGC",
                "description": "Sample DNA sequence"
            }
        }
    )

class Token(BaseModelWithConfig):
    access_token: str
    token_type: str

class TokenData(BaseModelWithConfig):
    username: Optional[str] = None

class User(BaseModelWithConfig):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    roles: List[str] = ["user"]

class UserInDB(User):
    hashed_password: str

class SequenceInput(BaseModelWithConfig):
    sequence: str
    description: Optional[str] = None
    
    @field_validator('sequence')
    @classmethod 
    def validate_sequence(cls, v: str) -> str:
        if not v:
            raise ValueError("Sequence cannot be empty")
        # Allow both DNA and protein sequences
        valid_dna = set('ACGT')
        valid_protein = set('ACDEFGHIKLMNPQRSTVWY')
        if all(c in valid_dna for c in v.upper()):
            return v.upper()
        elif all(c in valid_protein for c in v.upper()):
            return v.upper()
        raise ValueError("Invalid sequence - must be DNA (ACGT) or protein (standard amino acids)")

class AnalysisResult(BaseModelWithConfig):
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class BackgroundTask(BaseModelWithConfig):
    task_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Enhanced error response
class ErrorResponse(BaseModelWithConfig):
    detail: str
    status_code: int
    timestamp: str
    path: Optional[str] = None
    method: Optional[str] = None

# Security utilities
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_timestamp() -> str:
    return datetime.now(UTC).isoformat()

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception

    # Mock user for demo
    user = UserInDB(
        username=token_data.username,
        email=f"{token_data.username}@example.com",
        full_name="Test User",
        disabled=False,
        hashed_password="mock_hashed_password"
    )
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Background task functions
async def process_sequence_analysis(sequence: str, task_id: str):
    try:
        task_cache[task_id]["status"] = "processing"
        # Simulate processing time
        await asyncio.sleep(2)
        
        result = {
            'analysis': dna_sequence_analysis.analyze(sequence),
            'dynamics': molecular_dynamics.simulate(sequence),
            'variations': dna_generation.generate(sequence),
            'binding_sites': binding_analysis.predict(sequence),
            'mutations': mutation_effects.analyze(sequence),
            'nlp_insights': sequence_nlp.analyze(sequence)
        }
        
        task_cache[task_id].update({
            "status": "completed",
            "result": result,
            "completed_at": datetime.now(UTC)
        })
    except Exception as e:
        task_cache[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(UTC)
        })

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail),
            status_code=exc.status_code,
            timestamp=get_current_timestamp(),
            path=request.url.path,
            method=request.method
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            timestamp=get_current_timestamp(),
            path=request.url.path,
            method=request.method
        ).model_dump()
    )

# Web interface endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoints
@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": get_current_timestamp()
    }

@app.get("/info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": "DNA-Flex API",
        "version": API_VERSION,
        "description": "Advanced DNA sequence analysis and flexibility prediction",
        "endpoints": [
            {
                "path": "/predict",
                "method": "POST",
                "description": "Predict DNA sequence properties"
            },
            {
                "path": "/analyze",
                "method": "POST",
                "description": "Analyze DNA sequence"
            },
            {
                "path": "/batch",
                "method": "POST",
                "description": "Batch DNA sequence analysis"
            }
        ]
    }

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=AnalysisResult)
@limiter.limit("5/minute")
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    data: SequenceInput,
    current_user: User = Depends(get_current_active_user)
):
    """
    Predict DNA sequence properties with async background processing
    """
    task_id = str(uuid.uuid4())
    task_cache[task_id] = {
        "request_id": task_id,
        "status": "pending",
        "created_at": datetime.now(UTC)
    }
    
    background_tasks.add_task(
        process_sequence_analysis,
        data.sequence,
        task_id
    )
    
    return AnalysisResult(
        request_id=task_id,
        status="pending",
        created_at=datetime.now(UTC)
    )

@app.get("/tasks/{task_id}", response_model=AnalysisResult)
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of an analysis task"""
    if task_id not in task_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    return AnalysisResult(**task_cache[task_id])

@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze_dna(
    request: Request,
    data: SequenceInput,
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze DNA sequence properties
    """
    try:
        analysis = dna_sequence_analysis.analyze(data.sequence)
        return {
            'sequence': data.sequence,
            'analysis': analysis,
            'timestamp': get_current_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
@limiter.limit("2/minute")
async def batch_analyze(
    request: Request,
    file: UploadFile = File(...),
    max_sequences: int = Query(default=10, le=100),
    current_user: User = Depends(get_current_active_user)
):
    """
    Batch analyze multiple DNA sequences from a file
    """
    try:
        content = await file.read()
        sequences = content.decode().strip().split("\n")
        
        if len(sequences) > max_sequences:
            raise HTTPException(
                status_code=400,
                detail=f"Too many sequences. Maximum allowed: {max_sequences}"
            )
        
        results = []
        for seq in sequences:
            seq = seq.strip()
            if seq:
                try:
                    analysis = dna_sequence_analysis.analyze(seq)
                    results.append({
                        'sequence': seq,
                        'analysis': analysis
                    })
                except Exception as e:
                    results.append({
                        'sequence': seq,
                        'error': str(e)
                    })
        
        return {
            'total_sequences': len(sequences),
            'processed': len(results),
            'results': results,
            'timestamp': get_current_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(current_user: User = Depends(get_current_active_user)):
    """Get API usage statistics"""
    return {
        "total_requests": len(task_cache),
        "completed_tasks": len([t for t in task_cache.values() if t["status"] == "completed"]),
        "failed_tasks": len([t for t in task_cache.values() if t["status"] == "failed"]),
        "timestamp": get_current_timestamp()
    }

@app.post("/protein/predict_structure")
@limiter.limit("5/minute")
async def predict_protein_structure(
    request: Request,
    data: SequenceInput,
    current_user: User = Depends(get_current_active_user)
):
    """Predict protein structure from sequence"""
    try:
        # Call the protein model for structure prediction
        model = request.app.state.protein_model
        logger.info(f"Using protein model type: {model.model_type}")
        result = model.analyze_protein(data.sequence)
        
        # Get domains directly from structure prediction
        domains = result.get('structure', {}).get('domains', [])
        if not domains and len(data.sequence) >= 8:  # Ensure minimum sequence length
            # If no domains found, treat whole sequence as one domain
            domains = [{
                'start': 0,
                'end': len(data.sequence),
                'sequence': data.sequence,
                'type': 'structural'  # Default type for single domain
            }]
            
        return {
            'sequence': data.sequence,
            'secondary_structure': result.get('structure', {}).get('secondary_structure', {}),
            'domains': domains,
            'contacts': result.get('structure', {}).get('contacts', [])
        }
    except ValueError as e:
        logger.error(f"Validation error in predict_structure: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in predict_structure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/predict_function")
@limiter.limit("5/minute")
async def predict_protein_function(
    request: Request,
    data: SequenceInput,
    current_user: User = Depends(get_current_active_user)
):
    """Predict protein function from sequence"""
    try:
        model = request.app.state.protein_model
        logger.info(f"Using protein model type: {model.model_type}")
        result = model.analyze_protein(data.sequence)
        return {
            'sequence': data.sequence,
            'functional_sites': result.get('predicted_functions', {}).get('functional_sites', []),
            'structure_class': result.get('predicted_functions', {}).get('structure_class', {}),
            'predicted_functions': result.get('predicted_functions', {}).get('predicted_functions', {}),
            'localization': result.get('predicted_functions', {}).get('localization', {})
        }
    except ValueError as e:
        logger.error(f"Validation error in predict_function: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in predict_function: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/predict_localization")
@limiter.limit("5/minute")
async def predict_protein_localization(
    request: Request,
    data: SequenceInput, 
    current_user: User = Depends(get_current_active_user)
):
    """Predict protein subcellular localization"""
    try:
        model = request.app.state.protein_model
        result = model.analyze_protein(data.sequence)
        localization = result.get('predicted_functions', {}).get('localization', {})
        
        # Ensure probabilities sum to 1.0
        if localization:
            total = sum(localization.values())
            if total > 0:
                localization = {k: v/total for k, v in localization.items()}
            else:
                localization = {
                    'cytoplasmic': 0.3,
                    'nuclear': 0.2,
                    'membrane': 0.2,
                    'secreted': 0.2,
                    'mitochondrial': 0.1
                }
        
        return {
            'sequence': data.sequence,
            'localization': localization
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/analyze_domains")
@limiter.limit("5/minute")
async def analyze_protein_domains(
    request: Request,
    data: SequenceInput,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze protein domains"""
    try:
        model = request.app.state.protein_model
        result = model.analyze_protein(data.sequence)
        
        # Get domains from structure prediction
        domains = result.get('structure', {}).get('domains', [])
        
        # If no domains found, treat whole sequence as one domain
        if not domains:
            domains = [{
                'start': 0,
                'end': len(data.sequence),
                'sequence': data.sequence,
                'type': 'structural'  # Default type
            }]
            
        return {
            'sequence': data.sequence,
            'domains': [{
                'start': domain.get('start', 0),
                'end': domain.get('end', len(data.sequence)),
                'type': domain.get('type', 'unknown')
            } for domain in domains]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/predict_sites")
@limiter.limit("5/minute") 
async def predict_protein_sites(
    request: Request,
    data: SequenceInput,
    current_user: User = Depends(get_current_active_user)
):
    """Predict protein functional sites"""
    try:
        model = request.app.state.protein_model
        result = model.analyze_protein(data.sequence)
        return {
            'sequence': data.sequence,
            'active_sites': result.get('properties', {}).get('active_sites', []),
            'binding_sites': result.get('properties', {}).get('binding_sites', []),
            'ptm_sites': result.get('properties', {}).get('ptm_sites', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_protein")
@limiter.limit("5/minute")
async def analyze_protein(
    request: Request,
    data: SequenceInput,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze protein sequence"""
    try:
        logger.info(f"Analyzing protein sequence: {data.sequence}")
        model = request.app.state.protein_model
        logger.info(f"Using protein model type: {model.model_type}")
        result = model.analyze_protein(data.sequence)
        
        # Get embeddings and handle errors
        try:
            embeddings = model.generate_embeddings(data.sequence)
            embeddings_list = jax.device_get(embeddings).tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {str(e)}")
            embeddings_list = None
        
        response = {
            'sequence': data.sequence,
            'properties': result.get('properties', {}),
            'structure': result.get('structure', {}),
            'functions': result.get('predicted_functions', {})
        }
        
        if embeddings_list is not None:
            response['embeddings'] = embeddings_list
            
        return response
    except ValueError as e:
        logger.error(f"Validation error in analyze_protein: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_protein: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
