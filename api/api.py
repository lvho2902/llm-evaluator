import json
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
from database.data_provider import DataProvider
from fastapi.middleware.cors import CORSMiddleware
from evaluation.evaluator import run_evaluator

app = FastAPI()

# Define CORS origins
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Auto Evaluator!"}

@app.get("/experiments", response_model=List[Dict[str, int]])
async def get_all_experiments():
    try:
        experiments = DataProvider.get_all_experiments()
        return JSONResponse(content=experiments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluations/{experiment_id}", response_model=List[Dict[str, int]])
async def get_evaluations_by_experiment_id(experiment_id: int):
    try:
        evaluations = DataProvider.get_evaluations_by_experiment_id(experiment_id)
        if not evaluations:
            raise HTTPException(status_code=404, detail="No evaluations found for this experiment ID")
        return JSONResponse(content=evaluations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/evaluations", response_model=List[Dict[str, int]])
async def get_evaluations():
    try:
        evaluations = DataProvider.get_evaluations()
        if not evaluations:
            raise HTTPException(status_code=404, detail="No evaluations found")
        return JSONResponse(content=evaluations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/evaluator-stream")
async def create_response(
    files: List[UploadFile] = File(...),
    test_dataset: str = Form("[]"),
    number_of_question: int = Form(5),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    split_method: str = Form("RecursiveTextSplitter"),
    retriever_type: str = Form("similarity-search"),
    embedding_provider: str = Form("OpenAI"),
    model: str = Form("gpt-3.5-turbo"),
    evaluator_model: str = Form("openai"),
    num_neighbors: int = Form(3),
):
    test_dataset = json.loads(test_dataset)
    try:
        response_generator = run_evaluator(
            files=files,
            test_dataset=test_dataset,
            number_of_question=number_of_question,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_method=split_method,
            retriever_type=retriever_type,
            embedding_provider=embedding_provider,
            model=model,
            evaluator_model=evaluator_model,
            num_neighbors=num_neighbors,
        )
        return EventSourceResponse(
            response_generator,
            headers={"Content-Type": "text/event-stream", "Connection": "keep-alive", "Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)