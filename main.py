import cohere
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Control Validator API with Cohere is working."}

@app.post("/validate-control")
async def validate_control(
    process: str = Query(...),
    subprocess: str = Query(...),
    risk: str = Query(...),
    frequency: str = Query(...),
    risk_description: str = Query(...),
    control: str = Query(...),
    control_description: str = Query(...)
):
    try:
        prompt = f"""
You are an expert internal control validator.

Given the following:
Process: {process}
Subprocess: {subprocess}
Risk: {risk}
Frequency: {frequency}
Risk Description: {risk_description}
Control: {control}
Control Description: {control_description}

Your task:
1. Determine if the control description is correct (sufficiently mitigates the risk). Respond with either 'Correct' or 'Incorrect'.
2. If 'Correct': Say "No need to make changes in the control description."
3. If 'Incorrect': Suggest a corrected version of the control description.

Format your response exactly like this:
Correctness: <Correct/Incorrect>
Correction: <Your message or improved control description>
"""

        response = co.generate(
            model='command-xlarge',
            prompt=prompt,
            max_tokens=400,
            temperature=0.7
        )

        output = response.generations[0].text.strip()

        # Optional: Parse correctness and correction from the response
        if "Correctness:" in output and "Correction:" in output:
            parts = output.split("Correction:")
            correctness_line = parts[0].replace("Correctness:", "").strip()
            correction_line = parts[1].strip()
            return {
                "validity": correctness_line,
                "recommendation": correction_line
            }
        else:
            return {"result": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
