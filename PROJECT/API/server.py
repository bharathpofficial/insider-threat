from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from sse_starlette.sse import EventSourceResponse
import uvicorn
import asyncio
import json
import sys
sys.path.append('../find_anamoly')
from anamoly import AnomalyDetector #type: ignore

# Add shutdown flag
shutdown_event = asyncio.Event()

# Background task handler
async def process_new_records(detector):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        while not shutdown_event.is_set():
            try:
                await loop.run_in_executor(pool, detector.test_records, "/home/windass/Documents/autonomous-cybersecurity/datasets/originalDatasets/processed_combined/processed10percent/merged_dataset_batch_10.csv")
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Error processing records: {e}")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        detector = AnomalyDetector(
            model_path="../outputs/AutoEncoder_Trained_Final_Model.keras",
            threshold=8199062.75
        )
        task = asyncio.create_task(process_new_records(detector))
        app.state.detector = detector
        app.state.background_task = task  # Store task reference
        yield
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    finally:
        shutdown_event.set()
        if hasattr(app.state, 'background_task'):
            app.state.background_task.cancel()
            await asyncio.sleep(1)  # Give time for cleanup


app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def return_success():
    return {"result":"success"}

# Add new SSE endpoint

@app.get("/anomalies/stream")
async def message_stream():
    async def event_generator():
        try:
            while not shutdown_event.is_set():
                all_data = app.state.detector.detection_results
                latest_data = all_data[-1:]
                
                yield {
                    "data": json.dumps({
                        "status": "success",
                        "data": latest_data
                    })
                }
                
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            return
            
    return EventSourceResponse(
        event_generator(),
        ping=15,
        media_type="text/event-stream"
    )


@app.get("/anomalies/latest")
async def get_latest_anomalies():
    try:
        return {
            "status": "success",
            "data": app.state.detector.detection_results[-100:]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anomalies/threats")
async def get_threats():
    try:
        threats = [r for r in app.state.detector.detection_results if r["is_anomaly"]]
        return {
            "status": "success",
            "data": threats[-50:]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

"""
A digital dashboard displaying real-time insider threat detection results. The table features multiple rows, each containing a unique identifier (e.g., ORG-1234, IP addresses, a device name (e.g., pc-8924), and file paths. Rows are color-coded with green indicating safe files and red highlighting potentially dangerous files. A prominent header reads Real time Insider Threat detection against a dark, starry background.

I am using this data from the Back end Server: 
back end is charged with python's FastAPI, it will be providing these data continuously.
 
Typical Data from backend look like this. 
There are two endpoints:
1. /anomalies/threats
   provides top 50 detected anomalies or threats.
2. /anomalies/latest
   provides top 100 recent data with combined threats and non-threats.

json data from backend system looks like this:
{"status":"success","data":[{"timestamp":"2024-12-18T18:09:06.726270","is_anomaly":false,"mse":761552.1875,"original_data":{"date":"2010-02-13 11:41:59","user":2336,"pc":1921,"to_outside":true,"from_outside":false,"has_attachment":true,"is_exe":false,"file_open":true,"file_write":false,"file_copy":false,"file_delete":false,"connected":true,"logon":false}},{"timestamp":"2024-12-18T18:09:06.773624","is_anomaly":false,"mse":817030.25,"original_data":{"date":"2010-02-13 11:42:03","user":2060,"pc":2360,"to_outside":false,"from_outside":false,"has_attachment":true,"is_exe":false,"file_open":true,"file_write":false,"file_copy":false,"file_delete":false,"connected":false,"logon":false}}}

i would like to print original_data  on each row. in the table scrolling.

i would like to use react framework. I would like to do this project in structed way and provide me structed devlopment environment as well.
"""