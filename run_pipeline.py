"""
run_pipeline.py — Full pipeline: CV → RAG → result

Usage:
    python run_pipeline.py path/to/image.jpg
"""

import sys
import json
import httpx

CV_URL = "http://localhost:8001/inspection/inspect"
RAG_URL = "http://localhost:8000/inspect"


def run(image_path: str):
    print(f"\n[1] Sending image to CV system...")
    with open(image_path, "rb") as f:
        cv_response = httpx.post(CV_URL, files={"image": f}, timeout=60)
    if cv_response.status_code != 200:
        print(f"    CV error {cv_response.status_code}: {cv_response.text}")
        sys.exit(1)
    cv_result = cv_response.json()
    print(f"    Defect detected: {cv_result['vision_result']['defect_detected']}")
    if cv_result['vision_result'].get('primary_detection'):
        d = cv_result['vision_result']['primary_detection']
        print(f"    Defect: {d.get('defect_name')} | Confidence: {d.get('confidence')} | Severity: {d.get('severity_hint')}")

    print(f"\n[2] Sending to RAG system...")
    rag_response = httpx.post(RAG_URL, json=cv_result, timeout=300)
    rag_response.raise_for_status()
    result = rag_response.json()

    print(f"\n{'='*50}")
    print(f"  Defect explanation : {result['defect_explanation']}")
    print(f"  Severity           : {result['severity_assessment']}")
    print(f"  Action             : {result['recommended_action'].upper()}")
    print(f"  Justification      : {result['justification']}")
    print(f"  References         : {', '.join(result['sop_references'])}")
    print(f"  Confidence         : {result['confidence']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py path/to/image.jpg")
        sys.exit(1)
    run(sys.argv[1])
