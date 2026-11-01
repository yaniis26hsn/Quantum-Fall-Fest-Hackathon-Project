# 1. Use the BEST circuit and correlation found by our scanner
submission_circuit = best_circuit
submission_correlation = best_corr

# 2. Configuration (Make sure to paste your actual API key!)
API_KEY = "YOUR_TEAM_API_KEY_HERE"  # <--- DON'T FORGET TO CHANGE THIS
SUBMISSION_URL = "http://localhost:3000/api/submissions"
CHALLENGE_ID = "challenge_beginner"  

# 3. Create the payload
payload = {
    "qasm": qasm2.dumps(submission_circuit),
    "challengeId": CHALLENGE_ID,
    "obtainedResult": float(submission_correlation)
}

# 4. Send it to the server!
print(f"🚀 Submitting {submission_circuit.num_qubits}-qubit circuit to the server...")
try:
    response = requests.post(
        SUBMISSION_URL,
        json=payload,
        headers={"X-API-Key": API_KEY},
        timeout=20
    )
    if response.ok:
        data = response.json()
        print("\n✅ SUBMISSION SUCCESSFUL!")
        print(f"   Submission ID: {data['submission']['id']}")
        print(f"   🌟 OFFICIAL SERVER SCORE: {data['submission']['score']:,} POINTS 🌟")
    else:
        print(f"\n❌ Submission failed ({response.status_code}):")
        print("    ", response.json().get("error", response.text))
except Exception as e:
    print(f"\n⚠️ Network error: {e}")