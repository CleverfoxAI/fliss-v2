# Railway runs `release` during deploy; a non-zero exit aborts the deploy and
# keeps the current version live. This gates deploys on the model-resolution
# regression tests so a retired/invalid model can't reach production.
release: python test_model_resolution.py && python test_resilience.py && python test_sessions.py && python test_flows.py
web: uvicorn main:app --host 0.0.0.0 --port $PORT
