
EWCLv1-M (disorder):
curl -sS -X POST http://localhost:8000/ewcl/predict/ewclv1m   -H 'Content-Type: application/json'   --data-binary @/Users/lucascristino/ewcl-api-clean/samples/sample_payload_ewclv1m.json | tee /Users/lucascristino/ewcl-api-clean/samples/ewclv1m_predict.json

EWCLv1 (disorder):
curl -sS -X POST http://localhost:8000/ewcl/predict/ewclv1   -H 'Content-Type: application/json'   --data-binary @/Users/lucascristino/ewcl-api-clean/samples/sample_payload_ewclv1.json | tee /Users/lucascristino/ewcl-api-clean/samples/ewclv1_predict.json

ClinVar v1-C (plain):
curl -sS -X POST http://localhost:8000/clinvar/v7_3/predict   -H 'Content-Type: application/json'   --data-binary @/Users/lucascristino/ewcl-api-clean/samples/sample_payload_clinvar.json | tee /Users/lucascristino/ewcl-api-clean/samples/clinvar_predict.json

ClinVar v1-C (gated):
curl -sS -X POST http://localhost:8000/clinvar/v7_3/predict_gated   -H 'Content-Type: application/json'   --data-binary @/Users/lucascristino/ewcl-api-clean/samples/sample_payload_clinvar_gated.json | tee /Users/lucascristino/ewcl-api-clean/samples/clinvar_predict_gated.json
