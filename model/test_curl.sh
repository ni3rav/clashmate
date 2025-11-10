#!/bin/bash
# Test curl command for ClashMate API /predict endpoint

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "Troop",
    "hitpoints": 1766,
    "damage": 202,
    "hitSpeed": 1.2,
    "dps": 168,
    "range": 1.2,
    "count": 1
  }'

