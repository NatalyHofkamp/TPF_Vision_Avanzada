#!/bin/bash

# Script helper para ejecutar un experimento de entrenamiento y evaluación.

python training/finetune.py
python evaluation/evaluate.py
