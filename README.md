# Conditional Protein Generation with Flow Matching

Proyecto de investigación orientado a generación condicionada de estructuras proteicas utilizando modelos basados en Flow Matching.

## Objetivo

Explorar técnicas de conditioning sobre modelos generativos de proteínas para generar conformaciones estructurales asociadas a proteínas relacionadas con cáncer.

## Primera etapa

- Fine-tuning sobre proteínas kinasas.
- Conditioning activo/inactivo.
- Evaluación estructural mediante RMSD y diversidad conformacional.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset esperado

Colocar:

* archivos `.pdb`
* metadata CSV
* labels activo/inactivo

Dentro de:

```text
/data/raw/
```

---

## Entrenamiento

```bash
python training/finetune.py
```

---

## Evaluación

```bash
python evaluation/evaluate.py
```

---

## Estructura de conditioning

Actualmente el conditioning se realiza mediante labels:

* 0 = inactive
* 1 = active

Posteriormente puede extenderse a:

* flexibilidad
* mutation type
* pocket openness
* conformational states
