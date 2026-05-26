"""
Preprocesamiento de datasets de proteínas.

Este script:
1. Carga las estructuras PDB descargadas
2. Extrae coordenadas atómicas
3. Genera tensores para entrenamiento
4. Guarda datos procesados

IMPORTANTE: Splits por grupos
================================
Los nuevos splits (train.csv, val.csv, test.csv) usan estrategia GROUPED:
- Agrupa estructuras de la MISMA kinasa en un único split
- EVITA leakage: el modelo no ve la misma proteína en train y test
- Mejor generalización a kinasas completamente nuevas

Workflow:
1. download_klifs_dataset.py → genera data/metadata/kinase_labels.csv
2. download_klifs_dataset.py → genera data/splits/{train,val,test}.csv (GROUPED)
3. preprocess_dataset.py (este script) → genera tensores para TODAS las estructuras
4. training/finetune.py → FILTRA tensores usando los splits CSVs

Los splits CSVs se GENERAN ANTES del preprocesamiento y se USAN DESPUÉS.
No los elimines: contienen el mapeo de qué estructuras van a dónde.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from Bio.PDB import PDBParser, PDBIO, Select
except ImportError:
    raise ImportError("Instala BioPython: pip install biopython")

try:
    import torch
except ImportError:
    raise ImportError("Instala PyTorch: pip install torch")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProteinPreprocessor:
    """Procesa estructuras PDB a tensores."""

    def __init__(
        self,
        metadata_csv: Path = Path("data/metadata/kinase_labels.csv"),
        output_dir: Path = Path("data/processed")
    ):
        self.metadata_csv = Path(metadata_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata no encontrada: {self.metadata_csv}")

        self.df = pd.read_csv(self.metadata_csv)
        self.pdb_parser = PDBParser(QUIET=True)

        logger.info(f"Dataset cargado: {len(self.df)} estructuras")

    def extract_ca_coordinates(self, pdb_file: Path) -> Optional[np.ndarray]:
        """
        Extrae coordenadas de átomos Cα (carbono alfa).

        Cα es el átomo central de cada aminoácido en la cadena proteica.
        Estos átomos definen la estructura 3D de la proteína.
        """
        try:
            structure = self.pdb_parser.get_structure('protein', pdb_file)

            ca_coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca = residue['CA']
                            ca_coords.append(ca.coord)

            if not ca_coords:
                logger.warning(f"No Cα atoms encontrados en {pdb_file}")
                return None

            return np.array(ca_coords, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error procesando {pdb_file}: {e}")
            return None

    def compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancias entre átomos Cα.

        La matriz de distancias captura las relaciones espaciales entre
        residuos aminoácidos.
        """
        distances = np.zeros((len(coords), len(coords)), dtype=np.float32)

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def get_label(self, conformation: str) -> int:
        """Convierte conformación a label numérico."""
        return 1 if conformation == 'active' else 0

    def process_batch(self, batch_indices: list) -> list:
        """Procesa un batch de estructuras."""
        processed_items = []

        for idx in batch_indices:
            row = self.df.iloc[idx]
            pdb_path = Path(row['filepath'])

            if not pdb_path.exists():
                logger.warning(f"Archivo no encontrado: {pdb_path}")
                continue

            # Extraer coordenadas
            ca_coords = self.extract_ca_coordinates(pdb_path)
            if ca_coords is None:
                continue

            # Computar matriz de distancias
            distance_matrix = self.compute_distance_matrix(ca_coords)

            # Preparar datos
            item = {
                'pdb_id': row['pdb_id'],
                'kinase_name': row['kinase_name'],
                'label': self.get_label(row['conformational_state']),
                'ca_coords': torch.from_numpy(ca_coords),
                'distance_matrix': torch.from_numpy(distance_matrix),
                'sequence_length': len(ca_coords),
                'resolution': row['resolution'] if pd.notna(row['resolution']) else 0.0,
            }

            processed_items.append(item)

        return processed_items

    def run(self, save_tensors: bool = True):
        """Ejecuta el preprocesamiento completo."""
        logger.info(f"Procesando {len(self.df)} estructuras...")

        batch_size = 32
        all_items = []

        for i in tqdm(range(0, len(self.df), batch_size), desc="Procesando"):
            batch_indices = range(i, min(i + batch_size, len(self.df)))
            items = self.process_batch(batch_indices)
            all_items.extend(items)

        logger.info(f"Procesadas exitosamente: {len(all_items)} estructuras")

        if save_tensors:
            self._save_tensors(all_items)
            self._validate_preprocessing()

        return all_items

    def _save_tensors(self, items: list):
        """Guarda tensores procesados."""
        for item in tqdm(items, desc="Guardando tensores"):
            pdb_id = item['pdb_id']
            tensor_dir = self.output_dir / pdb_id

            tensor_dir.mkdir(parents=True, exist_ok=True)

            torch.save(item['ca_coords'], tensor_dir / 'ca_coords.pt')
            torch.save(item['distance_matrix'], tensor_dir / 'distance_matrix.pt')

            metadata = {
                'label': item['label'],
                'sequence_length': item['sequence_length'],
                'resolution': item['resolution'],
                'kinase_name': item['kinase_name'],
            }

            torch.save(metadata, tensor_dir / 'metadata.pt')

        logger.info(f"Tensores guardados en {self.output_dir}")

    def _validate_preprocessing(self):
        """
        Valida que los tensores preprocesados sean consistentes con metadata.
        """
        logger.info("\n" + "="*80)
        logger.info("VALIDACIÓN DE PREPROCESAMIENTO")
        logger.info("="*80)
        
        # Verificar que existen los splits CSVs
        splits_dir = Path("data/splits")
        if not splits_dir.exists():
            logger.warning("⚠️  No se encontraron splits en data/splits/")
            logger.warning("   Ejecute download_klifs_dataset.py primero")
            return
        
        # Cargar splits
        splits_info = {}
        for split_name in ['train', 'val', 'test']:
            split_path = splits_dir / f'{split_name}.csv'
            if split_path.exists():
                split_df = pd.read_csv(split_path)
                splits_info[split_name] = split_df
                logger.info(f"\n✅ Encontrado {split_name}.csv: {len(split_df)} estructuras")
                
                # Verificar que PDB IDs del split tienen tensores
                tensor_dir = self.output_dir
                missing_tensors = []
                for pdb_id in split_df['pdb_id']:
                    if not (tensor_dir / pdb_id).exists():
                        missing_tensors.append(pdb_id)
                
                if missing_tensors:
                    logger.warning(f"   ⚠️  Faltan tensores para {len(missing_tensors)} PDBs")
                else:
                    logger.info(f"   ✓ Todos los tensores disponibles")
            else:
                logger.warning(f"   ⚠️  No encontrado {split_name}.csv")
        
        # Información general
        logger.info(f"\n📊 RESUMEN GENERAL")
        logger.info(f"   Metadata: {len(self.df)} estructuras")
        logger.info(f"   Directorio de tensores: {self.output_dir}")
        
        if splits_info:
            total_in_splits = sum(len(df) for df in splits_info.values())
            logger.info(f"   Estructuras en splits: {total_in_splits}")
            if total_in_splits == len(self.df):
                logger.info(f"   ✓ Cobertura completa: 100%")
            else:
                logger.info(f"   ⚠️  Cobertura: {total_in_splits}/{len(self.df)} ({total_in_splits/len(self.df)*100:.1f}%)")
        
        logger.info("="*80 + "\n")

    def create_training_dataloader(self, batch_size: int = 32):
        """Crea dataloader para entrenamiento."""
        try:
            from torch.utils.data import DataLoader, Dataset
        except ImportError:
            logger.error("PyTorch DataLoader no disponible")
            return None

        class ProteinDataset(Dataset):
            def __init__(self, processed_items):
                self.items = processed_items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                item = self.items[idx]
                return {
                    'ca_coords': item['ca_coords'],
                    'distance_matrix': item['distance_matrix'],
                    'label': item['label'],
                    'pdb_id': item['pdb_id'],
                }

        processed_items = self.run(save_tensors=False)
        dataset = ProteinDataset(processed_items)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    """Ejecuta preprocesamiento del dataset."""
    try:
        logger.info("="*80)
        logger.info("PREPROCESAMIENTO DE DATASET")
        logger.info("="*80)
        logger.info("\nNOTA: Este script procesa TODAS las estructuras.")
        logger.info("Los splits (train/val/test) se GENERARON en el paso anterior.")
        logger.info("Los splits CSVs se USARÁN en el entrenamiento para filtrar datos.\n")
        
        preprocessor = ProteinPreprocessor()
        items = preprocessor.run(save_tensors=True)

        logger.info(f"Preprocesamiento completado: {len(items)} estructuras procesadas")

    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

