"""
Descargar y procesar dataset de kinasas desde KLIFS API.

KLIFS (Kinase-Ligand Interaction Fingerprints and Structures) es una base de datos
especializada en estructuras 3D de kinasas. Cada kinasa puede estar en dos conformaciones:
- ACTIVA: configuración donde el sitio activo está listo para catálisis
- INACTIVA: configuración bloqueada o replegada

Conformaciones se identifican por:
- DFG-state: "DFG-in" (activa) vs "DFG-out" (inactiva) - región de aspartico-fenilalanina-glicina
- alphaC-state: posición de la hélice alfa-C
- ligand_present: si hay moléculas pequeñas unidas
"""

import requests
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import logging

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



CANCER_KINASES = {
    # Receptor Tyrosine Kinases (RTKs)
    'EGFR': 'receptor tyrosine kinase',
    'ERBB2': 'receptor tyrosine kinase',   # HER2
    'FGFR1': 'receptor tyrosine kinase',
    'KIT': 'receptor tyrosine kinase',
    'MET': 'receptor tyrosine kinase',
    'PDGFRA': 'receptor tyrosine kinase',
    'ALK': 'receptor tyrosine kinase',

    # Non-receptor Tyrosine Kinases
    'ABL1': 'non-receptor tyrosine kinase',

    # Serine/Threonine Kinases
    'BRAF': 'serine/threonine kinase',
    'AKT1': 'serine/threonine kinase',

    # Cyclin-dependent kinases
    'CDK4': 'cyclin-dependent kinase',
    'CDK6': 'cyclin-dependent kinase',

    # Lipid kinase
    'PIK3CA': 'phosphoinositide 3-kinase',
}

class KLIFSDownloader:
    """Descarga y procesa estructuras desde KLIFS API."""

    BASE_URL = "https://klifs.net"
    SWAGGER_URL = "https://klifs.net/swagger/swagger.json"

    def __init__(
        self,
        output_dir: Path = Path("data/raw/pdbs"),
        metadata_dir: Path = Path("data/metadata"),
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Protein-ML-Pipeline/1.0'})

    def _request_with_retry(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Realiza request con reintentos automáticos."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=30, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Reintentando en {wait}s. Error: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Falló después de {self.max_retries} intentos: {e}")
                    return None

    def get_kinases(self) -> List[Dict]:
        """Obtiene lista de kinasas desde KLIFS."""
        logger.info("Obteniendo lista de kinasas...")

        url = f"{self.BASE_URL}/api/kinases"
        response = self._request_with_retry(url)

        if not response:
            raise RuntimeError("No se pudo descargar lista de kinasas")

        kinases = response.json()
        logger.info(f"Total kinasas en KLIFS: {len(kinases)}")

        return kinases

    def filter_cancer_kinases(self, kinases: List[Dict]) -> List[Dict]:
        """Filtra solo kinasas humanas relacionadas con cáncer."""
        filtered = []

        for k in kinases:
            # Extraer nombre de la kinasa (remover _XXX después del nombre)
            kinase_name = k.get('gene_name', '').upper().split('_')[0]
            species = k.get('species', '').lower()

            if kinase_name in CANCER_KINASES and 'homo sapiens' in species:
                filtered.append(k)

        logger.info(f"Kinasas humanas relacionadas con cáncer encontradas: {len(filtered)}")
        return filtered

    def get_structures_for_kinase(self, kinase_id: int) -> List[Dict]:
        """Obtiene estructuras disponibles para una kinasa."""
        url = f"{self.BASE_URL}/api/kinase_structures"
        params = {"kinase_id": kinase_id}

        response = self._request_with_retry(url, params=params)

        if not response:
            return []

        return response.json()

    def classify_conformation(self, structure: Dict) -> str:
        """
        Clasifica estructura como ACTIVA o INACTIVA.

        Lógica:
        - ACTIVA: DFG-in + alphaC helix presente
        - INACTIVA: DFG-out o alphaC replegada
        """
        dfg_state = structure.get('dfg_state', '').lower()
        alphac_state = structure.get('alphac_state', '').lower()

        # Si DFG está "out", es muy probable que sea inactiva
        if 'out' in dfg_state:
            return 'inactive'

        # Si alphaC está replegada, también es inactiva
        if 'out' in alphac_state or 'inactive' in alphac_state:
            return 'inactive'

        # Si DFG está "in" y alphaC está bien posicionada, es activa
        if 'in' in dfg_state and 'in' in alphac_state:
            return 'active'

        # Por defecto, usar DFG como indicador principal
        return 'active' if 'in' in dfg_state else 'inactive'

    def download_pdb(self, pdb_id: str) -> Optional[str]:
        """Descarga estructura PDB y retorna ruta."""
        try:
            url = f"{self.BASE_URL}/api/download_pdb?pdb_id={pdb_id}"
            response = self._request_with_retry(url)

            if not response:
                return None

            pdb_path = self.output_dir / f"{pdb_id.lower()}.pdb"

            with open(pdb_path, 'wb') as f:
                f.write(response.content)

            return str(pdb_path)

        except Exception as e:
            logger.error(f"Error descargando {pdb_id}: {e}")
            return None

    def build_metadata_dataframe(
        self,
        kinases: List[Dict],
        structures: List[Dict]
    ) -> pd.DataFrame:
        """Construye DataFrame con metadata completa."""
        rows = []

        for structure in structures:
            # Buscar información de kinasa
            kinase_info = next(
                (k for k in kinases if k['kinase_id'] == structure.get('kinase_id')),
                {}
            )

            conformation = self.classify_conformation(structure)

            row = {
                'pdb_id': structure.get('pdb_id', ''),
                'kinase_name': kinase_info.get('gene_name', ''),
                'kinase_family': kinase_info.get('kinase_family', ''),
                'species': kinase_info.get('species', ''),
                'conformational_state': conformation,
                'dfg_state': structure.get('dfg_state', ''),
                'alphac_state': structure.get('alphac_state', ''),
                'ligand_present': structure.get('ligand_present', 0),
                'resolution': structure.get('resolution', None),
                'filepath': None,  # Se actualiza después de descargar
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def download_all(self) -> pd.DataFrame:
        """Descarga todas las estructuras y construye dataset."""
        # Obtener kinasas
        all_kinases = self.get_kinases()
        cancer_kinases = self.filter_cancer_kinases(all_kinases)

        if not cancer_kinases:
            raise RuntimeError("No se encontraron kinasas relacionadas con cáncer")

        logger.info(f"Procesando {len(cancer_kinases)} kinasas...")

        # Obtener estructuras
        all_structures = []
        for kinase in tqdm(cancer_kinases, desc="Obteniendo estructuras"):
            structures = self.get_structures_for_kinase(kinase['kinase_id'])
            all_structures.extend(structures)

        logger.info(f"Total estructuras encontradas: {len(all_structures)}")

        # Construir metadata
        metadata_df = self.build_metadata_dataframe(cancer_kinases, all_structures)

        # Descargar PDBs
        logger.info("Descargando archivos PDB...")
        filepaths = []

        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            pdb_id = row['pdb_id']
            filepath = self.download_pdb(pdb_id)
            filepaths.append(filepath)
            time.sleep(0.1)  # Rate limiting

        metadata_df['filepath'] = filepaths
        metadata_df = metadata_df[metadata_df['filepath'].notna()]

        logger.info(f"Descargadas {len(metadata_df)} estructuras exitosamente")

        return metadata_df

    def save_metadata(self, df: pd.DataFrame, filename: str = "kinase_labels.csv"):
        """Guarda metadata en CSV."""
        output_path = self.metadata_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Metadata guardada en {output_path}")

    def create_train_val_test_splits(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ):
        """Divide dataset en train/val/test manteniendo balance de clases."""
        splits_dir = Path("data/splits")
        splits_dir.mkdir(parents=True, exist_ok=True)

        # Separar activas e inactivas
        active_indices = df[df['conformational_state'] == 'active'].index
        inactive_indices = df[df['conformational_state'] == 'inactive'].index

        # Split estratificado
        train_idx, temp_idx = train_test_split(
            df.index,
            train_size=train_size,
            random_state=random_state,
            stratify=df['conformational_state']
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size / (val_size + test_size),
            random_state=random_state,
            stratify=df.loc[temp_idx, 'conformational_state']
        )

        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]

        # Guardar splits
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        logger.info(f"Splits guardados en {splits_dir}/")

        return train_df, val_df, test_df


def main():
    """Ejecuta el pipeline completo."""
    downloader = KLIFSDownloader()

    try:
        # Descargar dataset
        metadata_df = downloader.download_all()

        # Guardar metadata completa
        downloader.save_metadata(metadata_df, "kinase_labels.csv")

        # Crear splits
        downloader.create_train_val_test_splits(metadata_df)

        logger.info("Pipeline completado exitosamente")

    except Exception as e:
        logger.error(f"Error en pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
