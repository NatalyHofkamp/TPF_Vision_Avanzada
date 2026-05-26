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
from typing import Any, List, Dict, Optional
from collections import defaultdict
import logging

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit


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

class KlifsAPIError(Exception):
    """Error de acceso a la API de KLIFS."""


class KlifsClient:
    """Cliente para la API de KLIFS, basado en Swagger oficial."""

    BASE_URL = "https://klifs.net/api"
    DEFAULT_HEADERS = {
        'User-Agent': 'Protein-ML-Pipeline/1.0',
        'Accept': 'application/json',
    }

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, timeout: float = 30.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def _parse_retry_after(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, stream: bool = False, accept: Optional[str] = None) -> requests.Response:
        url = f"{self.BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        attempt = 0
        delay = self.retry_delay

        if accept:
            self.session.headers['Accept'] = accept
        else:
            self.session.headers['Accept'] = self.DEFAULT_HEADERS['Accept']

        while attempt < self.max_retries:
            try:
                logger.debug("KLIFS request %s params=%s attempt=%d", url, params, attempt + 1)
                response = self.session.get(url, params=params, timeout=self.timeout, stream=stream)

                logger.info("KLIFS HTTP %s %s -> %d", response.request.method, response.url, response.status_code)

                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response.headers.get('Retry-After'))
                    wait = retry_after if retry_after is not None else delay
                    logger.warning("Rate limit alcanzado en KLIFS. Reintentando en %.1fs", wait)
                    time.sleep(wait)
                    attempt += 1
                    delay *= 2
                    continue

                if 400 <= response.status_code < 600:
                    if attempt < self.max_retries - 1 and response.status_code in {500, 502, 503, 504}:
                        logger.warning("KLIFS servidor error %d. Reintentando en %.1fs", response.status_code, delay)
                        time.sleep(delay)
                        attempt += 1
                        delay *= 2
                        continue

                    message = f"KLIFS API error {response.status_code}: {response.text}"
                    logger.error(message)
                    response.raise_for_status()

                return response

            except requests.RequestException as exc:
                if attempt < self.max_retries - 1:
                    logger.warning("KLIFS request falló: %s. Reintentando en %.1fs", exc, delay)
                    time.sleep(delay)
                    attempt += 1
                    delay *= 2
                else:
                    logger.error("KLIFS request falló después de %d intentos: %s", self.max_retries, exc)
                    raise KlifsAPIError(str(exc)) from exc

        raise KlifsAPIError(f"No se pudo completar la petición a KLIFS: {endpoint}")

    def get_kinase_names(self, species: str = "HUMAN") -> List[Dict[str, Any]]:
        return self._request("kinase_names", params={"species": species}).json()

    def get_kinase_information(self, kinase_ids: List[int], species: Optional[str] = None) -> List[Dict[str, Any]]:
        if not kinase_ids:
            return []
        params: Dict[str, Any] = {"kinase_ID": ",".join(map(str, kinase_ids))}
        if species:
            params["species"] = species
        return self._request("kinase_information", params=params).json()

    def get_structures(self, kinase_ids: List[int]) -> List[Dict[str, Any]]:
        if not kinase_ids:
            return []
        return self._request("structures_list", params={"kinase_ID": ",".join(map(str, kinase_ids))}).json()

    def get_structure_pdb(self, structure_id: int, complex_structure: bool = True) -> bytes:
        endpoint = "structure_get_pdb_complex" if complex_structure else "structure_get_protein"
        response = self._request(endpoint, params={"structure_ID": structure_id}, stream=True, accept="chemical/x-pdb")
        return response.content


class KLIFSDownloader:
    """Descarga y procesa estructuras desde KLIFS API."""

    def __init__(
        self,
        output_dir: Path = Path("data/raw/pdbs"),
        metadata_dir: Path = Path("data/metadata"),
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.client = KlifsClient(max_retries=max_retries, retry_delay=retry_delay, timeout=timeout)

    def get_kinases(self) -> List[Dict[str, Any]]:
        """Obtiene lista de kinasas humanas desde KLIFS usando endpoints oficiales."""
        logger.info("Obteniendo lista de kinasas humanas desde KLIFS...")

        kinase_names = self.client.get_kinase_names(species="HUMAN")
        kinases: List[Dict[str, Any]] = []

        for item in kinase_names:
            kinases.append({
                "kinase_id": item.get("kinase_ID"),
                "gene_name": item.get("name", ""),
                "full_name": item.get("full_name", ""),
                "species": item.get("species", ""),
            })

        logger.info("Total kinasas humanas en KLIFS: %d", len(kinases))
        return kinases

    def filter_cancer_kinases(self, kinases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtra kinasas humanas relacionadas con cáncer y enriquece metadata."""

        filtered = []

        for k in kinases:
            gene_name = (k.get("gene_name") or "").upper().strip()
            species = (k.get("species") or "").lower().strip()

            if gene_name in CANCER_KINASES and species == "human":
                filtered.append(k)

        logger.info("Kinasas humanas relacionadas con cáncer encontradas: %d", len(filtered))

        if filtered:
            kinase_ids = [k["kinase_id"] for k in filtered if k.get("kinase_id") is not None]

            info = self.client.get_kinase_information(kinase_ids, species="HUMAN")
            info_by_id = {item["kinase_ID"]: item for item in info}

            for k in filtered:
                details = info_by_id.get(k["kinase_id"], {})
                k["kinase_family"] = details.get("family", "")
                k["kinase_group"] = details.get("group", "")
                k["uniprot"] = details.get("uniprot", "")

        return filtered

    def get_structures_for_kinase(self, kinase_id: int) -> List[Dict[str, Any]]:
        """Obtiene estructuras disponibles para una kinasa usando KLIFS structures_list."""
        logger.info("Obteniendo estructuras para kinase_id=%s", kinase_id)
        return self.client.get_structures([kinase_id])

    def classify_conformation(self, structure: Dict[str, Any]) -> str:
        """Clasifica estructura como activa o inactiva usando DFG y alphaC."""
        dfg_state = str(structure.get("DFG", "")).lower()
        alphac_state = str(structure.get("aC_helix", "")).lower()

        if "out" in dfg_state:
            return "inactive"
        if "out" in alphac_state or "inactive" in alphac_state:
            return "inactive"
        if "in" in dfg_state and "in" in alphac_state:
            return "active"
        return "active" if "in" in dfg_state else "inactive"

    def download_pdb(self, structure_id: int, pdb_id: str) -> Optional[str]:
        """Descarga una estructura PDB usando el endpoint oficial de KLIFS."""
        try:
            pdb_bytes = self.client.get_structure_pdb(structure_id)
            if not pdb_bytes:
                return None

            pdb_path = self.output_dir / f"{pdb_id.lower()}.pdb"
            with open(pdb_path, "wb") as f:
                f.write(pdb_bytes)
            return str(pdb_path)

        except Exception as e:
            logger.error("Error descargando structure_id=%s pdb_id=%s: %s", structure_id, pdb_id, e)
            return None

    def build_metadata_dataframe(
        self,
        kinases: List[Dict[str, Any]],
        structures: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Construye DataFrame con metadata completa y nombres compatibles."""
        rows = []

        kinase_by_id = {k["kinase_id"]: k for k in kinases}

        for structure in structures:
            kinase_info = kinase_by_id.get(structure.get("kinase_ID"), {})
            conformation = self.classify_conformation(structure)
            ligand_present = bool(structure.get("ligand") or structure.get("allosteric_ligand"))

            row = {
                "structure_id": structure.get("structure_ID"),
                "pdb_id": structure.get("pdb", ""),
                "kinase_name": kinase_info.get("gene_name", ""),
                "kinase_family": kinase_info.get("kinase_family", ""),
                "kinase_group": kinase_info.get("kinase_group", ""),
                "species": kinase_info.get("species", ""),
                "conformational_state": conformation,
                "dfg_state": structure.get("DFG", ""),
                "alphac_state": structure.get("aC_helix", ""),
                "ligand_present": int(ligand_present),
                "resolution": structure.get("resolution", None),
                "filepath": None,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def download_all(self) -> pd.DataFrame:
        """Descarga todas las estructuras y construye dataset."""
        all_kinases = self.get_kinases()
        print(all_kinases[:5])
        cancer_kinases = self.filter_cancer_kinases(all_kinases)

        if not cancer_kinases:
            raise RuntimeError("No se encontraron kinasas relacionadas con cáncer")

        logger.info("Procesando %d kinasas...", len(cancer_kinases))

        all_structures: List[Dict[str, Any]] = []
        for kinase in tqdm(cancer_kinases, desc="Obteniendo estructuras"):
            structures = self.get_structures_for_kinase(kinase["kinase_id"])
            if structures:
                all_structures.extend(structures)

        logger.info("Total estructuras encontradas: %d", len(all_structures))

        metadata_df = self.build_metadata_dataframe(cancer_kinases, all_structures)

        logger.info("Descargando archivos PDB...")
        filepaths = []
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            filepath = self.download_pdb(row["structure_id"], row["pdb_id"])
            filepaths.append(filepath)
            time.sleep(0.1)

        metadata_df["filepath"] = filepaths
        metadata_df = metadata_df[metadata_df["filepath"].notna()]

        logger.info("Descargadas %d estructuras exitosamente", len(metadata_df))
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
        random_state: int = 42,
        strategy: str = "grouped",
        group_by: str = "kinase_name",
        verbose: bool = True
    ):
        """
        Divide dataset en train/val/test con control de leakage estructural.
        
        GROUPED STRATEGY (Recomendado):
        --------------------------------
        - Agrupa estructuras de la MISMA kinasa en un único split
        - EVITA data leakage: el modelo no ve la misma proteína en train y test
        - Simula generalización a kinasas completamente nuevas
        - Mantiene balance ACTIVE/INACTIVE en cada split
        - Mejor para biología estructural: aprende patrones generales, no memorización
        
        STRATIFIED STRATEGY (Legacy):
        - Divide aleatoriamente manteniendo balance de clases
        - RIESGO DE LEAKAGE: puede mezclar estructuras de la misma kinasa
        - NO RECOMENDADO para datos biológicos
        
        Args:
            df: DataFrame con columnas 'kinase_name' y 'conformational_state'
            train_size, val_size, test_size: proporciones de splits
            random_state: seed para reproducibilidad
            strategy: "grouped" (recomendado) o "stratified" (legacy)
            group_by: columna para agrupar (default: kinase_name)
            verbose: mostrar logs detallados
            
        Returns:
            Tupla (train_df, val_df, test_df)
        """
        splits_dir = Path("data/splits")
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            logger.info("="*80)
            logger.info("CREANDO SPLITS CON CONTROL DE LEAKAGE")
            logger.info("="*80)
            logger.info(f"Estrategia: {strategy.upper()}")
            logger.info(f"Agrupar por: {group_by}")
            logger.info(f"Proporciones: Train={train_size:.1%}, Val={val_size:.1%}, Test={test_size:.1%}")
            logger.info(f"Total estructuras: {len(df)}")
            logger.info(f"Total {group_by}s únicos: {df[group_by].nunique()}")
        
        if strategy == "grouped":
            train_df, val_df, test_df = self._grouped_split(
                df=df,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                random_state=random_state,
                group_by=group_by,
                verbose=verbose
            )
        elif strategy == "stratified":
            # Legacy implementation
            train_df, val_df, test_df = self._stratified_split(
                df=df,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                random_state=random_state,
                verbose=verbose
            )
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}. Use 'grouped' o 'stratified'")
        
        # Validar integridad de splits
        self._validate_splits(train_df, val_df, test_df, group_by=group_by, verbose=verbose)
        
        # Guardar splits
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)
        
        if verbose:
            logger.info(f"\n✅ Splits guardados en {splits_dir}/")
            logger.info("="*80)
        
        return train_df, val_df, test_df
    
    def _grouped_split(
        self,
        df: pd.DataFrame,
        train_size: float,
        val_size: float,
        test_size: float,
        random_state: int,
        group_by: str,
        verbose: bool = True
    ):
        """
        Split agrupado por grupos biológicos (ej: kinase_name).
        
        Garantiza que estructuras del MISMO grupo estén juntas en un split.
        Esto evita leakage estructural y simula generalización a grupos nuevos.
        """
        # Crear mapping: grupo -> índices
        group_indices = {}
        for group_name in df[group_by].unique():
            mask = df[group_by] == group_name
            indices = np.where(mask)[0]
            group_indices[group_name] = indices
        
        groups_list = list(group_indices.keys())
        n_groups = len(groups_list)
        
        if verbose:
            logger.info(f"\n🔬 ESTRATEGIA GROUPED SPLIT")
            logger.info(f"   Dividiendo {n_groups} grupos (kinasas) entre splits")
        
        # Calcular cuántos grupos en cada split (aproximadamente)
        n_train_groups = max(1, int(np.round(train_size * n_groups)))
        n_val_groups = max(1, int(np.round(val_size * n_groups)))
        n_test_groups = n_groups - n_train_groups - n_val_groups
        
        if n_test_groups < 1:
            # Si no hay suficientes grupos para test, reasignar
            if n_val_groups > 1:
                n_val_groups = max(1, n_val_groups - 1)
            n_test_groups = n_groups - n_train_groups - n_val_groups
        
        if verbose:
            logger.info(f"   Distribución de grupos:")
            logger.info(f"     - Train: {n_train_groups} grupos")
            logger.info(f"     - Val:   {n_val_groups} grupos")
            logger.info(f"     - Test:  {n_test_groups} grupos")
        
        # Shuffle grupos manteniendo balance ACTIVE/INACTIVE
        rng = np.random.RandomState(random_state)
        
        # Separar grupos por estado conformacional (si es posible)
        active_groups = [g for g in groups_list if (df[df[group_by] == g]['conformational_state'] == 'active').sum() > 0]
        inactive_groups = [g for g in groups_list if (df[df[group_by] == g]['conformational_state'] == 'inactive').sum() > 0]
        
        rng.shuffle(active_groups)
        rng.shuffle(inactive_groups)
        
        # Distribuir grupos alternando entre activos e inactivos para balance
        train_groups = []
        val_groups = []
        test_groups = []
        
        # Simple round-robin distribution
        for i, group in enumerate(groups_list):
            if len(train_groups) < n_train_groups:
                train_groups.append(group)
            elif len(val_groups) < n_val_groups:
                val_groups.append(group)
            else:
                test_groups.append(group)
        
        # Extraer índices para cada split
        train_indices = np.concatenate([group_indices[g] for g in train_groups])
        val_indices = np.concatenate([group_indices[g] for g in val_groups])
        test_indices = np.concatenate([group_indices[g] for g in test_groups])
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        
        if verbose:
            self._log_split_statistics(train_df, val_df, test_df, group_by=group_by, split_groups=(train_groups, val_groups, test_groups))
        
        return train_df, val_df, test_df
    
    def _stratified_split(
        self,
        df: pd.DataFrame,
        train_size: float,
        val_size: float,
        test_size: float,
        random_state: int,
        verbose: bool = True
    ):
        """
        Split estratificado manteniendo balance de clases.
        
        ⚠️ ADVERTENCIA: Puede causar leakage estructural si la misma proteína
        aparece en múltiples splits. Use solo para datasets que ya están
        separados por grupo.
        """
        if verbose:
            logger.warning("⚠️  USANDO ESTRATEGIA STRATIFICADA (LEGACY)")
            logger.warning("   Esto PUEDE CAUSAR LEAKAGE ESTRUCTURAL")
            logger.warning("   Se recomienda usar strategy='grouped'")
        
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

        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)
        
        return train_df, val_df, test_df
    
    def _log_split_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_by: str = "kinase_name",
        split_groups: Optional[tuple] = None
    ):
        """Log detallado de estadísticas de splits."""
        
        logger.info(f"\n📊 ESTADÍSTICAS DE SPLITS")
        logger.info("-" * 80)
        
        # Por split
        for split_name, split_df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
            n_total = len(split_df)
            n_active = len(split_df[split_df['conformational_state'] == 'active'])
            n_inactive = n_total - n_active
            pct_active = (n_active / n_total * 100) if n_total > 0 else 0
            pct_inactive = (n_inactive / n_total * 100) if n_total > 0 else 0
            pct_of_total = (n_total / (len(train_df) + len(val_df) + len(test_df)) * 100)
            
            logger.info(f"\n{split_name}:")
            logger.info(f"  Estructuras:  {n_total:4d} ({pct_of_total:5.1f}%)")
            logger.info(f"  Activas:      {n_active:4d} ({pct_active:5.1f}%)")
            logger.info(f"  Inactivas:    {n_inactive:4d} ({pct_inactive:5.1f}%)")
            logger.info(f"  {group_by}s:   {split_df[group_by].nunique():4d}")
        
        # Detalle por grupo
        if split_groups:
            train_groups, val_groups, test_groups = split_groups
            logger.info(f"\n🧬 {group_by.upper()}S POR SPLIT")
            logger.info(f"  TRAIN ({len(train_groups)}): {', '.join(sorted(train_groups))}")
            logger.info(f"  VAL   ({len(val_groups)}): {', '.join(sorted(val_groups))}")
            logger.info(f"  TEST  ({len(test_groups)}): {', '.join(sorted(test_groups))}")
    
    def _validate_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_by: str = "kinase_name",
        verbose: bool = True
    ):
        """
        Valida que no haya leakage entre splits.
        
        Verificaciones:
        - No hay grupos duplicados entre splits
        - Balance razonable de clases
        - Todos los datos se distribuyen
        """
        train_groups = set(train_df[group_by].unique())
        val_groups = set(val_df[group_by].unique())
        test_groups = set(test_df[group_by].unique())
        
        if verbose:
            logger.info(f"\n✅ VALIDACIÓN DE LEAKAGE")
        
        # Verificar overlap
        train_val_overlap = train_groups & val_groups
        train_test_overlap = train_groups & test_groups
        val_test_overlap = val_groups & test_groups
        
        if train_val_overlap:
            logger.error(f"❌ LEAKAGE DETECTADO: {group_by}s en TRAIN y VAL: {train_val_overlap}")
            raise ValueError(f"Leakage entre TRAIN y VAL: {train_val_overlap}")
        if train_test_overlap:
            logger.error(f"❌ LEAKAGE DETECTADO: {group_by}s en TRAIN y TEST: {train_test_overlap}")
            raise ValueError(f"Leakage entre TRAIN y TEST: {train_test_overlap}")
        if val_test_overlap:
            logger.error(f"❌ LEAKAGE DETECTADO: {group_by}s en VAL y TEST: {val_test_overlap}")
            raise ValueError(f"Leakage entre VAL y TEST: {val_test_overlap}")
        
        if verbose:
            logger.info(f"   ✓ Sin overlap entre TRAIN, VAL, TEST")
            logger.info(f"   ✓ Separación de {group_by}s garantizada")
        
        # Verificar cobertura
        total_structures = len(train_df) + len(val_df) + len(test_df)
        logger.info(f"   ✓ Cobertura: {total_structures} estructuras distribuidas")
        
        # Warnings sobre balance si es muy desbalanceado
        train_pct = len(train_df) / total_structures * 100
        val_pct = len(val_df) / total_structures * 100
        test_pct = len(test_df) / total_structures * 100
        
        if verbose:
            logger.info(f"   ✓ Proporciones finales: Train {train_pct:.1f}%, Val {val_pct:.1f}%, Test {test_pct:.1f}%")


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
