"""
Explorar y analizar dataset de kinasas.

Este script carga el CSV de metadata y genera estadísticas útiles para
entender la composición del dataset.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetExplorer:
    """Explora estadísticas del dataset de kinasas."""

    def __init__(self, metadata_csv: Path = Path("data/metadata/kinase_labels.csv")):
        self.csv_path = Path(metadata_csv)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Dataset cargado: {len(self.df)} estructuras")

    def print_basic_stats(self):
        """Imprime estadísticas básicas."""
        print("\n" + "="*80)
        print("ESTADÍSTICAS BÁSICAS DEL DATASET")
        print("="*80)

        print(f"\nTotal de estructuras: {len(self.df)}")
        print(f"Columnas: {list(self.df.columns)}")

        # Estados conformacionales
        print("\n--- ESTADOS CONFORMACIONALES ---")
        conformation_counts = self.df['conformational_state'].value_counts()
        for state, count in conformation_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {state.upper()}: {count} ({percentage:.1f}%)")

        # Kinasas
        print("\n--- KINASAS PRESENTES ---")
        kinases = self.df['kinase_name'].value_counts()
        for kinase, count in kinases.head(10).items():
            active = len(self.df[(self.df['kinase_name'] == kinase) &
                               (self.df['conformational_state'] == 'active')])
            inactive = len(self.df[(self.df['kinase_name'] == kinase) &
                                 (self.df['conformational_state'] == 'inactive')])
            print(f"  {kinase}: {count} (activas: {active}, inactivas: {inactive})")

    def print_conformation_details(self):
        """Analiza los estados DFG y alphaC."""
        print("\n" + "="*80)
        print("ANÁLISIS DE CONFORMACIONES")
        print("="*80)

        print("\n--- DFG States ---")
        dfg_states = self.df['dfg_state'].value_counts()
        for state, count in dfg_states.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {state}: {count} ({percentage:.1f}%)")

        print("\n--- AlphaC States ---")
        alphac_states = self.df['alphac_state'].value_counts()
        for state, count in alphac_states.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {state}: {count} ({percentage:.1f}%)")

        print("\n--- Ligands ---")
        ligand_counts = self.df['ligand_present'].value_counts()
        for ligand, count in ligand_counts.items():
            percentage = (count / len(self.df)) * 100
            status = "presentes" if ligand else "ausentes"
            print(f"  Ligandos {status}: {count} ({percentage:.1f}%)")

    def print_resolution_stats(self):
        """Analiza resolución cristalográfica."""
        print("\n" + "="*80)
        print("RESOLUCIÓN CRISTALOGRÁFICA")
        print("="*80)

        resolution = self.df['resolution'].dropna()

        if len(resolution) > 0:
            print(f"\n  Media: {resolution.mean():.2f} Å")
            print(f"  Mediana: {resolution.median():.2f} Å")
            print(f"  Min: {resolution.min():.2f} Å")
            print(f"  Max: {resolution.max():.2f} Å")
            print(f"  Desv. estándar: {resolution.std():.2f} Å")
            print(f"\n  Estructuras con resolución: {len(resolution)}")
            print(f"  Estructuras sin resolución: {len(self.df) - len(resolution)}")
        else:
            print("  No hay datos de resolución")

    def print_split_analysis(self):
        """Analiza los splits si existen."""
        splits_dir = Path("data/splits")

        print("\n" + "="*80)
        print("ANÁLISIS DE SPLITS")
        print("="*80)

        if splits_dir.exists():
            for split_file in ['train.csv', 'val.csv', 'test.csv']:
                path = splits_dir / split_file
                if path.exists():
                    split_df = pd.read_csv(path)
                    active = len(split_df[split_df['conformational_state'] == 'active'])
                    inactive = len(split_df[split_df['conformational_state'] == 'inactive'])
                    total = len(split_df)
                    percentage = (total / len(self.df)) * 100

                    print(f"\n{split_file}:")
                    print(f"  Total: {total} ({percentage:.1f}%)")
                    print(f"  Activas: {active} ({active/total*100:.1f}%)")
                    print(f"  Inactivas: {inactive} ({inactive/total*100:.1f}%)")
        else:
            print("  Splits no encontrados")

    def plot_distributions(self, output_dir: Path = Path("figures")):
        """Genera visualizaciones."""
        output_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dataset Analysis - Kinases', fontsize=16, fontweight='bold')

        # 1. Distribución de estados conformacionales
        conformation_counts = self.df['conformational_state'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0, 0].bar(conformation_counts.index, conformation_counts.values, color=colors)
        axes[0, 0].set_title('Conformational States Distribution')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(conformation_counts.values):
            axes[0, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')

        # 2. Top 10 kinasas
        top_kinases = self.df['kinase_name'].value_counts().head(10)
        axes[0, 1].barh(range(len(top_kinases)), top_kinases.values, color='#3498db')
        axes[0, 1].set_yticks(range(len(top_kinases)))
        axes[0, 1].set_yticklabels(top_kinases.index)
        axes[0, 1].set_title('Top 10 Kinases')
        axes[0, 1].set_xlabel('Count')

        # 3. Histograma de resolución
        resolution = self.df['resolution'].dropna()
        if len(resolution) > 0:
            axes[1, 0].hist(resolution, bins=30, color='#9b59b6', edgecolor='black')
            axes[1, 0].set_title('Resolution Distribution')
            axes[1, 0].set_xlabel('Resolution (Å)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(resolution.mean(), color='r', linestyle='--',
                             label=f'Mean: {resolution.mean():.2f}Å')
            axes[1, 0].legend()

        # 4. Ligandos presentes
        ligand_data = pd.Series({
            'With ligand': len(self.df[self.df['ligand_present'] == 1]),
            'Without ligand': len(self.df[self.df['ligand_present'] == 0])
        })
        axes[1, 1].pie(ligand_data.values, labels=ligand_data.index,
                       autopct='%1.1f%%', colors=['#f39c12', '#95a5a6'])
        axes[1, 1].set_title('Ligand Presence')

        plt.tight_layout()
        output_path = output_dir / "dataset_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figura guardada: {output_path}")
        plt.close()

    def generate_report(self):
        """Genera reporte completo."""
        self.print_basic_stats()
        self.print_conformation_details()
        self.print_resolution_stats()
        self.print_split_analysis()
        self.plot_distributions()

        print("\n" + "="*80)
        print("REPORTE COMPLETADO")
        print("="*80 + "\n")


def main():
    """Ejecuta exploración del dataset."""
    try:
        explorer = DatasetExplorer()
        explorer.generate_report()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
