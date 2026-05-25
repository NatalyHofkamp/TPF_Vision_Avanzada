"""
Script de validación del pipeline.

Verifica que todo esté configurado correctamente para ejecutar el pipeline.
"""

import sys
from pathlib import Path
import importlib.util
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_directories():
    """Verifica que los directorios existan."""
    logger.info("\n📁 Verificando directorios...")

    dirs_to_create = [
        "data/raw/pdbs",
        "data/metadata",
        "data/splits",
        "data/processed",
        "scripts",
        "notebooks",
    ]

    for dir_path in dirs_to_create:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"  ✅ {dir_path}")
        else:
            logger.warning(f"  ⚠️  {dir_path} - será creado al descargar")
            path.mkdir(parents=True, exist_ok=True)


def check_files():
    """Verifica que los archivos necesarios existan."""
    logger.info("\n📄 Verificando archivos...")

    required_files = [
        "requirements.txt",
        "Makefile",
        "scripts/download_klifs_dataset.py",
        "scripts/explore_dataset.py",
        "scripts/preprocess_dataset.py",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            logger.info(f"  ✅ {file_path}")
        else:
            logger.error(f"  ❌ {file_path} - FALTA")
            all_exist = False

    return all_exist


def check_python_packages():
    """Verifica que las dependencias estén instaladas."""
    logger.info("\n📦 Verificando dependencias de Python...")

    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'torch': 'PyTorch',
        'tqdm': 'tqdm',
        'requests': 'requests',
        'sklearn': 'scikit-learn',
        'Bio': 'BioPython',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
    }

    all_installed = True
    for module_name, package_name in required_packages.items():
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            logger.info(f"  ✅ {package_name}")
        else:
            logger.error(f"  ❌ {package_name} - Instala: pip install {package_name}")
            all_installed = False

    return all_installed


def check_internet():
    """Verifica conectividad con KLIFS API."""
    logger.info("\n🌐 Verificando conectividad con KLIFS API...")

    try:
        import requests
        response = requests.head("https://klifs.net", timeout=5)
        if response.status_code < 500:
            logger.info("  ✅ KLIFS API accesible")
            return True
        else:
            logger.error("  ❌ KLIFS API retorna error")
            return False
    except Exception as e:
        logger.error(f"  ❌ Sin conectividad: {e}")
        return False


def check_space():
    """Verifica espacio en disco."""
    logger.info("\n💾 Verificando espacio en disco...")

    import shutil
    path = Path("data")
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024 ** 3)

    if free_gb > 10:
        logger.info(f"  ✅ Espacio disponible: {free_gb:.1f} GB")
        return True
    else:
        logger.warning(f"  ⚠️  Espacio limitado: {free_gb:.1f} GB (recomendado: >10GB)")
        return False


def print_summary(checks):
    """Imprime resumen de verificaciones."""
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE VERIFICACIÓN")
    logger.info("="*80)

    total = len(checks)
    passed = sum(checks.values())

    logger.info(f"\nResultado: {passed}/{total} ✅")

    if passed == total:
        logger.info("\n🎉 ¡Todo configurado correctamente!")
        logger.info("\nPróximos pasos:")
        logger.info("  1. make download-dataset    # Descargar structures")
        logger.info("  2. make explore-dataset     # Explorar estadísticas")
        logger.info("  3. make preprocess          # Procesar a tensores")
        return True
    else:
        logger.error("\n⚠️  Hay problemas a resolver.")
        logger.error("Ver arriba para más detalles.")
        return False


def main():
    """Ejecuta todas las verificaciones."""
    logger.info("\n" + "="*80)
    logger.info("VERIFICACIÓN DEL PIPELINE KLIFS")
    logger.info("="*80)

    checks = {
        'Directorios': True,  # No falla
        'Archivos': check_files(),
        'Dependencias': check_python_packages(),
        'KLIFS API': check_internet(),
        'Espacio': check_space(),
    }

    check_directories()

    success = print_summary(checks)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
