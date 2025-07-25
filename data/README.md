# DEEP-TAO Dataset Information

This `data/` folder is a placeholder for the DEEP-TAO dataset used in the [DEEP-TAO Replication project](https://github.com/DreadPirateSamuel/Deep-TAO-Replication). Due to GitHub's file size limitations and the large scale of the dataset (~1.2 million FITS images), the full dataset is not included in this repository.

## Dataset Overview
The DEEP-TAO dataset, introduced in the paper ["DEEP-TAO: The Deep Learning Transient Astronomical Object Data Set for Astronomical Transient Event Classification"](https://arxiv.org/pdf/2503.16714), contains 1,249,079 FITS images, including 3,807 transient and 12,500 non-transient sequences across six classes: SN (supernovae), AGN (active galactic nuclei), BZ (blazars), CV (cataclysmic variables), OTHER (other transients), and NON-TRANSIENT. The dataset also includes the MANTRA dataset for lightcurve analysis.

## How to Access the Dataset
To obtain the full DEEP-TAO dataset, including the `TAO_transients-master`, `TAO_non-transients-master`, and `MANTRA-master` directories, please download it from the official source:

- **Source**: [MachineLearningUniandes GitHub Repository](https://github.com/MachineLearningUniandes)
- **Instructions**: Clone or download the dataset files and place them in this `data/` folder before running the project scripts. Ensure you have sufficient storage (~several GB) and computational resources to handle the dataset.

## Notes
- This project used a subset of ~2,000 FITS files (~17,136 images) due to computational constraints, as described in the main [README.md](../README.md).
- Sample FITS files may be included in this folder (if under GitHub's 100MB limit) for demonstration purposes. Check the repository for any available samples.

For further details on the dataset or project setup, refer to the main [README.md](../README.md) or contact the repository owner.
