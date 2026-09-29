# Bundled example spectra

These three spectra are the observations identified in the Ca II background plots used to choose the joint Si II and Ca II NIR example sample. [example_catalog.csv](../example_catalog.csv) supplies each observation's redshift, reddening, phase, smoothing width, and a Ca-only Si velocity prior.

| File | Observation MJD | Telescope / instrument | Pixels |
| --- | ---: | --- | ---: |
| `ZTF18aagrtxs_0_20180328_SEDM.DAT` | 58205.25 | P60 / SEDM | 208 |
| `ZTF18abtnbys_0_20180913_DBSP.DAT` | 58374.46 | P200 / DBSP | 4198 |
| `ZTF20aatzwgk_1_20200414_SEDM.DAT` | 58953.46 | P60 / SEDM | 220 |

Observation headers are preserved in the data files. Each file has wavelength, flux, statistical uncertainty, and systematic uncertainty columns. The default fitter uses the third column and ignores the fourth. All statistical uncertainties in the first two files are `-99` missing-data sentinels, so the fitter estimates them from the flux. The third file supplies positive statistical uncertainties.

All three spectra pass the default joint wavelength-coverage preflight at the redshifts in the example catalogue. The bundled sample is for demonstration and regression testing; fit quality still requires review of the diagnostic plots.
