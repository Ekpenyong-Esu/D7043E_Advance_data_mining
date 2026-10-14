# report/figures/

Figures included by the report sections.

Start with `eda_raw_ecg.png`. The EDA figures follow the order of report section 2.3:

1. `eda_raw_ecg.png`: ten seconds of one record.
2. `eda_class_examples.png`: beats from each class.
3. `eda_class_distribution.png`: beats per class in DS1 and DS2.
4. `eda_record_distribution.png`: beats per class per record.
5. `eda_amplitude.png`: R-peak height per record.
6. `eda_noisy.png`: noisy records and the NSTDB noise types.
7. `aug_amplitude_scale.png`: one beat at three amplitude scales, for section 3.7.

How they connect: `notebooks/EDA.ipynb` draws these with `src/ecg/eda/plots.py` into `results/eda/figures/`, and they are copied here by hand. The `.tex` files in `../sections/` include them.
