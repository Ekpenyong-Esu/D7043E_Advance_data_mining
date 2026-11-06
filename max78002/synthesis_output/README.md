# max78002/synthesis_output/

Start with `ai8xize.log`.

1. `ai8xize.log`: the command line and console output of `ai8xize.py`.
2. `sample_ecg_mitbih.npy`: the sample beat (one V beat from a validation record, as integers in [-128, 127]) that the synthesiser turns into a known-answer test.
3. `ecg-max78002/`: the generated project. `cnn.c` and `cnn.h` load the weights and run the network, `weights.h` holds the INT8 weights, `sampledata.h` the sample beat, `sampleoutput.h` the expected output, `main.c` the known-answer test, `softmax.c` the output softmax, `Makefile` the MSDK build. The verbose `log.txt` inside it (several hundred kB, every register write) is git-ignored; its resource summary is quoted in `results/deployment_evidence.csv`.

How they connect: section 8 of `notebooks/MODELING.ipynb` writes the sample with `src/ecg/deployment/sample_input.py`, runs `src/ecg/deployment/toolchain.run_synthesis`, and reads the log with `src/ecg/deployment/evidence.py`. The expected output in `sampleoutput.h` is compared with the Python INT8 simulation of the same beat.
