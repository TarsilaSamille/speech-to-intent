# S2I Classification

Intent Classification from Speech

## About

This is a fork from [skit-ai/speech-to-intent-dataset](https://github.com/skit-ai/speech-to-intent-dataset). It contains code to run Intent classification from human speech using the dataset provided in the aforementioned link. The dataset covers 14 coarse-grained intents from the Banking domain. This work is inspired by a similar release in the [Minds-14 dataset](https://huggingface.co/datasets/PolyAI/minds14) - here, we restrict ourselves to Indian English but with a much larger training set. The dataset is split into:
- test - `100` samples per intent
- train - `>650` samples per intent

The data was generated by 11 (Indian English) speakers, recording over a telephony line. We also provide access to anonymised speaker information - like gender, languages spoken, native language - so as to allow more structured discussions around robustness and bias, in the models you train.

## Download and Usage

The dataset can be downloaded by clicking on this [link](https://speech-to-intent-dataset.s3.ap-south-1.amazonaws.com/speech-to-intent.zip). Incase you face any issues please reach out to kmanas@skit.ai.

This dataset is shared under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) Licence. This places restrictions on commercial use of this dataset.

## Uses

Most spoken dialog-systems use a pipeline of speech recognition followed by intent classification, and optimise each individually. But this allows ASR errors to leak downstream. Instead, what if we train end-to-end intent models on speech ? More importantly, how well would such models generalise in a language like Indian English - given the diversity of speech behaviours ? This dataset is an attempt towards answering such questions around robustness and model bias.

## Structure

This release contains data of (Indian English) speech samples tagged with an intent from the Banking domain. Also includes the transcript template used to generate the sample.

Audio Quality : 8 Khz, 16-bit

Structure

```
- wav_audios          [contains the wav audio files]
- train.csv           [contains the train split, where each row contains "<id> | <intent-class> | <template> | <audio-path> | <speaker-id>"]
- test.csv            [contains the test split, where each row contains "<id> | <intent-class> | <template> | <audio-path> | <speaker-id>"]
- intent_info.csv     [contains information about the intents, where each row contains "<intent-class> | <intent-name> | <description>"]
- speaker_info.csv    [contains information about the speakers, where each row contains "<speaker-id> | <native-language> | <languages-spoken> | <places-lived> | <gender>"]

```

More information regarding the dataset can be found in the [datasheet](./datasheet.md).

## Baselines

The code for the baselines are provided in the [baselines](./baselines/) directory.

## Citation

If you are using this dataset, please cite using the link in the About section on the right.

## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
