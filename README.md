# KDSalBox

This repository contains the code for the toolbox presented in our paper at SVRHM 2021:

“KDSalBox: A toolbox of efficient knowledge-distilled saliency models”  
Ard Kastrati, Zoya Bylinskii, Eli Shechtman  
SVHRM @ NeurIPS ’21: 3rd Shared Visual Representations in Human & Machine Intelligence Workshop, Online, December 2021

The PDF of the article is available at this [link][1].

## Overview

The repository consists a toolbox to run, evaluate 10 different knowledge-distilled (KD) saliency models. We provide efficient neural networks for each model that was trained to mimic the behaviour of the original counterpart. The models that the toolbox currently includes are the following: AIM (Bruce & Tsotsos (2009)), IKN (Itti et al. (1998a)), GBVS (Harel et al. (2007)), BMS (Zhang & Sclaroff (2013)), IMSIG (Hou et al. (2012)), RARE2012 (Riche et al. (2013)), SUN (Zhang et al. (2008)), UniSal (Droste et al. (2020)), SAM (Cornia et al. (2018)), DGII (Kümmerer et al. (2016)).

Additionally we provide a GUI on top of this toolbox to facilitate the evaluation of the models for application-purposes.

We caution the users knowledge-distilled models are not meant to replace the original models for every case in absolute sense. As they are knowledge-distilled versions they should be used always with their limitations in mind. We invite the users to consult Table 2 in our paper for an estimate of how the different knowledge-distilled models generalize to different image types. The main goal in this work is to facilitate the evaluation and the selection of saliency models for different applications. The original models that are Matlab-based are not efficient and require a Matlab license to run. Our knowledge-distilled models can be beneficial in cases where the behaviour of the original models is better and efficiency is important.

## Installation (Environment)

There are dependencies in this toolbox and we propose to use anaconda as package manager.

You can install the dependencies inside a conda environment as follows:

```console
$ git clone 
$ cd make
$ conda create --name kdsalbox python=3.8
$ conda activate kdsalbox
```
(For details on how to install `conda`, please refer to the [official documentation][5].)

## Usage of Toolbox

You can use the toolbox mainly for the following three functions: train, test, run. With the train procedure you can use our pipeline to knowledge-distill our models or new models. Testing can be used to evaluate previous models. Finally, running can be used to run the models in a given dataset and produce the saliency maps. In the following we describe these in more details.


### Running
You can run a model by simply specifying the name of the model, the path to the input folder and the path to output folder as follows:

```bash
python main.py  -m "AIM" /path/to/input/folder/ /path/to/output/folder/
```

### Testing
You can evaluate the models for a given dataset 

```bash
python main.py test -m "AIM" --input_images ./path/to/input/images/ --input_saliencies ./path/to/saliency/images/
```

### Training
```bash
python main.py train -m "AIM" --input_images ./path/to/input/images/ --input_saliencies ./path/to/saliency/images/
```

## Configuration

There are more detailed configurations that you can do for each function (train, test, run) in more detail in the config.json file.

### config.json

We start by explaining the settings that can be made:

Parameter | Default | Description
------------ | ------------- | -------------
`verbose` | True | Whether the procedure should print the progress

[1]: https://tik-db.ee.ethz.ch/file/ce39d039b49f33a066d08e1c0ecb12f0/KDSalBox.pdf
[5]: https://conda.io/docs/user-guide/install/index.html
