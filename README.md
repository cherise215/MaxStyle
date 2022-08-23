# MaxStyle
[MICCAI 2022] MaxStyle: Adversarial Style Composition for Robust Medical Image Segmentation

## Introduction:
MaxStyle is a novel feature-space data augmentation, which maximizes the effectiveness of style augmentation for model out-of-domain performance. MaxStyle augments data with improved image style diversity and hardness, by expanding the style space with noise and searching for the worst-case style composition of latent features via adversarial training. 

Where to insert MaxStyle? Well, MaxStyle can be inserted after any conv blocks in a neural network, preferred to be applied to augmenting *low-level* features. In order to directly visualize the effect of MaxStyle and use it for standard image space data augmentation, in the paper, we design an auxiliary style-augmented image decoder. This decoder is attached to a segmentation network, turning the network to a dual-branch network for both image segmentation and image reconstruction/style augmentation. We found such a design can improve not only the interpretability of feture augmentation, but also the robustness of the segmentation network. We recommend to randomly insert MaxStyle in the low-level feature output from the decoder, e.g., insert MaxStyle after the last 1 st, 2nd, 3rd blocks.

Core implementation can be found at: 
- MaxStyle layer with learnable paprameters for smart style pertubation: [src/advanced/maxstyle.py](src/advanced/maxstyle.py).
- An example of an image decoder with MaxStyle layers inserted [src/models/ebm/encoder_decoder.py](src/models/ebm/encoder_decoder.py), see MyDecoder.apply_max_style().
- Code for generating and optimizing style augmentation at training can be found at [src/models/advanced_triplet_recon_segmentation_model.py](src/models/advanced_triplet_recon_segmentation_model.py), see generate_max_style_image(). 
- Main training and testing code: [src/train_adv_supervised_segmentation_triplet.py](src/train_adv_supervised_segmentation_triplet.py) 

## News:
- [x] MaxStyle training and testing code release [2022-8-23]

Incoming:
- [ ] Add jupyter notebook for ease of result analysis
- [ ] Add visualization of MaxStyle generated images

## Set Up
- Git clone this project
- (optional) Create a fresh Python 3.7.x virtual environment.
- Install PyTorch and other required python libraries with:
  `pip install -r requirements.txt`
- Install torchsample for data augmentation
  `pip install git+https://github.com/ozan-oktay/torchsample/`
## Data
### Cardiac multi-site, multi-sequence image segmentation dataset
In our paper, we trained our model on ACDC dataset and test it on a large-scale dataset, containing both corrupted and uncorrupted datasets from public benchmark datasets. However, due to limitations on redistribution put in place by the original data distributor, we cannot directly share data with you. But you can apply to download these cardiac data at:.
- [ACDC2017](https://acdc.creatis.insa-lyon.fr/#challenge/5846c3366a3c7735e84b67ec)
- [M&M](https://www.ub.edu/mnms/)
- [MS-CMRSeg19](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg19/data.html)

We are also happy to provide our preprocessed data with you, once you have been formally granted with all these permission to use these datasets.

### Prostate multi-site MRI segmentation
This large-scale multi-site prostate dataset contains prostate T2-weighted MRI data (with segmentation mask) collected from SEVEN different data sources out of FOUR public datasets, NCI-ISBI 2013 dataset [1], Initiative for Collaborative Computer Vision Benchmarking (I2CVB) dataset [2], Prostate MR Image Segmentation 2012 (PROMISE12) dataset [3], Medical Decathlon [4]. In our experiments, we use the last one for training and intra-domain testing, while the rest three are used for testing. The training dataset can be downloaded from the [medical segmentation decathlon website](http://medicaldecathlon.com/). Test datasets can be originally downloaded from https://liuquande.github.io/SAML/

 We are also happy to share our preprocessed data here, which can be downloaded at [Google Drive](). Please note that this dataset is for research purpose ONLY.All images have been resampled to have uniform voxel spacings [0.625,0.625,3.6]. We remove black slices that do not contain prostate. All labels have been converted to binary masks (covering PZ+TZ) for binary segmentation, with 0 for background and 1 for foreground. 

Once downloaded, unzip it and put files under the `MaxStyle/data` dir, you can find that images and labels are re-oargnized as follows:
 - MaxStyle/data/MICCAI2022_multi_site_prostate_dataset/reorganized
    - A-ISBI
        - patient_1 
            - t2_img_clipped.nii.gz
            - label_clipped.nii.gz
        ...
    - B-ISBI_1.5
        - patient_1 
            - t2_img_clipped.nii.gz
            - label_clipped.nii.gz
        ...
    - C-I2CVB
    - D-UCL
    - E-BIDMC
    - F-HK
    - G-MedicalDecathlon

## Training: Standard training with a dual-branch network
### Cardiac low-data regime segmentation (10 subjects)
- Standard training:
    ```
    cd path/to/MaxStyle
    cd /vol/biomedic3/cc215/Project/MaxStyle;
    source activate <your virtual env>;
    CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/ACDC/1500_epoch/standard_training.json --cval 0 --seed 40 --data_setting 10 --auto_test
    ```
- MaxStyle training with a dual-branch network
    ```
    CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/ACDC/1500_epoch/MICCAI2022_MaxStyle.json --cval 0 --seed 40 --data_setting 10 --auto_test
    ```
We also provide other baseline methods: Adversarial bias field and Latent space masking based data augmentation
```
   CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/ACDC/1500_epoch/MICCAI2020_AdvBias.json --cval 0 --seed 40 --data_setting 10 --auto_test
```
### Cardiac high-data regime segmentation (70 subjects)
simply change `--data_setting 10` to `--data_setting 'standard'`.
- Standard training:
    ```
    cd path/to/MaxStyle
    cd /vol/biomedic3/cc215/Project/MaxStyle;
    source activate <your virtual env>;
    CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/ACDC/1500_epoch/standard_training.json --cval 0 --seed 40 --data_setting 'standard' --auto_test
    ```
- MaxStyle training with a dual-branch network
    ```
    CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/ACDC/1500_epoch/MICCAI2022_MaxStyle.json --cval 0 --seed 40 --data_setting 'standard' --auto_test
    ```
### Prostate image segmentation (70 subjects)
Before running, please make sure the root path in the configuration file has been changed to your local path:
i.e., "root_dir": "path/to/prostate_multi_domain_data/reorganized/G-MedicalDecathlon",
- Standard training:
    ```
    cd path/to/MaxStyle
    cd /vol/biomedic3/cc215/Project/MaxStyle;
    source activate <your virtual env>;
    CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/Prostate/standard_training.json --cval 0 --seed 40 --data_setting 'all' --auto_test
    ```
- MaxStyle training with a dual-branch network
    ```
    CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/Prostate/MICCAI2022_MaxStyle.json  --cval 0 --seed 40 --data_setting 'all' --auto_test
    ```

# Evaluation
By default, we run model evaluation automatically after training with `--auto-test`. Model parameters and results will all be saved under `saved/`. To re-run inference of a trained model without training, simply run the same training command with additional `--no_train` option on:
```python
CUDA_VISIBLE_DEVICES=<gpu id> python src/train_adv_supervised_segmentation_triplet.py --cval <cval id> --seed <seed number> --data_setting <data setting identifier> --json_config_path <path/to/json_config_file> --data_setting 10 --auto_test --no_train ;
```
e.g., to test the above trained model, one can simply run:

``` python
CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --cval 0 --seed 40 --json_config_path ./config/ACDC/1500_epoch/MICCAI2022_MaxStyle.json --log --data_setting 10 --no_train;
```
This will auto test the model saved with the highest validation accuracy during training. To test the model saved from the last epoch, simply turn on '--use_last_epoch'

```python
CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --cval 0 --seed 40 --json_config_path ./config/ACDC/1500_epoch/MICCAI2022_MaxStyle.json --log --data_setting 10 --no_train --use_last_epoch;
```

## Visualization of loss curves and training progress:
```python
cd path/to/MaxStyle;
tensorboard --logdir ./saved --port 6066 --bind_all
```
Note: 6066 is a port number, which can be changed to other numbers. 

# Project structure and core files
- `config`: folder contains configuration for training deep learning models
    - ACDC: folder contains configurations for training cardiac segmentation using ACDC training data
    - Prostate: folder contains configurations for training cardiac segmentation using ACDC training data
- `src`: main code repository
    - `advanced`: contains implementation of advanced data augmentation
        - maxstyle.py: max style layer implementation
    - `dataset_loader`: contains code for loading different datasets
    - `models`: folder contains different basic neural nets and advanced neural networks 
        - advanced_triplet_recon_segmentation_model.py: dual-branch network implementation supports hard example training from latent space data augmentation
    - `train_adv_supervised_segmentation_triplet.py`: code for training and testing
- `saved`: contains saved checkpoints and test scores and log files (will appear after training)
   e.g.
   - `train_ACDC_10_n_cls_4/ACDC/1500_epoch/MICCAI2022_MaxStyle/0/model/best/checkpoints`: contains csv files reporting model performance. Here, the model was trained on ACDC dataset using 10 subjects for 1500 epochs and then evaludated on different testsets (only exists after testing). e.g. 
        - saved params: `image_decoder.pth`, `image_encoder.pth`, `segmentation_decoder.pth`
        - patient wise scores (Dice):`report/<dataset name>/<iter_1_detailed.csv>`
        - Average Dice scores for each dataset: `report/<dataset name>/<iter_1_summary.csv>`
        - Summary report contains scores for all test sets: `report/dataset_summary.csv`
   -  `train_ACDC_10_n_cls_4/ACDC/1500_epoch/MICCAI2022_MaxStyle/0/model/best/testing_segmentation_results/Seg_plots.png`: visualiztion of model prediction on validation set.
   - `train_ACDC_10_n_cls_4/config/ACDC/1500_epoch/MICCAI2022_MaxStyle/log`: contains training log file for tensorboard visualization


# Citation
To cite MaxStyle in your publications, please use the following bibtex entry
```
@inproceedings{Chen2022MaxStyle,
  title="{MaxStyle}: Adversarial Style Composition for Robust Medical Image Segmentation",
  author        = "Chen, Chen and Li, Zeju and Ouyang, Cheng and Sinclair, Matthew  
                   and Bai, Wenjia and Rueckert, Daniel",
  booktitle     =  {MICCAI},
  year          =  {2022},
  eprint        = "2206.01737"
}
```


# Acknowledgements
We sincerely thank the organizers and collaborators of NCI-ISBI13 Challenge [1], I2CVB dataset [2] and PROMISE12 Challenge [3], Medical Decathlon [4]for sharing the data for public use. We thank Quande Liu (qdliu@cse.cuhk.edu.hk) for sharing the public multi-site prostate segmentation datasets[1-3]. Basic information of these datasets can be found at https://liuquande.github.io/SAML/. We also thank Kaiyang Zhou for his pioneering [MixStyle](https://github.com/KaiyangZhou/mixstyle-release) work. 

# Reference
- [1] Bloch, N., Madabhushi, A., Huisman, H., Freymann, J., et al.: NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures. (2015)

- [2] Lemaitre, G., Marti, R., Freixenet, J., Vilanova. J. C., et al.: Computer-Aided Detection and diagnosis for prostate cancer based on mono and multi-parametric MRI: A review. In: Computers in Biology and Medicine, vol. 60, pp. 8-31 (2015)

- [3] Litjens, G., Toth, R., Ven, W., Hoeks, C., et al.: Evaluation of prostate segmentation algorithms for mri: The promise12 challenge. In: Medical Image Analysis. , vol. 18, pp. 359-373 (2014).

- [4] Antonelli, Michela, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Annette Kopp-Schneider, Bennett A. Landman, Geert Litjens, et al. 2022. “The Medical Segmentation Decathlon.” Nature Communications 13 (1): 1–13.