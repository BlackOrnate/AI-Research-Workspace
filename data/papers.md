# Local Paper Library

Simplified reading summaries of papers in the user's local library, written for learning purposes.
Each paper is one "##" section. For exact details, always check the original paper.

## HoVer-Net

- Full title: HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images
- Authors: Simon Graham et al.
- Year / Venue: 2019, Medical Image Analysis
- Task: Nuclear instance segmentation and nuclear type classification in H&E histology images
- Method: Predicts, for every nuclear pixel, its horizontal and vertical distance to the nucleus center of mass. Sharp changes in these distance maps mark boundaries between touching nuclei, which are then separated with a post-processing step. Separate branches predict nuclear pixels, horizontal/vertical maps, and nuclear type.
- Backbone: Pre-activated ResNet-50 encoder with multiple decoder branches
- Datasets: CoNSeP (introduced in this paper), Kumar, CPM-15, CPM-17, TNBC, CRCHisto
- Metrics: Dice, AJI, and Panoptic Quality (PQ = DQ x SQ)
- Limitations: Relies on hand-designed post-processing to split nuclei; convolutional encoder has a limited receptive field compared with Transformer-based models.

## CellViT

- Full title: CellViT: Vision Transformers for Precise Cell Segmentation and Classification
- Authors: Fabian Hörst et al.
- Year / Venue: 2024, Medical Image Analysis
- Task: Nuclear instance segmentation and classification in H&E histology images
- Method: Replaces the convolutional encoder of a HoVer-Net style network with a Vision Transformer, connected to a U-Net style decoder through skip connections. Keeps the HoVer-Net style horizontal/vertical distance maps for instance separation.
- Backbone: Vision Transformer encoders, including ViT-256 pretrained with HIPT on histology, and SAM image encoders (ViT-B, ViT-L, ViT-H)
- Datasets: PanNuke (main benchmark), MoNuSeg (generalization test)
- Metrics: Binary PQ (bPQ), multi-class PQ (mPQ), and detection F1 score
- Limitations: Large ViT encoders (especially SAM ViT-H) are heavy to train and run; still depends on HoVer-Net style post-processing.

## Cellpose

- Full title: Cellpose: a generalist algorithm for cellular segmentation
- Authors: Carsen Stringer, Tim Wang, Michalis Michaelos, Marius Pachitariu
- Year / Venue: 2021, Nature Methods
- Task: Generalist cell and nucleus segmentation across many microscopy image types, without retraining
- Method: Converts each ground-truth mask into a "flow field" produced by simulated heat diffusion from the cell center. The network predicts horizontal and vertical flows plus a cell probability map; pixels are grouped into cells by following the flows to their sinks.
- Backbone: U-Net style convolutional network with residual blocks and a global style vector
- Datasets: A new diverse dataset of over 70,000 segmented objects, including cytoplasm, nuclei and non-cell images
- Metrics: Average precision (AP) at different IoU thresholds
- Limitations: Generalist model may need fine-tuning for unusual cell shapes; not designed for nuclear type classification.

## StarDist

- Full title: Cell Detection with Star-convex Polygons
- Authors: Uwe Schmidt, Martin Weigert, Coleman Broaddus, Gene Myers
- Year / Venue: 2018, MICCAI
- Task: Nucleus detection and instance segmentation in microscopy images
- Method: For every pixel, predicts an object probability and the distances to the object boundary along a fixed set of radial directions, forming a star-convex polygon. Non-maximum suppression keeps the best polygon per object.
- Backbone: U-Net
- Datasets: Nuclei images from the 2018 Data Science Bowl (DSB2018), plus other microscopy datasets
- Metrics: Average precision (AP) at different IoU thresholds
- Limitations: Assumes star-convex shapes, so it struggles with irregular or elongated cells.

## Segment Anything (SAM)

- Full title: Segment Anything
- Authors: Alexander Kirillov et al. (Meta AI)
- Year / Venue: 2023, ICCV
- Task: Promptable segmentation of any object in natural images, from points, boxes or masks
- Method: A heavy image encoder computes an image embedding once; a lightweight prompt encoder and mask decoder then produce masks for each prompt in real time. Trained with a data engine that iteratively collected masks.
- Backbone: Vision Transformer image encoder pretrained with MAE (ViT-B, ViT-L, ViT-H)
- Datasets: SA-1B, with about 11 million images and 1.1 billion masks
- Metrics: mIoU on single-point prompts, and zero-shot transfer on many downstream tasks
- Limitations: Trained on natural images; performance drops on medical and histology images without adaptation, and it does not output semantic classes.

## MedSAM

- Full title: Segment Anything in Medical Images
- Authors: Jun Ma et al.
- Year / Venue: 2024, Nature Communications
- Task: Universal promptable segmentation for medical images across modalities
- Method: Fine-tunes SAM on a large medical dataset, using bounding-box prompts to indicate the target.
- Backbone: SAM ViT-B image encoder, prompt encoder and mask decoder
- Datasets: About 1.57 million medical image-mask pairs covering 10 imaging modalities (CT, MRI, endoscopy, ultrasound, pathology and others) and over 30 cancer types
- Metrics: Dice Similarity Coefficient (DSC) and Normalized Surface Distance (NSD)
- Limitations: Needs a bounding-box prompt for each target; dataset is imbalanced across modalities.
