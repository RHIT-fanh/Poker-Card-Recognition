---
annotations_creators:
- machine-generated
language: []
language_creators:
- machine-generated
license:
- mit
multilinguality: []
pretty_name: Playing cards
size_categories:
- 10K<n<100K
source_datasets:
- original
tags:
- attribute
- concepts
task_categories:
- image-classification
- image-segmentation
task_ids:
- multi-label-image-classification
- instance-segmentation
---

# Dataset Card for Playing cards

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

### Dataset Summary

A dataset containing four sets of playing card images. Each set contains 10,000 images and has a series of attributes. Cards are randomly rotated, flipped and scaled (within limits).

Train and test splits are provided in both JSON and pickle formats. Concept and task classification labels (both zero indexed) and names are provided in txt files.

## Dataset Structure

### Data Instances

Each set of samples have the following:
* A set number of playing cards in each sample
* A list of concepts present in the each sample (1 for concepts present and 0 otherwise)
* The task classification label
* coordinates for each of the corners of playing cards in each sample.

The basic structure of the JSON and pkl files describing each sample is as follows:

```
sample ID, {
	'img_path': string file path,
	'concept_label': list of 0s and 1s,
	'class_label': integer,
	'card_points': list of tuples and card class labels as integers
}
```

#### Single

Single playing card on a random background.

* **Number of playing cards**: 1
* **Concepts**: Suit and rank
* **Class label**: Card classification
* **Card points**: Coordinates of the card and card classification

##### Example

```
1599, {
	'img_path': 'imgs/single/1599.png',
	'concept_label': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
	'class_label': 11,
	'card_points': [[[(657, 517), (405, 609), (520, 139), (268, 231)], 11]]
}
```

#### Three

Three randomly selected playing cards on a random background. Class label is set to hand rank for the game Three card poker.

* **Number of playing cards**: 3
* **Concepts**: Cards present
* **Class label**: Hand rank
* **Card points**: Coordinates of the cards and card classifications

##### Example

```
5159, {
	'img_path': 'imgs/three/5159.png',
	'concept_label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	'class_label': 4,
	'card_points': [[[(35, 517), (68, 374), (250, 567), (283, 424)], 15], [[(70, 364), (103, 221), (285, 413), (318, 270)], 24], [[(106, 210), (139, 67), (321, 260), (354, 117)], 13]]
}

```

#### Three card poker

Three playing cards on a random background. Class label is set to hand rank for the game Three card poker.

* **Number of playing cards**: 3
* **Concepts**: Cards present
* **Class label**: Hand rank
* **Card points**: Coordinates of the cards and card classifications

##### Example

```
9259, {
	'img_path': 'imgs/three_card_poker/9259.png',
	'concept_label': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	'class_label': 4,
	'card_points': [[[(42, 478), (84, 347), (237, 541), (279, 411)], 8], [[(325, 271), (282, 401), (129, 208), (87, 338)], 13], [[(370, 132), (328, 262), (175, 68), (133, 198)], 10]]
}
```

#### Class-level Three card poker

Three playing cards on a random background. Class label is set to hand rank for the game Three card poker. This set of samples have concepts set to the class. Every instance of the same class will have the same concept vector.

* **Number of playing cards**: 3
* **Concepts**: Cards present
* **Class label**: Hand rank
* **Card points**: Coordinates of the cards and card classifications

##### Example

```
5992, {
	'img_path': 'imgs/three_card_poker_class_level/5992.png',
	'concept_label': [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
	'class_label': 5,
	'card_points': [[[(539, 98), (574, 247), (317, 150), (351, 298)], 5], [[(388, 457), (354, 309), (610, 406), (576, 257)], 10], [[(613, 416), (647, 565), (390, 468), (425, 616)], 7]]
}
```

### Data Fields

* String file path from the root of the dataset to a given samples image file
* A list of concepts present in the each sample (1 for concepts present and 0 otherwise). The index of each value in this list corresponds to the label in concepts.txt.
* The task classification label. This corresponds the the label in classes.txt
* list of playing cards present in a given sample. Each item in the list has a list of card coordinates (card coordinates are always in the order top left, top right, bottom left, bottom right) and the card classification label, either corresponding to concepts.txt for images with three cards per image or classes.txt for images with a single card present.

### Data Splits

#### Single

##### Task classifications

| Class name | Count train | Count val |
| --- | --- | --- |
| 2C | 135 | 58 |
| 2D | 135 | 58 |
| 2H | 135 | 58 |
| 2S | 135 | 58 |
| 3C | 135 | 58 |
| 3D | 135 | 58 |
| 3H | 135 | 58 |
| 3S | 135 | 58 |
| 4C | 135 | 58 |
| 4D | 135 | 58 |
| 4H | 135 | 58 |
| 4S | 135 | 58 |
| 5C | 135 | 58 |
| 5D | 135 | 58 |
| 5H | 135 | 58 |
| 5S | 135 | 58 |
| 6C | 134 | 58 |
| 6D | 134 | 58 |
| 6H | 134 | 58 |
| 6S | 134 | 58 |
| 7C | 134 | 58 |
| 7D | 134 | 58 |
| 7H | 134 | 58 |
| 7S | 134 | 58 |
| 8C | 134 | 58 |
| 8D | 134 | 58 |
| 8H | 134 | 58 |
| 8S | 134 | 58 |
| 9C | 134 | 58 |
| 9D | 134 | 58 |
| 9H | 134 | 58 |
| 9S | 134 | 58 |
| 10C | 134 | 58 |
| 10D | 134 | 58 |
| 10H | 134 | 58 |
| 10S | 134 | 58 |
| JC | 134 | 58 |
| JD | 134 | 58 |
| JH | 134 | 58 |
| JS | 134 | 58 |
| QC | 134 | 58 |
| QD | 134 | 58 |
| QH | 134 | 58 |
| QS | 134 | 58 |
| KC | 134 | 58 |
| KD | 134 | 58 |
| KH | 134 | 58 |
| KS | 134 | 58 |

##### Concepts

| Concept name | Count train | Count val |
| --- | --- | --- |
| 2 | 540 | 232 |
| 3 | 540 | 232 |
| 4 | 540 | 232 |
| 5 | 540 | 232 |
| 6 | 536 | 232 |
| 7 | 536 | 232 |
| 8 | 536 | 232 |
| 9 | 536 | 232 |
| 10 | 536 | 232 |
| J | 536 | 232 |
| Q | 536 | 232 |
| K | 536 | 232 |
| A | 536 | 232 |
| C | 1746 | 754 |
| D | 1746 | 754 |
| H | 1746 | 754 |
| S | 1746 | 754 |

#### Three

##### Task classification

| Class name | Count train | Count val |
| --- | --- | --- |
| straight_flush | 20 | 2 |
| three_of_a_kind | 17 | 11 |
| straight | 268 | 99 |
| flush | 332 | 149 |
| pair | 1171 | 524 |
| high_card | 5191 | 2216 |

##### Concepts

| Concept name | Count train | Count val |
| --- | --- | --- |
| 2C | 398 | 181 |
| 2D | 441 | 161 |
| 2H | 385 | 143 |
| 2S | 397 | 170 |
| 3C | 439 | 171 |
| 3D | 383 | 165 |
| 3H | 398 | 181 |
| 3S | 435 | 179 |
| 4C | 407 | 164 |
| 4D | 402 | 179 |
| 4H | 409 | 168 |
| 4S | 403 | 191 |
| 5C | 402 | 150 |
| 5D | 373 | 173 |
| 5H | 383 | 187 |
| 5S | 426 | 178 |
| 6C | 394 | 176 |
| 6D | 414 | 172 |
| 6H | 398 | 184 |
| 6S | 413 | 163 |
| 7C | 409 | 171 |
| 7D | 412 | 158 |
| 7H | 391 | 185 |
| 7S | 453 | 176 |
| 8C | 390 | 171 |
| 8D | 398 | 171 |
| 8H | 406 | 148 |
| 8S | 368 | 193 |
| 9C | 381 | 187 |
| 9D | 429 | 167 |
| 9H | 391 | 193 |
| 9S | 370 | 174 |
| 10C | 450 | 171 |
| 10D | 420 | 161 |
| 10H | 436 | 180 |
| 10S | 406 | 169 |
| JC | 416 | 160 |
| JD | 411 | 176 |
| JH | 409 | 182 |
| JS | 404 | 178 |
| QC | 403 | 171 |
| QD | 376 | 185 |
| QH | 407 | 182 |
| QS | 420 | 156 |
| KC | 414 | 180 |
| KD | 384 | 176 |
| KH | 377 | 157 |
| KS | 382 | 192 |
| AC | 348 | 168 |
| AD | 408 | 177 |
| AH | 427 | 174 |
| AS | 401 | 178 |

#### Three card poker

##### Task classification

| Class name | Count train | Count val |
| --- | --- | --- |
| straight_flush | 1166 | 501 |
| three_of_a_kind | 1166 | 501 |
| straight | 1166 | 501 |
| flush | 1166 | 501 |
| pair | 1166 | 500 |
| high_card | 1166 | 500 |

##### Concepts

| Concept name | Count train | Count val |
| --- | --- | --- |
| 2C | 344 | 171 |
| 2D | 368 | 181 |
| 2H | 386 | 161 |
| 2S | 359 | 163 |
| 3C | 400 | 186 |
| 3D | 421 | 181 |
| 3H | 414 | 181 |
| 3S | 407 | 172 |
| 4C | 409 | 185 |
| 4D | 388 | 199 |
| 4H | 408 | 185 |
| 4S | 411 | 195 |
| 5C | 403 | 176 |
| 5D | 409 | 191 |
| 5H | 401 | 198 |
| 5S | 392 | 206 |
| 6C | 422 | 177 |
| 6D | 405 | 194 |
| 6H | 444 | 161 |
| 6S | 412 | 175 |
| 7C | 420 | 181 |
| 7D | 404 | 189 |
| 7H | 429 | 156 |
| 7S | 433 | 158 |
| 8C | 422 | 173 |
| 8D | 436 | 159 |
| 8H | 412 | 167 |
| 8S | 431 | 166 |
| 9C | 407 | 186 |
| 9D | 405 | 174 |
| 9H | 432 | 169 |
| 9S | 398 | 185 |
| 10C | 416 | 164 |
| 10D | 413 | 174 |
| 10H | 430 | 171 |
| 10S | 402 | 176 |
| JC | 431 | 170 |
| JD | 462 | 158 |
| JH | 443 | 145 |
| JS | 405 | 186 |
| QC | 401 | 186 |
| QD | 432 | 163 |
| QH | 419 | 187 |
| QS | 397 | 172 |
| KC | 363 | 157 |
| KD | 358 | 178 |
| KH | 367 | 157 |
| KS | 364 | 155 |
| AC | 380 | 155 |
| AD | 363 | 151 |
| AH | 359 | 150 |
| AS | 351 | 156 |

#### Class-level Three card poker

##### Task classification

| Class name | Count train | Count val |
| --- | --- | --- |
| straight_flush | 1166 | 501 |
| three_of_a_kind | 1166 | 501 |
| straight | 1166 | 501 |
| flush | 1166 | 501 |
| pair | 1166 | 500 |
| high_card | 1166 | 500 |

##### Concepts

| Concept name | Count train | Count val |
| --- | --- | --- |
| 2H | 1166 | 501 |
| 3H | 2332 | 1002 |
| 4C | 2332 | 1002 |
| 4D | 2332 | 1002 |
| 4H | 1166 | 501 |
| 4S | 2332 | 1001 |
| 5C | 1166 | 500 |
| 5D | 3498 | 1501 |
| 6D | 1166 | 501 |
| 9D | 1166 | 501 |
| 10H | 2332 | 1000 |

## Dataset Creation

### Curation Rationale

This dataset was created to test Concept Bottleneck Models [1] with instance and class level concepts.

### Source Data

#### Initial Data Collection and Normalization

The dataset uses background from [2] and playing card images from [3]. The dataset is balanced to the task classification labels with concepts, backgrounds and card transformations being applied randomly. The code used to generate the dataset is available here [4].

### Annotations

#### Annotation process

The annotation process was completed during the generation of the dataset.

#### Who are the annotators?

Annotations were completed by a machine.

### Personal and Sensitive Information

This dataset does not contain personal and sensitive Information.

## Additional Information

### Licensing Information

This dataset is licenced with the [MIT licence](https://choosealicense.com/licenses/mit/).

### Citation Information

[1] Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B. &amp; Liang, P.. (2020). Concept Bottleneck Models. Proceedings of the 37th International Conference on Machine Learning, in Proceedings of Machine Learning Research 119:5338-5348 Available from https://proceedings.mlr.press/v119/koh20a.html.

[2] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed and A. Vedaldi, "Describing Textures in the Wild," 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 3606-3613, doi: 10.1109/CVPR.2014.461.

[3] j4p4n, "Full Deck Of Ornate Playing Cards - English", Available at: https://openclipart.org/download/315253/1550166858.svg

[4] J. Furby, "playing-card-concept-generator", Available at: https://github.com/JackFurby/playing-card-concept-generator
