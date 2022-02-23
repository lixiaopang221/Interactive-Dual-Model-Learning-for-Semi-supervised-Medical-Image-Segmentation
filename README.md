## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Download the data(ACDC LiTS BraTS2019) and put the data in `../data/BraTS2019` 、 `../data/ACDC` or `../data/LiTS`.

2. Train the model
```
python train_3D.py or python train_2D.py or python train_train_mobilenet_2D.py
```

3. Test the model
```
python test_XXXXX.py
```
# Related methods
* [Mean Teacher](https://papers.nips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf)
* [Entropy Minimization](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf)
* [Deep Adversarial Networks](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_47)
* [Uncertainty Aware Mean Teacher](https://arxiv.org/pdf/1907.07034.pdf)
* [Interpolation Consistency Training](https://arxiv.org/pdf/1903.03825.pdf)
* [Dual student: Breaking the limits of the teacher in semi-supervised learning](https://arxiv.org/abs/1909.01804)
* [Transformation-consistent self-ensembling model for semi-supervised medical image segmentation](https://arxiv.org/pdf/1903.00348.pdf)
* [Feature-map-level online adversarial knowledge distillation](http://proceedings.mlr.press/v119/chung20a/chung20a.pdf)
* [DualNet:Learn complementary features for image recognition](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_DualNet_Learn_Complementary_ICCV_2017_paper.pdf)
* [Deep mutual learning](https://link.zhihu.com/?target=https%3A//github.com/YingZhangDUT/Deep-Mutual-Learning)
## Acknowledgement
* Part of the code is adapted from open-source codebase and original implementations of algorithms, we thank these author for their fantastic and efficient codebase, such as, [UA-MT](https://github.com/yulequan/UA-MT). 
