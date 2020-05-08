# DeblurGAN

## Abstract:

Motion blur in videos and images is a fundamental problem in computer vision which leads to significant degradation of the image, impacting its quality. Many computer vision tasks like image/video analysis, object detection, and recognition rely on local features and descriptors and it is well known that motion blur decreases the performance of the traditional detectors and descriptors. Hence, in this project, we attempt to re-implement the DeblurGAN; an end-to-end learned method for motion deblurring, based on conditional GAN and content loss. We test the quality of de-blurred images by evaluating it on a real-world problem – Object detection. Here, we try to re-implement the image deblurring model presented by https://arxiv.org/pdf/1711.07064v4.pdf .

## Generator Architecture

![Generator Architecture](https://user-images.githubusercontent.com/53349721/78530733-34169280-77b2-11ea-9143-e042882782de.png)

## Discriminator Architecture

![Discriminator Architecture](https://user-images.githubusercontent.com/53349721/78531160-e6e6f080-77b2-11ea-9892-084a2bd17d46.png)

### The different files in this repository are as below:

1) Generator.py: Consists of the definition of the Generator class and the Resnet block.

2) Discriminator.py: Consists of the definition of the Discriminator class.

3) CreateDataLoader: Consists of the definition of custom dataloader used to load the images for training and testing.

4) Losses.py: 

5) utils.py: Consists of the definitions of various helper functions used throughout the code.

6) metrics.py: Consists of the definitions of metrics used to check the performance of the model.

7) constants.py: Consists of the values of different constants used throughout the project code.

8) train.py: Consists the code for actual training of model.

9) test.py: Consists the code for testing the model.