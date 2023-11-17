# Data Augmentation for Small Sample

Data augmentation is a technique that can enhance the quality and quantity of data for machine learning models. Data augmentation can help to overcome the challenges of limited or imbalanced data sets, reduce overfitting, and improve generalization. Data augmentation can be applied to different types of data, such as images, text, audio, and video.

This repository contains various examples and methods of data augmentation for different domains and tasks. The repository also provides some code snippets and libraries that can help you to implement data augmentation in your own projects. The goal of this repository is to provide a comprehensive and accessible resource for data augmentation, and to inspire you to explore new and creative ways of augmenting your data.

# Datasets 

## Spam 

SMS spam is a type of unsolicited message that is sent to mobile phones for various purposes, such as advertising, phishing, or scamming. SMS spam can be annoying, intrusive, and potentially harmful for the recipients. Therefore, it is important to develop methods and tools that can identify and filter out SMS spam messages.

One possible way to do this is to use machine learning techniques that can learn from a collection of labeled SMS messages and classify new messages as spam or ham (legitimate). To train and evaluate such machine learning models, we need a dataset that contains SMS messages with their corresponding labels.

These datasets can be useful for developing and testing SMS spam detection models, as they provide a variety of messages with different characteristics and styles. However, these datasets also have some limitations, such as:

- They may not reflect the current trends and patterns of SMS spam, as they were collected in different periods and regions.
- They may not cover all the possible types and categories of SMS spam, as they were based on limited sources and samples.
- They may have some errors or inconsistencies in the labeling process, as they were done manually or semi-automatically.

Additionally, it is important to apply proper data preprocessing and analysis techniques, such as cleaning, filtering, tokenizing, and feature extraction, to improve the quality and usability of the data.

## Library 

This dataset includes book titles, and category for each respective book. The categories are divided into 7 classes. The format of the dataset is a text file, where each line has the correct class followed by the raw title. 

This dataset can be used to train and evaluate machine learning models that can classify books by their titles. However, some challenges and limitations of this dataset are:

- The titles may not be sufficient to determine the class of the book, as some titles may be ambiguous, vague, or misleading.
- The classes may not be mutually exclusive, as some books may belong to more than one category or subcategory.
- The dataset may not reflect the current trends and preferences of the readers, as it was collected in different periods and regions.

# Techniques

## Snorkel 

Snorkel is a Python package that helps you to create and manage large-scale labeled datasets for machine learning. Snorkel allows you to programmatically label, augment, and transform your data using weak supervision sources, such as heuristics, knowledge bases, or other models. Snorkel also provides tools for analyzing, debugging, and improving your data and models.

Snorkel is designed to help you with the following tasks:

![Snorkel Functions](/refs/images/snorkel_f.png)

- Labeling: You can use Snorkel to generate noisy labels for your unlabeled data using various weak supervision sources, such as labeling functions, crowdsourcing, or distant supervision. Snorkel then combines these labels into a single probabilistic label for each data point using a generative model.
- Augmentation: You can use Snorkel to augment your data by applying transformations that preserve the label and increase the diversity of your data. Snorkel supports various types of transformations, such as text replacement, image rotation, or data synthesis.
- Slicing: You can use Snorkel to slice your data into different subsets based on certain criteria, such as keywords, patterns, or features. Snorkel helps you to identify and monitor the performance of your model on different slices of your data.

Snorkel is an open-source project that is actively developed and maintained by a community of researchers and practitioners. Snorkel is compatible with popular machine learning frameworks, such as PyTorch, TensorFlow, and Scikit-learn. Snorkel has been used for various applications, such as information extraction, natural language processing, computer vision, and healthcare.

To install Snorkel, you can use pip¹ or conda². To learn more about Snorkel, you can visit the official website³, read the documentation, or check out the tutorials.

[Snorkel Project](www.google.com)

## Pytorch

PyTorch is a Python package that provides two high-level features: **Tensor computation** (like NumPy) with strong GPU acceleration and **Deep neural networks** built on a tape-based autograd system¹. PyTorch is designed to be flexible, intuitive, and expressive, making it a popular choice for researchers and developers who want to build dynamic and scalable machine learning applications.

PyTorch has the following key features and capabilities:

- **Production Ready**: PyTorch supports seamless transition between eager and graph modes with TorchScript, and accelerates the path to production with TorchServe¹.
- **Distributed Training**: PyTorch enables scalable distributed training and performance optimization in research and production using the torch.distributed backend¹.
- **Robust Ecosystem**: PyTorch has a rich ecosystem of tools and libraries that extend its functionality and support development in various domains, such as computer vision, natural language processing, audio, and more¹.
- **Cloud Support**: PyTorch is well supported on major cloud platforms, providing frictionless development and easy scaling¹.
- **torch.package**: PyTorch adds support for creating packages containing both artifacts and arbitrary PyTorch code. These packages can be saved, shared, used to load and execute models at a later date or on a different machine, and can even be deployed to production using torch::deploy².

[PyTorch](https://pytorch.org/)

