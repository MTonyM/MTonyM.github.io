---
layout: archive
title: "About my ROI"
permalink: /roi/
author_profile: true
redirect_from:
  - /roi
  - /researchofinterests
  - /yanjiuxingqu
---

Tell You About the Nowadays' CV
======
In this page, I will present several topics in Computer Vision
that I considered as important tasks. Each topic gives my insights towards it.
If you think it is really interesting and want to discuss with me, please contact 
[me](mailto:maoym.troy@gmail.com) without hesitation.

Classic Tasks in Computer Vision.
======

Interpretable models/XAI: how black box works?
---

As we know, `neural networks` are not as simple as `linear regression`, 
and complex, deep layers and skip connections make the 
structure confusing (unlike the decision tree, the partitions 
on each node are interpretable). Before the gradient-descent based
neural network, a semantic convolution kernel, such as the `Sobel` operator 
and the `Robert cross-gradient` operator, are manually set 
for specific tasks (edge detection). During the era of these operators,
people are *hand-designing* these kernels with wisdom. 
When the convolutional neural network came out, these multi-layered 
convolution kernels of self-learning became an unsolved mystery. 
***What actually did they learn?***

There are many work on interpretable deep neural networks, 
such as Professor Bolei Zhou interpretability for a network
defined as the number of interpretable kernels. 
What is an interpretable kernel? Bolei found that after the 
activated layers, the extracted feature map has the 
characteristics of high activation value in specific
semantics. For example, the feature map of the last layer
of VGG-19 has peaks on specific objects such as the 
dog's ears and nose. What is funny is that, the kernel which highly activate
in the dog's ears also activate in the cat's ears(they are 
in similar shapes and textures). This might be the knowledge that the
VGG-19 learned.

Another very interesting work about interpretable models is 
Professor Songchun Zhu's architecture with graph embedded, aiming to 
design a structure with interpretable models. My past-supervisor Xuming He
also very interested in this topic, his idea is to distill network 
knowledge into a interpretable model via Teacher-Student network method.

*My Idea:* Maybe there is some distribution deep into the kernels 
or the activated map. I am going to use unsupervised learning 
method to analysis the distribution. Details under editing.

Generative models: detailed wins!
---
`under construction`

Image-based tasks: backbones win!
---
`under construction`

