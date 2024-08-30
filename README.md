# DSCN
Hi! You are welcome to visit here! This repository will be used to release the code of a novel tracking method called the Discriminative Sparse Convolution Network (DSCN), which has been proposed in our paper entitled "Online Learning Discriminative Sparse Convolution Networks for Robust UAV Object Tracking" submitted to the International journal of Knowledge-Based Systems (KBS). **NOTE that once our manuscript is accepted by the journal, we will release the code and experimental results immediately**. 

We propose a Discriminative Sparse Convolution Network (DSCN) for robust UAV object tracking, which introduces convolutional layers with explicit data meaning and can be learned online without using any auxiliary training data. By imposing distinct sparsity constraints on the foreground and background image blocks, we develop a sparse filter learning method to acquire a discriminative sparse dictionary online and employ its atoms as convolution kernels termed discriminative sparse filters. These discriminative sparse filters have explicit data constraints, that is, they are close to the foreground data and far away from the background data, making the convolutional layers have explicit data meaning — obtaining discriminative features that can distinguish between foreground and background regions. On the other hand, we propose a simple and effective sequential learning method based on recursive least square (RLS), which directly learns the weights of fully connected layers in DSCN online from sequentially appearing data streams. This sequential learning method eliminates the backpropagation process that requires parameter adjustments (such as learning rate and weight decay), and improves the robustness of the tracking algorithm

# Experimental Datasets
* **UAV123@10\_fps** [1]. (https://cemse.kaust.edu.sa/ivul/uav123 ).  UAV123@10fps contains 123 fully annotated high-definition video sequences captured from low-altitude drones, spanning a wide range of scenes and objects from an aerial perspective. 

* **DBT70** [2]. (https://github.com/flyers/drone-tracking).  DTB70 aims to solve the difficult problem of significant camera motion under various extreme conditions. The design purpose of UAVDT is to focus on complex situations such as weather, altitude, camera views, vehicle types, and occlusion.

* **UAVDT** [3]. (https://sites.google.com/site/daviddo0323/projects/uavdt). UAVDT is to focus on complex situations such as weather, altitude, camera views, vehicle types, and occlusion.

* **UAVTrack112** [4]. (https://github.com/vision4robotics/SiamAPN).  UAVTrack112 is a long-term tracking benchmark, where the video sequences were collected and annotated from the real world with aerial-specific challenges. 

* **VisDrone-SOT2020-test** [5]. ( http://aiskyeye.com/). VisDrone-SOT2020-test is based on visually satisfying the challenge of single object tracking for unmanned aerial vehicles. This dataset was collected using various drone platforms under different weather and lighting conditions in various real-world environments.

* **Anti-UAV410** [6]. (https://github.com/HwangBo94/Anti-UAV410).  Anti-UAV410 is a Thermal Infrared (TIR) benchmark used for anti-UAV systems. The tracking videos were captured in diverse and complex scenarios, such as varying lighting conditions, different seasons, and a wide range of backgrounds.




References:

[1] Matthias Mueller, Neil Smith, Bernard Ghanem: A Benchmark and Simulator for UAV Tracking. ECCV (1) 2016: 445-461 https://doi.org/10.1007/978-3-319-46448-0\_27

[2] Siyi Li, Dit-Yan Yeung: Visual Object Tracking for Unmanned Aerial Vehicles: A Benchmark and New Motion Models. AAAI 2017: 4140-4146 https://doi.org/10.1609/aaai.v31i1.11205

[3] Hongyang Yu, Guorong Li, Weigang Zhang, Qingming Huang, Dawei Du, Qi Tian, Nicu Sebe: The Unmanned Aerial Vehicle Benchmark: Object Detection, Tracking and Baseline. Int. J. Comput. Vis. 128(5): 1141-1159 (2020) https://doi.org/10.1007/s11263-019-01266-1

[4] Changhong Fu, Ziang Cao, Yiming Li, Junjie Ye, Chen Feng: Onboard Real-Time Aerial Tracking With Efficient Siamese Anchor Proposal Network. IEEE Trans. Geosci. Remote. Sens. 60: 1-13 (2022) https://doi.org/10.1109/TGRS.2021.3083880

[5] Heng Fan, Longyin Wen, Dawei Du, Pengfei Zhu, Qinghua Hu, Haibin Ling, et al.: VisDrone-SOT2020: The Vision Meets Drone Single Object Tracking Challenge Results. ECCV Workshops (4) 2020: 728-749 https://doi.org/10.1007/978-3-030-66823-5\_44

[6] Bo Huang, Jianan Li, Junjie Chen, Gang Wang, Jian Zhao, Tingfa Xu: Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild. IEEE Trans. Pattern Anal. Mach. Intell. 46(5): 2852-2865 (2024)  https://doi.org/10.1109/TPAMI.2023.3335338
