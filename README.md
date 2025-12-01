
<div align="center">

# B2N3D: Progressive Learning from Binary to N-ary Relationships for 3D Object Grounding
**Feng Xiao** · **Hongbin Xu** · **Hai Ci** · **Wenxiong Kang**  
*South China University of Technology*  
*ByteDance Seed*  
*National University of Singapore*

[Code](https://github.com/onmyoji-xiao/B2N3D) | [Paper](#) | [Project Page](#)  

</div>

## Environment
```
git clone https://github.com/onmyoji-xiao/B2N3D.git
cd B2N3D/

conda create -n b2n3d python=3.10
conda activate b2n3d

pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt

cd external_tools/pointnet2
python setup.py install

cd external_tools/open_clip
pip install -e .
```

## Data Preparation
### ScanNet v2
Download the [ScanNet V2](http://www.scan-net.org/) dataset.  
Prepare for ScanNet data and package it into "scannet_00_views.pkl" or "scannet_0x_views.pkl"
```
cd data
python prepare_scannet_data.py --process-only-zero-view [1/0] -scans_dir [scannet_path]
```
