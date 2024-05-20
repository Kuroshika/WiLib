# WiLib

A toolbox to test deeplearning models on Wi-Fi Human Recognition.

## Installation

Try to execute below script on a linux computer with conda and git:

```bash
git clone [url]
cd [wilib]
conda create -n wifilib python==3.7
conda activate wifilib
pip install -r requirements.txt
```

## Run

```bash
python ./run.py -config [path to config] -debug False
```

The example configurations are under the 'config' directory.

## Model Zoo

<table>
    <thead>
    <tr>
        <th>Model</th>
        <th>Paper</th>
        <th>Code</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>THAT</td>
        <td>
            <a href="https://ojs.aaai.org/index.php/AAAI/article/view/16103">
                Two-Stream Convolution Augmented Transformer for Human Activity Recognition</a>
        </td>
        <td><a href="https://github.com/windofshadow/THAT">
            https://github.com/windofshadow/THAT</a></td>
    </tr>
    <tr>
        <td>SenseFi</td>
        <td><a href="https://arxiv.org/abs/2207.07859">
            SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing</a></td>
        <td><a href="https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark">https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark</a>
        </td>
    </tr>
    <tr>
        <td>ST-GCN (AAAI 2018)</td>
        <td><a href="https://arxiv.org/abs/1801.07455">
            Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition</a></td>
        <td><a href="https://github.com/kennymckormick/pyskl">https://github.com/kennymckormick/pyskl</a></td>
    </tr>
    <tr>
        <td>KAN</td>
        <td><a href="https://arxiv.org/abs/2404.19756">
            KAN: Kolmogorov-Arnold Networks</a></td>
        <td><a href="https://github.com/Blealtan/efficient-kan">https://github.com/Blealtan/efficient-kan</a></td>
    </tr>
    </tbody>
</table>

## Datasets

<table>
    <thead>
    <tr>
        <th>Dataset Name</th>
        <th>Paper</th>
        <th>Data resource</th>
        <th>Benchmark</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>UT-HAR</td>
        <td><a href="https://ieeexplore.ieee.org/document/8067693">A Survey on Behavior Recognition Using WiFi Channel State Information</a></td>
        <td><a href="https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt">Downloading URL</a></td>
         <td><a href="https://github.com/Kuroshika/WiLib/blob/master/doc/benchmark/UTHAR.md"> UTHAR Benchmark</a></td>
</tr>
<tr>
        <td>NTU-Fi HAR</td>
        <td><a href="https://arxiv.org/abs/2207.07859">
            SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing</a></td>
        <td><a href="https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt">Downloading URL</a></td>
    <td><a href="https://github.com/Kuroshika/WiLib/blob/master/doc/benchmark/NTU-Fi.md"> NTU-Fi Benchmark</a></td>
</tr>
<tr>
        <td>MM-Fi</td>
        <td><a href="https://arxiv.org/abs/2305.10345">
            MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing</a></td>
        <td><a href="https://github.com/ybhbingo/MMFi_dataset">Downloading URL</a></td>
<td><a href="https://github.com/Kuroshika/WiLib/blob/master/doc/benchmark/MM_Fi.md"> MM-Fi Benchmark</a></td>
</tr>
    </tbody>
</table>
