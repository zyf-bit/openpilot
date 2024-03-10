# openpilot

openpilot原项目地址https://github.com/commaai/openpilot

在ubuntu20.04上仿真运行步骤如下

1.网页上进入tools文件夹，选择执行
``` bash
git clone --recurse-submodules https://github.com/commaai/openpilot.git
```
2.配置环境
```bash
cd openpilot
git lfs pull
tools/ubuntu_setup.sh
```
3.激活运行环境
``` bash
poetry shell
```
