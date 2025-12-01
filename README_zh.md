<h4 align="center">
    <p>
        <a href="./README.md">English</a> |
        <b>简体中文</b> |
    </p>
</h4>

# MuseTalk Docker 部署
本目录提供适用于 Intel Gaudi（Habana）硬件的 MuseTalk Docker 与 docker compose 部署方式，开箱即用、便于快速上手。

## MuseTalk 简介
MuseTalk 是一个实时、高保真、基于音频驱动的口型同步模型，可根据输入音频修改视频中的人脸区域，并支持多语言与实时推理。更多细节请参考官方仓库与论文。
- 官方仓库：https://github.com/TMElyralab/MuseTalk
- 技术报告：https://arxiv.org/abs/2410.10122

## 开始使用

### 1. 克隆仓库
```bash
git clone https://gitee.com/intel-china/aisolution-musetalk.git
cd aisolution-musetalk
```

### 2. （可选）配置代理
若处于代理环境，构建镜像前请设置以下环境变量：
```bash
export http_proxy=<your-http-proxy>
export https_proxy=<your-https-proxy>
export no_proxy=localhost,127.0.0.1
```

### 3. 构建 Docker 镜像
```bash
docker compose build
```
或使用手动传递代理参数的方式：
```bash
docker build $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
    -f Dockerfile \
    -t aisolution-musetalk:latest .
```

### 4. 启动 WebUI 服务
推荐后台运行并实时查看日志：
```bash
docker compose up -d
docker compose logs -f
```
当日志出现如下信息，表示 WebUI 已就绪：
```
* Running on local URL:  http://0.0.0.0:7860
```
随后可在浏览器访问：`http://<host-ip>:7860`

停止 WebUI 服务：
```bash
# 若通过 docker compose 启动
docker compose down

# 或直接停止指定容器
docker stop aisolution-musetalk
```

### 5. 模型权重
镜像构建时会通过 `download_weights.sh` 自动下载所需权重；也可按官方仓库说明手动下载：
https://github.com/TMElyralab/MuseTalk#download-weights

### 6. FFmpeg
容器中已安装 FFmpeg。如需自定义版本，可设置环境变量 `FFMPEG_PATH`。

## 自定义运行
与 docker compose 保持一致（入口与代理设置）：
```bash
docker run -itd --rm \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--env /') \
    --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --net=host \
    --name aisolution-musetalk \
    -v /data:/data \
    --workdir /workspace/MuseTalk \
    aisolution-musetalk:latest \
    python app.py --ip 0.0.0.0 --port 7860
```
查看容器日志以确认服务就绪：
```bash
docker logs -f aisolution-musetalk
```
出现如下内容即表示 WebUI 已启动：
```
* Running on local URL:  http://0.0.0.0:7860
```

## 参考
- 官方仓库：https://github.com/TMElyralab/MuseTalk
- 技术报告：https://arxiv.org/abs/2410.10122
