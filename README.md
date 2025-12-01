<h4 align="center">
    <p>
        <b>English</b> |
        <a href="./README_zh.md">简体中文</a> |
    </p>
</h4>

# MuseTalk Docker Deployment

This directory provides a ready-to-use Docker and docker compose setup for [MuseTalk](https://github.com/TMElyralab/MuseTalk): a real-time, high-fidelity audio-driven lip-sync model. The setup is adapted for Intel Gaudi hardware and supports convenient deployment and usage.

## What is MuseTalk?

MuseTalk is a real-time, high-quality audio-driven lip-syncing model. It modifies a face in video according to input audio, supporting multiple languages and real-time inference. See the [official repo](https://github.com/TMElyralab/MuseTalk) for details and research paper.




## Getting Started

### 1. Clone the Repository

```bash
git clone https://gitee.com/intel-china/aisolution-musetalk.git
cd aisolution-musetalk
```

### 2. (Optional) Configure Proxy
If you are behind a proxy, set these environment variables before building:

```bash
export http_proxy=<your-http-proxy>
export https_proxy=<your-https-proxy>
export no_proxy=localhost,127.0.0.1
```

### 3. Build the Docker Image

```bash
docker compose build
```
Or, to pass proxy args manually:
```bash
docker build $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
    -f Dockerfile \
    -t aisolution-musetalk:latest .
```

### 4. Start the WebUI Server

Recommended: run in background and check logs
```bash
docker compose up -d
docker compose logs -f
```
When you see the following log, the WebUI server is ready:
```
* Running on local URL:  http://0.0.0.0:7860
```
Access the WebUI at: `http://<host-ip>:7860`

To stop the WebUI server:
```bash
# If started with docker compose
docker compose down

# Or stop the specific container
docker stop aisolution-musetalk
```

### 5. Model Weights
Required model weights are automatically downloaded during build via `download_weights.sh`. You can also download them manually as described in the [MuseTalk repo](https://github.com/TMElyralab/MuseTalk#download-weights).

### 6. FFmpeg
FFmpeg is installed in the container. For a custom version, set the `FFMPEG_PATH` environment variable.






## Example: Custom Run

To run the container with custom options (same entrypoint and proxy settings as docker compose):
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
To check logs for readiness:
```bash
docker logs -f aisolution-musetalk
```
Look for:
```
* Running on local URL:  http://0.0.0.0:7860
```


## References
- [MuseTalk Official Repo](https://github.com/TMElyralab/MuseTalk)
- [MuseTalk Paper](https://arxiv.org/abs/2410.10122)

