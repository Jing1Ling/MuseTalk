import os
import time
import re
import sys
import subprocess
import multiprocessing
import argparse
import glob
import pickle
import copy

import cv2
import torch
import numpy as np
import imageio
from tqdm import tqdm
from argparse import Namespace
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import WhisperModel
import gradio as gr

from musetalk.utils.blending import get_blending_mask, get_image, get_image_blending, get_image_blending_from_mask_info

inpainting_infer_ps = None
task_queue = None
result_queue = None
task_event = None
shutdown_event = None

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

def inpainting(pe, vae, unet, timesteps,
               video_path, bbox_shift, extra_margin, parsing_mode,
               left_cheek_width, right_cheek_width,
               device, weight_dtype):
    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import get_file_type
    from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder

    """Debug inpainting parameters, only process the first frame"""
    # Set default parameters
    args_dict = {
        "result_dir": './results/debug', 
        "fps": 25, 
        "batch_size": 1, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # Create debug directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Read first frame
    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Save first frame
    debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    # Get face coordinates
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox = coord_list[0]
    frame = frame_list[0]
    
    if bbox == coord_placeholder:
        return None, "No face detected, please adjust bbox_shift parameter"
    
    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    # Process first frame
    x1, y1, x2, y2 = bbox
    y2 = y2 + args.extra_margin
    y2 = min(y2, frame.shape[0])
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    
    # Generate random audio features
    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    
    # Get latents
    latents = vae.get_latents_for_unet(crop_frame)
    latents = latents.to(dtype=weight_dtype)
    
    # Generate prediction results
    pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    
    # Inpaint back to original image
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    
    # Save results (no need to convert color space again since get_image already returns RGB format)
    debug_result_path = os.path.join(args.result_dir, "debug_result.png")
    cv2.imwrite(debug_result_path, combine_frame)
    
    # Create information text
    info_text = f"Parameter information:\n" + \
                f"bbox_shift: {bbox_shift}\n" + \
                f"extra_margin: {extra_margin}\n" + \
                f"parsing_mode: {parsing_mode}\n" + \
                f"left_cheek_width: {left_cheek_width}\n" + \
                f"right_cheek_width: {right_cheek_width}\n" + \
                f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"

    return combine_frame, info_text

@torch.no_grad()
def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw",
                    left_cheek_width=90, right_cheek_width=90):
    global task_queue, result_queue, task_event

    task = {
        'is_testing': True,
        'video_path': video_path,
        'bbox_shift': bbox_shift,
        'extra_margin': extra_margin,
        'parsing_mode': parsing_mode,
        'left_cheek_width': left_cheek_width,
        'right_cheek_width': right_cheek_width,
    }
    task_queue.put(task)
    # Signal child process immediately after enqueuing the task
    task_event.set()

    try:
        result = result_queue.get(timeout=None)
        return cv2.cvtColor(result['frame'], cv2.COLOR_RGB2BGR), result['info_text']

    except Exception:
        print("Fatal error!!!")
        return None, None

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)

def download_model():
    # 检查必需的模型文件是否存在
    required_models = {
        "MuseTalk UNet": f"{CheckpointsDir}/musetalkV15/unet.pth",
        "MuseTalk Config": f"{CheckpointsDir}/musetalkV15/musetalk.json",
        "SD VAE": f"{CheckpointsDir}/sd-vae/config.json",
        "Whisper": f"{CheckpointsDir}/whisper/config.json",
        "DWPose": f"{CheckpointsDir}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CheckpointsDir}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth",
        "ResNet": f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
    }
    
    missing_models = []
    for model_name, model_path in required_models.items():
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        # 全用英文
        print("The following required model files are missing:")
        for model in missing_models:
            print(f"- {model}")
        print("\nPlease run the download script to download the missing models:")
        if sys.platform == "win32":
            print("Windows: Run download_weights.bat")
        else:
            print("Linux/Mac: Run ./download_weights.sh")
        sys.exit(1)
    else:
        print("All required model files exist.")




download_model()  # for huggingface deployment.


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def infer(pe, vae, unet, timesteps, audio_processor, whisper,
          audio_path, video_path, bbox_shift, extra_margin, parsing_mode,
          left_cheek_width, right_cheek_width, progress,
          device, weight_dtype):
    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range

    # Set default parameters, aligned with inference.py
    args_dict = {
        "result_dir": './results/output', 
        "fps": 25, 
        "batch_size": 8, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",  # Fixed use v15 version
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # Check ffmpeg
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    # Create temporary directory
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set result save path
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join(temp_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)
        
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        # Read video
        reader = imageio.get_reader(video_path)

        # Save images
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
        
    ############################################## extract audio feature ##############################################
    # Extract audio features
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, 
        device, 
        weight_dtype, 
        whisper, 
        librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
        
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    
    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
        device="hpu"
    )
    
    i = 0
    input_latent_list = []
    input_blending_mask_info_list = []
    for bbox, frame in  tqdm(zip(coord_list, frame_list), total=len(coord_list)):
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        # use v15 blending
        blending_mask, crop_bbox = get_blending_mask(frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        input_blending_mask_info_list.append((blending_mask, crop_bbox))
        if bbox == coord_placeholder:
            continue
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)


    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    input_blending_mask_info_list = input_blending_mask_info_list + input_blending_mask_info_list[::-1]
    
    ############################################## inference batch by batch ##############################################
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    delay_frame = 0
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=delay_frame,
        device=device,
    )
    res_frame_list = []
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = pe(whisper_batch)
        # Ensure latent_batch is consistent with model weight type
        latent_batch = latent_batch.to(dtype=weight_dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    ############################################## pad to full image ##############################################
    height, width, _ = frame_list[0].shape
    temp_vid_path = 'temp.mp4'
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",                    # stdin 输入
        "-an",
        "-vcodec", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        temp_vid_path
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[(i+delay_frame)%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[(i+delay_frame)%(len(frame_list_cycle))])
        blending_mask, crop_bbox = input_blending_mask_info_list[(i+delay_frame)%(len(coord_list_cycle))]
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue

        # Use v15 version blending
        combine_frame = get_image_blending_from_mask_info(ori_frame, res_frame, [x1, y1, x2, y2], blending_mask, crop_bbox)

        # combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        process.stdin.write(combine_frame.astype(np.uint8).tobytes())
        # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)

    process.stdin.close()
    process.wait()
    print("Video saved to", temp_vid_path)

    input_video = './temp.mp4'
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
    print("Audio combination command:", cmd_combine_audio) 
    os.system(cmd_combine_audio)

    os.remove("temp.mp4")
    # shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name, bbox_shift_text

@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, extra_margin=10, parsing_mode="jaw",
              left_cheek_width=90, right_cheek_width=90, progress=gr.Progress(track_tqdm=True)):
    global task_queue, result_queue, task_event

    task = {
        'is_testing': False,
        'audio_path': audio_path,
        'video_path': video_path,
        'bbox_shift': bbox_shift,
        'extra_margin': extra_margin,
        'parsing_mode': parsing_mode,
        'left_cheek_width': left_cheek_width,
        'right_cheek_width': right_cheek_width,
        'progress': progress
    }
    task_queue.put(task)
    # Signal child process immediately after enqueuing the task
    task_event.set()

    try:
        result = result_queue.get(timeout=None)
        return result['output_vid_name'], result['bbox_shift_text']
    except Exception:
        print("Fatal error!!!")
        return None, None


def check_video(video):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)


    # read video
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # get fps from original video

    # conver fps to 25
    frames = [im for im in reader]
    target_fps = 25
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while t_idx < L and target_t / target_fps > original_t[t_idx]:
            t_idx += 1
        if t_idx >= L:
            t_idx = L - 1
        target_frames.append(frames[t_idx])

    # save video
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video

def inpainting_infer_ps_fn(use_float16: bool,
                           task_queue: multiprocessing.Queue,
                           result_queue: multiprocessing.Queue,
                           task_event,
                           shutdown_event):
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.audio_processor import AudioProcessor

    # load model weights
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "hpu"
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )

    # Set data type
    if use_float16:
        # Convert models to half precision for better performance
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    timesteps = torch.tensor([0], device=device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    print("Starting the main loop in the child process")
    while not shutdown_event.is_set():
        task_event.wait(timeout=2)

        if shutdown_event.is_set():
            break
        # Clear the event before draining the queue to avoid losing a set() that
        # happens while we are processing tasks.
        task_event.clear()

        # Drain all available tasks.
        while True:
            try:
                task = task_queue.get_nowait()
            except Exception:
                break
            if task is None:
                print("Got an empty task in the loop !!!")
                continue

            is_testing = task['is_testing']
            if is_testing:
                video_path = task['video_path']
                bbox_shift = task['bbox_shift']
                extra_margin = task['extra_margin']
                parsing_mode = task['parsing_mode']
                left_cheek_width = task['left_cheek_width']
                right_cheek_width = task['right_cheek_width']

                frame, info_text = inpainting(pe, vae, unet, timesteps,
                                              video_path, bbox_shift, extra_margin, parsing_mode,
                                              left_cheek_width, right_cheek_width,
                                              device, weight_dtype)
                result = {
                    'frame': frame,
                    'info_text': info_text,
                }
            else:
                audio_path = task['audio_path']
                video_path = task['video_path']
                bbox_shift = task['bbox_shift']
                extra_margin = task['extra_margin']
                parsing_mode = task['parsing_mode']
                left_cheek_width = task['left_cheek_width']
                right_cheek_width = task['right_cheek_width']
                progress = task['progress']
                output_vid_name, bbox_shift_text = infer(pe, vae, unet, timesteps, audio_processor, whisper,
                                                         audio_path, video_path, bbox_shift,
                                                         extra_margin, parsing_mode,
                                                         left_cheek_width, right_cheek_width, progress,
                                                         device, weight_dtype)
                result = {
                    'output_vid_name': output_vid_name,
                    'bbox_shift_text': bbox_shift_text,
                }

            result_queue.put(result)


# Parse command line arguments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\\bin", help="Path to ffmpeg executable")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    args = parser.parse_args()

    global task_queue, result_queue, task_event, shutdown_event, inpainting_infer_ps
    task_queue = multiprocessing.Queue(maxsize=3)
    result_queue = multiprocessing.Queue(maxsize=3)
    task_event = multiprocessing.Event()
    shutdown_event = multiprocessing.Event()

    inpainting_infer_ps = multiprocessing.Process(
        target=inpainting_infer_ps_fn,
        args=(args.use_float16, task_queue, result_queue, task_event, shutdown_event),
    )
    inpainting_infer_ps.start()

    time.sleep(5)

    css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """<div align='center'> <h1>MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling</h1> \
                        <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        </br>\
                        Yue Zhang <sup>*</sup>,\
                        Zhizhou Zhong <sup>*</sup>,\
                        Minhao Liu<sup>*</sup>,\
                        Zhaokang Chen,\
                        Bin Wu<sup>†</sup>,\
                        Yubin Zeng,\
                        Chao Zhang,\
                        Yingjie He,\
                        Junxin Huang,\
                        Wenjiang Zhou <br>\
                        (<sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author, benbinwu@tencent.com)\
                        Lyra Lab, Tencent Music Entertainment\
                    </h2> \
                    <a style='font-size:18px;color: #000000' href='https://github.com/TMElyralab/MuseTalk'>[Github Repo]</a>\
                    <a style='font-size:18px;color: #000000' href='https://github.com/TMElyralab/MuseTalk'>[Huggingface]</a>\
                    <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2410.10122'> [Technical report] </a>"""
        )

        with gr.Row():
            with gr.Column():
                audio = gr.Audio(label="Drving Audio", type="filepath")
                video = gr.Video(label="Reference Video", sources=['upload'])
                bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
                extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
                parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
                left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5)
                right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)
                bbox_shift_scale = gr.Textbox(label="'left_cheek_width' and 'right_cheek_width' parameters determine the range of left and right cheeks editing when parsing model is 'jaw'. The 'extra_margin' parameter determines the movement range of the jaw. Users can freely adjust these three parameters to obtain better inpainting results.")

                with gr.Row():
                    debug_btn = gr.Button("1. Test Inpainting ")
                    btn = gr.Button("2. Generate")
            with gr.Column():
                debug_image = gr.Image(label="Test Inpainting Result (First Frame)")
                debug_info = gr.Textbox(label="Parameter Information", lines=5)
                out1 = gr.Video()

        video.change(
            fn=check_video, inputs=[video], outputs=[video]
        )
        btn.click(
            fn=inference,
            inputs=[
                audio,
                video,
                bbox_shift,
                extra_margin,
                parsing_mode,
                left_cheek_width,
                right_cheek_width
            ],
            outputs=[out1, bbox_shift_scale]
        )
        debug_btn.click(
            fn=debug_inpainting,
            inputs=[
                video,
                bbox_shift,
                extra_margin,
                parsing_mode,
                left_cheek_width,
                right_cheek_width
            ],
            outputs=[debug_image, debug_info]
        )

    if not fast_check_ffmpeg():
        print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        demo.queue(max_size=8 * 4, default_concurrency_limit=1).launch(
            share=args.share,
            debug=True,
            server_name=args.ip,
            server_port=args.port)
    except KeyboardInterrupt:
        print("Main process received KeyboardInterrupt")
    finally:
        if inpainting_infer_ps and shutdown_event:
            shutdown_event.set()
            inpainting_infer_ps.join(timeout=5)
            if inpainting_infer_ps.is_alive():
                inpainting_infer_ps.terminate()
                inpainting_infer_ps.join()

if __name__ == "__main__":
    main()
