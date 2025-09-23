import gradio as gr
import os
import glob
import cv2
import numpy as np
import torch
import matplotlib
from tqdm import tqdm
import subprocess
from depth_anything_v2.dpt import DepthAnythingV2
import io
import sys
import contextlib
import tempfile



# === Sharpening functions ===
def unsharp_mask(image, kernel_size=(3,3), amount=1.0):
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    maxval = 65535 if image.dtype == np.uint16 else 255
    return np.clip(sharpened, 0, maxval).astype(image.dtype)

def edge_only_sharpen(depth, amount=1.0):
    depth8 = cv2.convertScaleAbs(depth, alpha=(255.0 / max(1, depth.max())))
    edges = cv2.Canny(depth8, 50, 150)
    edge_mask = cv2.dilate(edges, None, iterations=1) > 0
    sharpened = unsharp_mask(depth, amount=amount)
    return np.where(edge_mask, sharpened, depth).astype(depth.dtype)

def anime_sharpen(depth, amount=1.0):
    depth8 = cv2.convertScaleAbs(depth, alpha=(255.0 / max(1, depth.max())))
    edges = cv2.Canny(depth8, 80, 180)
    warped = cv2.erode(depth, np.ones((3, 3), np.uint8), iterations=1)
    sharpened = unsharp_mask(warped, amount=amount * 1.3)
    edge_mask = cv2.dilate(edges, None, iterations=1) > 0
    return np.where(edge_mask, sharpened, depth).astype(depth.dtype)

def load_model(encoder='vitl', device='cuda'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    checkpoint = f'checkpoints/depth_anything_v2_{encoder}.pth'
    depth_anything.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    return depth_anything

def get_video_fps(video_path, fpsdetect):
    if fpsdetect == "opencv":
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    else:
        result = subprocess.run(
            ['ffmpeg', '-i', video_path],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        for line in result.stderr.split('\n'):
            if 'Stream #0' in line and fpsdetect in line:
                try:
                    fps = float(line.split(fpsdetect)[0].split()[-1])
                    return fps
                except:
                    continue
        return 30

def mux_audio(outvideo_path, original_video, audio_codec="copy", audio_bitrate="0k"):
    muxed_path = os.path.splitext(outvideo_path)[0] + "_muxed.mkv"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", outvideo_path,
        "-i", original_video,
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-map", "0:v:0",
        "-map", "1:a:0",
        muxed_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.replace(muxed_path, outvideo_path)

# === Video Processing ===
def process_video_folder(
    vid_folder, out_folder, encoder, input_size, video_bitrate, audio_bitrate,
    ffmpeg, ffmpeg_codec, ffmpeg_extension, bit16, color, pred_only, useheight, usewidth,
    codec, fps, extras, device, sharpen_edgeonly, sharpen_anime, sharpen_strength,
    output_vertical, audio_codec, fpsdetect
):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_anything = load_model(encoder, device)
    os.makedirs(out_folder, exist_ok=True)
    vid_files = glob.glob(os.path.join(vid_folder, "*"))
    for k, filename in enumerate(vid_files):
        cap = cv2.VideoCapture(filename)
        frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if useheight:
            input_size = frame_height
        if usewidth:
            input_size = frame_width
        frame_rate = fps if fps else get_video_fps(filename, fpsdetect)
        if pred_only: 
            output_width = frame_width
            output_height = frame_height
        else: 
            output_height = frame_height * 2
            output_width = frame_width
        # == FFMPEG mode ==
        if ffmpeg:
            output_basename = os.path.splitext(os.path.basename(filename))[0] + '_IS_' + str(input_size) + '_Touchly1'
            output_path = os.path.join(out_folder, output_basename + '.' + ffmpeg_extension)
            frames_dir = os.path.join(out_folder, output_basename + '_frames')
            os.makedirs(frames_dir, exist_ok=True)
            totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_idx in tqdm(range(totalFrameCount), desc=f"Processing {filename}"):
                ret, raw_frame = cap.read()
                if not ret:
                    break
                if bit16:
                    temppics = 'png'
                    raw_frame16 = (raw_frame.astype(np.uint16) * 255)
                    depth = depth_anything.infer_image(raw_frame, input_size)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65536.0
                    depth = depth.astype(np.uint16)
                    if sharpen_edgeonly:
                        depth = edge_only_sharpen(depth, amount=sharpen_strength)
                    elif sharpen_anime:
                        depth = anime_sharpen(depth, amount=sharpen_strength)
                else:
                    temppics = 'jpg'
                    raw_frame16 = raw_frame
                    depth = depth_anything.infer_image(raw_frame, input_size)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                    depth = depth.astype(np.uint8)
                    if sharpen_edgeonly:
                        depth = edge_only_sharpen(depth, amount=sharpen_strength)
                    elif sharpen_anime:
                        depth = anime_sharpen(depth, amount=sharpen_strength)
                if color:
                    raw_frame16 = (raw_frame.astype(np.uint16) * 255)
                    depth = depth_anything.infer_image(raw_frame, input_size)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65536.0
                    depth = depth.astype(np.uint16)
                    if sharpen_edgeonly:
                        depth = edge_only_sharpen(depth, amount=sharpen_strength)
                    elif sharpen_anime:
                        depth = anime_sharpen(depth, amount=sharpen_strength)
                    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint16)
                else:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                if pred_only:
                    frame_to_save = depth
                else:
                    frame_to_save = cv2.vconcat([raw_frame16, depth])
                frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:06d}.' + temppics)
                cv2.imwrite(frame_filename, frame_to_save)
            cap.release()
            additional_ffmpeg_args = extras if extras else ""
            additional_ffmpeg_args_list = additional_ffmpeg_args.split()
            ffmpeg_cmd = [
                'ffmpeg', '-framerate', str(frame_rate), '-i',
                os.path.join(frames_dir, 'frame_%06d.' + temppics),
                *additional_ffmpeg_args_list, output_path
            ]
            subprocess.run(ffmpeg_cmd)
            mux_audio(output_path, filename, audio_codec, audio_bitrate)
            for file in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, file))
            os.rmdir(frames_dir)
        # == OpenCV mode ==
        else:
            temp_output_path = os.path.join(out_folder, os.path.splitext(os.path.basename(filename))[0] + '_temp.'+ ffmpeg_extension)
            final_output_path = os.path.join(out_folder, os.path.splitext(os.path.basename(filename))[0] + '_IS_' + str(input_size) + '_Touchly1' + '.' + ffmpeg_extension)
            out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*codec), frame_rate, (output_width, output_height))
            totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in tqdm(range(totalFrameCount), desc=f"Processing {filename}"):
                ret, raw_frame = cap.read()
                if not ret:
                    break
                depth = depth_anything.infer_image(raw_frame, input_size)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                if sharpen_edgeonly:
                    depth = edge_only_sharpen(depth, amount=sharpen_strength)
                elif sharpen_anime:
                    depth = anime_sharpen(depth, amount=sharpen_strength)
                if color:
                    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                else:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)            
                if pred_only:
                    combined_frame = depth
                    out.write(combined_frame)
                else:
                    combined_frame = cv2.vconcat([raw_frame, depth])
                    out.write(combined_frame)
            cap.release()
            out.release()
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_output_path, '-i', filename, 
                '-c:v', ffmpeg_codec, '-b:v', video_bitrate, '-c:a', audio_codec, '-b:a', audio_bitrate, '-map', '0:v:0', '-map', '1:a:0', 
                final_output_path
            ])
            os.remove(temp_output_path)
    return f"Processed {len(vid_files)} videos. See {out_folder}."

# === Image Processing ===
def process_image_folder(
    img_folder, out_folder, encoder, input_size, color, bit16, exr, exronly, pred_only, 
    sharpen_edgeonly, sharpen_anime, sharpen_strength, output_vertical, device, imagetovideo, ffmpeg_extension, extras
):
    depth_anything = load_model(encoder, device)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    os.makedirs(out_folder, exist_ok=True)
    img_files = glob.glob(os.path.join(img_folder, "*"))
    for filename in img_files:
        raw_image = cv2.imread(filename)
        raw_image16 = (raw_image.astype(np.uint16) * 255)
        depth = depth_anything.infer_image(raw_image, input_size)
        if exr or exronly:
            cv2.imwrite(os.path.join(out_folder, os.path.splitext(os.path.basename(filename))[0] + '_IS_' + str(input_size) + 'cv2' + '.exr'), depth)
            if exronly:
                continue
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65536.0
        depth = depth.astype(np.uint16) if bit16 else (depth / 256).astype(np.uint8)
        if sharpen_edgeonly:
            depth = edge_only_sharpen(depth, amount=sharpen_strength)
        elif sharpen_anime:
            depth = anime_sharpen(depth, amount=sharpen_strength)
        if color:
            depth = (cmap(depth)[:, :, :3] * 65536)[:, :, ::-1].astype(np.uint16 if bit16 else np.uint8)
        else:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        topimage = raw_image16 if bit16 else raw_image
        bottomimage = depth
        output_img_path = os.path.join(out_folder, os.path.splitext(os.path.basename(filename))[0] + '_IS_' + str(input_size) + '.png')
        if pred_only:
            cv2.imwrite(output_img_path, depth)
        else:
            combined_result = cv2.vconcat([topimage, bottomimage])
            cv2.imwrite(output_img_path, combined_result)
        if imagetovideo:
            output_video_path = os.path.join(out_folder, os.path.splitext(os.path.basename(filename))[0] + '_IS_' + str(input_size) + '_pic_Touchly1.' + ffmpeg_extension)
            additional_ffmpeg_args = extras if extras else ""
            additional_ffmpeg_args_list = additional_ffmpeg_args.split()
            cmd = [
                'ffmpeg',
                '-loop', '1',
                '-framerate', '1',
                '-i', output_img_path,
                *additional_ffmpeg_args_list,
                '-t', '30',
                '-y', output_video_path
            ]
            subprocess.run(cmd)
    return f"Processed {len(img_files)} images. See {out_folder}."

# === Utility Functions ===
import subprocess

OPENCV_FOURCCS = [
    "H264", "h264", "X264", "x264", "avc1", "DAVC", "SMV2", "VSSH", "Q264", "V264", "GAVC", "UMSV", "tshd", "INMC",
    "H263", "X263", "T263", "L263", "VX1K", "ZyGo", "M263", "lsvm", "I263", "H261", "U263", "VSM4", "FMP4", "DIVX",
    "DX50", "XVID", "MP4S", "M4S2", "ZMP4", "DIV1", "BLZ0", "mp4v", "UMP4", "WV1F", "SEDG", "RMP4", "3IV2", "WAWV",
    "FFDS", "FVFW", "DCOD", "MVXM", "PM4V", "SMP4", "DXGM", "VIDM", "M4T3", "GEOX", "G264", "HDX4", "DM4V", "DMK2",
    "DYM4", "DIGI", "EPHV", "EM4A", "M4CC", "SN40", "VSPX", "ULDX", "GEOV", "SIPP", "SM4V", "XVIX", "DreX", "QMP4",
    "PLV1", "GLV4", "GMP4", "MNM4", "GTM4", "MP43", "DIV3", "MPG3", "DIV5", "DIV6", "DIV4", "DVX3", "AP41", "COL1",
    "COL0", "MP42", "DIV2", "MPG4", "MP41", "WMV1", "WMV2", "GXVE", "dvsd", "dvhd", "dvh1", "dvsl", "dv25", "dv50",
    "cdvc", "CDVH", "CDV5", "dvcs", "dvis", "pdvc", "SL25", "SLDV", "AVd1", "mpg1", "mpg2", "MPEG", "PIM1", "PIM2",
    "VCR2", "DVR ", "MMES", "LMP2", "slif", "EM2V", "M701", "M702", "M703", "M704", "M705", "mpgv", "BW10", "XMPG",
    "MJPG", "MSC2", "LJPG", "dmb1", "mjpa", "JR24", "JPGL", "MJLS", "jpeg", "IJPG", "AVRn", "ACDV", "QIVG", "SLMJ",
    "CJPG", "IJLV", "MVJP", "AVI1", "AVI2", "MTSJ", "ZJPG", "MMJP", "HFYU", "FFVH", "CYUV", "I420", "YUY2", "Y422",
    "V422", "YUNV", "UYNV", "UYNY", "uyv1", "2Vu1", "2vuy", "yuvs", "yuv2", "P422", "YV12", "YV16", "YV24", "UYVY",
    "VYUY", "IYUV", "Y800", "HDYC", "YVU9", "VDTZ", "Y411", "NV12", "NV21", "Y41B", "Y42B", "YUV9", "YVYU", "YUYV",
    "I410", "I411", "I422", "I440", "I444", "J420", "J422", "J440", "J444", "YUVA", "I40A", "I42A", "RGB2", "RV15",
    "RV16", "RV24", "RV32", "RGBA", "AV32", "GREY", "I09L", "I09B", "I29L", "I29B", "I49L", "I49B", "I0AL", "I0AB",
    "I2AL", "I2AB", "I4AL", "I4AB", "I4FL", "I4FB", "I0CL", "I0CB", "I2CL", "I2CB", "I4CL", "I4CB", "I0FL", "I0FB",
    "FRWU", "R10k", "r210", "v210", "C210", "v308", "v408", "AYUV", "v410", "yuv4", "IV31", "IV32", "IV41", "IV50",
    "VP31", "VP30", "VP40", "VP50", "VP60", "VP61", "VP62", "VP6A", "VP6F", "FLV4", "VP70", "VP71", "VP80", "VP90",
    "ASV1", "ASV2", "VCR1", "FFV1", "Xxan", "LM20", "mrle", "MSVC", "msvc", "CRAM", "cram", "WHAM", "wham", "cvid",
    "DUCK", "PVEZ", "MSZH", "ZLIB", "SNOW", "4XMV", "FLV1", "S263", "FSV1", "svq1", "tscc", "ULTI", "VIXL", "QPEG",
    "Q1.0", "Q1.1", "WMV3", "WMVP", "WVC1", "WMVA", "WVP2", "LOCO", "WNV1", "YUV8", "AAS4", "AASC", "RT21", "FPS1",
    "theo", "TM20", "TR20", "CSCD", "ZMBV", "KMVC", "CAVS", "AVS2", "mjp2", "MJ2C", "LJ2C", "LJ2K", "IPJ2", "AVj2",
    "VMnc", "TGA ", "MPNG", "PNG1", "CLJR", "drac", "azpr", "RPZA", "rpza", "SP54", "AURA", "AUR2", "KGV1", "LAGS",
    "AMVF", "ULRA", "ULRG", "ULY0", "ULY2", "ULY4", "ULH0", "ULH2", "ULH4", "UQY0", "UQY2", "UQRA", "UQRG", "UMY2",
    "UMH2", "UMY4", "UMH4", "UMRA", "UMRG", "VBLE", "E130", "xtor", "ZECO", "Y41P", "AFLC", "MSS1", "MSA1", "TSC2",
    "MTS2", "CLLC", "MSS2", "SVQ3", "012v", "a12v", "G2M2", "G2M3", "G2M4", "G2M5", "FICV", "CHQX", "TDSC", "CUVC",
    "RV40", "SPV1", "RSCC", "ISCC", "CFHD", "M101", "M102", "MAGY", "M8RG", "M8RA", "M0RA", "M0RG", "M0G0", "M0Y0",
    "M0Y2", "M0Y4", "M2RA", "M2RG", "YLC0", "SHQ0", "SHQ1", "SHQ2", "SHQ3", "SHQ4", "SHQ5", "SHQ7", "SHQ9", "FMVC",
    "SCPR", "UCOD", "AV01", "MSCC", "SRGC", "IMM4", "BT20", "MWSC", "WCMV", "RASC", "HYMT", "ARBC", "AGM0", "AGM1",
    "AGM2", "AGM3", "AGM4", "AGM5", "AGM6", "AGM7", "LSCR", "IMM5", "MVDV", "MVHA", "MV30", "nlc1",
]

def show_codecs(extension):
    codecs_info = "***** OpenCV FourCC video codecs *****\n"
    # Display all FourCCs, 8 per line for readability
    for i in range(0, len(OPENCV_FOURCCS), 8):
        codecs_info += ", ".join(OPENCV_FOURCCS[i:i+8]) + "\n"
    codecs_info += "\n***** FFmpeg codecs *****\n"
    ffmpegvideocodes = ['ffmpeg', '-encoders', '-hide_banner']
    try:
        result = subprocess.run(
            ffmpegvideocodes,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        dummy_path = f"dummy.{extension}"
        vw = cv2.VideoWriter(dummy_path, -1, 30, (1920, 1080))
        vw.release()
        codecs_info += result.stdout
    except Exception as e:
        codecs_info += f"(Error running ffmpeg: {e})"
    return codecs_info
    
def show_ffmpeg_version():
    ffmpegvideocodes = ['ffmpeg','-version']
    return subprocess.run(ffmpegvideocodes, capture_output=True, text=True).stdout

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("# Depth Anything V2 — Full-Feature Gradio UI")

    with gr.Tab("Batch Videos"):
        gr.Markdown("Process all videos in a folder using Depth Anything V2. Results will be saved to your output folder, with all original audio muxed.")
        vid_folder = gr.Textbox(value="inputvideo", label="Input Video Folder")
        out_folder_vid = gr.Textbox(value="outputvideo", label="Output Video Folder")
        encoder = gr.Dropdown(['vits', 'vitb', 'vitl', 'vitg'], value='vitl', label='Encoder')
        input_size = gr.Slider(256, 1024, 518, step=1, label="Input Size")
        video_bitrate = gr.Textbox("0k", label="Video Bitrate")
        audio_bitrate = gr.Textbox("0k", label="Audio Bitrate")
        ffmpeg = gr.Checkbox(False, label="Use FFMPEG for Encoding")
        ffmpeg_codec = gr.Textbox('copy', label="FFmpeg Video Codec")
        ffmpeg_extension = gr.Textbox('mkv', label="FFmpeg Container/Extension")
        bit16 = gr.Checkbox(False, label="16-bit Output")
        color = gr.Checkbox(False, label="Colorful Output")
        pred_only = gr.Checkbox(False, label="Prediction Only (no stack)")
        useheight = gr.Checkbox(False, label="Set Input Size to Video Height")
        usewidth = gr.Checkbox(False, label="Set Input Size to Video Width")
        codec = gr.Textbox('HFYU', label='OpenCV Video Codec (fourcc)')
        fps = gr.Number(None, label="Override FPS")
        extras = gr.Textbox('-pix_fmt p010le -c:v hevc_nvenc -crf 18', label="Extra FFMPEG Flags")
        device = gr.Dropdown(['cuda', 'cpu'], value='cuda', label='Device')
        sharpen_edgeonly = gr.Checkbox(False, label="Sharpen Edges Only")
        sharpen_anime = gr.Checkbox(False, label="Anime-style Sharpen")
        sharpen_strength = gr.Slider(0.1, 3.0, 1.0, step=0.1, label="Sharpening Strength")
        output_vertical = gr.Checkbox(True, label="Vertical Stack (input+depth)")
        audio_codec = gr.Textbox('copy', label="Audio Codec (ffmpeg)")
        fpsdetect = gr.Dropdown(['fps', 'tbr', 'opencv'], value='fps', label="FPS Detection Method")
        btn_vid = gr.Button("Process All Videos in Folder")
        out_msg_vid = gr.Textbox(label="Status / Message")
        btn_vid.click(
            process_video_folder,
            inputs=[
                vid_folder, out_folder_vid, encoder, input_size, video_bitrate, audio_bitrate, ffmpeg, ffmpeg_codec,
                ffmpeg_extension, bit16, color, pred_only, useheight, usewidth, codec, fps, extras, device,
                sharpen_edgeonly, sharpen_anime, sharpen_strength, output_vertical, audio_codec, fpsdetect
            ],
            outputs=[out_msg_vid],
        )
    with gr.Tab("Batch Images"):
        gr.Markdown("Process all images in a folder using Depth Anything V2. Results will be saved to your output folder.")
        img_folder = gr.Textbox(value="inputpics", label="Input Image Folder")
        out_folder_img = gr.Textbox(value="outputpics", label="Output Image Folder")
        encoder2 = gr.Dropdown(['vits', 'vitb', 'vitl', 'vitg'], value='vitl', label='Encoder')
        input_size2 = gr.Slider(256, 1024, 518, step=1, label="Input Size")
        color2 = gr.Checkbox(False, label="Colorful Output")
        bit16_2 = gr.Checkbox(False, label="16-bit Output")
        exr2 = gr.Checkbox(False, label="Save OpenEXR (float)")
        exronly2 = gr.Checkbox(False, label="Save Only OpenEXR (no PNG output)")
        pred_only2 = gr.Checkbox(False, label="Prediction Only (no stack)")
        sharpen_edgeonly2 = gr.Checkbox(False, label="Sharpen Edges Only")
        sharpen_anime2 = gr.Checkbox(False, label="Anime-style Sharpen")
        sharpen_strength2 = gr.Slider(0.1, 3.0, 1.0, step=0.1, label="Sharpening Strength")
        output_vertical2 = gr.Checkbox(True, label="Vertical Stack (input+depth)")
        device2 = gr.Dropdown(['cuda', 'cpu'], value='cuda', label='Device')
        imagetovideo2 = gr.Checkbox(False, label="Convert Output Image to Video")
        ffmpeg_extension2 = gr.Textbox('mkv', label="Video Extension")
        extras2 = gr.Textbox('-pix_fmt p010le -c:v hevc_nvenc -crf 18', label="Extra FFMPEG Flags")
        btn_img = gr.Button("Process All Images in Folder")
        out_msg_img = gr.Textbox(label="Status / Message")
        btn_img.click(
            process_image_folder,
            inputs=[
                img_folder, out_folder_img, encoder2, input_size2, color2, bit16_2, exr2, exronly2, pred_only2,
                sharpen_edgeonly2, sharpen_anime2, sharpen_strength2, output_vertical2, device2, imagetovideo2, ffmpeg_extension2, extras2
            ],
            outputs=[out_msg_img],
        )
    with gr.Tab("Show Codecs"):
        extension = gr.Textbox('mkv', label="Extension")
        btn_codecs = gr.Button("Show Available Codecs")
        out_codecs = gr.Textbox(label="Codecs Info")
        btn_codecs.click(show_codecs, [extension], [out_codecs])
    with gr.Tab("Show FFmpeg Version"):
        btn_ffmpeg = gr.Button("Show FFmpeg Version")
        out_ffmpeg = gr.Textbox(label="FFmpeg Version")
        btn_ffmpeg.click(show_ffmpeg_version, [], [out_ffmpeg])

if __name__ == "__main__":
    demo.launch()
