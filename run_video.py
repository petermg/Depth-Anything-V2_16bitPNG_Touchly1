import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from tqdm import tqdm
import subprocess

from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str, default='inputvideo', help='default is "inputvideo"')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='outputvideo', help='default is "outputvideo"')
    parser.add_argument('--video-bitrate', type=str, default='0k', help='Set the video bitrate when using --ffmpeg. Default value is "0k".')
    parser.add_argument('--audio-bitrate', type=str, default='0k', help='Set the audio bitrate when using --ffmpeg. Default value is "0k".')
    parser.add_argument('--ffmpeg', dest='ffmpeg', action='store_true', help='Encode using FFMPEG instead of cv2.VideoWriter_fourcc. This first generates temp image files.')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--audio-codec', type=str, default='copy', help='Specify the audio codec to be used when ffmpeg adds the audio. By default this is set to "copy" which just remuxes the audio from the original video file without any re-encoding. The --ffmpeg option is NOT required for this.')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--color', dest='color', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--ffmpeg-codec', type=str, default='copy', help='Sets the ffmpeg video codec for the ffmpeg output. To be used in conjunction with the --ffmpeg option.')
    parser.add_argument('--ffmpeg-extension', type=str, default='mkv', help='Sets the file extension/container for the final output by ffmpeg. Default is "mkv". Note, different containers support different codecs.')
    parser.add_argument('--bit16', dest='bit16', action='store_true', help='Used with --ffmpeg, creates 16bit grayscale png files as intermediates before encoding video file. Does not work with --color option.')
    parser.add_argument('--pix-fmt', type=str, default='yuv420p', help='Sets the video input pixel format. Default is "yuv420p". To be used with the --ffmpeg option.')
    parser.add_argument('--useheight', dest='useheight', action='store_true', help='Sets the input height to match the height of the input video.')
    parser.add_argument('--usewidth', dest='usewidth', action='store_true', help='Sets the input height to match the width of the input video.')
    parser.add_argument('--codec', type=str, default='HFYU', help='Sets the ffmpeg video codec for the ffmpeg output. To be used in conjunction with the --ffmpeg option.')

    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 0
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        print('Video Height:', frame_height)
        print('Video Width:', frame_width)
        print('Input Size:', args.input_size) 
        
        if args.pred_only: 
            output_width = frame_width
            output_height = frame_height
        else: 
            output_height = frame_height * 2
            output_width = frame_width
            
        if args.useheight:
            args.input_size=frame_height
            
        if args.usewidth:
            args.input_size=frame_width    
        
    


            
        if args.ffmpeg:
            output_basename = os.path.splitext(os.path.basename(filename))[0] + '_Touchly1'
            output_path = os.path.join(args.outdir, output_basename + '.' + args.ffmpeg_extension)
            frames_dir = os.path.join(args.outdir, output_basename + '_frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            totalFrameCount = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
           
            for frame_idx in tqdm(range(totalFrameCount)):
                ret, raw_frame = raw_video.read()
                if not ret:
                    break
                
                if args.bit16:
                    temppics = 'png'
                    raw_frame16 = (raw_frame.astype(np.uint16) * 255)
                    depth = depth_anything.infer_image(raw_frame, args.input_size)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65536.0
                    #depth = depth.cpu().numpy().astype(np.uint16)
                    depth = depth.astype(np.uint16)
                    #depth = depth.cpu().numpy().astype(np.uint16)
                
                else:
                    temppics = 'jpg'
                    raw_frame16 = raw_frame
                    depth = depth_anything.infer_image(raw_frame, args.input_size)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                    #depth = depth.cpu().numpy().astype(np.uint16)
                    depth = depth.astype(np.uint8)
                    #depth = depth.cpu().numpy().astype(np.uint16)

                if args.color:
                    raw_frame16 = (raw_frame.astype(np.uint16) * 255)
                    depth = depth_anything.infer_image(raw_frame, args.input_size)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65536.0
                    #depth = depth.cpu().numpy().astype(np.uint8)
                    depth = depth.astype(np.uint16)
                    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint16)


                else:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                    
                if args.pred_only:
                    frame_to_save = depth
                else:
                    #split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    frame_to_save = cv2.vconcat([raw_frame16, depth])
                
                frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:06d}.' + temppics)
                cv2.imwrite(frame_filename, frame_to_save)
            
            raw_video.release()
            
            # Encode video using ffmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-framerate', str(frame_rate), '-i',
                os.path.join(frames_dir, 'frame_%06d.' + temppics),
                '-c:v', args.ffmpeg_codec, '-pix_fmt', args.pix_fmt, '-b:v', args.video_bitrate, output_path
            ]
            subprocess.run(ffmpeg_cmd)
            
            # Mux audio into the video using ffmpeg
            temp_output_path = os.path.join(args.outdir, output_basename + '_temp.' + args.ffmpeg_extension)
            mux_command = [
                 'ffmpeg', '-i', output_path, '-i', filename, '-c:v', 'copy', '-c:a', args.audio_codec, '-b:a:', args.audio_bitrate, '-map', '0:v:0', '-map', '1:a:0', temp_output_path
            ]
            subprocess.run(mux_command)
            os.replace(temp_output_path, output_path)
            
            # Clean up frames directory
            for file in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, file))
            os.rmdir(frames_dir)

        else:         
            temp_output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_temp.mkv')
            final_output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_Touchly1' + '.' + 'mkv')
            out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*args.codec), frame_rate, (output_width, output_height))
            totalFrameCount = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for _ in tqdm(range(totalFrameCount)):
                ret, raw_frame = raw_video.read()
                if not ret:
                    break
            
                depth = depth_anything.infer_image(raw_frame, args.input_size)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                
                if args.color:
                    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                else:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)            
                
                if args.pred_only:
                    combined_frame = depth
                    out.write(combined_frame)
                else:
                    combined_frame = cv2.vconcat([raw_frame, depth])
                    out.write(combined_frame)
            
            raw_video.release()
            out.release()
            
            # Use ffmpeg to combine the video and audio
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_output_path, '-i', filename, 
                '-c:v', args.ffmpeg_codec, '-b:v', args.video_bitrate, '-c:a', args.audio_codec, '-b:a', args.audio_bitrate, '-map', '0:v:0', '-map', '1:a:0', 
                final_output_path
            ])
            
            # Remove the temporary video file
            os.remove(temp_output_path)
                
