
#args
import argparse, os
parser = argparse.ArgumentParser(description='SAMTrack CLI')

import warnings
warnings.filterwarnings("ignore")

parser.add_argument('--video_path',
                        help='Input video path',
                        required=True,
                        type=str)
parser.add_argument('--outdir',
                        help='output directory',
                        required=True,
                        type=str)
parser.add_argument('--caption',
                        
                        default='person',
                        help='Text prompt to detect objects in key-frames',
                        type=str)
parser.add_argument('--save_gif',
                        help='Save predicted masks as gif',
                        action='store_true')
parser.add_argument('--save_video',
                        help='Save predicted masks as video',
                        action='store_true')
parser.add_argument('--save_mask',
                        help='Save predicted joint mask images',
                        action='store_true')
parser.add_argument('--save_separate_masks',
                        help='Save separate masks per instance',
                        action='store_true')
parser.add_argument('--save_sam',
                        help='Save intermediary SAM predictions',
                        action='store_true')
parser.add_argument('--no_reset_image',
                        help='reset the image embeddings for SAM',
                        action='store_false')
parser.add_argument('--grounding_caption',
                        help='grounding_caption weight',
                        default=0.35, 
                        
                        type=float)
parser.add_argument('--text_threshold',
                        default=0.5, 
                        help='threshold for text caption',
                        type=float)
parser.add_argument('--box_threshold',
                        default=0.35, 
                        type=float)
parser.add_argument('--box_size_threshold',
                        default=0.8, 
                        help='If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.',
                        type=float)
parser.add_argument('--points_per_side',
                        default=30, 
                        type=int)
parser.add_argument('--pred_iou_thresh',
                        default=0.8, 
                        type=float)
parser.add_argument('--stability_score_thresh',
                        default=0.9, 
                        type=float)
parser.add_argument('--crop_n_layers',
                        default=1, 
                        type=int)
parser.add_argument('--crop_n_points_downscale_factor',
                        default=2, 
                        type=int)
parser.add_argument('--min_mask_region_area',
                        default=200, 
                        type=int)
parser.add_argument('--sam_gap',
                        default=50, 
                        help='the interval to run sam to segment new objects',
                        type=int)
parser.add_argument('--min_area',
                        default=200, 
                        help='minimal mask area to add a new mask as a new object',
                        type=int)
parser.add_argument('--max_obj_num',
                        default=255, 
                        help='maximal object number to track in a video',
                        type=int)
parser.add_argument('--min_new_obj_iou',
                        default=0.8, 
                        help='the area of a new object in the background should > 80%',
                        type=float)
parser.add_argument('--threads',
                        default=12, 
                        help='Num threads for mask export',
                        type=int)

args = parser.parse_args()

from model_args import aot_args,sam_args,segtracker_args

grounding_caption = args.caption
box_threshold, text_threshold, box_size_threshold, reset_image = args.box_threshold, args.text_threshold, args.box_size_threshold, not args.no_reset_image
sam_args['generator_args'] = {
        'points_per_side': args.points_per_side,
        'pred_iou_thresh': args.pred_iou_thresh,
        'stability_score_thresh': args.stability_score_thresh,
        'crop_n_layers': args.crop_n_layers,
        'crop_n_points_downscale_factor': args.crop_n_points_downscale_factor,
        'min_mask_region_area': args.min_mask_region_area,
    }

video_name_full = args.video_path
video_name = video_name_full.split('/')[-1][:-4]
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
io_args = {
    'input_video': video_name_full,
    'output_mask_dir': f'{outdir}/{video_name}_masks' if args.save_mask else '', # save pred masks
    'output_video': f'{outdir}/{video_name}_seg.mp4' if args.save_video else '', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif': f'{outdir}/{video_name}_seg.gif' if args.save_gif else '', # mask visualization
}

segtracker_args = {
    'sam_gap': args.sam_gap,
    'min_area': args.min_area,
    'max_obj_num': args.max_obj_num,
    'min_new_obj_iou': args.min_new_obj_iou
}

import os
import cv2
from SegTracker import SegTracker

from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.7, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

# For every sam_gap frames, we use SAM to find new objects and add them for tracking
# larger sam_gap is faster but may not spot new objects in time


# source video to segment
cap = cv2.VideoCapture(io_args['input_video'])
fps = cap.get(cv2.CAP_PROP_FPS)
# output masks
if args.save_mask:
  output_dir = io_args['output_mask_dir']
  os.makedirs(output_dir, exist_ok=True)
pred_list = []
masked_pred_list = []

torch.cuda.empty_cache()
gc.collect()
sam_gap = segtracker_args['sam_gap']
frame_idx = 0
segtracker = SegTracker(segtracker_args, sam_args, aot_args)
segtracker.restart_tracker()

from tqdm import tqdm
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=frame_count)
progress_bar.set_description("Processing frames...")

import os 
os.makedirs('./debug/seg_result', exist_ok=True)
os.makedirs('./debug/aot_result', exist_ok=True)

with torch.cuda.amp.autocast():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if frame_idx == 0:
            pred_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
            torch.cuda.empty_cache()
            gc.collect()
            segtracker.add_reference(frame, pred_mask)
        elif (frame_idx % sam_gap) == 0:
            seg_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, 
                                                    box_size_threshold, reset_image)
            save_prediction(seg_mask, './debug/seg_result', str(frame_idx)+'.png')
            torch.cuda.empty_cache()
            gc.collect()
            track_mask = segtracker.track(frame)
            save_prediction(track_mask, './debug/aot_result', str(frame_idx)+'.png')

            # find new objects, and update tracker with new objects
            new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
            if np.sum(new_obj_mask > 0) >  frame.shape[0] * frame.shape[1] * 0.4:
                new_obj_mask = np.zeros_like(new_obj_mask)
            if args.save_mask: save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
            pred_mask = track_mask + new_obj_mask
            segtracker.add_reference(frame, pred_mask)
        else:
            pred_mask = segtracker.track(frame,update_memory=True)
        torch.cuda.empty_cache()
        gc.collect()
        
        if args.save_mask: save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')

        pred_list.append(pred_mask)
        
        print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
        frame_idx += 1
        progress_bar.update(1)
    cap.release()
    print('\nfinished')

if args.save_video:
  # draw pred mask on frame and save as a video
  cap = cv2.VideoCapture(io_args['input_video'])
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  if io_args['input_video'][-3:]=='mp4':
      fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
  elif io_args['input_video'][-3:] == 'avi':
      fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
      # fourcc = cv2.VideoWriter_fourcc(*"XVID")
  else:
      fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
  out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

  frame_idx = 0

  progress_bar = tqdm(total=frame_count)
  progress_bar.set_description("Processing frames...")

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      pred_mask = pred_list[frame_idx]
      masked_frame = draw_mask(frame,pred_mask)
      # masked_frame = masked_pred_list[frame_idx]
      masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
      out.write(masked_frame)
      print('frame {} writed'.format(frame_idx),end='\r')
      frame_idx += 1
      progress_bar.update(1)
  out.release()
  cap.release()
  print("\n{} saved".format(io_args['output_video']))
  print('\nfinished')

if args.save_gif:
  # save colorized masks as a gif
  imageio.mimsave(io_args['output_gif'],pred_list,fps=fps)
  print("{} saved".format(io_args['output_gif']))

from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import PIL

threads = args.threads

def write_masks_frame(frame_num,  predicted_masks, output_folder, max_ids=255):
  predicted_masks_frame = predicted_masks[frame_num]
  for i in range(max_ids):
    img_out = PIL.Image.fromarray(((predicted_masks_frame==i+1)*255).astype('uint8'))
    img_out.save(os.path.join(output_folder, f'mask{i:03}', f'alpha_{frame_num:06}.jpg'))

def write_masks_frame_multi(predicted_masks, output_folder, max_ids):
  for i in range(max_ids):
    os.makedirs(os.path.join(output_folder, f'mask{i:03}'), exist_ok=True)

  with Pool(threads) as p:
    fn = partial(write_masks_frame, predicted_masks=predicted_masks, output_folder=output_folder, max_ids=max_ids)
    result = list(tqdm(p.imap(fn, range(len(predicted_masks))), total=len(predicted_masks)))

if args.save_separate_masks:
  print('Saving Separate masks')
  write_masks_frame_multi(predicted_masks=pred_list, output_folder=args.outdir, max_ids=segtracker.get_obj_num())
