import time
import json
from pathlib import Path
from argparse import Namespace

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .pd import PhotometricDistort
from lib.datasets.utils import angle2class
from lib.datasets.kitti.utils import get_objects_from_label, Calibration, get_affine_transform, affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti


class KITTI():
    def __init__(self, root_dir, split: str):
        # load dataset
        self.labels, self.imgs, self.cailbs = dict(), dict(), dict()
        print('loading annotations into memory...')
        tic = time.time()
        root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        assert split in ['train', 'val', 'trainval', 'test']
        split_file = root_dir / 'ImageSets' / f'{split}.txt'
        data_dir = root_dir / f'{"testing" if split == "test" else "training"}'
        image_id_list = [x.strip() for x in open(split_file).readlines()]
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        print('creating index...')
        self.img_path = data_dir / 'image_2'
        self.calib_path = data_dir / 'calib'
        self.label_path = data_dir / 'label_2'
        pbar = tqdm(enumerate(image_id_list))
        for i, im_id in pbar:
            pbar.set_description(f'Index {i}')
            im_id = int(im_id)
            self.labels[i] = data_dir / 'label_2' / f'{im_id:06d}.txt'
            self.imgs[i] = data_dir / 'image_2' / f'{im_id:06d}.png'
            self.cailbs[i] = data_dir / 'calib' / f'{im_id:06d}.txt'
        print('index created!')
    
    @property
    def label2id(self) -> dict:
        return {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
    
    def loadImg(self, ids) -> Path:
        return self.imgs[ids]
    
    def loadCalib(self, id) -> Path:
        return self.cailbs[id]
    
    def loadLabel(self, id) -> Path:
        return self.labels[id]


class KITTI_Dataset(Dataset):
    def __init__(self, split, cfg: Namespace):
        
        self.kitti = KITTI(cfg.root_dir, split)
        self.ids = list(sorted(self.kitti.imgs.keys()))
        self.idx_list = [self.kitti.imgs[id].name.split('.')[0] for id in self.ids]
        self.label2id = self.kitti.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.split = split
        self.max_objs = 50
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = getattr(cfg, 'use_3d_center', True)
        self.writelist = getattr(cfg, 'writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = getattr(cfg, 'bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = getattr(cfg, 'meanshape', False)
        
        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        
        self.aug_pd = getattr(cfg, 'aug_pd', False)
        self.aug_crop = getattr(cfg, 'aug_crop', False)
        self.aug_calib = getattr(cfg, 'aug_calib', False)
        
        self.random_mixup3d = getattr(cfg, 'random_mixup3d', 0.5)
        self.random_flip = getattr(cfg, 'random_flip', 0.5)
        self.random_crop = getattr(cfg, 'random_crop', 0.5)
        self.scale = getattr(cfg, 'scale', 0.4)
        self.shift = getattr(cfg, 'shift', 0.1)
        
        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([
            [1.76255119    ,0.66068622   , 0.84422524   ],
            [1.52563191462 ,1.62856739989, 3.88311640418],
            [1.73698127    ,0.59706367   , 1.76282397   ]])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = getattr(cfg, 'clip_2d', False)
    
    def _load_image(self, id):
        image_path = self.kitti.loadImg(id)
        assert image_path.exists(), f'Image path {image_path} does not exist'
        return Image.open(image_path)   # (H, W, 3) RGB mode
    
    def _load_label(self, id):
        label_path = self.kitti.loadLabel(id)
        assert label_path.exists(), f'Label path {label_path} does not exist'
        return get_objects_from_label(label_path)
    
    def _load_calib(self, id) -> Calibration:
        calib_path = self.kitti.loadCalib(id)
        assert calib_path.exists(), f'Calib path {calib_path} does not exist'
        return Calibration(calib_path)
    
    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.kitti.label_path, img_ids)

        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            for split_name in ['aos', '3d', 'bev', 'image']:
                split_name = f'{category}_{split_name}'
                if f'{split_name}_easy_R40' not in results_dict:
                    continue
                logger.info(f"------------{split_name}------------")
                msg = (
                    f'{split_name}_easy_R40: {results_dict[f"{split_name}_easy_R40"]:.2f}%\t'
                    f'{split_name}_moderate_R40: {results_dict[f"{split_name}_moderate_R40"]:.2f}%\t'
                    f'{split_name}_hard_R40: {results_dict[f"{split_name}_hard_R40"]:.2f}%\t'
                )
                logger.info(msg)
        return car_moderate
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index: int):
        #  ============================   get inputs   ===========================
        img_id = int(self.idx_list[index])
        img = self._load_image(index)
        img_size = np.array(img.size) # (W, H)
        features_size = self.resolution // self.downsample    # W * H
        img_without_pd = img.copy()
        
        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False
        random_mix_flag = False
        
        if self.data_augmentation:
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)
            if np.random.random() < self.random_flip:
                random_flip_flag = True
            if self.aug_crop and np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                crop_size = img_size * crop_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            if np.random.random() < self.random_mixup3d:
                random_mix_flag, random_index = self._random_mix_img(img, index)
        
        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        if random_mix_flag:
            img_temp = self._load_image(random_index)
            img_blend = Image.blend(img, img_temp, 0.5)
            img = img_blend
            img_random_mix = self._affine_transform_img(img_temp, trans_inv, random_flip_flag)
        img = self._affine_transform_img(img, trans_inv, random_flip_flag)
        img_without_pd = self._affine_transform_img(img_without_pd, trans_inv, random_flip_flag)
        
        info = {'img_id': img_id,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        
        if self.split == 'test':
            calib = self._load_calib(index)
            return {
                'pixel_values': img,
                'calibs': calib.P2,
                'info': info,
            }

        #  ============================   get labels   ==============================
        objects = self._load_label(index)
        calib = self._load_calib(index)
        
        # add corners_3d to object
        for object in objects:
            object.corners_3d = object.generate_corners3d()  # (8, 3)
        
        # data augmentation for labels
        if random_flip_flag:
            objects, calib = self._aug_labels(objects, calib, img_size)
            
        if random_mix_flag:
            random_objects = self._load_label(random_index)
            # add corners_3d to object
            for random_object in random_objects:
                random_object.corners_3d = random_object.generate_corners3d()  # (8, 3)
            if random_flip_flag:
                random_objects, _ = self._aug_labels(random_objects, calib, img_size)
            objects.extend(random_objects)

        # labels encoding
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        corners_3d = np.zeros((self.max_objs, 8, 2), dtype=np.float32)
        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool) # (H, W)
        
        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        
        labels, mask_2d, depth, heading_bin, heading_res, size_2d, size_3d, src_size_3d, boxes, boxes_3d, corners_3d, obj_region = self._encode_label(
            objects, calib, object_num, trans, img_size, crop_scale, random_flip_flag,
            labels, mask_2d, depth, heading_bin, heading_res, size_2d, size_3d, src_size_3d, boxes, boxes_3d, corners_3d, obj_region)
        
        return {
            'pixel_values': img,
            'pixel_values_without_pd': img_without_pd,
            'pixel_values_random_mix': img_random_mix if random_mix_flag else np.zeros_like(img),
            'calibs': calib.P2,
            'targets': {
                'labels': labels,
                'boxes': boxes,
                'boxes_3d': boxes_3d,
                'depth': depth,
                'size_3d': size_3d,
                'heading_bin': heading_bin,
                'heading_res': heading_res,
                'mask_2d': mask_2d,
                'obj_region': obj_region,
            },
            'info': {
                'corners_3d': corners_3d,
                **info,
            },
        }

    def _affine_transform_img(self, img, trans_inv, random_flip_flag):
        if random_flip_flag:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.transform(
            tuple(self.resolution.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR
        )
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W
        return img

    def _random_mix_img(self, img, dst_index):
        count_num = 0
        random_mix_flag = False
        while count_num < 50:
            count_num += 1
            random_index = int(np.random.choice(self.ids))
            calib_temp = self._load_calib(random_index)
            calib = self._load_calib(dst_index)
            
            if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                img_temp = self._load_image(random_index)
                dst_W_temp, dst_H_temp = np.array(img_temp.size)
                dst_W, dst_H = np.array(img.size)
                if dst_W_temp == dst_W and dst_H_temp == dst_H:
                    objects_1 = self._load_label(dst_index)
                    objects_2 = self._load_label(random_index)
                    if len(objects_1) + len(objects_2) < self.max_objs:
                        random_mix_flag = True
                        break
        return random_mix_flag, random_index
    
    def _aug_labels(self, objects, calib, img_size):
        if self.aug_calib:
            calib.flip(img_size)
        for object in objects:
            [x1, _, x2, _] = object.box2d
            object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
            object.alpha = np.pi - object.alpha
            object.ry = np.pi - object.ry
            if self.aug_calib:
                object.pos[0] *= -1
            if object.alpha > np.pi:  object.alpha -= 2 * np.pi # check range
            if object.alpha < -np.pi: object.alpha += 2 * np.pi
            if object.ry > np.pi:  object.ry -= 2 * np.pi
            if object.ry < -np.pi: object.ry += 2 * np.pi
            
        return objects, calib

    def _filter_object(self, object):
        is_filter = False
        # filter objects by writelist
        if object.cls_type not in self.writelist:
            is_filter = True
        # filter inappropriate samples
        if object.level_str == 'UnKnown' or object.pos[-1] < 2:
            is_filter = True
        # ignore the samples beyond the threshold [hard encoding]
        threshold = 65
        if object.pos[-1] > threshold:
            is_filter = True
        return is_filter

    def _encode_label(self, objects, calib, object_num, trans, img_size, crop_scale, random_flip_flag,
                      labels, mask_2d, depth, heading_bin, heading_res, size_2d, size_3d, src_size_3d, boxes, boxes_3d, corners_3d, obj_region):
        for i in range(object_num):
            if self._filter_object(objects[i]):
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            
            # create object region
            ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], img_size[1]))
            xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], img_size[0]))
            obj_region[ymin:ymax, xmin:xmax] = 1

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], 
                                 dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)
            
            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]: 
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]: 
                proj_inside_img = False

            if proj_inside_img == False:
                    continue
        
            # class
            cls_id = self.label2id[objects[i].cls_type]
            labels[i] = cls_id
        
            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b
        
            # encoding depth
            depth[i] = objects[i].pos[-1] * crop_scale
        
            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)
        
            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.label2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size
        
            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1
            
            # corners_3d
            corners_3d[i], _ = calib.rect_to_img(objects[i].corners_3d) # (8, 2)
            if random_flip_flag:
                corners_3d[i][:, 0] = img_size[0] - corners_3d[i][:, 0]
            corners_3d[i] = np.stack([affine_transform(corner, trans) for corner in corners_3d[i]], axis=0)
        
        return labels, mask_2d, depth, heading_bin, heading_res, size_2d, size_3d, src_size_3d, boxes, boxes_3d, corners_3d, obj_region
