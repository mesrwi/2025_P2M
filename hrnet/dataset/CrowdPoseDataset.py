# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path

import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset

# from crowdposetools.cocoeval import COCOeval
from utils import zipreader
# from utils.rescore import CrowdRescoreEval

logger = logging.getLogger(__name__)


class CrowdPoseDataset(Dataset):
    def __init__(self, cfg, dataset):
        from crowdposetools.coco import COCO
        self.root = cfg.DATASET.ROOT
        self.dataset = dataset
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

    def _get_anno_file_name(self):
        # example: root/json/crowdpose_{train,val,test}.json
        dataset = 'trainval' if 'rescore' in self.dataset else self.dataset
        return os.path.join(
            self.root,
            'json',
            'crowdpose_{}.json'.format(
                dataset
            )
        )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        if self.data_format == 'zip':
            return images_dir + '.zip@' + file_name
        else:
            return os.path.join(images_dir, file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        image_info = coco.loadImgs(img_id)[0]

        file_name = image_info['file_name']

        if self.data_format == 'zip':
            img = zipreader.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            img = cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if 'train' in self.dataset:
            return img, [obj for obj in target], image_info
        else:    
            return img

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def evaluate(self, cfg, preds, scores, output_dir, tag,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % (self.dataset+tag))

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)

                kpts[int(file_name.split('.')[0])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': int(file_name.split('.')[0]),
                        'area': area
                    }
                )

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        # CrowdPose `test` set has annotation.
        info_str = self._do_python_keypoint_eval(
            res_file, res_folder
        )
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        num_joints = 14

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                # keypoints score.
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AR', 'AR .5',
                       'AR .75', 'AP (easy)', 'AP (medium)', 'AP (hard)']
        stats_index = [0, 1, 2, 5, 6, 7, 8, 9, 10]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[stats_index[ind]]))
            # info_str.append(coco_eval.stats[ind])

        return info_str


class CrowdPoseRescoreDataset(CrowdPoseDataset):
    def __init__(self, cfg, dataset):
        CrowdPoseDataset.__init__(self, cfg, dataset)

    def evaluate(self, cfg, preds, scores, output_dir,
                 *args, **kwargs):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.dataset)

        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)

                kpts[int(file_name.split('.')[0])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': int(file_name.split('.')[0]),
                        'area': area
                    }
                )

        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        self._do_python_keypoint_eval(
            cfg.RESCORE.DATA_FILE, res_file, res_folder
        )

    def _do_python_keypoint_eval(self, data_file, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = CrowdRescoreEval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.dumpdataset(data_file)