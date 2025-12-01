import torch
from PIL import Image, ImageDraw, ImageOps
import open_clip
import json
import os
import argparse
import h5py
import numpy as np
import pickle
import torch_scatter


def get_obj_boxes(points, point2objects, intrinsic, pose_files, h, w):
    device = points.device
    poses = np.array([np.loadtxt(os.path.join(pose_dir, pose_filename)) for pose_filename in pose_files])
    poses = torch.from_numpy(poses).float().to(device)

    object_image_dict = {}

    coords = poses.new(4, len(points))
    coords[:3, :] = torch.t(points)
    coords[3, :].fill_(1)

    # project world (coords) to camera
    world_to_cameras = torch.stack([torch.inverse(pose) for pose in poses])
    camera = torch.bmm(world_to_cameras, coords.repeat(len(world_to_cameras), 1, 1))

    # # project camera to image
    xys = torch.zeros_like(camera)
    xys[:, 0] = (camera[:, 0] * intrinsic[0, 0]) / camera[:, 2] + intrinsic[0, 2]
    xys[:, 1] = (camera[:, 1] * intrinsic[1, 1]) / camera[:, 2] + intrinsic[1, 2]
    xys = torch.round(xys[:, :2]).long()
    depth_vals = camera[:, 2]

    valid_masks = torch.ge(xys[:, 0], 0) * torch.ge(xys[:, 1], 0) * torch.lt(xys[:, 0], w) * torch.lt(xys[:, 1], h)

    for xy, dv, mask, filename in zip(xys, depth_vals, valid_masks, pose_files):
        wrap_depth_image = torch.zeros(h * w, device=device)
        valid_xy = xy[:, mask]
        valid_p2o = point2objects[mask]
        valid_inds = valid_xy[1] * w + valid_xy[0]
        valid_image_z = dv[mask]
        depmask = valid_image_z > 0
        new_inds, mapping = torch.unique(valid_inds[depmask], return_inverse=True)
        new_values = torch_scatter.scatter(valid_image_z[depmask], mapping, dim=0, reduce='min')
        wrap_depth_image[new_inds] = new_values

        for obj_id in torch.unique(valid_p2o):
            if obj_id == -1:
                continue
            obj_mask = valid_p2o == obj_id
            xys = valid_xy[:, obj_mask]
            match_mask = wrap_depth_image[valid_inds[obj_mask]] == valid_image_z[obj_mask]
            obj_xy = xys[:, match_mask]

            if len(obj_xy[0]) > 10:
                obj_box = torch.stack([obj_xy[0].min(), obj_xy[1].min(), obj_xy[0].max(), obj_xy[1].max()])
                if obj_box[2] - obj_box[0] < 5 or obj_box[3] - obj_box[1] < 5:
                    continue
                # # 计算凸包
                # pps = obj_xy.transpose(-1, 0)
                # hull = ConvexHull(pps)
                # # 获取凸包的顶点
                # vertices = pps[hull.vertices].tolist()
                info = [filename.split('.')[0], len(obj_xy[0]), obj_box.numpy().tolist()]
                if int(obj_id) not in object_image_dict:
                    object_image_dict[int(obj_id)] = [info]
                else:
                    object_image_dict[int(obj_id)].append(info)
    return object_image_dict


def fill_pad(image):
    w, h = image.size
    if w > h:
        tmp1 = (w - h) // 2
        tmp2 = w - h - tmp1
        border = (0, tmp1, 0, tmp2)
        image = ImageOps.expand(image, border=border, fill=0)
    elif w < h:
        tmp1 = (h - w) // 2
        tmp2 = h - w - tmp1
        border = (tmp1, 0, tmp2, 0)
        image = ImageOps.expand(image, border=border, fill=0)
    return image


def get_object_feature(top_dir, model, preprocess, mapping):
    object_ids = []
    object_feats = []
    for obj_id in mapping:
        box_lists = mapping[obj_id]
        pnum = np.array([bb[1] for bb in box_lists])
        inds = np.argsort(pnum)[::-1]

        object_crop_images = []
        for i in inds[:10]:
            frame_name, _, box = box_lists[i]
            xmin, ymin, xmax, ymax = box
            img = Image.open(os.path.join(top_dir, 'color', frame_name + '.jpg'))

            cxmin = max(0, xmin - 1)
            cymin = max(0, ymin - 1)
            cxmax = min(img.size[0], xmax + 1)
            cymax = min(img.size[1], ymax + 1)

            img = img.crop((cxmin, cymin, cxmax, cymax))
            img = fill_pad(img)

            try:
                object_crop_images.append(preprocess(img))
            except:
                continue
        if len(object_crop_images) > 0:
            with torch.no_grad():
                image_features = model.encode_image(torch.stack(object_crop_images).to(device))
            object_feats.append(image_features.cpu().numpy())
            object_ids.append(int(obj_id))

    return object_feats, object_ids


def get_boxes_per_object(all_points, objects, intrinsic, pose_files, h, w):
    device = all_points.device
    poses = np.array([np.loadtxt(os.path.join(pose_dir, pose_filename)) for pose_filename in pose_files])
    poses = torch.from_numpy(poses).float().to(device)

    object_image_dict = {}
    # project world (coords) to camera
    world_to_cameras = torch.stack([torch.inverse(pose) for pose in poses])

    for obj in objects:
        obj_id = obj.object_id
        points = all_points[obj.points]
        coords = poses.new(4, len(points))
        coords[:3, :] = torch.t(points)
        coords[3, :].fill_(1)

        camera = torch.bmm(world_to_cameras, coords.repeat(len(world_to_cameras), 1, 1))

        # # project camera to image
        xys = torch.zeros_like(camera)
        xys[:, 0] = (camera[:, 0] * intrinsic[0, 0]) / camera[:, 2] + intrinsic[0, 2]
        xys[:, 1] = (camera[:, 1] * intrinsic[1, 1]) / camera[:, 2] + intrinsic[1, 2]
        xys = torch.round(xys[:, :2]).long()
        depth_vals = camera[:, 2]

        valid_masks = torch.ge(xys[:, 0], 0) * torch.ge(xys[:, 1], 0) * torch.lt(xys[:, 0], w) * torch.lt(xys[:, 1], h)
        valid_masks = valid_masks * (depth_vals > 0)

        object_image_dict[int(obj_id)] = []
        for xy, dv, mask, filename in zip(xys, depth_vals, valid_masks, pose_files):
            obj_xy = xy[:, mask]
            if len(obj_xy[0]) > 10:
                obj_box = torch.stack([obj_xy[0].min(), obj_xy[1].min(), obj_xy[0].max(), obj_xy[1].max()])
                if obj_box[2] - obj_box[0] < 5 or obj_box[3] - obj_box[1] < 5:
                    continue
                info = [filename.split('.')[0], len(obj_xy[0]), obj_box.numpy().tolist()]
                object_image_dict[int(obj_id)].append(info)

    return object_image_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data process')

    parser.add_argument('-scans_dir', type=str, default='',
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('-clip_path', type=str,
                        default='./pretrained/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin')
    parser.add_argument('--save_dir', default='', type=str, help='preprocess data path')
    parser.add_argument('--pkl_path', default='./data/scannet_00_views.pkl', type=str,
                        help='preprocess data path')
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=args.clip_path)

    device = torch.device('cuda:3')
    model = model.to(device)

    pkls = args.pkl_path.split(';')
    all_scans = dict()
    for pkl_f in pkls:
        with open(pkl_f, 'rb') as f:
            scans = pickle.load(f)
        scans = {scan.scan_id: scan for scan in scans}
        all_scans.update(scans)

    scan_ids = sorted(os.listdir(args.scans_dir))
    scan_ids = [s for s in scan_ids if '_00' in s]

    INTRINSICS = [[577.590698, 0.0, 318.905426, 0.0],
                  [0.0, 578.729797, 242.683609, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]]
    intrinsics = np.array(INTRINSICS)
    intrinsics = torch.from_numpy(intrinsics)
    with h5py.File(os.path.join(args.save_dir, 'clip_feats_pad0.hdf5'), "a") as f:
        # if True:
        for scan_id in scan_ids:
            scan_i = all_scans[scan_id]
            pose_dir = os.path.join(args.scans_dir, scan_id, 'pose')
            pose_files = os.listdir(pose_dir)
            pose_files.sort(key=lambda e: (int(e.split('.')[0]), e))
            pc = torch.from_numpy(scan_i.pc).float()

            # object_image_dict = get_obj_boxes(pc, scan_i.three_d_objects, intrinsics, pose_files, 480, 640)
            object_image_dict = get_boxes_per_object(pc, scan_i.three_d_objects, intrinsics, pose_files, 480, 640)
            print(scan_id, len(object_image_dict.keys()))
            # for obj in object_image_dict:
            #     infos = object_image_dict[obj]
            #     for i, info in enumerate(infos):
            #         try:
            #             img = Image.open(os.path.join(args.scans_dir, scan_id, 'color', info[0] + '.jpg'))
            #             x1, y1, x2, y2 = info[2]
            #             img = img.crop((x1, y1, x2, y2))
            #             img.save(f'./out/{i}.jpg')
            #         except:
            #             x = 1
            object_feats, object_ids = get_object_feature(os.path.join(args.scans_dir, scan_id), model, preprocess,
                                                          object_image_dict)
            g = f.create_group(scan_id)
            for id, ff in zip(object_ids, object_feats):
                g.create_dataset(str(id), data=ff, compression="gzip")
            print(scan_id, ' done')
