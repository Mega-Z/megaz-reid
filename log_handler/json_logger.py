# -*- coding=utf-8 -*-
import os
import json
import datetime
import numpy as np
import cv2
import wandb
import torchvision.transforms as T
from PIL import Image

JSON_LOG_DIR = "../logs/"
if not os.path.exists(JSON_LOG_DIR):
	os.mkdir(JSON_LOG_DIR)

class JsonLogger:
	"""
		json format:
		{
			"run_name": str
			"config": dict
			"states": [dict]
		}
	"""
	def __init__(self, cfg=None, log_path=None, test=False):
		if cfg is not None:
			self.transform = T.Compose([
				T.Normalize(mean=[-1*cfg.INPUT.PIXEL_MEAN[i]/cfg.INPUT.PIXEL_STD[i] for i in range(3)], std=[1/x for x in cfg.INPUT.PIXEL_STD]),
				T.ToPILImage()
			])
			self.log_dir = cfg.OUTPUT_DIR
			if not os.path.exists(self.log_dir):
				os.mkdir(self.log_dir)
		else:
			self.log_dir = JSON_LOG_DIR
		if log_path is not None:
			self.run_name = log_path.split('/')[-2]
			self.json_path = JSON_LOG_DIR + log_path
			self.load_log()
		else:
			assert cfg is not None
			ls = os.listdir(self.log_dir)
			if not test:
				run_ls = [d for d in ls if d.startswith("run_")]
				self.run_name = "run_{:0>3d}_".format(len(run_ls)+1)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
				os.mkdir(os.path.join(self.log_dir, self.run_name))
				self.json_log = dict()
				self.json_log["run_name"] = self.run_name
				self.json_log["config"] = cfg
				self.json_log["states"] = []
				self.json_path = os.path.join(self.log_dir, self.run_name, "log.json")
				self.dump_log()
			else:
				test_ls = [d for d in ls if d.startswith("test_")]
				self.run_name = "test_{:0>3d}_".format(len(test_ls)+1)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
				os.mkdir(os.path.join(self.log_dir, self.run_name))
				self.json_log = dict()
				self.json_log["run_name"] = self.run_name
				self.json_log["config"] = cfg
				self.json_log["states"] = []
				self.json_path = os.path.join(self.log_dir, self.run_name, "log.json")
				self.dump_log()


	def load_log(self):
		with open(self.json_path, "r") as f:
			self.json_log = json.load(f)

	def dump_log(self):
		with open(self.json_path, "w") as f:
			json.dump(self.json_log, f)

	def append_state(self, state_dict, new_epoch=False, dump=False):
		if new_epoch:
			self.json_log["states"].append(state_dict)
		else:
			self.json_log["states"][-1].update(state_dict)
		if dump:
			self.dump_log()

	def save_images(self, imgs, epoch, suffix, prefix='img', id_plus=0):
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, "../imgs")):
			os.mkdir(os.path.join(self.log_dir, self.run_name, "../imgs"))
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, 'imgs/epoch_{:0>3d}/'.format(epoch))):
			os.mkdir(os.path.join(self.log_dir, self.run_name, 'imgs/epoch_{:0>3d}/'.format(epoch)))
		for i, img in enumerate(imgs):
			path = os.path.join(self.log_dir, self.run_name,
			                    'imgs/epoch_{:0>3d}/'.format(epoch)+prefix+'{:0>5d}{}.jpg'.format(i+id_plus, suffix))
			img = self.transform(img)
			img.save(path)

	def visualize_attn(self, imgs, attn, epoch, suffix, prefix='img', id_plus=0):
		"""

		Args:
			imgs: [batch_size, C, H, W]
			attn: [batch_size, patches]
			epoch: str

		Returns:

		"""
		attn_size = (16, 8)
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, "attns")):
			os.mkdir(os.path.join(self.log_dir, self.run_name, "attns"))
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, 'attns/epoch_{:0>3d}/'.format(epoch))):
			os.mkdir(os.path.join(self.log_dir, self.run_name, 'attns/epoch_{:0>3d}/'.format(epoch)))
		for i, img in enumerate(imgs):
			name = os.path.join(self.log_dir, self.run_name,
			                    'attns/epoch_{:0>3d}/{}{:0>5d}{}.jpg'.format(epoch, prefix, i+id_plus, suffix))
			img = self.transform(img)
			np_img = np.array(img)[:, :, ::-1]
			mask = attn[i].reshape(attn_size).numpy()
			max_mask = np.max(mask)
			mask = mask / max_mask
			mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]), interpolation=cv2.INTER_NEAREST)
			img = np.float32(img) / 255
			heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
			heatmap = np.float32(heatmap) / 255
			cam = heatmap + np.float32(img)
			cam = cam / np.max(cam)
			masked_img = np.uint8(255 * cam)
			cv2.imwrite(name, masked_img)
		'''
		# print(attn.min(), attn.max())
		# attn = (attn - attn.min()) / (attn.max() - attn.min())
		for b in range(attn.shape[1]):
			blk_root = JSON_LOG_DIR+self.run_name+'/attns/epoch_{:0>3d}/block_{:0>2d}/'.format(epoch, b)
			if not os.path.exists(blk_root):
				os.mkdir(blk_root)
			for i in range(attn.shape[0]):
				for h in range(attn.shape[2]):
					# print(attn[i][b][h].min(), attn[i][b][h].max())
					attn[i][b][h] = (attn[i][b][h] - attn[i][b][h].min()) / (attn[i][b][h].max() - attn[i][b][h].min())
					img = attn[i][b][h].reshape(attn_size).cpu().numpy() * 255
					img = img.astype(np.uint8)
					img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
					cv2.imwrite(blk_root + prefix + "{:0>3d}_h{:0>2d}{}.jpg".format(i, h, suffix), img)'''


	def visualize_matches(self, idx5, val_dataset, path_list, id_annotation=False):
		visualize_root = os.path.join(self.log_dir, self.run_name, 'match_results')
		if not os.path.exists(visualize_root):
			os.mkdir(visualize_root)
		# query_root = "../../datasets/Occluded_Duke/query/"
		# gallery_root = "../../datasets/Occluded_Duke/bounding_box_test/"
		# query_root = "../../datasets/Occluded_ReID/occluded_body_images/"
		# gallery_root = "../../datasets/Occluded_ReID/whole_body_images/"
		num_query = idx5.shape[0]
		for q_idx in range(num_query):
			if not os.path.exists(path_list[q_idx]):
				print(path_list[q_idx], "doesn't exist!")
			q_id = val_dataset[q_idx][1]
			q_img = cv2.resize(cv2.imread(path_list[q_idx]), (128, 256))
			g_ids = [val_dataset[num_query+g_idx][1] for g_idx in idx5[q_idx]]
			g_imgs = [cv2.resize(cv2.imread(path_list[num_query+g_idx]), (128, 256)) for g_idx in idx5[q_idx]]
			if id_annotation:
				output = np.ones((300, 808, 3)) * 255
			else:
				output = np.ones((256, 808, 3)) * 255
			if id_annotation:
				cv2.putText(output, "{:0>4d}".format(q_id), (20, 295), 0, 1, (255, 255, 255))
			cv2.rectangle(q_img, (0, 0), (127, 255), (127, 127, 127), 2)
			for m_idx in range(5):
				if id_annotation:
					cv2.putText(output, "{:0>4d}".format(g_ids[m_idx]), (128*(m_idx+1)+20, 295), 0, 1, (255, 255, 255))
				if g_ids[m_idx] == q_id:
					cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 255, 0), 4)
				else:
					cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 0, 255), 4)

			output[:256, :128] = q_img
			output[:256, 168:] = np.concatenate(g_imgs, axis=1)
			cv2.imwrite(visualize_root + '{:0>5d}.jpg'.format(q_idx+1), output)

	def wandb_sync(self):
		wandb.init(project="zzw_reid", entity="mega_z", name=self.run_name)
		wandb.config.dataset = self.json_log["config"]["DATASETS"]["NAMES"]
		wandb.config.base_lr = self.json_log["config"]["SOLVER"]["BASE_LR"]
		wandb.config.batch_size = self.json_log["config"]["SOLVER"]["IMS_PER_BATCH"]
		if "mae" in self.json_log["config"]["MODEL"]["PRETRAIN_PATH"]:
			wandb.config.pretrain = "MAE"
		wandb.config.id_loss_weight = self.json_log["config"]["MODEL"]["ID_LOSS_WEIGHT"]
		wandb.config.triplet_loss_weight = self.json_log["config"]["MODEL"]["TRIPLET_LOSS_WEIGHT"]
		if self.json_log["config"]["MODEL"]["ZZWTRY"] or self.json_log["config"]["MODEL"]["ZZWEXP"]:
			wandb.config.branch_blocks = self.json_log["config"]["MODEL"]["BRANCH_BLOCKS"]
			wandb.config.occ_decoder = self.json_log["config"]["MODEL"]["OCCDECODER"]

			wandb.config.occlude_type = self.json_log["config"]["MODEL"]["OCC_TYPE"]
			if self.json_log["config"]["MODEL"]["OCC_TYPE"] == 'img_block':
				wandb.config.occlude_ratio = self.json_log["config"]["MODEL"]["OCC_RATIO"]
				# wandb.config.align_bottom_occlude = self.json_log["config"]["MODEL"]["OCC_ALIGN_BTM"]
				wandb.config.align_bound = self.json_log["config"]["MODEL"]["OCC_ALIGN_BOUND"]
				wandb.config.occlude_ulrd = self.json_log["config"]["MODEL"]["OCC_ULRD"]
				wandb.config.patch_align_occlude = self.json_log["config"]["MODEL"]["PATCH_ALIGN_OCC"]

			if self.json_log["config"]["MODEL"]["OCCDECODER"]:
				wandb.config.use_decoder_feat = self.json_log["config"]["MODEL"]["USE_DECODER_FEAT"]

			wandb.config.occlusion_aware = self.json_log["config"]["MODEL"]["OCC_AWARE"]
			if self.json_log["config"]["MODEL"]["OCC_AWARE"]:
				wandb.config.fix_alpha = self.json_log["config"]["MODEL"]["FIX_ALPHA"]
				wandb.config.occlusion_loss_weight = self.json_log["config"]["MODEL"]["OCC_LOSS_WEIGHT"]

			wandb.config.inference = self.json_log["config"]["MODEL"]["IFRC"]
			if self.json_log["config"]["MODEL"]["IFRC"]:
				wandb.config.inference_loss_weight = self.json_log["config"]["MODEL"]["IFRC_LOSS_WEIGHT"]
				wandb.config.inference_loss_type = self.json_log["config"]["MODEL"]["IFRC_LOSS_TYPE"]
				wandb.config.pretext = self.json_log["config"]["MODEL"]["PRETEXT"]
				if self.json_log["config"]["MODEL"]["PRETEXT"] == 'rgb_avg':
					wandb.config.pretext_rgb_pix = self.json_log["config"]["MODEL"]["PRETEXT_RGB_PIX"]

		for step in self.json_log["states"]:
			step.pop("Epoch")
			wandb.log(step)



if __name__ == "__main__":
	json_logger = JsonLogger(log_path="occ_duke/run_036_fix_0.9/log.json")
	json_logger.wandb_sync()
