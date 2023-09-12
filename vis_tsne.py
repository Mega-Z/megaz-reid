# -*- coding=utf-8 -*-

import numpy as np
from openTSNE import TSNE
from matplotlib import pyplot as plt


tsne = TSNE(perplexity=9)
num_query = 2210
id_num = 8
color_set = np.asarray([
	[1,0,0],
	[0,1,0],
	[0,0,1],
	[1,0.5,0],
	[1,0,1],
	[0,1,1],
	[0.7,1,0],
	[0.8,0,0.3]]
)

def sample_n_fit(feat, pid, camid, pid_sel, max_sample=10):
	qf = feat[:num_query]
	q_pids = pid[:num_query]
	q_camids = camid[:num_query]
	gf = feat[num_query:]
	g_pids = pid[num_query:]
	g_camids = camid[num_query:]
	q_f_sel = []
	q_id_sel = []
	q_colors = []
	g_f_sel = []
	g_id_sel = []
	g_colors = []
	q_num  = 0
	for i in range(len(pid_sel)):
		id = pid_sel[i]
		# random color
		if i < color_set.shape[0]:
			rand_color = color_set[i]
		else:
			rand_color = np.random.rand(1,3)
		q_cam_same_id = q_camids[q_pids==id]
		most_cam = np.bincount(q_cam_same_id).argmax()
		# append query samples
		qf_same_id = qf[(q_pids==id) & (q_camids==most_cam)]
		np.random.shuffle(qf_same_id)
		qf_id_sel = qf_same_id[:min(qf_same_id.shape[0], max_sample)]
		q_f_sel.extend(qf_id_sel.tolist())
		q_id_sel.extend([id]*qf_id_sel.shape[0])
		q_colors.extend([rand_color]*qf_id_sel.shape[0])
		q_num += qf_id_sel.shape[0]
		# append gallery samples
		gf_same_id = gf[(g_pids==id) & (g_camids!=most_cam)]
		np.random.shuffle(gf_same_id)
		gf_id_sel = gf_same_id[:min(gf_same_id.shape[0], max_sample)]
		g_f_sel.extend(gf_id_sel.tolist())
		g_id_sel.extend([id]*gf_id_sel.shape[0])
		g_colors.extend([rand_color]*gf_id_sel.shape[0])
	embeddings = tsne.fit(np.asarray(q_f_sel+g_f_sel))
	return embeddings, q_colors+g_colors, q_num



if __name__ == "__main__":
	feats2 = np.load("./feats/no_inf_feat.npy")
	feats1 = np.load("./feats/zzw_feat.npy")
	feats0 = np.load("./feats/base_feat.npy")
	feats = [feats1, feats2, feats0]
	pid = np.load("./feats/pid.npy")
	camid = np.load("./feats/camid.npy")
	# embeds = []
	# colors = []
	# q_nums = []


	pid_set = set(pid[:num_query])
	while True:
		pid_sel = np.random.choice(list(pid_set), id_num)
		print("id select: ", pid_sel)
		plt.figure(figsize=(15, 5))
		plt.cla()
		for i in range(3):
			embeddings, colors, q_num = sample_n_fit(feats[i], pid, camid, pid_sel)
			# embeds.append(e)
			# colors.append(c)
			# q_nums.append(q)
			plt.subplot(1, 3, i+1)
			plt.scatter(embeddings[q_num:,0], embeddings[q_num:, 1], s=20, c='w', edgecolors=colors[q_num:], marker='o')
			plt.scatter(embeddings[:q_num,0], embeddings[:q_num, 1], s=20, c=colors[:q_num], marker='x')
		plt.show()

	print()