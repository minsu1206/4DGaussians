from plyfile import PlyData, PlyElement
import os
import numpy as np

path = "/workspace/lustre/datasets/nerf_team/taekwondo"

bg_ply_path = os.path.join(path, "background/0.ply")
fg_ply_path1 = os.path.join(path, "frame1/pointclouds/1.ply")
fg_ply_path2 = os.path.join(path, "frame1/pointclouds/2.ply")

bg_ply = PlyData.read(bg_ply_path)

bg_xyz = np.vstack([bg_ply['vertex']['x'], bg_ply['vertex']['y'], bg_ply['vertex']['z']]).T
# print("[DEBUG] : ", bg_ply['face']['red'])
# exit()
# bg_color = np.vstack([bg_ply['vertex']['red'], bg_ply['vertex']['green'], bg_ply['vertex']['blue']]).T / 255.0
# bg_color = np.vstack([bg_ply['face']['red'], bg_ply['face']['green'], bg_ply['face']['blue']]).T / 255.0
bg_color = np.random.rand(len(bg_xyz),3)

fg_ply1 = PlyData.read(fg_ply_path1)
fg_xyz1 = np.vstack([fg_ply1['vertex']['x'], fg_ply1['vertex']['y'], fg_ply1['vertex']['z']]).T
# print(fg_ply1.__dict__)
# fg_color1 = np.vstack([fg_ply1['face']['red'], fg_ply1['face']['green'], fg_ply1['face']['blue']]).T / 255.0
fg_color1 = np.vstack([fg_ply1['vertex']['red'], fg_ply1['vertex']['green'], fg_ply1['vertex']['blue']]).T / 255.0
fg_ply2 = PlyData.read(fg_ply_path2)
fg_xyz2 = np.vstack([fg_ply2['vertex']['x'], fg_ply2['vertex']['y'], fg_ply2['vertex']['z']]).T
# fg_color2 = np.vstack([fg_ply2['face']['red'], fg_ply2['face']['green'], fg_ply2['face']['blue']]).T / 255.0
fg_color2 = np.vstack([fg_ply2['vertex']['red'], fg_ply2['vertex']['green'], fg_ply2['vertex']['blue']]).T / 255.0
print("[DEBUG] : bg ply check : ", bg_xyz.shape, bg_color.shape)
print("[DEBUG] : fg ply 1 check : ", fg_xyz1.shape, fg_color1.shape)
print("[DEBUG] : fg ply 2 check : ", fg_xyz2.shape, fg_color2.shape)

merged_xyz = np.vstack([bg_xyz, fg_xyz1, fg_xyz2])
merged_color = np.vstack([bg_color, fg_color1, fg_color2])
merged_normals = np.ones((len(merged_xyz), 3)) / 2
pts_idx = np.random.choice(len(merged_xyz), 100_000)
merged_xyz = merged_xyz[pts_idx, :]
merged_color = merged_color[pts_idx, :]
merged_normals = merged_normals[pts_idx, :]

dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
elements = np.empty(merged_xyz.shape[0], dtype=dtype)
attributes = np.concatenate((merged_xyz, merged_normals, merged_color), axis=1)
elements[:] = list(map(tuple, attributes))

# Create the PlyData object and write to file
vertex_element = PlyElement.describe(elements, 'vertex')
ply_data = PlyData([vertex_element])
save_path = os.path.join(path, "fused.ply")
ply_data.write(save_path)
print("DONE ! save fused ply at ", save_path)
