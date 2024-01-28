# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch

from . import util
from . import texture
from . import mesh
from . import material

######################################################################################
# 工具函数（Utility functions）
######################################################################################

def _write_weights(folder, mesh):
    if mesh.v_weights is not None:
        file = os.path.join(folder, 'mesh.weights')
        np.save(file, mesh.v_weights.detach().cpu().numpy())

def _write_bones(folder, mesh):
    if mesh.bone_mtx is not None:
        file = os.path.join(folder, 'mesh.bones')
        np.save(file, mesh.bone_mtx.detach().cpu().numpy())

def _find_mat(materials, name):
    for mat in materials:
        if mat['name'] == name:
            return mat
    return materials[0] # Materials 0 is the default

######################################################################################
# 从obj文件创建网格对象（Create mesh object from objfile）
# obj数据前缀：
#   - mtllib: 关联材质文件
#        e.g: mtllib mesh.mtl
#
#   - v:  几何顶点（geometric vertices） (x, y, z, [w]), w 可选，默认为 1.0.；
#        e.g.: v 0.123 0.234 0.345 1.0
#
#   - vt: 纹理坐标（texture coordinates) (u, [v, w]), 范围（0，1），w 可选，默认为 0.0；
#        e.g.: vt 0.500 1 [0]
#
#   - vn: 顶点法向量（vertex normal）(x,y,z) 法向量可能不为单位向量；
#        e.g.: vn 0.707 0.000 0.707
#
#   - vp: 参数空间顶点（parameter space vertices） (u, [v, w]) ，Freeform几何表述（参数化表面）;
#        e.g.: vp 0.310000 3.210000 2.100000
#
#   - f:  多边形面元素（Polygonal face element）
#        e.g.: f 6/4/1 3/5/3 7/6/5
#
#   - l:  线元素（Line element）
#        e.g.: l 5 8 1 2 4 9
######################################################################################

def load_obj(filename, clear_ks=True, mtl_override=None):
    obj_path = os.path.dirname(filename)

    # 读取文件（Read entire file）
    with open(filename) as f:
        lines = f.readlines()

    # 加载材质（Load materials）
    all_materials = [
        {
            'name' : '_default_mat',
            'bsdf' : 'falcor',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }
    ]
        
    # 材质重写（Material Override）
    if mtl_override is None: 
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'mtllib':
                all_materials += material.load_mtl(os.path.join(obj_path, line.split()[1]), clear_ks) # Read in entire material library
    else:
        all_materials += material.load_mtl(mtl_override)
        
    # 顶点加载（load vertices），分离顶点，材质，法向量数据。
    vertices, texcoords, normals  = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])

    # 面加载（load faces）
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'usemtl': # Track used materials
            mat = _find_mat(all_materials, line.split()[1])
            if not mat in used_materials:
                used_materials.append(mat)
            activeMatIdx = used_materials.index(mat)
        elif prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if vv[2] != "" else -1
            for i in range(nv - 2): # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)

    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    else:
        uber_material = used_materials[0]

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda') if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device='cuda') if len(normals) > 0 else None
    
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None

    # Read weights and bones if available
    try:
        v_weights = torch.tensor(np.load(os.path.splitext(filename)[0] + ".weights.npy"), dtype=torch.float32, device='cuda')
        bone_mtx = torch.tensor(np.load(os.path.splitext(filename)[0] + ".bones.npy"), dtype=torch.float32, device='cuda')
    except:
        v_weights, bone_mtx = None, None

    return mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, v_weights=v_weights, bone_mtx=bone_mtx, material=uber_material)

######################################################################################
# 将网格保存为obj文件（Save mesh object to objfile）
######################################################################################

def write_obj(folder, mesh):
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        print("    writing %d texcoords" % len(v_tex))
        if v_tex is not None:
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        print("    writing %d normals" % len(v_nrm))
        if v_nrm is not None:
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    mtl_file = os.path.join(folder, 'mesh.mtl')
    print("Writing material: ", mtl_file)
    material.save_mtl(mtl_file, mesh.material)

    _write_weights(folder, mesh)
    _write_bones(folder, mesh)

    print("Done exporting mesh")
