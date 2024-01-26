import os
import sys
import time
import argparse
import json

from . import util


def load_args(args):
    parser = argparse.ArgumentParser(description='diffmodeling')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', type=int, default=512)
    parser.add_argument('-rtr', '--random-train-res', action='store_true', default=False)
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=None)
    parser.add_argument('-lp', '--light-power', type=float, default=5.0)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-sd', '--subdivision', type=int, default=0)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-lf', '--laplacian-factor', type=float, default=None)
    parser.add_argument('-rl', '--relative-laplacian', type=bool, default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relativel2'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-rm', '--ref-mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str)
    parser.add_argument('--radius', type=float, default=3.5, help="Camera eye radius")
    return parser.parse_args(args)

def init_flags(args, show=True):
    FLAGS = load_args(args)
    FLAGS.camera_eye = [0.0, 0.0, FLAGS.radius]
    FLAGS.camera_up  = [0.0, 1.0, 0.0]
    FLAGS.skip_train = []
    FLAGS.displacement = 0.15
    FLAGS.mtl_override = None
    FLAGS.proj_mtx = util.projection(x=0.4, f=1000.0)
    
    # 加载配置文件
    if FLAGS.config is not None:
        with open(FLAGS.config) as f:
            data = json.load(f)
            for key in data:
                FLAGS.__dict__[key] = data[key]
    
    # 设置分辨率
    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    
    # 设置输出文件目录
    if FLAGS.out_dir is None:
        out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        out_dir = 'out/' + FLAGS.out_dir
    FLAGS.out_dir = out_dir

    # 设置默认学习率
    if FLAGS.learning_rate is None:
        FLAGS.learning_rate = 0.01

    # 参数检查
    assert not FLAGS.random_train_res or FLAGS.custom_mip, "Random training resolution requires custom mip."

    # 打印配置信息
    if show:
        print("----配置信息----")
        print(f"迭代次数(iter):{FLAGS.iter}")
        print(f"批大小(batch):{FLAGS.batch}")
        print(f"单像素采样数(spp):{FLAGS.spp}")
        print(f"层数(layers):{FLAGS.layers}")
        print(f"训练分辨率(train_res):{FLAGS.train_res}")
        print(f"随机训练分辨率(random_train_res):{FLAGS.random_train_res}")
        print(f"显示分辨率(display_res):{FLAGS.display_res}")
        print(f"材质分辨率(texture_res):{FLAGS.texture_res}")
        print(f"显示间隔(display_interval):{FLAGS.display_interval}")
        print(f"保存间隔(save_interval):{FLAGS.save_interval}")
        print(f"学习率(learning_rate):{FLAGS.learning_rate}")
        print(f"光功率(light_power):{FLAGS.light_power}")
        print(f"最小粗糙度(min_roughness):{FLAGS.min_roughness}")
        print(f"细分(subdivision):{FLAGS.subdivision}")
        print(f"自定义mip(custom_mip):{FLAGS.custom_mip}")
        print(f"随机纹理(random_textures):{FLAGS.random_textures}")
        print(f"拉普拉斯因子(laplacian_factor):{FLAGS.laplacian_factor}")
        print(f"相对拉普拉斯(relative_laplacian):{FLAGS.relative_laplacian}")
        print(f"背景(background):{FLAGS.background}")
        print(f"损失函数(loss):{FLAGS.loss}")
        print(f"输出文件目录(out_dir):{FLAGS.out_dir}")
        print(f"配置文件(config):{FLAGS.config}")
        print(f"参考网格(ref_mesh):{FLAGS.ref_mesh}")
        print(f"基础网格(base_mesh):{FLAGS.base_mesh}")
        print(f"相机位置(camera_eye):{FLAGS.camera_eye}")
        print(f"相机朝向(camera_up):{FLAGS.camera_up}")
        print(f"弧度(radius):{FLAGS.radius}")
        print(f"投影矩阵(proj_mtx):{FLAGS.proj_mtx}")
        print(f"跳过训练(skip_train):{FLAGS.skip_train}")    
        print(f"位移(displacement):{FLAGS.displacement}")    
        print(f"材质重写(mtl_override):{FLAGS.mtl_override}")   
    return FLAGS

