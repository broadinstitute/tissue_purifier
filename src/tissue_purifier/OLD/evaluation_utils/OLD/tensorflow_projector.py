import tarfile
import os
import shutil
import numpy as np
import pandas as pd
import pathlib
import matplotlib
from PIL import Image


def to_image(img):
    return img.cpu().numpy().transpose((1, 2, 0))[:, :, :3]


def scale_image(tensor):
    min_, max_ = tensor.min(), tensor.max()
    dist = max_ - min_
    dist[dist == 0.] = 1.
    scale = 1.0 / dist
    tensor.mul_(scale).sub_(min_)
    return tensor


def create_sprite(images_path, sprite_save_path):
    images = [Image.open(f"{images_path}/{i}.png").resize((200, 200)) for i in range(700)]
    image_width, image_height = images[0].size
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = (image_width * one_square_size)

    master_height = image_height * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0))  # fully transparent

    for count, image in enumerate(images):
        div, mod = divmod(count, one_square_size)
        h_loc = image_width * div
        w_loc = image_width * mod
        spriteimage.paste(image, (w_loc, h_loc))

    spriteimage.convert("RGB").save(sprite_save_path, transparency=0)


def create_tarfile(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def create_projector(dataloader, embeddings, meta_kwargs, apply_compress=True) -> None:
    """
    Create TensorBoard log directory with Projector data
    compressed [optionally] in a tar zip array


    Args:
        dataloader: Dataloader with images. It is used to create simplified
            versions of pictures to put them in a projector space
        embeddings: Array of embedded pictures
        meta_kwargs: Metadata which is passed to projector (it can be labels,
            score metrics etc)
        apply_compress: Boolean flag indicating that we need to compress projector data

    Examples:
        >>> create_projector(dataloader, embeddings, {"labels": y})
        >>> # Will create projector data in the current directory
        >>> # To create a tensorboard you will need to use a command-line tool:
        >>> # `tensorboard --logdir ./projector`
    """
    projector_dir = "./projector"
    tmp_dir = f"{projector_dir}/tmp"

    pathlib.Path("projector/tmp").mkdir(parents=True, exist_ok=True)
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.to_csv(f"{projector_dir}/embeddings.tsv", sep='\t', index=False, header=False)
    df_meta = pd.DataFrame(meta_kwargs)
    df_meta.to_csv(f"{projector_dir}/embeddings_meta.tsv", sep='\t', index=False)

    i = 0
    for batch in dataloader:
        imgs, labels, fnames = batch[:3]
        
        for tensor, label in zip(imgs, labels):
            x = scale_image(tensor)
            x = to_image(x)
            matplotlib.image.imsave(f"{tmp_dir}/{i}.png", x)
            i += 1

    create_sprite(tmp_dir, f"{projector_dir}/sprite.jpg")
    with open(f"{projector_dir}/projector_config.pbtxt", "w+") as f:
        f.write(
            'embeddings {'
            '  tensor_path: "embeddings.tsv"'
            '  metadata_path: "embeddings_meta.tsv"'
            '  sprite {'
            '    image_path: "sprite.jpg"'
            '    single_image_dim: 200'
            '    single_image_dim: 200'
            '  }'
            '}'
        )

    shutil.rmtree(tmp_dir)

    if apply_compress:
        create_tarfile("projector", "projector.tar.gz")
        shutil.rmtree(projector_dir)
