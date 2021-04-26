import argparse
from os import listdir
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from medzoo_imports import create_model, DiceLoss


def vnet(args):
    """Run inference with VNet

    Args:
        model (str): The 3D segmentation model to use
        opt (str): The type of optimizer
        lr (float): The learning rate
        inChannels (int): The desired modalities/channels that you want to use
        classes (int): The number of classes
        dataset_name (str): The name of the dataset
        pretrained (str): Location of the pretrained model
        save (str): The saved checkpoint
        test_dir (str): The directory of the test data

    Returns:
        The training and validation data loaders
    """
    model, optimizer = create_model(args)
    criterion = DiceLoss(classes=args.classes)
    model.restore_checkpoint(args.pretrained)
    model.eval()
    for file in listdir(args.test_dir+"Input/"):
        orig_data = fits.getdata(args.test_dir+"Input/"+file)[:64, :128, :128]
        prepared_data = prepare_data(orig_data)
        realseg_data = fits.getdata(args.test_dir+"Target/mask_"+file.split("_")[-1])[:64, :128, :128]
        
        data_loader_tensor = torch.FloatTensor(prepared_data.astype(np.float32)).unsqueeze(0)[None, ...]
        with torch.no_grad():
            out_cube = model.inference(data_loader_tensor)
        
        loss_dice, per_ch_score = criterion(out_cube, torch.FloatTensor(np.moveaxis(realseg_data, 0, 2).astype(np.float32)).unsqueeze(0)[None, ...])
        
        print(loss_dice)
        fig, axes = plt.subplots(1, 4, figsize=(10, 10))
        axes[0].imshow(orig_data[1])
        axes[1].imshow(prepared_data[..., 1])
        axes[2].imshow(realseg_data[1])
        axes[3].imshow(out_cube.squeeze()[..., 1])
        plt.show()

def main(args):
    vnet(args)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        '--model', type=str, nargs='?', const='default', default='VNET',
        help='The 3D segmentation model to use')
    parser.add_argument(
        '--opt', type=str, nargs='?', const='default', default='sgd',
        help='The type of optimizer')
    parser.add_argument(
        '--lr', type=float, nargs='?', const='default', default=1e-3,
        help='The learning rate')
    parser.add_argument(
        '--inChannels', type=int, nargs='?', const='default', default=1,
        help='The desired modalities/channels that you want to use')
    parser.add_argument(
        '--classes', type=int, nargs='?', const='default', default=1,
        help='The number of classes')
    parser.add_argument(
        '--dataset_name', type=str, nargs='?', const='default', default='hi_source',
        help='The name of the dataset')
    parser.add_argument(
        '--save', type=str, nargs='?', const='default',
        default='./inference_checkpoints/VNET_checkpoints/VNET_hi_source',
        help='The checkpoint location')
    parser.add_argument(
        '--pretrained', type=str, nargs='?', const='default',
        default="../saved_models/VNET_checkpoints/VNET_/VNET__BEST.pth",
        help='The checkpoint location')
    parser.add_argument(
        '--cuda', type=bool, nargs='?', const='default', default=False,
        help='Memory allocation')
    parser.add_argument(
        '--test_dir', type=bool, nargs='?', const='default', default=False,
        help='Memory allocation')
    args = parser.parse_args()

    main(args)