import argparse
from os import listdir
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from medzoo_imports import create_model, DiceLoss
import gc


def load_vnet(args, data_loader_tensor):
    """Run inference with VNet

    Args:
        args (str): The model arguments
        data_loader_tensor (tensor): The input tensor

    Returns:
        The training and validation data loaders
    """
    model, optimizer = create_model(args)
    model.restore_checkpoint(args.pretrained)
    model.eval()
    with torch.no_grad():
        out_cube = model.inference(data_loader_tensor)
    return out_cube

def main(args):
    criterion = DiceLoss(classes=args.classes)
    interval = ZScaleInterval()
    orig_data = fits.getdata(args.test_dir+"Input/"+file)[:64, :128, :128]
    prepared_data = interval(np.nan_to_num(np.moveaxis(orig_data, 0, 2)))
    del orig_data
    gc.collect()
    data_loader_tensor = torch.FloatTensor(prepared_data.astype(np.float32)).unsqueeze(0)[None, ...]
    del prepared_data
    gc.collect()
    realseg_data = fits.getdata(args.test_dir+"Target/mask_"+file.split("_")[-1])[:64, :128, :128]
    mask_tensor = torch.FloatTensor(np.moveaxis(realseg_data, 0, 2).astype(np.float32)).unsqueeze(0)[None, ...]
    del realseg_data
    gc.collect()
    # VNET
    vnet_out_tensor = load_vnet(args, data_loader_tensor)
    vnet_loss_dice, vnet_per_ch_score = criterion(vnet_out_tensor, mask_tensor)
    # MTO
    mto_seg_data = fits.getdata(args.test_dir+"MTO/c%s_mto_lvq"%(file.split("_")[-1]))[:64, :128, :128]
    mto_out_tensor = torch.FloatTensor(mto_seg_data.astype(np.float32)).unsqueeze(0)[None, ...]
    del mto_seg_data
    gc.collect()
    mto_loss_dice, mto_per_ch_score = criterion(mto_out_tensor, mask_tensor)

    return vnet_loss_dice, mto_loss_dice


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