import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_path", type=int, default=8, help="input image path")
    parser.add_argument("-s", "--step", type=int, default=8, help="time slice")
    parser.add_argument("-os", "--output_size", type=tuple, default=(128, 128), help="size of images(H, W)")
    args = parser.parse_args()

    setup_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    img_path = args.img_path
    step = args.step
    output_size = args.output_size

    # get image
    cv2.imread()

    net = SegmentModel(output_size=output_size, out_cls=train_dataset.num_class, node=BiasLIFNode, step=step)