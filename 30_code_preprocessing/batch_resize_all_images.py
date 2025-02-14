from utils.image_manipulation import batch_resize_images

if __name__ == "__main__":
    input_dir = (
        "/home/matthias/workspace/Coding/20_helmet_reco/data_hh_3/train/images_orig_size"  # Update this path
        # "/home/matthias/workspace/Coding/20_helmet_reco/data_hh_3/val/images_orig_size"  # Update this path
    )
    output_dir = (
        "/home/matthias/workspace/Coding/20_helmet_reco/data_hh_3/train/images"  # Update this path
    )
    # output_dir = "/home/matthias/workspace/Coding/20_helmet_reco/data_hh_3/test/images"  # Update this path
    target_width = 640
    target_height = target_width
    keep_aspect = False

    batch_resize_images(
        path_input_dir=input_dir,
        path_output_dir=output_dir,
        target_width=target_width,
        target_height=target_height,
        keep_aspect_ratio=keep_aspect,
    )
