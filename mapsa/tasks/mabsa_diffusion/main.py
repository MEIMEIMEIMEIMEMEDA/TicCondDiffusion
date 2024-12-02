from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Union
from tqdm import tqdm

from transformers import AutoImageProcessor
from transformers import PretrainedConfig
import numpy as np

from PIL import Image
from mapsa.core import CoreFactory
from mapsa.core import CoreFactoryConfig
from mapsa.core import get_diffusion_args
from mapsa.data.data_types import MetricType, TwitterSample
from mapsa.data.data_types import TaskType
from mapsa.data.data_types import TrainerType

WORKER_NUMS = 32


def create_model(config: PretrainedConfig):
    from mapsa.model.modeling_mabsa_diffusion import MAPSADiffusion

    model = MAPSADiffusion(config)

    return model


def get_img_processor(name):
    try:
        img_processor = AutoImageProcessor.from_pretrained(name)
        return img_processor
    except:
        print(f"{name} image processor not have")


class ImageProcessor:

    def __init__(self, name) -> None:
        self._processor = get_img_processor(name)

    def __call__(self, data: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(data, np.ndarray):
            data = Image.fromarray(data)
        return self._processor.preprocess(data, return_tensors="np")["pixel_values"][0]


def replace_field(
    config: CoreFactoryConfig,
    src_field="image",
    dst_field="input_image",
    processor=None,
):
    # Create a ThreadPoolExecutor with worker threads
    if processor is None:
        processor = lambda x, y: y

    src_field_idx = TwitterSample._fields.index(src_field)
    dst_field_idx = TwitterSample._fields.index(dst_field)

    def _call_processor(fn, src_idx, dst_idx, processor):
        data = np.load(fn, allow_pickle=True)["data"].tolist()
        data[dst_idx] = processor(data[src_idx])
        with open(fn, "wb") as fp:
            np.savez_compressed(
                fp, data=np.array(data, dtype=object), meta=TwitterSample._fields
            )

    start = time.time()
    all_npz_files = []
    for root in [config.train_root, config.val_root, config.test_root]:
        all_npz_files += list(Path(root).glob("*.npz"))

    pbar = tqdm(all_npz_files, f"Replace {dst_field} in {dst_field_idx}")
    with ThreadPoolExecutor(max_workers=WORKER_NUMS) as executor:
        futures = [
            executor.submit(
                _call_processor, filename, src_field_idx, dst_field_idx, processor
            )
            for filename in pbar
        ]
    # Wait for all threads to complete
    for future in futures:
        future.result()
    print(f"Runtime: {time.time() - start:.2f} seconds, data:")


def main(args):
    dataset = Path(args.dataset)
    if not dataset.exists():
        raise ValueError(f"Please check the path {dataset}")
    dataset_useful_fields = [
        "image_id",  # Image
        # "image",
        "input_image",
        "words",  # Text
        'image_caption',
        "raw_target",
    ]
    core_config = CoreFactoryConfig(
        lm_name=args.lm_name,
        im_name=args.im_name,
        task_type=TaskType.MABSA_DIFFUSION,
        trainer_type=TrainerType.DIFFUSION,
        label_type=args.label_type,
        module_type=args.module_type,
        loss_type=MetricType.ABSA_DIFFUSION,
        eval_type=MetricType.ABSA_DIFFUSION,
        train_root=str(dataset / "train"),
        val_root=str(dataset / "valid"),
        test_root=str(dataset / "test"),
        output_root_dir=f"/home/haomei/raid1/vscodeproject/MAPSA/mapsa_exp/{dataset.parent.stem}",
        dataset_useful_fields=dataset_useful_fields,
        max_length=60,
    )
    # NOTE: dynamic replacement by experiment lm to accelerate data loading.
    # if args.re_gen_input_image:
    #     img_processor = ImageProcessor(args.im_name)
    #     replace_field(core_config, processor=img_processor)

    core_factory = CoreFactory(core_config)
    model = create_model(core_config.create_hf_config(core_factory.tokenizer))
    
    # Calculate and print the number of parameters in the model
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters in the model: {total_params}")
    
    core_factory.run(
        model=model,
        data_collator=core_factory.get_data_collator(
            repeat_gt_nums=args.repeat_gt_nums,
        ),
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main(get_diffusion_args())
