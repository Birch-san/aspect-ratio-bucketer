from __future__ import annotations
from argparse import ArgumentParser, Namespace
from typing import TypedDict, Generator, NamedTuple, Optional, List, Callable, Dict
from webdataset import WebDataset, ShardWriter
from os import listdir, makedirs
from os.path import exists, join
from pathlib import Path
from dataclasses import dataclass
# from numpy import frombuffer
from numpy.typing import NDArray
import numpy as np
import cv2
import json
import torch
from cv2 import Mat
import torch
from torch import FloatTensor, LongTensor, IntTensor, tensor, argmin, abs, load, frombuffer
from tqdm import tqdm
import fnmatch
import wandb
import datetime as dt
from time import sleep
from contextlib import ExitStack
from enum import Enum
from PIL import Image
from io import BytesIO
from torchvision.io import write_png, write_jpeg, decode_jpeg, encode_jpeg, ImageReadMode
from torchvision.transforms.v2 import Resize, Compose, CenterCrop, Transform, InterpolationMode

class HWDims(NamedTuple):
  height: int
  width: int

class InputRecordJson(TypedDict):
  width: int
  height: int

class ImgEncodeResult(NamedTuple):
  success: bool
  buffer: NDArray

class InputRecord(TypedDict):
  # '00000005'
  __key__: str
  # '/home/birch/ml-data/cc12m/data/00000.tar'
  __url__: str
  # will be NDArray only if you used .decode()
  jpg: bytes | NDArray
  json: bytes
  txt: bytes

class MappedRecord(NamedTuple):
  # __key__: str
  # __url__: str
  bucket_ix: int
  cc12m_key: str
  img: Image.Image
  txt: bytes
  crop_top: int
  crop_left: int

OutputRecord = TypedDict('OutputRecord', {
  '__key__': str,
  'cc12m_key': str,
  'img.jpg': Image.Image,
  'txt': bytes,
  'crop_top': str,
  'crop_left': str,
})

class DiscardReason(Enum):
  ShorterSideInsufficient='shorter_side_insufficient'
  ASideSmallerThanBucket='a_side_smaller_than_bucket'
  FitNoBucketAdequately='fit_no_bucket_adequately'

@dataclass
class BucketerArgs:
  wds_input_dir: str
  wds_output_dir: str
  use_wandb: bool
  wandb_run_name: Optional[str]
  wandb_entity: Optional[str]
  samples_limit: Optional[int]
  solar_only: bool
  square_side_len_px: int
  bucket_overflow_discard_pct: float

class Bucketer:
  args: BucketerArgs
  use_cv2_cuda: bool
  dtype: torch.dtype
  device: torch.device
  discarded: int
  discard_reasons: Dict[DiscardReason, int]
  buckets: List[HWDims]
  imgs_written_by_bucket: Dict[str, int]

  @staticmethod
  def add_argparse_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group('Bucketer')
    group.add_argument('--wds_input_dir', type=str, help='directory into which you have cloned https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/data', required=True)
    group.add_argument('--wds_output_dir', type=str, help='where we will output the processed data', required=True)
    group.add_argument('--use_wandb', default=False, action='store_true')
    group.add_argument('--wandb_entity', default=None, type=str, required=False)
    group.add_argument('--wandb_run_name', default=None, type=str, required=False)
    group.add_argument('--samples_limit', default=None, type=int, required=False)
    group.add_argument('--solar_only', default=False, action='store_true')
    group.add_argument('--square_side_len_px', default=512, type=int)
    group.add_argument('--bucket_overflow_discard_pct', default=1.2, type=float, help="if either dimension exceeds the best-fit bucket by this fraction, then we discard the image (concluding that it doesn't fit well into the best-fit bucket)")
  
  @staticmethod
  def map_parsed_args(namespace: Namespace) -> BucketerArgs:
    return BucketerArgs(
      wds_input_dir=namespace.wds_input_dir,
      wds_output_dir=namespace.wds_output_dir,
      use_wandb=namespace.use_wandb,
      wandb_entity=namespace.wandb_entity,
      wandb_run_name=namespace.wandb_run_name,
      samples_limit=namespace.samples_limit,
      solar_only=namespace.solar_only,
      square_side_len_px=namespace.square_side_len_px,
      bucket_overflow_discard_pct=namespace.bucket_overflow_discard_pct,
    )

  def __init__(
    self,
    args: BucketerArgs,
  ) -> None:
    self.args = args
    self.use_cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    self.dtype = torch.bfloat16
    self.device = torch.device('cuda')
    self.discarded = 0
    self.discard_reasons = { e: 0 for e in DiscardReason }

    # note: these are the aspects *prior* to rounding
    # they're from SDXL:
    # https://github.com/Stability-AI/generative-models/blob/45c443b316737a4ab6e40413d7794a7f5657c19f/scripts/demo/sampling.py#L7
    # but I didn't take anything wider than 2
    aspects: FloatTensor = tensor([
      .5,
      .52,
      .57,
      .6,
      .68,
      .72,
      .78,
      .82,
      .88,
      .94,
      1.,
      1.07,
      1.13,
      1.21,
      1.29,
      1.38,
      1.46,
      1.67,
      1.75,
      1.91,
      2.,
    ], dtype=torch.float32, device=self.device)

    thetas: FloatTensor = aspects.atan()

    thetas_sin: FloatTensor = thetas.sin()
    thetas_cos: FloatTensor = thetas.cos()

    hypotenuses: FloatTensor = args.square_side_len_px/(2*thetas_sin*thetas_cos)**.5

    heights: FloatTensor = hypotenuses*2**.5*thetas_sin
    widths: FloatTensor = hypotenuses*2**.5*thetas_cos

    buckets = torch.column_stack([heights, widths])
    buckets_rounded: IntTensor = 8*(buckets / 8).round().int()

    # you can check how close we got to them having same area as square:
    # buckets_rounded.prod(dim=-1)-args.square_side_len_px**2

    self.bucket_aspects = (buckets_rounded[:,0]/buckets_rounded[:,1]).cpu()
    self.buckets = [HWDims(height=bucket[0].item(), width=bucket[1].item()) for bucket in buckets_rounded.cpu()]

    self.imgs_written_by_bucket = {
      Bucketer.get_bucket_metric_key(bucket_h, bucket_w): 0 for bucket_h, bucket_w in self.buckets
    }
  
  @staticmethod
  def encode(img: Mat) -> NDArray:
    encoded: ImgEncodeResult = cv2.imencode('.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 95])
    success, encoded_buff = encoded
    assert success
    return encoded_buff

  @staticmethod
  def get_bucket_fit_dims(orig: HWDims, bucket: HWDims) -> HWDims:
    height_quotient: float = orig.height / bucket.height
    width_quotient: float = orig.width / bucket.width

    if width_quotient < height_quotient:
      return HWDims(height=round(orig.height/width_quotient), width=bucket.width)
    return HWDims(height=bucket.height, width=round(orig.width/height_quotient))
  
  def compose(self, records: Generator[InputRecord, None, None]) -> Generator[MappedRecord, None, None]:
    for record in records:
      meta: InputRecordJson = json.loads(record['json'])
      shorter_side: int = min(meta['height'], meta['width'])

      if shorter_side < self.args.square_side_len_px:
        self.discarded += 1
        self.discard_reasons[DiscardReason.ShorterSideInsufficient] += 1
        continue

      aspect: float = meta['width'] / meta['height']
      closest_bucket_ix: LongTensor = argmin(abs(self.bucket_aspects - tensor(aspect)))
      closest_bucket: HWDims = self.buckets[closest_bucket_ix]

      bucket_w, bucket_h = closest_bucket

      if meta['height'] < bucket_h or meta['width'] < bucket_w:
        self.discarded += 1
        self.discard_reasons[DiscardReason.ASideSmallerThanBucket] += 1
        continue

      encoded: IntTensor = frombuffer(record['jpg'], dtype=torch.uint8)
      img: FloatTensor = decode_jpeg(encoded, mode=ImageReadMode.RGB, device=self.device)

      meta: InputRecordJson = json.loads(record['json'])
      orig_height, orig_width = meta['height'], meta['width']
      orig_dims = HWDims(height=orig_height, width=orig_width)

      resize_dims: HWDims = Bucketer.get_bucket_fit_dims(orig=orig_dims, bucket=closest_bucket)
      excess_pct: float = max(resize_dims.height / bucket_h, resize_dims.width / bucket_w)
      if excess_pct > self.args.bucket_overflow_discard_pct:
        # we lost more than 20% of the image along some dimension. it's not a good fit for even the closest bucket.
        self.discarded += 1
        self.discard_reasons[DiscardReason.FitNoBucketAdequately] += 1
        continue

      tform = Compose([
        Resize(size=resize_dims, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(size=closest_bucket)
      ])

      bucketed_img: FloatTensor = tform(img)
      re_encoded: IntTensor = encode_jpeg(bucketed_img.cpu(), quality=95)

      pil: Image.Image = Image.open(BytesIO(bytearray(re_encoded)))

      height_diff = resize_dims.height - closest_bucket.height
      width_diff = resize_dims.width - closest_bucket.width

      crop_top = height_diff//2
      crop_left = width_diff//2

      mapped = MappedRecord(
        bucket_ix=closest_bucket_ix,
        cc12m_key=record['__key__'],
        img=pil,
        txt=record['txt'],
        crop_top=crop_top,
        crop_left=crop_left,
      )

      yield mapped
  
  @staticmethod
  def get_discard_reason_metric_name(discard_reason: DiscardReason) -> str:
    return f'imgs_discarded_{discard_reason.value}'
  
  @staticmethod
  def get_bucket_metric_key(bucket_h: int, bucket_w: int) -> str:
    return f'imgs_written_bucket_w{bucket_w}_h{bucket_h}'

  @staticmethod
  def get_bucket_dirname(bucket_h: int, bucket_w: int) -> str:
    return f'bucket_w{bucket_w}_h{bucket_h}'
  
  def main(self) -> None:
    assert exists(self.args.wds_input_dir), f'Please ensure existence of the directory indicated by --wds_input_dir option: {self.args.wds_input_dir}'
    input_dir_files: List[str] = listdir(self.args.wds_input_dir)
    input_shard_files_unsorted: List[str] = fnmatch.filter(input_dir_files, f'*.tar')
    input_shard_count: int = len(input_shard_files_unsorted)
    assert input_shard_count > 0
    shard_keyer: Callable[[str], int] = lambda fname: int(Path(fname).stem)
    shard_files: List[str] = [join(self.args.wds_input_dir, file) for file in sorted(input_shard_files_unsorted, key=shard_keyer)]
    del input_shard_files_unsorted
    first_ix = int(Path(shard_files[0]).stem)
    last_ix = int(Path(shard_files[-1]).stem)
    url = join(self.args.wds_input_dir, f'{{{first_ix:05d}..{last_ix:05d}}}.tar')

    log_interval_iterations=20
    log_interval_current=log_interval_iterations
    written=0
    input_shard_size=10000
    imgs_estimate=input_shard_count*input_shard_size

    if self.args.use_wandb:
      wandb.config.update({
        'imgs_estimate': imgs_estimate,
        'input_shard_count': input_shard_count,
        'input_shard_size': input_shard_size,
      })
      wandb.log({
        'imgs_written': written,
        'imgs_discarded': self.discarded,
        **self.imgs_written_by_bucket,
        **{ Bucketer.get_discard_reason_metric_name(key): value for key, value in self.discard_reasons.items() }
      }, commit=False)
    
    def on_solar(now: dt.datetime) -> bool:
      hour = now.hour
      minute = now.minute
      if hour < 8:
        return False
      if hour == 8:
        # don't want to be woken up by fans
        return minute > 45
      if hour < 20:
        return True
      if hour == 20:
        return minute < 30
      return False

    makedirs(self.args.wds_output_dir, exist_ok=True)

    dataset = WebDataset(url).compose(self.compose)

    bucket_dirnames: List[str] = [Bucketer.get_bucket_dirname(bucket_h, bucket_w) for bucket_h, bucket_w in self.buckets]
    bucket_dirs: List[str] = [join(self.args.wds_output_dir, name) for name in bucket_dirnames]
    print(f"making bucket dirs under {self.args.wds_output_dir}: {bucket_dirnames}")
    for bucket_dir in bucket_dirs:
      makedirs(bucket_dir, exist_ok=True)

    with ExitStack() as stack:
      shard_writers: List[ShardWriter] = [
        ShardWriter(join(bucket_out, '%05d.tar'),maxcount=10000) for bucket_out in bucket_dirs
      ]
      for mgr in shard_writers:
        stack.enter_context(mgr)

      for record in tqdm(dataset, total=imgs_estimate, unit=f'img'):
        record: MappedRecord = record
        bucket_ix, cc12m_key, img, text, crop_top, crop_left = record

        bucket: HWDims = self.buckets[bucket_ix]
        bucket_h, bucket_w = bucket

        sink: ShardWriter = shard_writers[bucket_ix]
        out_record: OutputRecord = {
          '__key__': f'{written:08}',
          'cc12m_key': cc12m_key,
          'img.jpg': img,
          'txt': text,
          'crop_top': str(crop_top),
          'crop_left': str(crop_left),
        }
        sink.write(out_record)

        metric_key: str = Bucketer.get_bucket_metric_key(bucket_h, bucket_w)
        self.imgs_written_by_bucket[metric_key] += 1
        written += 1

        log_interval_current -= 1
        if log_interval_current == 0:
          log_interval_current = log_interval_iterations
          if self.args.use_wandb:
            wandb.log({
              'imgs_written': written,
              'imgs_discarded': self.discarded,
              metric_key: self.imgs_written_by_bucket[metric_key],
              **{ Bucketer.get_discard_reason_metric_name(key): value for key, value in self.discard_reasons.items() }
            })
        if self.args.solar_only:
          while not on_solar(dt.datetime.today()):
            sleep(60)
        if self.args.samples_limit is not None and written >= self.args.samples_limit:
          print(f'wrote {written} samples, which exceeds our samples_limit of {self.args.samples_limit}')
          break
      
    print(f'imgs_written final: {written}')
    if self.args.use_wandb:
      wandb.finish()
