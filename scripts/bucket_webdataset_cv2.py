from argparse import ArgumentParser, Namespace
import wandb

from support.cv2_bucketer import Bucketer, BucketerArgs

if __name__ == '__main__':
  parser = ArgumentParser()
  Bucketer.add_argparse_args(parser)
  args: Namespace = parser.parse_args()

  dm_args: BucketerArgs = Bucketer.map_parsed_args(args)
  mapper = Bucketer(dm_args)

  if dm_args.use_wandb:
    wandb.login()
    wandb.init(
      project="wds-arb-cc12m",
      entity=dm_args.wandb_entity,
      name=dm_args.wandb_run_name,
    )
  mapper.main()