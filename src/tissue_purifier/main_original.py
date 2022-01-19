#!/usr/bin/env python
import argparse
import torch
import sys
from typing import List
from datetime import timedelta
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.profiler import SimpleProfiler, PassThroughProfiler, AdvancedProfiler
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from tissue_purifier.misc_utils.misc import smart_bool
from tissue_purifier.model_utils.dino import DinoModel
from tissue_purifier.model_utils.vae import VaeModel
from tissue_purifier.data_utils.datamodule import DinoDM, SlideSeqTestisDM, SlideSeqKidneyDM, DummyDM
from tissue_purifier.model_utils.logger import NeptuneLoggerCkpt


def initialization(
        args_dict: dict,
        ckpt_file: str,
        initialization_type: str,
        ) -> (pl.LightningModule, pl.Trainer, dict, str):
    """
    Initialize all that is necessary for the simulation

    Args:
        args_dict: the command-line arguments obtained from parser.parse_args()
        ckpt_file: path to a ckpt file (obtained from trainer.save_checkpoint or the CheckpointCallback)
        initialization_type: str, either 'resume', 'extend', 'scratch', 'pretraining', 'predict_only'

    Returns:
        pl_model: the pytorch_lightning model either a scratch or one loaded from a checkpoint
        pl_trainer:  the pytorch_lightning trainer
        new_args_dict: dict with the configurations. This is identical to :attr:'args_dict'
            if :attr:'initialization_type' == 'scratch'.
            In other cases it can be different from :attr:'args_dict' because some entries are overwritten.
        ckpt_file_for_trainer: None or a str pointing to a ckpt_file to be used to resume the training.
    """
    assert initialization_type in {'resume', 'extend', 'scratch', 'pretraining', 'predict_only'}

    if initialization_type in {'resume', 'extend', 'pretraining', 'predict_only'}:
        assert ckpt_file is not None
    elif initialization_type == 'scratch':
        ckpt_file = None

    if initialization_type in {'resume', 'extend'}:
        ckpt_file_for_trainer = ckpt_file
    else:
        ckpt_file_for_trainer = None

    if initialization_type in {'resume'}:
        # use ckpt_file only
        if args_dict["model"] == "dino":
            pl_model = DinoModel.load_from_checkpoint(checkpoint_path=ckpt_file)
        elif args_dict["model"] == "vae":
            pl_model = VaeModel.load_from_checkpoint(checkpoint_path=ckpt_file)
        else:
            raise Exception("Invalid model value. Received {0}".format(args_dict["model"]))
        # TODO: I am having trouble using the same run_id
        neptune_run_id = None  # pl_model.neptune_run_id
        new_dict = pl_model.__dict__['_hparams'].copy()

    elif initialization_type in {'predict_only'}:
        # use ckpt_file only but overwrite stuff relative to the duration of the training
        if args_dict["model"] == "dino":
            pl_model = DinoModel.load_from_checkpoint(checkpoint_path=ckpt_file)
        elif args_dict["model"] == "vae":
            pl_model = VaeModel.load_from_checkpoint(checkpoint_path=ckpt_file)
        else:
            raise Exception("Invalid model value. Received {0}".format(args_dict["model"]))
        # TODO: I am having trouble using the same run_id
        neptune_run_id = None  # pl_model.neptune_run_id
        new_dict = pl_model.__dict__['_hparams'].copy()
        new_dict['training'] = False

    elif initialization_type in {'extend'}:
        # use ckpt_file only but overwrite stuff relative to the duration of the training
        if args_dict["model"] == "dino":
            pl_model = DinoModel.load_from_checkpoint(checkpoint_path=ckpt_file)
        elif args_dict["model"] == "vae":
            pl_model = VaeModel.load_from_checkpoint(checkpoint_path=ckpt_file)
        else:
            raise Exception("Invalid model value. Received {0}".format(args_dict["model"]))
        # TODO: I am having trouble using the same run_id
        neptune_run_id = None  # pl_model.neptune_run_id
        new_dict = pl_model.__dict__['_hparams'].copy()
        for key, value in args_dict.items():
            if key in {"max_epochs", "max_time_minutes"}:
                new_dict[key] = value

    elif initialization_type in {'scratch'}:
        # use args only
        if args_dict["model"] == "dino":
            pl_model = DinoModel(**args_dict)
        elif args_dict["model"] == "vae":
            pl_model = VaeModel(**args_dict)
        else:
            raise Exception("Invalid model value. Received {0}".format(args_dict["model"]))
        neptune_run_id = None
        new_dict = args_dict.copy()

    elif initialization_type in {'pretraining'}:
        # use checkpoint but overwrite
        if args_dict["model"] == "dino":
            pl_model = DinoModel.load_from_checkpoint(checkpoint_path=ckpt_file, **args_dict)
        elif args_dict["model"] == "vae":
            pl_model = VaeModel.load_from_checkpoint(checkpoint_path=ckpt_file, **args_dict)
        else:
            raise Exception("Invalid model value. Received {0}".format(args_dict["model"]))
        neptune_run_id = None
        new_dict = args_dict.copy()

    else:
        raise Exception("You should not be here initialization_type = {0}".format(initialization_type))

    print("new_dict ->", new_dict)

    pl_neptune_logger = NeptuneLoggerCkpt(
        project='cellarium/tissue-purifier',  # args_dict["neptune_project"],  # change this to your project
        run=neptune_run_id,  # pass something here to keep logging onto an existing run
        log_model_checkpoints=True,  # copy the checkpoints into Neptune
        # neptune kargs
        mode="async" if args_dict.get("logging", False) else "offline",
        tags=[str(args_dict.get("model", "no model")), str(args_dict.get("dataset", "no_dataset"))],
        source_files=["main*.py", "*.yaml"],
        fail_on_exception=True,  # it does not good if you are not logging anything but simulation keeps going
    )
    assert isinstance(pl_neptune_logger, NeptuneLoggerCkpt)

    if new_dict["profiler"] == 'advanced':
        profiler = AdvancedProfiler(dirpath="./", filename="advanced_profiler.out", line_count_restriction=1.0)
    elif new_dict["profiler"] == "simple":
        profiler = SimpleProfiler(dirpath="./", filename="simple_profiler.out")
    else:
        profiler = PassThroughProfiler()

    if torch.cuda.device_count() == 0:
        # cpu emulating ddp process
        # TODO: This hangs. Lightining removed the ddp_cpu 
        strategy = 'ddp'
        accelerator = 'cpu'
        num_processes = 2
        sync_batchnorm = False
        precision = 32
    elif torch.cuda.device_count() == 1:
        # single gpu
        accelerator = None
        strategy = None
        num_processes = 1
        sync_batchnorm = False
        precision = new_dict["precision"]
    else:
        # more that 1 gpu
        accelerator = None
        if args_dict["model"] == "dino":
            # dino uses automatic optimization. I can set this flag to False for speed.
            strategy = DDPPlugin(find_unused_parameters=False)
        elif args_dict["model"] == "vae":
            # vae uses manual optimization. I need to set this flag to true
            strategy = DDPPlugin(find_unused_parameters=True)
        else:
            raise Exception("Model is not recognized. Received {0}".format(args_dict["model"]))
        num_processes = 1
        sync_batchnorm = True
        precision = new_dict["precision"]

    # monitor the learning rate. This will work both when manual or scheduler is used to change the learning rate.
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)

    # save the best ckpt according to the monitor quantity
    ckpt_save_best = ModelCheckpoint(
        filename="best_checkpoint-{epoch}",  # the extension .ckpt will be added automatically
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=False,
        monitor='train_loss',
        save_top_k=1,
        mode='min',
        every_n_epochs=100,  # this is how frequently I check the monitor quantity
    )

    # save ckpts every XXX minutes or YYY epochs (useful for resuming after pre-emption)
    ckpt_train_interval = ModelCheckpoint(
        filename="periodic_checkpoint-{epoch}",  # the extension .ckpt will be added automatically
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=False,
        # the following 2 are mutually exclusive. Determine how frequently to save
        train_time_interval=timedelta(minutes=new_dict["checkpoint_interval_minutes"]),
        # every_n_epochs=3,
    )

    # save ckpt at the end of training
    ckpt_train_end = ModelCheckpoint(
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=True,
        every_n_epochs=0,
    )
    ckpt_train_end.CHECKPOINT_NAME_LAST = 'my_checkpoint_last'  # the extension .ckpt will be added automatically

    pl_trainer = Trainer(
        weights_save_path="saved_ckpt",
        profiler=profiler,
        num_nodes=num_processes,  # uses a single machine possibly with many gpus,
        accelerator=accelerator,
        gpus=torch.cuda.device_count(),  # number of gpu cards on a single machine to use
        check_val_every_n_epoch=new_dict["check_val_every_n_epoch"],
        callbacks=[ckpt_train_end, ckpt_train_interval, ckpt_save_best, lr_monitor],
        strategy=strategy,
        num_sanity_val_steps=0,
        # debugging
        # fast_dev_run=True,
        overfit_batches=new_dict["overfit_batches"],
        # Select how long to train for
        max_epochs=new_dict["max_epochs"],
        max_time=timedelta(minutes=new_dict["max_time_minutes"]),
        # other stuff
        logger=pl_neptune_logger,
        log_every_n_steps=100,
        num_processes=num_processes,
        sync_batchnorm=sync_batchnorm,
        precision=precision,  # if using P100 GPU reduce to 16 (half-precision) to have massive speedup
        # If model uses automatic_optimization -> you can use gradient clipping in the trainer.
        # Otherwise gradient clipping need to be performed internally in the model
        gradient_clip_val=new_dict["gradient_clip_val"] if pl_model.automatic_optimization else None,
        gradient_clip_algorithm=new_dict["gradient_clip_algorithm"] if pl_model.automatic_optimization else None,
        # these are for debug and make the model slower
        detect_anomaly=new_dict["detect_anomaly"],
        deterministic=new_dict["deterministic"],
        # track_grad_norm='inf',
    )

    return pl_model, pl_trainer, new_dict, ckpt_file_for_trainer


def run_simulation(config_dict: dict, datamodule: DinoDM):
    """
    This is where most of the work ois done.
    Log info, train the model, save the checkpoint.

    Args:
        config_dict: dictionary with all the config parameters
        datamodule: a DinoDM datamodule (includes, train_dataloader and val_dataloaders)
    """

    pl.seed_everything(seed=config_dict['random_seed'], workers=True)

    # Initialization need to handle 4 situations: "resume", "extend", "pretraining", "scratch"
    try:
        print("trying to restart from pre-emption checkpoint")
        model, trainer, hparam_dict, ckpt_file_trainer = initialization(
            ckpt_file="./preemption_ckpt.pt",
            args_dict=config_dict,
            initialization_type='resume',
        )
    except (Exception, LookupError) as e2:
        print(e2)
        # this can be: resume, pretraining or scratch
        print("trying to initialize from:", config_dict["initialization_type"])
        model, trainer, hparam_dict, ckpt_file_trainer = initialization(
            ckpt_file=None if config_dict["initialization_type"] == 'scratch' else "./old_run_ckpt.pt",
            args_dict=config_dict,
            initialization_type=config_dict["initialization_type"],
        )
    print("initialization done. I have a model, trainer, hparam_dict")

    if model.global_rank == 0:
        # log model and hyperparameters
        trainer.logger.log_hyperparams(hparam_dict)
        trainer.logger.log_model_summary(model=model, log_name="training/model/summary")
        trainer.logger.log_long_text(
            text=model.__str__(),
            log_name="training/model/architecture",
        )

        # log the transform into neptune
        trainer.logger.log_long_text(
            text=datamodule.trsfm_train_local.__repr__(),
            log_name="transform/train_local",
        )
        trainer.logger.log_long_text(
            text=datamodule.trsfm_train_global.__repr__(),
            log_name="transform/train_global",
        )
        trainer.logger.log_long_text(
            text=datamodule.trsfm_test.__repr__(),
            log_name="transform/test",
        )

    if hparam_dict["training"]:
        print("training begins")
        # this use train and validation dataset
        if ckpt_file_trainer is not None:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_file_trainer)
        else:
            trainer.fit(model=model, datamodule=datamodule)

    if hparam_dict["predict"]:
        print("prediction begins, will be here")  # will be written to file using the PredictionWriter callback
        # trainer.predict(model=model,
        #                 datamodule=datamodule,
        #                 return_predictions=False)

    # at the end close connection to neptune database
    if model.global_rank == 0:
        trainer.logger.finalize(status='completed')


def parse_args(argv: List[str]) -> dict:
    """
    Read argv from command-line and produce a configuration dictionary.
    If the command-line arguments include

    If the command-line arguments include '--to_yaml my_yaml_file.yaml' the configuration dictionary is written to file.

    Args:
        argv: the parameter passed from the command line. If argv includes '--from_yaml input.yaml' all
            other CL parameters are neglected. The parameters will be read from file instead.
            Parameters missing in input.yaml will be set to their default values.
            If argv includes '--to_yaml output.yaml' the configuration dictionary is written to file.

    Returns:
        config_dict: a dictionary with all the configuration parameters.

    Note:
        Parameters which are missing from argv or input.yaml will be set to their default values.
    """
    def write_to_yaml(_my_dict, _yaml_file):
        import yaml
        with open(_yaml_file, 'w') as file:
            yaml.dump(_my_dict, file)

    def read_args_from_yaml(_yaml_file):
        import yaml
        with open(_yaml_file, 'r') as stream:
            config_tmp = yaml.safe_load(stream)
        _args_from_file = []
        for k, v in config_tmp.items():
            _args_from_file.append('--'+k)
            if isinstance(v, list):
                _args_from_file += [str(vi) for vi in v]
            else:
                _args_from_file.append(str(v))
        return _args_from_file

    parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
    parser.add_argument("--to_yaml", type=str, default=None,
                        help="Write a yaml file with the chosen (or default) parameters and exit")
    parser.add_argument("--from_yaml", type=str, default=None,
                        help="If given ALL the args will be read from the YAML file specified here. \
                        Command Line arguments will be ignored. Missing arguments will be set to default values")

    # parameters for Neptune and random seed
    parser.add_argument("--random_seed", type=int, default=1, help="Integer specifying the global random seed")
    parser.add_argument("--neptune_project", type=str, default="cellarium/tissue-purifier",
                        help="neptune project name. This is where the data will be logged")

    # parameters for the trainer
    parser.add_argument("--max_epochs", type=int, default=3000, help="maximum number of training epochs")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=25, help="run validation every N epochs")
    parser.add_argument("--profiler", default="passthrough", type=str, choices=["passthrough", "simple", "advanced"])
    parser.add_argument("--max_time_minutes", type=int, default=1440,
                        help="Training will be stopped after this amount of time. \
                        It is safety mechanism to make sure that runs will not be going forever.")
    parser.add_argument("--checkpoint_interval_minutes", type=int, default=25,
                        help="checkpoint the simulation every N minutes")
    parser.add_argument("--precision", type=int, default=16, choices=[32, 16],
                        help="full or half precision. On CPU only machines must be 32")
    parser.add_argument("--overfit_batches", type=int, default=0,
                        help="Specify the number of batches to use to overfit the model. \
                        Usefull for debugging. If 0 no overfit.")
    parser.add_argument("--gradient_clip_val", type=float, default=0.5,
                        help="Clip the gradients to this value. If 0 no clipping")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value", choices=["norm", "value"],
                        help="Algorithm to use for gradient clipping.")
    parser.add_argument("--detect_anomaly", type=smart_bool, default=False, help="Detect anomaly, i.e. Nans")
    parser.add_argument("--deterministic", type=smart_bool, default=False, help="Deterministic operation in CUDA?")

    # select the model and dataset
    parser.add_argument("--model", default="dino", type=str, choices=["dino", "vae"],
                        help="methodology for representation learning")
    parser.add_argument("--dataset", default="slide_seq_testis", type=str,
                        choices=["slide_seq_testis", "slide_seq_kidney", "dummy_dm"],
                        help="datamodule to use for train and validation")

    # simulation parameters
    parser.add_argument("--initialization_type", type=str, default="scratch",
                        choices=["resume", "extend", "predict_only", "pretraining", "scratch"])
    parser.add_argument("--predict", default=False, type=smart_bool,
                        help="use trained model for prediction? Set to false if interested in training only")
    parser.add_argument("--training", default=True, type=smart_bool,
                        help="train the model? Set to false if interested in evaluation only")
    parser.add_argument("--logging", default=True, type=smart_bool,
                        help="Logging to neptune? Set to false for quicker development")

    # parse the known args to decide if I am going to use the command_line or config_file
    (args, _) = parser.parse_known_args(argv)

    if args.from_yaml is not None:
        argv = read_args_from_yaml(args.from_yaml)
        (args, _) = parser.parse_known_args(args=argv)

    # Decide which model to use
    if args.model == "dino":
        parser = DinoModel.add_specific_args(parser)
    elif args.model == 'vae':
        parser = VaeModel.add_specific_args(parser)
    else:
        raise Exception("Invalid model {0}".format(args.model))

    # Decide which dataset to use
    if args.dataset == "slide_seq_testis":
        parser = SlideSeqTestisDM.add_specific_args(parser)
        datamodule_ch_in = SlideSeqTestisDM.ch_in
    elif args.dataset == "slide_seq_kidney":
        parser = SlideSeqKidneyDM.add_specific_args(parser)
        datamodule_ch_in = SlideSeqKidneyDM.ch_in
    elif args.dataset == 'dummy_dm':
        parser = DummyDM.add_specific_args(parser)
        datamodule_ch_in = DummyDM.ch_in
    else:
        raise Exception("Invalid dataset {0}".format(args.dataset))

    # add the help at the very end so that all the options are present
    parser = argparse.ArgumentParser(parents=[parser], add_help=True)

    # read from command_line if args_from_file is None, missing_value will be set to defaults values
    args = parser.parse_args(args=argv)

    # overwrite so that ch_in is the one specified by the datamodule
    args.image_in_ch = datamodule_ch_in

    config_dict_tmp = args.__dict__
    if args.to_yaml is not None:
        yaml_file = args.to_yaml
        config_to_file = config_dict_tmp.copy()
        config_to_file.pop('to_yaml', None)
        config_to_file.pop('from_yaml', None)
        write_to_yaml(config_to_file, yaml_file)

    return config_dict_tmp


if __name__ == '__main__':
    """ This is executed when run from the command line """
    config_dict_ = parse_args(sys.argv[1:])
    to_yaml_ = config_dict_.get('to_yaml', None)
    dataset_ = config_dict_.get('dataset', None)

    if to_yaml_ is None:
        if dataset_ == "slide_seq_testis":
            datamodule_ = SlideSeqTestisDM(**config_dict_)
        elif dataset_ == "slide_seq_kidney":
            datamodule_ = SlideSeqKidneyDM(**config_dict_)
        elif dataset_ == "dummy_dm":
            datamodule_ = DummyDM(**config_dict_)
        else:
            raise Exception("Invalid dataset {0}".format(dataset_))

        # Run the simulation with all the command-line params passed as a dictionary
        run_simulation(config_dict=config_dict_, datamodule=datamodule_)
