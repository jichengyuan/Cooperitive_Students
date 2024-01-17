import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from coStudents import add_teacher_config
from coStudents.engine.trainer_cos import CoSTrainer, BaselineTrainer

from coStudents.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
from coStudents.modeling.proposal_generator.rpn import PseudoLabRPN
from coStudents.modeling.roi_heads.roi_head import StandardROIHeadsPseudoLab
import coStudents.data.datasets.builtin

from coStudents.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from coStudents.engine.hooks import BestCheckpointer


def setup_config(args):
    """
    Sets up the configuration from arguments.
    """
    cfg = get_cfg()
    add_teacher_config(cfg)
    print("Config File:", args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def select_trainer(cfg):
    """
    Selects the trainer based on configuration.
    """
    if cfg.SEMISUPNET.Trainer == "studentteacher":
        return CoSTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        return BaselineTrainer
    else:
        raise ValueError(f"Unsupported Trainer: {cfg.SEMISUPNET.Trainer}")


def evaluate_model(cfg, args, Trainer):
    """
    Evaluates the model.
    """
    if cfg.SEMISUPNET.Trainer == "studentteacher":
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensemble_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(ensemble_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, ensemble_model.modelTeacher)
    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, model)


def main(args):
    cfg = setup_config(args)
    Trainer = select_trainer(cfg)

    if args.eval_only:
        return evaluate_model(cfg, args, Trainer)

    trainer = Trainer(cfg)
    trainer.register_hooks([BestCheckpointer()])
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
