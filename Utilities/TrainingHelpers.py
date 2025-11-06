from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from Models.Timer import Timer
from Models.RunningAverage import RunningAverage
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Utilities.Logger import Logger
from Utilities.ModelAttributesManager import ModelAttributesManager
from Utilities.PredictionHelpers import PredictionHelpers
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch

class TrainingHelpers:
    @staticmethod
    def RunEpoch(
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: Optimizer, 
        loss_fn: nn.Module, 
        metrics: list, 
        epoch: int, 
        configurations: TrainingDataConfigurations, 
        writer = SummaryWriter
    ):
        # 1. Run the single epoch
        model.train()

        # 2. Init the input image patches normalizers
        inp_sub = torch.FloatTensor(configurations.input_nomrlizer_range.sub).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(configurations.input_nomrlizer_range.div).view(1, -1, 1, 1).cuda()

        # 3. Calculate steps for the current epoch
        total_steps = int(configurations.total_examples / configurations.batch_size * configurations.repeat)
        step = 0

        runningAvg = RunningAverage()
        # 4. Process loss for each batch
        for batch in tqdm(train_loader, leave=False, desc='train'):
            # 4.1. convert all items to cuda
            for k, v in batch.items():
                batch[k] = v.cuda()

            # 4.2. Normalize the input image patch and get prediction from the model
            inp = (batch['inp'] - inp_sub) / inp_div
            pred = model(inp, batch['coord'], batch['cell'])

            # 4.3. Normalize the gt and calculate loss and metrics
            gt = (batch['gt'] - inp_sub) / inp_div
            loss = loss_fn(pred, gt)

            evals = {i.__name__: i(pred, gt) for i in metrics}

            # 4.4. Log the epoch-step, loss, and metric
            evals['loss'] = loss
            Logger.LogSummaryWriter(writer, 'train', evals, epoch, step, total_steps)
            step += 1

            # 4.5. Accumulate the loss for the step
            runningAvg.SetItem(loss.item())
            
            # 4.6. Back propogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = None
            loss = None
            evals = None

        # 5. Return the accumulated loss
        return runningAvg.GetItem()

    @staticmethod
    def Validation(
        model, 
        pipeline: BaseTrainingPipeline, 
        log_info: list, 
        epoch: int, 
        max_validation_metric: float
    ):
        validation_metrics: dict = PredictionHelpers.EvaluateForTrainigData(
            pipeline.validation_data_loader, 
            model, 
            pipeline.configurations.validation_data_configurations.input_nomrlizer_range, 
            pipeline.configurations.validation_data_configurations.eval_batch_size, 
            pipeline.configurations.validation_data_configurations.eval_scale, 
            pipeline.configurations.validation_data_configurations.benchmark_type
        )

        for key, value in validation_metrics.items():
            log_info.append('Validation ' + key + '= {:.4f}'.format(value))
        if validation_metrics[pipeline.configurations.monitor_metric] >= max_validation_metric:
            ModelAttributesManager.SaveModel(
                model, 
                pipeline.optimizer, 
                epoch, 
                pipeline.configurations.save_path, 
                'best'
            )

    @staticmethod
    def LogLoadingInformation(start_epoch: int, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, logger: Logger.Log):
        # Set that start to 1 if fresh start, else add +1 if resuming        
        logger('Current epoch: ', start_epoch)
        # Print LR(s)
        for i, group in enumerate(optimizer.param_groups):
            logger(f"Param group {i} -> current LR: {group['lr']} | initial LR: {group.get('initial_lr', 'N/A')}")

        # Print decay info
        if hasattr(lr_scheduler, "gamma"):
            logger(f"Decay factor (gamma): {lr_scheduler.gamma}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        logger(f"Trainable parameters: {trainable_params:,}")
        logger(f"Non-trainable parameters: {non_trainable_params:,}")
        logger(f"Total parameters: {trainable_params + non_trainable_params:,}")

    @staticmethod
    def Train(
        pipeline: BaseTrainingPipeline, 
        logger: Logger.Log,
        writer: SummaryWriter, 
        n_gpus: int, 
        allow_multi_gpu: bool = True
    ):
        # 1. Check if multiple GPUs can be used for training
        if n_gpus > 1 and allow_multi_gpu:
            pipeline.model = nn.parallel.DataParallel(pipeline.model)

        # 2. Init the timer
        timer = Timer()
        max_validation_metric = -1e18

        logger('Last completed epoch: ', pipeline.start_epoch)
        pipeline.start_epoch = 1 if pipeline.start_epoch == -1 else pipeline.start_epoch + 1

        TrainingHelpers.LogLoadingInformation(pipeline.start_epoch, pipeline.model, pipeline.optimizer, pipeline.lr_scheduler, logger)

        # 3. Train the model for the required epochs
        for epoch in range(pipeline.start_epoch, pipeline.configurations.epochs + 1):
            # 3.1. Start logging the epoch
            epoch_start = timer.Elapsed()
            log_info = ['epoch {}/{}'.format(epoch, pipeline.configurations.epochs)]

            writer.add_scalar('lr', pipeline.optimizer.param_groups[0]['lr'], epoch)

            # 3.2. Run the training steps for the epoch, and get loss
            loss = TrainingHelpers.RunEpoch(
                pipeline.model, 
                pipeline.training_data_loader, 
                pipeline.optimizer, 
                pipeline.loss, 
                pipeline.metrics, 
                epoch, 
                pipeline.configurations.data_configurations, 
                writer
            )

            # 3.3. Adjust the learning rate
            pipeline.lr_scheduler.step()

            # 3.4. Log loss and lr info
            log_info.append('train: loss={:.4f} lr={:.4f}'.format(loss, pipeline.optimizer.param_groups[0]['lr']))

            # 3.5. Check if the model was paralellized across multiple gpus
            if n_gpus > 1 and allow_multi_gpu:
                model_ = pipeline.model.module
            else:
                model_ = pipeline.model

            # 3.6. Save the current epoch model
            ModelAttributesManager.SaveModel(
                model_, 
                pipeline.optimizer, 
                epoch, 
                pipeline.configurations.save_path, 
                'last'
            )

            # 3.7. Save model if required for this epoch
            if epoch % pipeline.configurations.epoch_save == 0:
                ModelAttributesManager.SaveModel(
                    model_, 
                    pipeline.optimizer, 
                    epoch, 
                    pipeline.configurations.save_path, 
                    'epoch_' + str(epoch)
                )

            # 3.8. Incase validation needs to be run for this epoch
            if epoch % pipeline.configurations.epoch_val == 0:
                TrainingHelpers.Validation(
                    model_, 
                    pipeline, 
                    log_info, 
                    epoch, 
                    max_validation_metric
                )

            # 3.9. Print Epoch time, Total time spent so far, and time left for training completion
            progress = (epoch - pipeline.start_epoch + 1) / (pipeline.configurations.epochs - pipeline.start_epoch + 1)
            elapsed_total = timer.Elapsed()
            log_info.append('{} {}/{}'.format(
                    Timer.ConvertTimeToText(elapsed_total - epoch_start), 
                    Timer.ConvertTimeToText(elapsed_total),
                    Timer.ConvertTimeToText(elapsed_total / progress)
                )
            )

            logger(', '.join(log_info))
            writer.flush()

        