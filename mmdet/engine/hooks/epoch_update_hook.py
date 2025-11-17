from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class KDWeightUpdateHook(Hook):
    """动态更新知识蒸馏权重的Hook"""

    def __init__(self, update_interval=1, strategy='adaptive'):
        self.update_interval = update_interval
        self.strategy = strategy
        self.best_map = 0.0
        self.patience = 0
        self.max_patience = 5

    def before_train_epoch(self, runner):
        """在每个epoch开始前更新KD权重"""
        current_epoch = runner.epoch

        # 更新模型中的KD权重
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model

        if hasattr(model, 'update_kd_weights'):
            kd_weights = model.update_kd_weights(current_epoch, self.strategy)

            # 记录日志
            if current_epoch % 5 == 0:
                runner.logger.info(f'Epoch {current_epoch}: KD Weights - {kd_weights}')


@HOOKS.register_module()
class GradientCoordinatorHook(Hook):
    """梯度协调Hook，防止蒸馏损失主导训练"""

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """在每次迭代后检查梯度平衡"""
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model

        if hasattr(model, 'check_gradient_balance'):
            model.check_gradient_balance(runner.logger)