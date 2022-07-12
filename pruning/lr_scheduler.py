from torch.optim.lr_scheduler import LambdaL


def exp_lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=40, minimum_lr = 0.000001):
	"""Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
	lr = init_lr * (0.1**(epoch // lr_decay_epoch))
	if lr < minimum_lr:
		lr = minimum_lr
	if epoch % lr_decay_epoch == 0:
		print('LR is set to {}'.format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer, lr
def adaptive_lr_scheduler(optimizer, epoch, cur_loss, init_lr=0.1, lr_decay_epoch=40, minimum_lr = 0.002):
	"""Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
	if not hasattr(adaptive_lr_scheduler,"lately_losses"):
		adaptive_lr_scheduler.lately_losses = []
	adaptive_lr_scheduler.lately_losses.append()

	lr = init_lr * (0.5**(epoch // lr_decay_epoch))
	if lr < minimum_lr:
		lr = minimum_lr
	if epoch % lr_decay_epoch == 0:
		print('LR is set to {}'.format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer
